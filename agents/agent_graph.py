import json

from PIL import Image
from core.prompt_loader import get_system_prompt
from agents.agent_tools import local_editor_node, global_editor_node
from agents.agent_types import AgentState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from core.config_loader import ConfigLoader

def quality_checker_node(state:AgentState, model_registry):

    original = Image.open(state["previous_image_path"]).convert('RGB')
    edited = Image.open(state["current_image_path"]).convert('RGB')

    if state["intent"] == "local_edit":

        mask = state["last_mask"]
        prompt = state["prompt"]
        scores = model_registry.validator.evaluate_local(
            original,
            edited,
            mask,
            prompt
        )

    else:

        prompt = state["prompt"]
        scores = model_registry.validator.evaluate_global(
            original,
            edited,
            state["depth_original"],
            state["depth_generated"],
            prompt
        )

    state["quality_score"] = scores #float(scores['final'])

    return state

def planner_node(state, model_registry):
    history = "\n".join(state.get("conversation_history", []))
    original = state.get("original_image_path")
    current = state.get("current_image_path", original)
    SYSTEM_PROMPT = get_system_prompt()    

    prompt = f"""
        {SYSTEM_PROMPT}

        Context:
        - Original Base Image: {original}
        - Your Current Progress: {current}
        - Past Requests: {history}

        User request: {state['prompt']}
    """

    response = model_registry.llm.invoke(prompt)
    text = response.content
    print(text)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        plan = json.loads(text[start:end])
    except:
        try:
            plan = json.loads(text)
        except:
            plan = {"source": "last_edit", "intent": "global_edit", "global_prompt": state['prompt'], "strength": 0.5}

    source_choice = plan.get("source", "last_edit")
    state["image_path"] = original if source_choice == "original" else current
    state["intent"] = plan.get("intent", "global_edit").lower()
    state["global_prompt"] = plan.get("global_prompt", "")
    state["objects"] = plan.get("objects", [])
    state["strength"] = max(plan.get("strength", 0.4), 0.4)

    state["reply"] = plan.get("reply", "Configuring the edit...")

    print(f"Planner chose source: {source_choice} with intent: {state['intent']} and strength: {state['strength']}")
    print('AI reply: ', state['reply'])
    print('state', state)
    return state

def update_history_node(state):
    if "conversation_history" not in state: state["conversation_history"] = []
    state["conversation_history"].append(state["prompt"])
    if state.get('output_image_path'): state['current_image_path'] = state['output_image_path']
    state['retry_count'] = 0

    return state

def retry_logic_node(state: AgentState):
    """
    Increments the retry counter and prepares the state for a re-planning turn.
    """
    retry_count = state.get("retry_count", 0)
    state["retry_count"] = retry_count + 1
    
    # Revert current_image_path to previous to ensure we don't build on a failed edit
    state['current_image_path'] = state.get('previous_image_path', state.get('original_image_path'))
    
    print(f"Retry Logic Node: Incrementing retry count to {state['retry_count']}. Resetting image path.")
    return state

def router(state: AgentState):

    if state["intent"] == "local_edit":
        return "local_editor"

    return "global_editor"

def quality_router(state: AgentState):
    """
    Routes to either 'update_history' if quality is met or retries exhausted,
    otherwise routes to the 'retry_logic' node.
    """

    config = ConfigLoader("config/pipeline_config.yaml").get('pipeline')
    QUALITY_THRESHOLD = config['quality_threshold']
    MAX_RETRIES = config['retry_limit']
    
    quality_score = state.get("quality_score", 0)
    retry_count = state.get("retry_count", 0)

    print(f"Quality Router Check: Score = {quality_score:.2f}, Current Retry = {retry_count}")

    if quality_score >= QUALITY_THRESHOLD or retry_count >= MAX_RETRIES:
        return "update_history"

    return "retry_logic"

def retry_logic_node(state: AgentState):
    """
    Increments the retry counter and prepares the state for a re-planning turn.
    """
    retry_count = state.get("retry_count", 0)
    state["retry_count"] = retry_count + 1
    
    # Revert current_image_path to previous to ensure we don't build on a failed edit
    state['current_image_path'] = state.get('previous_image_path', state.get('original_image_path'))
    
    print(f"Retry Logic Node: Incrementing retry count to {state['retry_count']}. Resetting image path.")
    return state

def build_graph(model_registry):

    builder = StateGraph(AgentState)

    # 3. Add all necessary nodes
    builder.add_node("planner", lambda state: planner_node(state, model_registry))
    builder.add_node("local_editor", lambda state: local_editor_node(state, model_registry) )
    builder.add_node("global_editor", lambda state: global_editor_node(state, model_registry))
    builder.add_node("quality_checker", lambda state: quality_checker_node(state, model_registry))
    builder.add_node("update_history", update_history_node)

    # 4. Set the entry point
    builder.set_entry_point("planner")
    # 5. Add existing conditional edges from planner to editors
    builder.add_conditional_edges(
        "planner",
        router,
        {
            "local_editor": "local_editor",
            "global_editor": "global_editor",
        }
    )

    # 6. Add regular edges from editors to quality_checker
    builder.add_edge("local_editor", "quality_checker")
    builder.add_edge("global_editor", "quality_checker")

    # 7. Add conditional edge from quality_checker using quality_router
    builder.add_conditional_edges(
        "quality_checker",
        quality_router,
        {
            "planner": "planner",
            "update_history": "update_history",
        }
    )
    # 8. Set finish point
    builder.set_finish_point("update_history")

    # 9. Compile the graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph