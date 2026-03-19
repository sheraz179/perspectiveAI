from typing import TypedDict, List

class AgentState(TypedDict):

    prompt: str

    conversation_history: List[str]
    original_image_path: str
    image_path: str
    current_image_path: str
    previous_image_path: str
    last_mask: str
    depth_generated: str
    depth_original: str
    quality_score: int

    source:str
    intent: str
    global_prompt: str
    strength: float
    objects: List[dict]
    reply: str
    retry_count:int
    seed: int
    message_index: int