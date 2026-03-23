# 🏠 GenAI Interior Editor

A **multi-modal generative AI system** that enables intelligent interior image editing using natural language prompts.

This project combines **computer vision, diffusion models, and agent-based orchestration** to perform both **local object edits** and **global room transformations**.

---

## 🚀 Features

* 🧠 **LLM-based Planning**

  * Understands user prompts
  * Decides between local vs global edits

* 🎯 **Local Object Editing**

  * Detect room decor objects (YOLO)
  * Segment regions (SAM)
  * Inpaint using diffusion

* 🏠 **Global Room Styling**

  * Whole-scene transformation
  * Controlled using depth & structure

* 🔁 **Iterative Editing (Conversation-like)**

  * Supports multiple edits on the same image

* ✅ **AI Quality Validation**

  * Semantic alignment (CLIP)
  * Structural consistency (edges)
  * Mask leakage detection
  * Depth consistency (for global edits)

* 🔄 **Automatic Retry System**

  * Regenerates output if quality is low

---

## 🧠 System Architecture

```
User Prompt
     ↓
Planner Node
     ↓
Router
     ↓
Editor Tool (Local / Global)
     ↓
Quality Checker
     ↓
Retry OR Final Output
```

---

## 🧩 Tech Stack

* **Diffusion Models** – Stable Diffusion + ControlNet
* **Object Detection** – YOLO (Ultralytics) - custom trained on homeobjects-3K dataset that detects 11 objects.
* **Segmentation** – SAM2
* **Depth Estimation** – MiDaS
* **Line Detection** – MLSD
* **Validation** – CLIP + custom metrics
* **Agent Framework** – LangGraph

---

## 📁 Project Structure

```
genai-interior-editor/

agent/

   ├── agent_type.py        #agent states
   ├── agent_graph.py      # LangGraph nodes + graph
   ├── quality_checker.py   #quality nodes
   └── agent_tools.py      # Tool wrappers

core/
   ├── config_loader.py
   ├── model_registry.py 
   ├── prompt_loader.py 
   └── logger.py           # Global logging system

vision/
   ├── decor_detector.py 
   ├── depth.py 
   ├── edges.py 
   └── vision_models.py    # All model loaders

editors/
   ├── local_editor.py
   └── global_editor.py

config/
   ├── prompts #contains different prompts versions
      ├── v1.yaml
      ├── v2.yaml  
   ├── model_config.yaml
   └── pipeline_config.yaml
utils.py
requirements.txt
```

---

## 🧠 Key Concepts

### Local Editing Pipeline

```
Image → YOLO → SAM → Mask → Inpainting
```

---

### Global Editing Pipeline

```
Image → Depth + Edges → ControlNet → Diffusion
```

---

### Agent Behavior

* Parses user intent
* Selects appropriate tool
* Executes edit
* Validates output
* Retries if needed

---

## 🏆 Highlights

* Multi-model vision pipeline
* Agent-based orchestration
* Automated quality control
* Modular and extensible design
* Supports iterative editing

---

## 🔮 Future Improvements

* Prompt-based segmentation (Grounding models)
* Perspective consistency validation
* FastAPI deployment
* UI for interactive editing
* Parallel processing for vision modules


