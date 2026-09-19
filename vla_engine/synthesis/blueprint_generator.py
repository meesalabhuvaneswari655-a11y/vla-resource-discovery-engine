"""Synthesizes ranked components into an end-to-end VLA training blueprint."""

from datetime import datetime
from typing import Dict, List, Optional
from vla_engine.evaluation.compatibility import CompatibilityEngine
from vla_engine.models.blueprint import PipelineStep, VLABlueprint
from vla_engine.models.query import ParsedQuery
from vla_engine.models.resource import Category, Resource


class BlueprintGenerator:
    """Generates an actionable, step-by-step VLA training blueprint from ranked components."""

    def __init__(self, compatibility_engine: Optional[CompatibilityEngine] = None):
        self.compatibility_engine = compatibility_engine or CompatibilityEngine()

    def generate(
        self,
        ranked_categories: Dict[Category, List[Resource]],
        query: ParsedQuery,
    ) -> VLABlueprint:
        """Create a complete VLABlueprint from ranked resources."""
        top_stack: Dict[Category, Resource] = {}

        # Pick the top candidate for each available category
        for cat, resources in ranked_categories.items():
            if resources:
                top_stack[cat] = resources[0]

        # Stack evaluation
        eval_result = self.compatibility_engine.evaluate_stack(top_stack)
        compatibility_score = eval_result["score"]
        compatibility_verdict = eval_result["verdict"]

        # Build pipeline steps
        steps = self._build_pipeline_steps(top_stack, query)

        # Consolidate hardware summary
        hw_summary = self._consolidate_hardware(top_stack)

        return VLABlueprint(
            query=query,
            top_stack=top_stack,
            compatibility_score=compatibility_score,
            compatibility_verdict=compatibility_verdict,
            pipeline_steps=steps,
            hardware_summary=hw_summary,
            all_ranked_resources=ranked_categories,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    def _build_pipeline_steps(
        self,
        top_stack: Dict[Category, Resource],
        query: ParsedQuery,
    ) -> List[PipelineStep]:
        vlm = top_stack.get(Category.VLM_BACKBONE)
        vla = top_stack.get(Category.VLA_POLICY)
        dataset = top_stack.get(Category.ROBOTICS_DATASET)
        sim = top_stack.get(Category.SIMULATION)
        framework = top_stack.get(Category.TRAINING_FRAMEWORK)

        vlm_name = vlm.name if vlm else "SigLIP / DINOv2 Backbone"
        vla_name = vla.name if vla else "OpenVLA (7B)"
        dataset_name = dataset.name if dataset else "Open X-Embodiment"
        sim_name = sim.name if sim else "ManiSkill 3 / SIMPLER"
        framework_name = framework.name if framework else "LeRobot / OpenVLA Codebase"

        tasks_str = ", ".join(query.target_tasks) if query.target_tasks else "object manipulation"

        return [
            PipelineStep(
                step_number=1,
                title="Vision-Language Representation Initialization & Grounding",
                description=(
                    f"Initialize the multi-modal transformer backbone using pre-trained weights from {vlm_name}. "
                    "Extract spatial patch tokens and project visual features into the language embedding space, "
                    "ensuring high-resolution visual grounding for fine-grained object manipulation."
                ),
                recommended_tools=[vlm_name, "Hugging Face Transformers", "PyTorch"],
                inputs="High-resolution RGB camera streams + Natural language prompt embeddings",
                outputs="Pre-aligned visual-text token embeddings ready for action discretization or flow matching",
                hardware_notes="1x NVIDIA GPU with >= 24GB VRAM (e.g. RTX 4090 or A100)",
            ),
            PipelineStep(
                step_number=2,
                title="Trajectory Ingestion & Policy Co-Fine-Tuning",
                description=(
                    f"Stream multi-embodiment demonstrations from {dataset_name} using RLDS or Zarr data loaders. "
                    f"Fine-tune {vla_name} using {framework_name}. Employ Parameter-Efficient Fine-Tuning (PEFT / LoRA) "
                    "or FSDP to adapt action tokens (Cartesian delta poses + gripper state) while preserving general knowledge."
                ),
                recommended_tools=[framework_name, dataset_name, vla_name, "PyTorch FSDP / PEFT"],
                inputs=f"Multi-view robot trajectories (RGB + Proprioception) conditioned on instructions for '{tasks_str}'",
                outputs="Trained checkpoint weights with action chunking / auto-regressive action generation",
                hardware_notes=(
                    framework.hardware_requirements.get("training", "1-8x NVIDIA A100/H100 or 1x RTX 4090 with LoRA")
                    if framework else "1x RTX 4090 for LoRA fine-tuning"
                ),
            ),
            PipelineStep(
                step_number=3,
                title="Simulation Validation & Sim-to-Real Benchmark Rollout",
                description=(
                    f"Deploy policy checkpoints into {sim_name} for zero-shot and domain-randomized policy rollout. "
                    "Evaluate success rates across pick-and-place, articulated drawer opening, and unseen language instructions. "
                    "Calculate Pearson correlation with physical benchmarks to prevent sim-to-real divergence."
                ),
                recommended_tools=[sim_name, vla_name, "Vulkan / PhysX GPU Backend"],
                inputs="Policy checkpoint weights + Standardized evaluation task environments",
                outputs="Task success rate metrics (SR%), trajectory completion time, and safety violation logs",
                hardware_notes=sim.hardware_requirements.get("gpu", "NVIDIA GPU with Vulkan ray tracing support") if sim else "1x RTX 3080+",
            ),
            PipelineStep(
                step_number=4,
                title="Physical Hardware Deployment & Edge Inference",
                description=(
                    f"Export fine-tuned policy to ONNX or TensorRT-LLM for low-latency edge inference on the physical robot. "
                    f"Interface with robot controllers at 10Hz-50Hz using {framework_name}'s hardware connectors. "
                    "Deploy closed-loop safety filters with gripper force feedback and trajectory smoothing."
                ),
                recommended_tools=[framework_name, "ROS 2 / gRPC", "TensorRT / bitsandbytes"],
                inputs="Real-time camera feed (30 FPS) + Operator natural-language instructions",
                outputs="End-effector delta position commands (x, y, z, roll, pitch, yaw, gripper state)",
                hardware_notes=(
                    vla.hardware_requirements.get("inference", "1x NVIDIA RTX 3090/4090 or Jetson Orin with quantization")
                    if vla else "1x RTX 3080/4090"
                ),
            ),
        ]

    def _consolidate_hardware(self, top_stack: Dict[Category, Resource]) -> Dict[str, str]:
        """Summarize hardware recommendations across the selected stack."""
        summary = {
            "Minimum Training Rig": "1x NVIDIA RTX 3090 / 4090 (24GB VRAM) for LoRA fine-tuning",
            "Recommended Production Cluster": "4x or 8x NVIDIA A100 / H100 (80GB) with NVLink & InfiniBand",
            "Minimum Storage": "500 GB NVMe SSD for demonstration trajectory subsets",
            "Simulation Environment": "NVIDIA GPU supporting Vulkan / PhysX 5 for 30,000+ FPS parallel rollout",
            "Edge Deployment": "NVIDIA Jetson AGX Orin (64GB) or local workstation with 1x RTX 4070/4080 (int4/bfloat16)",
        }

        # Enrich from actual resources
        vla = top_stack.get(Category.VLA_POLICY)
        if vla and vla.hardware_requirements.get("training"):
            summary["Policy Training Specifics"] = vla.hardware_requirements["training"]

        return summary
