from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os, json
class GradValueClippingCallback(TrainerCallback):
    "A callback that helps deals with initialize memory to save grad value clip results and dump it at the end of training"
    # def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if hasattr(self, "gradclipmemory"):
    #         self.gradClipMemory.clear()
    #     else:
    #         self.gradClipMemory = {}
    #     self.gradClipMemorySavePath = os.path.join(args.output_dir, "gradClipMemoryJsons")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # if (state.global_step + 1) % args.grad_clip_data_save_period == 0:
        # steps_in_epoch = args.max_steps * args.gradient_accumulation_steps
        steps_in_epoch = int((state.global_step + 1) / (state.epoch - int(state.epoch) + 1e-6))
        if (state.global_step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                steps_in_epoch <= args.gradient_accumulation_steps
                and (state.global_step + 1) == steps_in_epoch
        ):
            # self.gradClipMemory["step"] = state.global_step
            if hasattr(state, "gradClipMemory"):
                state.gradClipMemory.clear()
            else:
                state.gradClipMemory = {}
            state.gradClipMemory["step"] = state.global_step + 1


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not os.path.exists(args.gradClipMemorySavePath):
            os.makedirs(args.gradClipMemorySavePath)
        if hasattr(state, "gradClipMemory") and len(state.gradClipMemory) > 0:
            if (state.global_step + 1) % args.grad_clip_data_save_period == 0:
                with open(os.path.join(args.gradClipMemorySavePath, f"status_{state.global_step}.json"), "w", encoding='utf-8') as f_out:
                    json.dump(state.gradClipMemory, f_out)
            state.gradClipMemory.clear()
