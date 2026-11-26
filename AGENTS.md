# Project intent: DroneDetect

DroneDetect classifies drone RF signals from raw IQ data. Its checkpoint
format, label order, and classifier output dimension must stay aligned.

- Preserve training data, checkpoints, and existing scan outputs. Do not
  delete or rewrite them during routine code review.
- Validate the checkpoint format and class count before changing model load
  logic. Never claim a model loads correctly from syntax checks alone.
- Use CPU fixtures for small tests when possible. GPU training and real RF
  capture require separate, explicit validation.
- Record which checkpoint was tested, its class count, and the exact command.
  If dependencies or weights are absent, mark runtime behavior unverified.
