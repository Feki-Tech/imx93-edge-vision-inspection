#!/bin/sh
# Compile the quantized TFLite model for the Ethos-U65 NPU using the Vela compiler.
#   pip install ethos-u-vela
set -e
MODEL=${1:-models/defect_int8.tflite}
vela "$MODEL" \
  --accelerator-config ethos-u65-256 \
  --system-config Ethos_U65_High_End \
  --memory-mode Dedicated_Sram \
  --output-dir models/
echo "Vela-compiled model written to models/"
