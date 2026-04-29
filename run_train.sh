#!/bin/bash

# ----------------------------------------
# GPU Training Script (Interactive + Safe)
# ----------------------------------------
# Usage:
#   ./run_train.sh [GPU_IDs]
#
# Examples:
#   ./run_train.sh 1        -> use GPU 1
#   ./run_train.sh 0        -> use GPU 0
#   ./run_train.sh 0,1      -> use both GPUs
#
# If no argument is provided, user will be prompted.
# Script ALWAYS asks for confirmation before running.
# ----------------------------------------

# Function to ask GPU input
get_gpu_input() {
    echo ""
    echo "Available GPU options:"
    echo "  0     -> GPU 0"
    echo "  1     -> GPU 1"
    echo "  0,1   -> Both GPUs"
    echo ""

    read -p "Enter GPU ID(s): " GPU_IDS
}

# Step 1: Check if argument provided
if [ -z "$1" ]; then
    echo "No GPU ID provided."
    get_gpu_input
else
    GPU_IDS=$1
fi

# Step 2: Confirmation loop
while true; do
    echo ""
    echo "Selected GPU(s): $GPU_IDS"

    # Dynamic warning
    echo "⚠️ WARNING: Make sure GPU(s) [$GPU_IDS] are allowed before proceeding!"

    read -p "Confirm and proceed? (Y/N): " CONFIRM

    case $CONFIRM in
        [Yy]* )
            break
            ;;
        [Nn]* )
            echo ""
            echo "Re-select GPU..."
            get_gpu_input
            ;;
        * )
            echo "Please enter Y or N."
            ;;
    esac
done

# Step 3: Set environment
export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo ""
echo "----------------------------------------"
echo "Final GPU selection: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"

# Optional sanity check
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Visible devices:', torch.cuda.device_count())"

# Step 4: Run training
echo ""
echo "Starting training..."
START_TIME=$(date +%s)

# provide the model to train - check script/train.py
# train.py --lang {de,fr,es,it,sv,multi}
python -m scripts.train --lang multi
STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""

if [ $STATUS -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "⏱️ Time taken: ${DURATION} seconds"
else
    echo "❌ Training failed. Please check logs/errors."
    echo "⏱️ Ran for: ${DURATION} seconds"
fi