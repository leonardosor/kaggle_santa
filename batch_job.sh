#!/bin/bash

################################################################################
# GPU-Accelerated Tree Packing Batch Job
# 
# This script runs GPU-accelerated optimization with automatic environment
# setup and result management. Works on any system with NVIDIA GPUs.
#
# Usage:
#   bash batch_job.sh [num_trees] [num_gpus]
#
# Examples:
#   bash batch_job.sh              # Default: 200 trees, auto-detect GPUs
#   bash batch_job.sh 100          # 100 trees, auto-detect GPUs
#   bash batch_job.sh 200 1        # 200 trees, force single GPU
#   bash batch_job.sh 50 2         # 50 trees, use 2 GPUs
################################################################################

# Configuration
NUM_TREES=${1:-200}              # Default to 200 trees
NUM_GPUS=${2:-"auto"}            # Auto-detect GPUs or use specified number
MAX_TIME_HOURS=4                 # Maximum runtime in hours
EMAIL_NOTIFICATIONS="false"      # Set to "true" to enable email alerts
EMAIL_ADDRESS="your.email@domain.com"

# Derived settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_ID="job_${TIMESTAMP}"
LOG_DIR="${SCRIPT_DIR}/logs"
OUTPUT_DIR="${SCRIPT_DIR}/output"

################################################################################
# Setup and Validation
################################################################################

echo "========================================================================"
echo "GPU-Accelerated Santa 2025 Tree Packing"
echo "========================================================================"
echo "Job ID:        ${JOB_ID}"
echo "Start Time:    $(date)"
echo "Num Trees:     ${NUM_TREES}"
echo "Requested GPUs: ${NUM_GPUS}"
echo "Working Dir:   ${SCRIPT_DIR}"
echo "========================================================================"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"

# Log file paths
STDOUT_LOG="${LOG_DIR}/${JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/${JOB_ID}.err"
COMBINED_LOG="${LOG_DIR}/${JOB_ID}.log"

# Redirect output to log files
exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

################################################################################
# GPU Detection and Validation
################################################################################

echo ""
echo "[SETUP] Detecting GPUs..."

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. CUDA drivers may not be installed."
    exit 1
fi

# Detect available GPUs
DETECTED_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "[INFO] Detected ${DETECTED_GPUS} GPU(s)"

# Auto-detect or validate GPU count
if [ "${NUM_GPUS}" = "auto" ]; then
    NUM_GPUS=${DETECTED_GPUS}
    echo "[INFO] Auto-selected ${NUM_GPUS} GPU(s)"
elif [ ${NUM_GPUS} -gt ${DETECTED_GPUS} ]; then
    echo "[WARN] Requested ${NUM_GPUS} GPUs but only ${DETECTED_GPUS} available"
    NUM_GPUS=${DETECTED_GPUS}
    echo "[INFO] Using ${NUM_GPUS} GPU(s)"
fi

# Display GPU information
echo ""
echo "[GPU] Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Set CUDA device visibility
if [ ${NUM_GPUS} -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "[INFO] Using GPU 0 only"
elif [ ${NUM_GPUS} -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[INFO] Using GPUs 0,1"
else
    echo "[INFO] Using all ${NUM_GPUS} GPUs"
fi

################################################################################
# Python Environment Setup
################################################################################

echo ""
echo "[SETUP] Configuring Python environment..."

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "[ERROR] Python not found"
    exit 1
fi

echo "[INFO] Using: $($PYTHON_CMD --version)"

# Check if in conda environment
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "[INFO] Conda environment: ${CONDA_DEFAULT_ENV}"
elif [ ! -z "$VIRTUAL_ENV" ]; then
    echo "[INFO] Virtual environment: ${VIRTUAL_ENV}"
else
    echo "[WARN] No virtual environment detected (using system Python)"
fi

# Verify required packages
echo ""
echo "[VERIFY] Checking Python packages..."

check_package() {
    ${PYTHON_CMD} -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        VERSION=$(${PYTHON_CMD} -c "import $1; print($1.__version__)" 2>/dev/null)
        echo "[OK] $1 ${VERSION}"
        return 0
    else
        echo "[MISSING] $1"
        return 1
    fi
}

MISSING_PACKAGES=()
check_package "cupy" || MISSING_PACKAGES+=("cupy-cuda12x")
check_package "numpy" || MISSING_PACKAGES+=("numpy")
check_package "pandas" || MISSING_PACKAGES+=("pandas")
check_package "shapely" || MISSING_PACKAGES+=("shapely")

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "[INSTALL] Installing missing packages: ${MISSING_PACKAGES[*]}"
    ${PYTHON_CMD} -m pip install --user "${MISSING_PACKAGES[@]}"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Package installation failed"
        exit 1
    fi
fi

# Verify CUDA/CuPy compatibility
echo ""
echo "[VERIFY] Checking CUDA/CuPy compatibility..."
${PYTHON_CMD} -c "
import cupy as cp
try:
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f'[OK] CuPy can access {device_count} GPU(s)')
    for i in range(device_count):
        with cp.cuda.Device(i):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode('utf-8')
            mem_gb = props['totalGlobalMem'] / (1024**3)
            print(f'  GPU {i}: {name} ({mem_gb:.2f} GB)')
except Exception as e:
    print(f'[ERROR] CuPy GPU access failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "[ERROR] CUDA/CuPy verification failed"
    exit 1
fi

################################################################################
# Timeout Handler
################################################################################

# Set up timeout
TIMEOUT_SECONDS=$((MAX_TIME_HOURS * 3600))

timeout_handler() {
    echo ""
    echo "[TIMEOUT] Job exceeded ${MAX_TIME_HOURS} hour limit"
    echo "[INFO] Attempting to save partial results..."
    pkill -TERM -P $$
    sleep 5
    pkill -KILL -P $$
    exit 124
}

################################################################################
# Run Optimization
################################################################################

echo ""
echo "========================================================================"
echo "[RUN] Starting GPU-accelerated tree packing optimization"
echo "========================================================================"
echo "Trees to place: ${NUM_TREES}"
echo "GPUs to use:    ${NUM_GPUS}"
echo "Max runtime:    ${MAX_TIME_HOURS} hours"
echo "========================================================================"
echo ""

# Change to script directory
cd "${SCRIPT_DIR}"

# Record start time
START_TIME=$(date +%s)

# Run with timeout
(
    trap timeout_handler SIGTERM
    
    # Execute the GPU optimization
    ${PYTHON_CMD} src/run_gpu.py \
        --num-trees ${NUM_TREES} \
        --num-gpus ${NUM_GPUS} \
        2>&1 | tee -a "${COMBINED_LOG}"
    
    exit ${PIPESTATUS[0]}
) &

# Get PID
RUN_PID=$!

# Monitor with timeout
(
    sleep ${TIMEOUT_SECONDS}
    if ps -p ${RUN_PID} > /dev/null 2>&1; then
        echo "[TIMEOUT] Killing job after ${MAX_TIME_HOURS} hours"
        kill -TERM ${RUN_PID} 2>/dev/null
        sleep 10
        kill -KILL ${RUN_PID} 2>/dev/null
    fi
) &

TIMEOUT_PID=$!

# Wait for completion
wait ${RUN_PID}
EXIT_CODE=$?

# Kill timeout monitor
kill ${TIMEOUT_PID} 2>/dev/null

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

################################################################################
# Results and Summary
################################################################################

echo ""
echo "========================================================================"
echo "[COMPLETE] Job Finished"
echo "========================================================================"
echo "End Time:      $(date)"
echo "Elapsed Time:  ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "Exit Code:     ${EXIT_CODE}"
echo "========================================================================"

# Check for output files
echo ""
echo "[RESULTS] Generated files:"
if ls output/submission_gpu_*.csv 1> /dev/null 2>&1; then
    LATEST_SUBMISSION=$(ls -t output/submission_gpu_*.csv | head -n 1)
    FILE_SIZE=$(du -h "${LATEST_SUBMISSION}" | cut -f1)
    NUM_LINES=$(wc -l < "${LATEST_SUBMISSION}")
    echo "  Submission: ${LATEST_SUBMISSION}"
    echo "  Size:       ${FILE_SIZE}"
    echo "  Trees:      $((NUM_LINES - 1))"  # Subtract header
else
    echo "  [WARN] No submission files found"
fi

if ls output/scores_gpu_*.csv 1> /dev/null 2>&1; then
    LATEST_SCORES=$(ls -t output/scores_gpu_*.csv | head -n 1)
    echo "  Scores:     ${LATEST_SCORES}"
fi

# GPU final status
echo ""
echo "[GPU] Final GPU status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Generate summary report
SUMMARY_FILE="${LOG_DIR}/${JOB_ID}_summary.txt"
cat > "${SUMMARY_FILE}" << EOF
====================================================================
GPU Tree Packing Job Summary
====================================================================
Job ID:           ${JOB_ID}
Start Time:       $(date -d @${START_TIME} 2>/dev/null || date -r ${START_TIME})
End Time:         $(date -d @${END_TIME} 2>/dev/null || date -r ${END_TIME})
Elapsed Time:     ${ELAPSED_MIN}m ${ELAPSED_SEC}s
Exit Code:        ${EXIT_CODE}
====================================================================
Configuration:
  Trees:          ${NUM_TREES}
  GPUs Used:      ${NUM_GPUS}
  Max Time:       ${MAX_TIME_HOURS} hours
====================================================================
Results:
  Submission:     ${LATEST_SUBMISSION:-"Not found"}
  Scores:         ${LATEST_SCORES:-"Not found"}
====================================================================
Logs:
  Combined:       ${COMBINED_LOG}
  Stdout:         ${STDOUT_LOG}
  Stderr:         ${STDERR_LOG}
====================================================================
EOF

echo "[INFO] Summary saved to: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

# Send email notification if enabled
if [ "${EMAIL_NOTIFICATIONS}" = "true" ] && command -v mail &> /dev/null; then
    echo ""
    echo "[INFO] Sending email notification to ${EMAIL_ADDRESS}"
    mail -s "GPU Job ${JOB_ID} Completed (Exit: ${EXIT_CODE})" "${EMAIL_ADDRESS}" < "${SUMMARY_FILE}"
fi

# Exit with the same code as the optimization
exit ${EXIT_CODE}
