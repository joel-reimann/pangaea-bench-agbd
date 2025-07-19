#!/bin/bash

# Test script for PANGAEA-bench SLURM Manager v3.0
# This script validates the new features and ensures backward compatibility

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGER_V3="$SCRIPT_DIR/pangaea_slurm_manager_v3.sh"
TEST_OUTPUT_DIR="/tmp/pangaea_test_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Testing PANGAEA-bench SLURM Manager v3.0"
echo "================================================"
echo "Test output: $TEST_OUTPUT_DIR"
echo ""

# Create test output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Function to run test and capture output
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"
    
    echo "Testing: $test_name"
    echo "Command: $command"
    
    if eval "$command" > "$TEST_OUTPUT_DIR/${test_name}.log" 2>&1; then
        if grep -q "$expected_pattern" "$TEST_OUTPUT_DIR/${test_name}.log"; then
            echo "✅ PASS: $test_name"
        else
            echo "❌ FAIL: $test_name (pattern not found: $expected_pattern)"
            echo "Output:"
            cat "$TEST_OUTPUT_DIR/${test_name}.log" | head -10
        fi
    else
        echo "❌ FAIL: $test_name (command failed)"
        echo "Output:"
        cat "$TEST_OUTPUT_DIR/${test_name}.log" | head -10
    fi
    echo ""
}

# Test 1: Help message
run_test "help_message" \
    "$MANAGER_V3 help" \
    "PANGAEA-bench SLURM Manager v3.0"

# Test 2: List models
run_test "list_models" \
    "$MANAGER_V3 list-models" \
    "Available Models in PANGAEA-bench"

# Test 3: Generate minimal debug job
run_test "generate_minimal" \
    "$MANAGER_V3 generate -d -e minimal -o $TEST_OUTPUT_DIR" \
    "Generated.*job files"

# Test 4: Generate standard job with specific models
run_test "generate_standard" \
    "$MANAGER_V3 generate -m satmae_base,dofa -e debug -o $TEST_OUTPUT_DIR" \
    "Generated.*job files"

# Test 5: Validate experiment types
run_test "invalid_experiment" \
    "$MANAGER_V3 generate -d -e invalid_type -o $TEST_OUTPUT_DIR" \
    "Invalid experiment type"

# Test 6: Test fine-grained control
run_test "fine_grained_control" \
    "$MANAGER_V3 generate -m satmae_base --vis-interval 100 --batch-size 64 -e minimal -o $TEST_OUTPUT_DIR" \
    "Generated.*job files"

echo "================================================"
echo "Test Summary"
echo "================================================"

# Check if any SLURM files were generated
if find "$TEST_OUTPUT_DIR" -name "*.slurm" | head -5; then
    echo ""
    echo "Sample generated SLURM job:"
    echo "=========================="
    slurm_file=$(find "$TEST_OUTPUT_DIR" -name "*.slurm" | head -1)
    if [[ -n "$slurm_file" ]]; then
        echo "File: $(basename "$slurm_file")"
        echo ""
        # Show key sections of the SLURM file
        grep -A 5 -B 2 "PANGAEA-bench SLURM Job v3.0" "$slurm_file" || true
        echo "..."
        grep -A 10 "CODEBASE SYNCHRONIZATION" "$slurm_file" || true
        echo "..."
        grep -A 5 "torchrun\|python pangaea/run.py" "$slurm_file" || true
    fi
fi

echo ""
echo "All test logs available in: $TEST_OUTPUT_DIR"
echo "To test actual submission (requires SLURM):"
echo "  $MANAGER_V3 submit -d -e minimal"
echo ""
echo "To test with custom paths:"
echo "  $MANAGER_V3 generate -d -e minimal \\"
echo "    -o /custom/output \\"
echo "    -p /custom/pangaea-bench \\"
echo "    -v /custom/venv"
