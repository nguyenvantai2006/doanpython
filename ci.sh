#!/bin/bash

set -e

echo "🚀 Starting CI pipeline..."

# =========================================
# ENVIRONMENT
# =========================================
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export CI_MODE=1

export HF_HOME=/cache
export TRANSFORMERS_CACHE=/cache
export HF_DATASETS_CACHE=/cache
export TOKENIZERS_PARALLELISM=false

# =========================================
# HF CACHE
# =========================================
export HF_CACHE_DIR=/tmp/hf-cache

mkdir -p "$HF_CACHE_DIR"
mkdir -p reports
mkdir -p models

echo "📦 HF Cache: $HF_CACHE_DIR"

# =========================================
# DETECT CHANGES
# =========================================
echo "🔍 Detecting changes..."

DIFF_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || true)

API_CHANGED=$(echo "$DIFF_FILES" | grep -E '^api/' || true)
MODEL_CHANGED=$(echo "$DIFF_FILES" | grep -E '^models/|^src/' || true)
UI_CHANGED=$(echo "$DIFF_FILES" | grep -E '^ui/' || true)
TEST_CHANGED=$(echo "$DIFF_FILES" | grep -E '^tests/' || true)
DOCKER_CHANGED=$(echo "$DIFF_FILES" | grep -E 'Dockerfile|docker-compose|compose.yaml' || true)

echo ""
echo "Changed files:"
echo "$DIFF_FILES"

# =========================================
# CLEANUP
# =========================================
cleanup() {
    echo ""
    echo "🧹 Cleaning up containers..."
    docker compose down -v || true
}

trap cleanup EXIT

# =========================================
# WAIT HELPER
# =========================================
wait_for_service() {

    URL=$1
    NAME=$2

    echo "⏳ Waiting for $NAME..."

    for i in {1..30}; do
        if curl -sf "$URL" >/dev/null; then
            echo "✅ $NAME ready"
            return 0
        fi

        sleep 2
    done

    echo "❌ $NAME failed"

    docker compose logs || true

    return 1
}

# =========================================
# BUILD IMAGES
# =========================================
echo ""
echo "🏗️ Building images..."

if [ -n "$DOCKER_CHANGED" ] || [ -n "$MODEL_CHANGED" ]; then
    docker compose build model-base
else
    echo "⚡ model-base unchanged"
fi

if [ -n "$DOCKER_CHANGED" ] || [ -n "$API_CHANGED" ]; then
    docker compose build api
else
    echo "⚡ api unchanged"
fi

if [ -n "$DOCKER_CHANGED" ] || [ -n "$UI_CHANGED" ]; then
    docker compose build ui
else
    echo "⚡ ui unchanged"
fi

if [ -n "$DOCKER_CHANGED" ] || [ -n "$TEST_CHANGED" ]; then
    docker compose build test-base
else
    echo "⚡ test-base unchanged"
fi

# =========================================
# TRAIN MODELS (PARALLEL)
# =========================================
echo ""
echo "🧠 Training models..."

MODEL_STATUS=0
PHOBERT_STATUS=0
MBERT_STATUS=0
DEMO_STATUS=0

if [ -n "$MODEL_CHANGED" ] || [ -n "$DOCKER_CHANGED" ]; then

    set +e

    docker compose run --rm train-model &
    PID_MODEL=$!

    docker compose run --rm train-phobert &
    PID_PHOBERT=$!

    docker compose run --rm train-mbert &
    PID_MBERT=$!

    docker compose run --rm train-demo-bert &
    PID_DEMO=$!

    wait $PID_MODEL
    MODEL_STATUS=$?

    wait $PID_PHOBERT
    PHOBERT_STATUS=$?

    wait $PID_MBERT
    MBERT_STATUS=$?

    wait $PID_DEMO
    DEMO_STATUS=$?

    set -e

else
    echo "⚡ Skipping training (no model changes)"
fi


# =========================================
# START API + UI
# =========================================
echo ""
echo "🚀 Starting API + UI..."

docker compose up -d api ui

wait_for_service http://localhost:8000/docs API
API_READY_STATUS=$?

wait_for_service http://localhost:8501 UI
UI_READY_STATUS=$?

# =========================================
# MODEL TESTS (PARALLEL)
# =========================================
echo ""
echo "🧪 Running model tests..."

set +e

docker compose run --rm test-model &
PID_TEST_MODEL=$!

docker compose run --rm \
    -e MODEL_KEY=phobert \
    test-bert &
PID_TEST_PHOBERT=$!

docker compose run --rm \
    -e MODEL_KEY=mbert \
    test-bert &
PID_TEST_MBERT=$!

docker compose run --rm \
    -e MODEL_KEY=demo \
    test-bert &
PID_TEST_DEMO=$!

wait $PID_TEST_MODEL
TEST_MODEL_STATUS=$?

wait $PID_TEST_PHOBERT
TEST_PHOBERT_STATUS=$?

wait $PID_TEST_MBERT
TEST_MBERT_STATUS=$?

wait $PID_TEST_DEMO
TEST_DEMO_STATUS=$?

set -e

# =========================================
# API TESTS (PARALLEL)
# =========================================
echo ""
echo "🌐 Running API tests..."

set +e

run_api_test() {
    MODEL=$1

    docker compose run --rm \
        -e MODEL_TYPE=$MODEL \
        test-api
}

run_api_test model &
PID_API_MODEL=$!


wait $PID_API_MODEL
API_MODEL_STATUS=$?


set -e

# =========================================
# UI TEST
# =========================================
echo ""
echo "🖥️ Running UI tests..."

set +e

docker compose run --rm test-ui
TEST_UI_STATUS=$?

set -e

# =========================================
# RESULTS
# =========================================
echo ""
echo "========== RESULTS =========="

echo ""
echo "🧠 TRAINING"
echo "Model train  : $MODEL_STATUS"
echo "PhoBERT train: $PHOBERT_STATUS"
echo "mBERT train  : $MBERT_STATUS"
echo "Demo train   : $DEMO_STATUS"

echo ""
echo "🚀 SERVICES"
echo "API ready    : $API_READY_STATUS"
echo "UI ready     : $UI_READY_STATUS"

echo ""
echo "🧪 MODEL TESTS"
echo "Model test   : $TEST_MODEL_STATUS"
echo "PhoBERT test : $TEST_PHOBERT_STATUS"
echo "mBERT test   : $TEST_MBERT_STATUS"
echo "Demo test    : $TEST_DEMO_STATUS"

echo ""
echo "🌐 API TESTS"
echo "Model API    : $API_MODEL_STATUS"

echo ""
echo "🖥️ UI TEST"
echo "UI test      : $TEST_UI_STATUS"

# =========================================
# FINAL STATUS
# =========================================
FAILED=0

for s in \
$MODEL_STATUS \
$PHOBERT_STATUS \
$MBERT_STATUS \
$DEMO_STATUS \
$API_READY_STATUS \
$UI_READY_STATUS \
$TEST_MODEL_STATUS \
$TEST_PHOBERT_STATUS \
$TEST_MBERT_STATUS \
$TEST_DEMO_STATUS \
$API_MODEL_STATUS \
$TEST_UI_STATUS
do
    if [ "$s" -ne 0 ]; then
        FAILED=1
    fi
done

echo ""

if [ "$FAILED" -eq 1 ]; then
    echo "❌ CI FAILED"
    exit 1
fi

echo "✅ CI pipeline completed successfully!"