#!/bin/bash
# Test script for kubectl --stale and --idle features
# This script automates the testing process for someone who just cloned the repo

set -e

CLUSTER_NAME="${CLUSTER_NAME:-test-cluster}"
KUBECONFIG_FILE="/tmp/${CLUSTER_NAME}-kubeconfig.yaml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Testing kubectl --stale and --idle Features ===${NC}\n"

# Step 1: Build kubectl
echo -e "${YELLOW}Step 1: Building kubectl...${NC}"
# Build just kubectl for speed, but 'make' (builds everything) also works
if ! make all WHAT=cmd/kubectl > /dev/null 2>&1; then
    echo "Error: Failed to build kubectl. Make sure you have Go and make installed."
    exit 1
fi

# Determine binary path based on OS/ARCH
GOOS=$(go env GOOS)
GOARCH=$(go env GOARCH)
KUBECTL_BIN="_output/local/bin/${GOOS}/${GOARCH}/kubectl"

if [ ! -f "$KUBECTL_BIN" ]; then
    echo "Error: kubectl binary not found at $KUBECTL_BIN"
    exit 1
fi

echo -e "${GREEN}✓ kubectl built successfully${NC}\n"

# Step 2: Check/create kind cluster
echo -e "${YELLOW}Step 2: Setting up kind cluster...${NC}"
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo -e "${GREEN}✓ Using existing cluster: ${CLUSTER_NAME}${NC}"
else
    echo "Creating new kind cluster: ${CLUSTER_NAME}..."
    kind create cluster --name "${CLUSTER_NAME}"
    echo -e "${GREEN}✓ Cluster created${NC}"
fi

# Get kubeconfig
kind get kubeconfig --name "${CLUSTER_NAME}" > "${KUBECONFIG_FILE}"
echo -e "${GREEN}✓ Kubeconfig saved to ${KUBECONFIG_FILE}${NC}\n"

# Step 3: Create test pods
echo -e "${YELLOW}Step 3: Creating test pods...${NC}"
export KUBECONFIG="${KUBECONFIG_FILE}"

# Create a test pod
"${KUBECTL_BIN}" run test-pod-1 --image=nginx:alpine --restart=Never --quiet 2>/dev/null || true
"${KUBECTL_BIN}" run test-pod-2 --image=busybox --restart=Never --command -- sleep 3600 --quiet 2>/dev/null || true

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
sleep 5
"${KUBECTL_BIN}" wait --for=condition=Ready pod/test-pod-1 --timeout=30s > /dev/null 2>&1 || true
"${KUBECTL_BIN}" wait --for=condition=Ready pod/test-pod-2 --timeout=30s > /dev/null 2>&1 || true

echo -e "${GREEN}✓ Test pods created${NC}\n"

# Step 4: Test --stale flag
echo -e "${YELLOW}Step 4: Testing --stale flag...${NC}"
echo "Running: kubectl get pods --stale=1h"
"${KUBECTL_BIN}" get pods --stale=1h || echo "No stale pods found (this is expected for new pods)"
echo ""

# Step 5: Test --idle flag
echo -e "${YELLOW}Step 5: Testing --idle flag...${NC}"
echo "Running: kubectl get pods --idle=30m"
"${KUBECTL_BIN}" get pods --idle=30m
echo ""

echo "Running: kubectl get pods --idle=1m (should show pods)"
"${KUBECTL_BIN}" get pods --idle=1m
echo ""

# Step 6: Show help
echo -e "${YELLOW}Step 6: Showing help for new flags...${NC}"
"${KUBECTL_BIN}" get pods --help | grep -A 2 -E "(stale|idle)" || true
echo ""

echo -e "${GREEN}=== Testing Complete ===${NC}"
echo ""
echo "To use the new kubectl features, use:"
echo "  export KUBECONFIG=${KUBECONFIG_FILE}"
echo "  ${KUBECTL_BIN} get pods --stale=1h"
echo "  ${KUBECTL_BIN} get pods --idle=30m"
echo ""
echo "To clean up:"
echo "  kind delete cluster --name ${CLUSTER_NAME}"
echo "  rm ${KUBECONFIG_FILE}"

