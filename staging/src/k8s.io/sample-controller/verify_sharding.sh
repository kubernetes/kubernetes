#!/bin/bash
set -e
export KUBECONFIG=/var/run/kubernetes/admin.kubeconfig

# Cleanup on exit
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

echo "Building sample-controller..."
go build -o sample-controller .

echo "Applying CRD..."
kubectl apply -f artifacts/examples/crd.yaml

# Clear old logs
rm -f shard1.log shard2.log

# Ensure we have a clean slate (delete existing foos/deployments matching our pattern)
kubectl delete foo -l app=sharding-test 2>/dev/null || true
kubectl delete deployment -l app=sharding-test 2>/dev/null || true

echo "Starting Shard 1 (Range: 0... - 7...)..."
./sample-controller \
  --kubeconfig=$KUBECONFIG \
  --sharding-strategy=uid \
  --shard-range-end=8 \
  --v=4 \
  > shard1.log 2>&1 &

echo "Starting Shard 2 (Range: 8... - f...)..."
./sample-controller \
  --kubeconfig=$KUBECONFIG \
  --sharding-strategy=uid \
  --shard-range-start=8 \
  --v=4 \
  > shard2.log 2>&1 &

sleep 5

echo "Creating Foo resources..."
for i in {1..20}; do
    cat <<EOF | kubectl apply -f -
apiVersion: samplecontroller.k8s.io/v1alpha1
kind: Foo
metadata:
  name: test-foo-$i
  labels:
    app: sharding-test
spec:
  deploymentName: test-dep-$i
  replicas: 1
EOF
done

echo "Waiting for sync..."
sleep 10

echo "Verifying logs..."
# Extract unique object names processed by each shard
grep "Successfully synced" shard1.log | grep -o "test-foo-[0-9]*" | sort | uniq > shard1_items.txt
grep "Successfully synced" shard2.log | grep -o "test-foo-[0-9]*" | sort | uniq > shard2_items.txt

COUNT1=$(wc -l < shard1_items.txt)
COUNT2=$(wc -l < shard2_items.txt)
TOTAL=$((COUNT1 + COUNT2))

echo "Shard 1 synced $COUNT1 items"
echo "Shard 2 synced $COUNT2 items"
echo "Total synced: $TOTAL / 20"

# Check overlap
OVERLAP=$(comm -12 shard1_items.txt shard2_items.txt | wc -l)

if [ "$OVERLAP" -ne 0 ]; then
    echo "FAILURE: Overlap detected! $OVERLAP items handled by both shards."
    comm -12 shard1_items.txt shard2_items.txt
    exit 1
fi

if [ "$TOTAL" -ne 20 ]; then
    echo "FAILURE: Not all items synced. Expected 20, got $TOTAL."
    echo "Shard 1 items:"
    cat shard1_items.txt
    echo "Shard 2 items:"
    cat shard2_items.txt
    exit 1
fi

echo "SUCCESS: Verification Passed! Workload distributed (approx split: $COUNT1 vs $COUNT2)."
