# Testing Static Pod Priority Warnings

This directory contains test manifests for validating static pod priority/priorityClassName warning logic.

## Test Cases

1. **case1-priority-without-classname.yaml**: Pod with priority set but no priorityClassName
   - Expected: Warning about priority set without priorityClassName

2. **case2-valid-priority-and-classname.yaml**: Pod with correct values (2000001000/system-node-critical)
   - Expected: No warnings

3. **case3-invalid-priority.yaml**: Pod with non-standard priority value
   - Expected: Warning about non-standard priority value

4. **case4-invalid-classname.yaml**: Pod with non-standard priorityClassName
   - Expected: Warning about non-standard priorityClassName

## Instructions for Testing in Kind Cluster

### 1. Find your kind node container
```bash
docker ps | grep kind
```

### 2. Exec into the kind control-plane node
```bash
docker exec -it <kind-control-plane-container-name> bash
# Or if your cluster is named "kind" (default):
docker exec -it kind-control-plane bash
```

### 3. Navigate to the static pod manifests directory
```bash
cd /etc/kubernetes/manifests
```

### 4. Copy test manifests into the container

From your host machine (in a new terminal), copy each manifest:
```bash
# Copy all test cases
docker cp case1-priority-without-classname.yaml kind-control-plane:/etc/kubernetes/manifests/
docker cp case2-valid-priority-and-classname.yaml kind-control-plane:/etc/kubernetes/manifests/
docker cp case3-invalid-priority.yaml kind-control-plane:/etc/kubernetes/manifests/
docker cp case4-invalid-classname.yaml kind-control-plane:/etc/kubernetes/manifests/
```

Or copy them one at a time to test individually.

### 5. Check kubelet logs for warnings on the control-plane node

**Important**: Since you copied the manifests to the control-plane node, the static pods will be scheduled on that node, so you need to check the kubelet logs **on the control-plane node**.

```bash
# Watch kubelet logs on control-plane (where the static pods will run)
docker exec kind-control-plane journalctl -u kubelet -f

# Or use crictl to view kubelet container logs:
docker exec kind-control-plane crictl logs -f $(docker exec kind-control-plane crictl ps --name kubelet -q) 2>&1 | grep -i priority

# Or check logs directly from docker:
docker logs -f kind-control-plane 2>&1 | grep -i priority
```

Look for log messages containing:
- "Priority set without PriorityClassName"
- "Priority set to non-standard value"
- "PriorityClassName set to non-standard value"

### 6. Verify mirror pods were created

```bash
# From your host (with kubectl configured for kind):
kubectl get pods -A | grep test-

# You should see mirror pods with "-kind-control-plane" suffix
# The suffix matches the node name where the static pod manifests were placed
kubectl get pods -n kube-system -o wide | grep test-
```

### 7. Clean up test pods

```bash
# From host:
docker exec kind-control-plane rm /etc/kubernetes/manifests/case*.yaml

# Or from inside the container:
cd /etc/kubernetes/manifests
rm case*.yaml
```

## Testing on Worker Nodes (Optional)

If you have a multi-node kind cluster and want to test on worker nodes:

```bash
# Copy to worker node instead
docker cp case1-priority-without-classname.yaml kind-worker:/etc/kubernetes/manifests/

# Check logs on that worker node
docker logs -f kind-worker 2>&1 | grep -i priority

# Mirror pod will have "-kind-worker" suffix
kubectl get pods -A | grep kind-worker
```

## Expected Behavior

- **Case 1**: Should log warning but create mirror pod
- **Case 2**: Should create mirror pod without warnings
- **Case 3**: Should log warning but create mirror pod
- **Case 4**: Should log warning but create mirror pod

All pods should be created despite warnings, as these are warnings, not errors.
