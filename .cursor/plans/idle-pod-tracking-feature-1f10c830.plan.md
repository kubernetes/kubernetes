<!-- 1f10c830-742a-4a1c-9f49-e31e744a9fed 675682c5-a80f-4de1-81a0-b8429aab3f9a -->
# Idle Pod Tracking Implementation Plan

## Status: Tests Already Written

The following test files are already implemented and define the expected interfaces:

- `pkg/kubelet/activity_tracker_test.go` - ActivityTracker unit tests with implementation stubs
- `pkg/kubelet/server/idle_endpoint_test.go` - /pods/idle endpoint tests  
- `pkg/kubelet/status/last_activity_test.go` - Status manager activity tracking
- `staging/src/k8s.io/kubectl/pkg/cmd/get/idle_test.go` - kubectl --idle flag tests
- `test/e2e/kubectl/idle_pods_test.go` - E2E tests
- `test/integration/kubelet/activity_tracking_test.go` - Integration tests

---

## Phase 1: Kubelet-Side Implementation

### 1. Add Feature Gate

**File:** `pkg/features/kube_features.go`

Add `IdlePodTracking` feature gate (alpha, default disabled).

### 2. Extract ActivityTracker from Test File

**File:** `pkg/kubelet/activity_tracker.go` (new, move from test)

The test file contains implementation stubs - extract to production code:

- `ActivityType` enum (exec, port-forward, logs, attach, copy)
- `ActivityTracker` struct with thread-safe map
- `RecordActivity()`, `GetLastActivity()`, `GetIdlePods()`, `FormatIdleDuration()`

### 3. Hook into Kubelet Server Endpoints

**File:** `pkg/kubelet/server/server.go`

Call `RecordActivity()` in: `getExec()`, `getAttach()`, `getPortForward()`, `getContainerLogs()`

### 4. Add /pods/idle Endpoint

**File:** `pkg/kubelet/server/server.go`

Readonly endpoint returning `IdlePodsResponse{Pods: map[string]metav1.Time}`

### 5. Integrate with Status Manager

**File:** `pkg/kubelet/status/status_manager.go`

Persist activity to annotation `kubernetes.io/last-activity`

---

## Phase 2: kubectl Integration

### 6. Add --idle Flag

**File:** `staging/src/k8s.io/kubectl/pkg/cmd/get/get.go`

Add `Idle time.Duration` to `GetOptions`, register `--idle` flag (default 30m).

### 7. Add Idle Filtering Logic

**File:** `staging/src/k8s.io/kubectl/pkg/cmd/get/get.go`

Filter pods by `kubernetes.io/last-activity` annotation when `--idle` is set.

Add `formatIdleDuration()` and `parseIdleDuration()` helpers (defined in test file).

### 8. Add IDLE-SINCE Column (Optional)

**File:** `pkg/printers/internalversion/printers.go`

Add column showing idle duration when `--idle` flag is used.

---

## Key Interfaces (from tests)

```go
// ActivityTracker (pkg/kubelet/activity_tracker_test.go:290-302)
type ActivityTracker struct {
    clock      clock.Clock
    activities map[types.UID]metav1.Time
}

// IdlePodsResponse (pkg/kubelet/server/idle_endpoint_test.go:34-37)
type IdlePodsResponse struct {
    Pods map[string]metav1.Time `json:"pods"`
}

// Annotation key (used throughout tests)
const LastActivityAnnotationKey = "kubernetes.io/last-activity"
```

### To-dos

- [ ] Add IdlePodTracking feature gate to pkg/features/kube_features.go
- [ ] Extract ActivityTracker from test file to pkg/kubelet/activity_tracker.go
- [ ] Hook activity tracker into exec/attach/portforward/logs handlers
- [ ] Add /pods/idle readonly endpoint to kubelet server
- [ ] Integrate activity tracking with status manager for persistence
- [ ] Add --idle flag to kubectl get command
- [ ] Implement client-side idle filtering using annotation
- [ ] Add IDLE-SINCE column to pod printers (optional)