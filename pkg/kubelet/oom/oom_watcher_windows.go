//go:build windows

/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package oom

import (
	"context"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubelettypes "k8s.io/kubelet/pkg/types"
)

// oomKilledExitReason is the CRI container exit reason reported when a
// container was terminated by the out-of-memory killer. It matches the value
// the kubelet kuberuntime layer already looks for (see
// pkg/kubelet/kuberuntime/helpers.go).
const oomKilledExitReason = "OOMKilled"

// windowsOOMEventReason is the Kubernetes event reason emitted when the Windows
// CRI reports that a container crossed into the OOMKilled state.
const windowsOOMEventReason = "ContainerOOMKilled"

// windowsOOMPollInterval is how often the watcher reconciles the CRI container
// state. It is a variable so tests can shrink the interval to avoid sleeping
// for the real period.
var windowsOOMPollInterval = time.Minute

type windowsWatcher struct {
	// recorder emits Kubernetes events for OOMKilled containers.
	recorder record.EventRecorder
	// containers polls the CRI container state.
	containers containerStatusGetter
	// pollInterval controls the reconciliation period.
	pollInterval time.Duration

	// mu guards seen so replay once-per-container bookkeeping is safe even if
	// Start is invoked concurrently.
	mu sync.Mutex
	// seen records container IDs already reported as OOMKilled so a single
	// terminated container is surfaced exactly once across polling intervals.
	seen map[string]struct{}
	// inspected records container IDs whose terminal status we have already
	// looked up. Exited containers never change their exit reason, so once we
	// have inspected one we must not query its status again on later polls;
	// otherwise exited non-OOM containers accumulate and are re-polled every
	// interval without bound.
	inspected map[string]struct{}
}

var _ Watcher = &windowsWatcher{}

// NewWatcher creates a Windows OOM watcher that reconciles CRI container
// statuses to surface containers that were terminated by the out-of-memory
// killer. The runtime service may be nil, in which case the watcher logs the
// skipped reconcile and stays safe.
func NewWatcher(recorder record.EventRecorder, containers containerStatusGetter) (Watcher, error) {
	return &windowsWatcher{
		recorder:     recorder,
		containers:   containers,
		pollInterval: windowsOOMPollInterval,
		seen:         make(map[string]struct{}),
		inspected:    make(map[string]struct{}),
	}, nil
}

// Start reconciles CRI container statuses until ctx is cancelled, emitting a
// Kubernetes event and incrementing container_oom_events_total for every
// container that crosses into the OOMKilled state.
func (ow *windowsWatcher) Start(ctx context.Context, _ *v1.ObjectReference) error {
	logger := klog.FromContext(ctx)
	go func() {
		defer runtime.HandleCrashWithContext(ctx)
		ow.pollLoop(ctx, logger)
	}()
	return nil
}

// pollLoop periodically lists CRI containers and reports newly OOMKilled ones.
// It is extracted from Start so tests can exercise a single reconcile without
// driving the full timer loop.
func (ow *windowsWatcher) pollLoop(ctx context.Context, logger klog.Logger) {
	ticker := time.NewTicker(ow.pollInterval)
	defer ticker.Stop()

	// Seed the seen set with containers that are already OOMKilled so a kubelet
	// restart does not replay historical kills as fresh events. Reconcile only
	// after a successful seed (i.e. once a reliable baseline exists): if the
	// initial list fails, an empty seen set would treat every surviving
	// historical OOM as a fresh kill on the first pass. If the runtime is nil or
	// the initial list errors, retry the seed on subsequent intervals rather
	// than reconciling against that incomplete baseline.
	for {
		seeded := seedOOMKills(ctx, ow.containers, ow.seen, logger)
		if seeded {
			ow.reconcile(ctx, logger)
			break
		}
		logger.V(2).Info("Windows OOM watcher baseline seed incomplete; retrying next interval")
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ow.reconcile(ctx, logger)
		}
	}
}

// reconcile inspects the current CRI container state and reports any container
// that newly crossed into OOMKilled. It is called from the single pollLoop
// goroutine.
func (ow *windowsWatcher) reconcile(ctx context.Context, logger klog.Logger) {
	if ow.containers == nil {
		logger.V(2).Info("Windows OOM watcher has no CRI runtime service; skipping reconcile")
		return
	}

	containers, err := ow.containers.ListContainers(ctx, nil)
	if err != nil {
		logger.Error(err, "Windows OOM watcher failed to list containers")
		return
	}

	// Prune inspection/seen bookkeeping against the current container list.
	// CRI garbage collection removes completed containers, so without pruning the
	// inspected set would retain a per-container entry for every exited container
	// the watcher has ever seen, growing without bound on a long-lived kubelet
	// even though the runtime has reclaimed the objects.
	ow.pruneBookkeeping(containers)

	// Only containers that are no longer running can carry a terminal exit
	// reason, so short-circuit the per-container status lookup for anything
	// still running or not yet started.
	for _, c := range containers {
		if c.GetState() != runtimeapi.ContainerState_CONTAINER_EXITED {
			continue
		}

		ow.mu.Lock()
		_, inspected := ow.inspected[c.GetId()]
		_, saw := ow.seen[c.GetId()]
		ow.mu.Unlock()
		if inspected || saw {
			continue
		}

		status, err := ow.containers.ContainerStatus(ctx, c.GetId(), false)
		if err != nil {
			// A container can be removed between ListContainers and
			// ContainerStatus; retry next interval rather than failing the poll.
			logger.V(2).Info("Windows OOM watcher failed to get container status", "containerID", c.GetId(), "err", err)
			continue
		}
		if status == nil || status.GetStatus() == nil {
			continue
		}

		cs := status.GetStatus()

		// The exit reason is terminal, so record the container as inspected
		// regardless of outcome. This stops exited non-OOM containers from being
		// re-queried on every subsequent poll.
		ow.mu.Lock()
		ow.inspected[c.GetId()] = struct{}{}
		ow.mu.Unlock()

		if cs.GetReason() != oomKilledExitReason {
			continue
		}

		// The Windows cAdvisor metrics path keys per-container info by the CRI
		// container ID (see cadvisor.GetRequestedContainersInfo), and the server
		// reads the count with OOMEventsForContainer(info.Name). Record under the
		// container ID so the count is readable back on Windows; on Linux the
		// in-process counter is unused because the kernel cgroup OOM path also
		// keys by containerName-equivalent cgroup path.
		recordOOMKill(c.GetId())

		// The human-facing event still uses the Kubernetes container name.
		containerName := cs.GetLabels()[kubelettypes.KubernetesContainerNameLabel]
		if containerName == "" {
			containerName = cs.GetMetadata().GetName()
		}

		// Remember the container id so the kill is surfaced exactly once.
		ow.mu.Lock()
		ow.seen[c.GetId()] = struct{}{}
		ow.mu.Unlock()

		podName := cs.GetLabels()[kubelettypes.KubernetesPodNameLabel]
		podNamespace := cs.GetLabels()[kubelettypes.KubernetesPodNamespaceLabel]
		ref := &v1.ObjectReference{
			Kind:      "Pod",
			Namespace: podNamespace,
			Name:      podName,
			UID:       types.UID(cs.GetLabels()[kubelettypes.KubernetesPodUIDLabel]),
		}
		ow.recorder.Eventf(ref, v1.EventTypeWarning, windowsOOMEventReason,
			"Container %s in pod %s/%s was terminated by the out-of-memory killer", containerName, podNamespace, podName)
		logger.Info("Windows OOM watcher reported OOMKilled container", "containerID", c.GetId(), "container", containerName, "pod", podNamespace+"/"+podName)
	}
}

// pruneBookkeeping drops inspected/seen entries for container IDs that are no
// longer present in the latest successful ListContainers result, so the maps
// stay bounded to currently-managed containers while retaining deduplication
// for containers that still exist. pruneBookkeeping is called only after a
// successful full list, so a transient list failure never clears state.
// Callers must already hold the reconcile path; this locks ow.mu itself.
func (ow *windowsWatcher) pruneBookkeeping(containers []*runtimeapi.Container) {
	ow.mu.Lock()
	defer ow.mu.Unlock()
	if len(containers) == 0 {
		// No containers currently exist; the runtime has reclaimed them all.
		ow.inspected = make(map[string]struct{})
		ow.seen = make(map[string]struct{})
		return
	}
	live := make(map[string]struct{}, len(containers))
	for _, c := range containers {
		live[c.GetId()] = struct{}{}
	}
	for id := range ow.inspected {
		if _, ok := live[id]; !ok {
			delete(ow.inspected, id)
		}
	}
	for id := range ow.seen {
		if _, ok := live[id]; !ok {
			delete(ow.seen, id)
		}
	}
}

// seedOOMKills records the ids of containers that are already OOMKilled when
// the watcher starts so a kubelet restart does not replay historical kills. It
// returns false when it could not establish a reliable baseline (no runtime, or
// the initial list failed), so the caller can avoid reconciling against an
// incomplete seen set that would replay historical kills as fresh events.
func seedOOMKills(ctx context.Context, containers containerStatusGetter, seen map[string]struct{}, logger klog.Logger) bool {
	if containers == nil {
		return false
	}
	list, err := containers.ListContainers(ctx, nil)
	if err != nil {
		logger.V(2).Info("Windows OOM watcher failed to seed OOMKilled containers", "err", err)
		return false
	}
	for _, c := range list {
		if c.GetState() != runtimeapi.ContainerState_CONTAINER_EXITED {
			continue
		}
		status, err := containers.ContainerStatus(ctx, c.GetId(), false)
		if err != nil {
			continue
		}
		if status != nil && status.GetStatus() != nil && status.GetStatus().GetReason() == oomKilledExitReason {
			seen[c.GetId()] = struct{}{}
		}
	}
	return true
}
