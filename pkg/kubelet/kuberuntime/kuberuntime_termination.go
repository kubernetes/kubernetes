/*
Copyright The Kubernetes Authors.

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

package kuberuntime

import (
	"context"
	"fmt"
	"math"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierror "k8s.io/cri-api/pkg/errors"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// These records deduplicate in-flight CRI calls; they do not define desired
// state. After a kubelet restart, spec and runtime observations determine the
// same actions and StopContainer can safely be issued again.
// Only the pod worker accesses records. Asynchronous calls return through done.
type podTerminationOperations struct {
	containers map[kubecontainer.ContainerID]*containerTerminationOperations
	restarted  map[kubecontainer.ContainerID]bool
}

type containerTerminationOperations struct {
	hook         *terminationOperation
	hookComplete bool
	stop         *terminationOperation
	stopComplete bool
}

type terminationOperation struct {
	done     chan error
	cancel   context.CancelFunc
	deadline time.Time
}

func startTerminationOperation(ctx context.Context, deadline time.Time, run func(context.Context) error) *terminationOperation {
	// The worker's context is cancelled when a grace period is shortened. Keep
	// the call alive across ordinary reconciles and explicitly replace it when
	// the new deadline arrives; cancellation alone does not stop a CRI container.
	ctx, cancel := context.WithDeadline(context.WithoutCancel(ctx), deadline)
	op := &terminationOperation{done: make(chan error, 1), cancel: cancel, deadline: deadline}
	go func() {
		defer cancel()
		op.done <- run(ctx)
	}()
	return op
}

// containersToKeepDuringTermination derives the sidecars still needed by live
// application containers or later sidecars. Exited and never-started containers
// cannot hold up termination. Unknown states are conservatively treated as live.
func containersToKeepDuringTermination(pod *v1.Pod, status *kubecontainer.PodStatus) map[string]bool {
	active := func(name string) bool {
		s := status.FindContainerStatusByName(name)
		return s != nil && s.State != kubecontainer.ContainerStateExited
	}
	waiting := false
	for _, c := range pod.Spec.Containers {
		waiting = waiting || active(c.Name)
	}
	for _, c := range pod.Spec.InitContainers {
		if !podutil.IsRestartableInitContainer(&c) {
			waiting = waiting || active(c.Name)
		}
	}
	keep := make(map[string]bool)
	for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
		c := &pod.Spec.InitContainers[i]
		if !podutil.IsRestartableInitContainer(c) {
			continue
		}
		keep[c.Name] = waiting
		waiting = waiting || active(c.Name)
	}
	return keep
}

// SyncTerminatingPod performs one reconciliation. PLEG and the pod worker's
// timer provide subsequent observations; no kill operation watches pod status.
func (m *kubeGenericRuntimeManager) SyncTerminatingPod(ctx context.Context, pod *v1.Pod, status *kubecontainer.PodStatus, pullSecrets []v1.Secret, backOff *flowcontrol.Backoff, deadline time.Time) (bool, error) {
	if deadline.IsZero() {
		return false, fmt.Errorf("cannot reconcile termination of pod %s/%s without a deadline", pod.Namespace, pod.Name)
	}
	m.terminationLock.Lock()
	if m.terminations == nil {
		m.terminations = make(map[types.UID]*podTerminationOperations)
	}
	ops := m.terminations[pod.UID]
	if ops == nil {
		ops = &podTerminationOperations{containers: make(map[kubecontainer.ContainerID]*containerTerminationOperations), restarted: make(map[kubecontainer.ContainerID]bool)}
		m.terminations[pod.UID] = ops
	}
	m.terminationLock.Unlock()

	keep := containersToKeepDuringTermination(pod, status)
	var errs []error
	active := false
	for _, cs := range status.ContainerStatuses {
		if cs.State == kubecontainer.ContainerStateExited {
			if op := ops.containers[cs.ID]; op != nil {
				if op.hook != nil {
					op.hook.cancel()
				}
				if op.stop != nil {
					op.stop.cancel()
				}
				delete(ops.containers, cs.ID)
			}
			continue
		}
		active = true
		c := kubecontainer.GetContainerSpec(pod, cs.Name)
		if c == nil {
			if time.Now().Before(deadline) {
				errs = append(errs, fmt.Errorf("cannot gracefully terminate container %q: missing spec in pod %s/%s", cs.Name, pod.Namespace, pod.Name))
				continue
			}
			// A missing spec must not prevent enforcement of the pod deadline.
			c = &v1.Container{Name: cs.Name}
		}
		if cs.State == kubecontainer.ContainerStateCreated {
			// A failed StartContainer can leave a created replacement behind.
			// Keep it for the restart action only if this sidecar has run before.
			if !keep[c.Name] || cs.RestartCount == 0 {
				if err := m.removeContainer(ctx, cs.ID.ID, true); err != nil && !crierror.IsNotFound(err) {
					errs = append(errs, fmt.Errorf("remove unstarted container %q during pod termination: %w", c.Name, err))
				}
			}
			continue
		}
		op := ops.containers[cs.ID]
		if op == nil {
			op = &containerTerminationOperations{}
			ops.containers[cs.ID] = op
		}
		// Preserve preStop timing: hooks can help application containers drain, so
		// sidecar hooks must not wait for application containers to finish.
		hookDone := m.reconcileTerminationHook(ctx, pod, c, cs, op, deadline)
		if time.Now().Before(deadline) && (keep[c.Name] || !hookDone) {
			continue
		}
		if err := m.reconcileContainerStop(ctx, pod, c, cs, op, deadline); err != nil {
			errs = append(errs, err)
		}
	}

	// Starts are serialized with reconciles, just as in SyncPod. The deadline
	// bounds image pulls and postStart hooks as well as container creation.
	for i := range pod.Spec.InitContainers {
		c := &pod.Spec.InitContainers[i]
		if !keep[c.Name] || time.Until(deadline) <= time.Second {
			continue
		}
		cs := status.FindContainerStatusByName(c.Name)
		if cs == nil || ops.restarted[cs.ID] ||
			(cs.State != kubecontainer.ContainerStateExited && (cs.State != kubecontainer.ContainerStateCreated || cs.RestartCount == 0)) {
			continue
		}
		startCtx, cancel := context.WithDeadline(ctx, deadline)
		started, err := m.restartTerminatingSidecar(startCtx, pod, c, status, pullSecrets, backOff)
		cancel()
		if err != nil {
			errs = append(errs, err)
		}
		if started {
			ops.restarted[cs.ID] = true
		}
	}
	if !active {
		// Runtime observations, rather than completed RPCs, authorize resource cleanup.
		for _, op := range ops.containers {
			if op.hook != nil {
				op.hook.cancel()
			}
			if op.stop != nil {
				op.stop.cancel()
			}
		}
		m.terminationLock.Lock()
		delete(m.terminations, pod.UID)
		m.terminationLock.Unlock()
	}
	return !active, utilerrors.NewAggregate(errs)
}

func (m *kubeGenericRuntimeManager) reconcileTerminationHook(ctx context.Context, pod *v1.Pod, c *v1.Container, cs *kubecontainer.Status, ops *containerTerminationOperations, deadline time.Time) bool {
	if ops.hookComplete || c.Lifecycle == nil || c.Lifecycle.PreStop == nil || cs.State != kubecontainer.ContainerStateRunning {
		return true
	}
	if !time.Now().Before(deadline) {
		if ops.hook != nil {
			ops.hook.cancel()
		}
		return true
	}
	if ops.hook == nil {
		ops.hook = startTerminationOperation(ctx, deadline, func(ctx context.Context) error {
			_, err := m.runner.Run(ctx, cs.ID, pod, c, c.Lifecycle.PreStop)
			return err
		})
	}
	select {
	case err := <-ops.hook.done:
		ops.hookComplete = true
		if err != nil {
			klog.FromContext(ctx).Error(err, "PreStop hook failed during pod termination", "pod", klog.KObj(pod), "containerName", c.Name)
			m.recordContainerEvent(ctx, pod, c, cs.ID.ID, v1.EventTypeWarning, events.FailedPreStopHook, "PreStopHook failed")
		}
		return true
	default:
		return false
	}
}

func (m *kubeGenericRuntimeManager) reconcileContainerStop(ctx context.Context, pod *v1.Pod, c *v1.Container, cs *kubecontainer.Status, ops *containerTerminationOperations, deadline time.Time) error {
	if ops.stopComplete {
		return nil
	}
	// Allow transport overhead after the runtime's absolute shutdown deadline.
	callDeadline := deadline.Add(2 * time.Second)
	if callDeadline.Before(time.Now()) {
		callDeadline = time.Now().Add(2 * time.Second)
	}
	if ops.stop != nil {
		select {
		case err := <-ops.stop.done:
			ops.stop = nil
			if err == nil || crierror.IsNotFound(err) {
				ops.stopComplete = true
				return nil
			}
			return fmt.Errorf("stop container %q in terminating pod %s/%s: %w", c.Name, pod.Namespace, pod.Name, err)
		default:
			if !callDeadline.Before(ops.stop.deadline) {
				return nil
			}
			ops.stop.cancel()
			ops.stop = nil
		}
	}
	grace := int64(math.Ceil(max(0, time.Until(deadline).Seconds())))
	m.recordContainerEvent(ctx, pod, c, cs.ID.ID, v1.EventTypeNormal, events.KillingContainer, "Stopping container %s", c.Name)
	ops.stop = startTerminationOperation(ctx, callDeadline, func(ctx context.Context) error {
		return m.runtimeService.StopContainer(ctx, cs.ID.ID, grace)
	})
	return nil
}

func (m *kubeGenericRuntimeManager) restartTerminatingSidecar(ctx context.Context, pod *v1.Pod, c *v1.Container, status *kubecontainer.PodStatus, pullSecrets []v1.Secret, backOff *flowcontrol.Backoff) (bool, error) {
	if len(status.SandboxStatuses) == 0 || status.SandboxStatuses[0].State != runtimeapi.PodSandboxState_SANDBOX_READY {
		return false, nil
	}
	if backedOff, _, err := m.doBackOff(ctx, pod, c, status, backOff); backedOff {
		return false, err
	}
	if cs := status.FindContainerStatusByName(c.Name); cs.State == kubecontainer.ContainerStateCreated {
		if err := m.removeContainer(ctx, cs.ID.ID, true); err != nil && !crierror.IsNotFound(err) {
			return false, fmt.Errorf("remove failed sidecar start %q before retrying: %w", c.Name, err)
		}
	}
	sandbox := status.SandboxStatuses[0]
	config, err := m.generatePodSandboxConfig(ctx, pod, sandbox.GetMetadata().GetAttempt())
	if err != nil {
		return false, err
	}
	pulls, err := m.getImageVolumes(ctx, pod, config, pullSecrets)
	if err != nil {
		return false, err
	}
	result := kubecontainer.NewSyncResult(kubecontainer.StartContainer, c.Name)
	volumes, err := m.toKubeContainerImageVolumes(ctx, pulls, c, pod, result)
	if err != nil {
		return false, err
	}
	podIP := ""
	if len(status.IPs) > 0 {
		podIP = status.IPs[0]
	}
	msg, err := m.startContainer(ctx, sandbox.Id, config, containerStartSpec(c), pod, status, pullSecrets, podIP, status.IPs, volumes)
	incrementImageVolumeMetrics(err, msg, c, volumes)
	if err != nil {
		return false, fmt.Errorf("restart sidecar %q during termination of pod %s/%s: %w: %s", c.Name, pod.Namespace, pod.Name, err, msg)
	}
	metrics.SidecarRestartsDuringTerminationTotal.Inc()
	return true, nil
}
