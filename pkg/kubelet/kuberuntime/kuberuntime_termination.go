/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	internalapi "k8s.io/cri-api/pkg/apis"
	crierror "k8s.io/cri-api/pkg/errors"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/cri/remote"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	sc "k8s.io/kubernetes/pkg/securitycontext"
)

// SyncTerminatingPod syncs the terminating pod into the desired state. This
// returns a PodSyncResult and a bool indicating whether the pod is terminated.
//
// 1. computeTerminationPodActions
// 2. restart restartable init containers if its dependent containers are not stopped yet
// 3. kill containers asynchronously to not block the sync loop
func (m *kubeGenericRuntimeManager) SyncTerminatingPod(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, gracePeriodOverride *int64, pullSecrets []v1.Secret, backOff *flowcontrol.Backoff) (kubecontainer.PodSyncResult, bool) {
	var result kubecontainer.PodSyncResult
	containersToStop := make(map[kubecontainer.ContainerID]containerToKillInfo)
	var initContainersToStart []int

	hasInitialized := false
	for _, container := range pod.Spec.Containers {
		status := podStatus.FindContainerStatusByName(container.Name)
		if status == nil {
			continue
		}

		if status.State != kubecontainer.ContainerStateUnknown {
			hasInitialized = true
		}

		if status.State == kubecontainer.ContainerStateExited {
			continue
		}

		containersToStop[status.ID] = containerToKillInfo{
			name:      container.Name,
			container: &container,
			reason:    reasonUnknown,
			canStop:   true,
		}
	}

	canStop := false
	if len(containersToStop) == 0 {
		canStop = true
	}

	if hasInitialized {
		for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
			container := pod.Spec.InitContainers[i]
			status := podStatus.FindContainerStatusByName(container.Name)
			if status == nil {
				continue
			}

			if status.State == kubecontainer.ContainerStateExited {
				if canStop {
					continue
				}

				if !types.IsRestartableInitContainer(&container) {
					continue
				}

				klog.V(4).InfoS("Added the restartable init container to containersToStart since its dependent containers are not stopped yet", "pod", klog.KObj(pod), "podUID", pod.UID, "container", container.Name)
				initContainersToStart = append(initContainersToStart, i)
			}

			klog.V(4).InfoS("Added the the init container to containersToStop", "pod", klog.KObj(pod), "podUID", pod.UID, "container", container.Name, "canStop", canStop)
			containersToStop[status.ID] = containerToKillInfo{
				name:      container.Name,
				container: &container,
				reason:    reasonUnknown,
				canStop:   canStop,
			}
			canStop = false
		}
	} else {
		for _, container := range pod.Spec.InitContainers {
			status := podStatus.FindContainerStatusByName(container.Name)
			if status == nil {
				continue
			}

			if status.State == kubecontainer.ContainerStateExited {
				continue
			}

			containersToStop[status.ID] = containerToKillInfo{
				name:      container.Name,
				container: &container,
				reason:    reasonUnknown,
				canStop:   true,
			}
		}
	}

	_, attempt, podSandboxID := runtimeutil.PodSandboxChanged(pod, podStatus)
	// Get podSandboxConfig for containers to start.
	configPodSandboxResult := kubecontainer.NewSyncResult(kubecontainer.ConfigPodSandbox, podSandboxID)
	result.AddSyncResult(configPodSandboxResult)
	podSandboxConfig, err := m.generatePodSandboxConfig(pod, attempt)
	if err != nil {
		message := fmt.Sprintf("GeneratePodSandboxConfig for pod %q failed: %v", format.Pod(pod), err)
		klog.ErrorS(err, "GeneratePodSandboxConfig for pod failed", "pod", klog.KObj(pod))
		configPodSandboxResult.Fail(kubecontainer.ErrConfigPodSandbox, message)
		return result, false
	}

	var podIPs []string
	if podStatus != nil {
		podIPs = podStatus.IPs
	}

	podIP := ""
	if len(podIPs) != 0 {
		podIP = podIPs[0]
	}

	// Helper containing boilerplate common to starting all types of containers.
	// typeName is a description used to describe this type of container in log messages,
	// currently: "container", "init container" or "ephemeral container"
	// metricLabel is the label used to describe this type of container in monitoring metrics.
	// currently: "container", "init_container" or "ephemeral_container"
	start := func(ctx context.Context, typeName, metricLabel string, spec *startSpec) error {
		startContainerResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, spec.container.Name)
		result.AddSyncResult(startContainerResult)

		isInBackOff, msg, err := m.doBackOff(pod, spec.container, podStatus, backOff)
		if isInBackOff {
			startContainerResult.Fail(err, msg)
			klog.V(4).InfoS("Backing Off restarting container in pod", "containerType", typeName, "container", spec.container.Name, "pod", klog.KObj(pod))
			return err
		}

		metrics.StartedContainersTotal.WithLabelValues(metricLabel).Inc()
		if sc.HasWindowsHostProcessRequest(pod, spec.container) {
			metrics.StartedHostProcessContainersTotal.WithLabelValues(metricLabel).Inc()
		}
		klog.V(4).InfoS("Creating container in pod", "containerType", typeName, "container", spec.container, "pod", klog.KObj(pod))
		// NOTE (aramase) podIPs are populated for single stack and dual stack clusters. Send only podIPs.
		if msg, err := m.startContainer(ctx, podSandboxID, podSandboxConfig, spec, pod, podStatus, pullSecrets, podIP, podIPs); err != nil {
			// startContainer() returns well-defined error codes that have reasonable cardinality for metrics and are
			// useful to cluster administrators to distinguish "server errors" from "user errors".
			metrics.StartedContainersErrorsTotal.WithLabelValues(metricLabel, err.Error()).Inc()
			if sc.HasWindowsHostProcessRequest(pod, spec.container) {
				metrics.StartedHostProcessContainersErrorsTotal.WithLabelValues(metricLabel, err.Error()).Inc()
			}
			startContainerResult.Fail(err, msg)
			// known errors that are logged in other places are logged at higher levels here to avoid
			// repetitive log spam
			switch {
			case err == images.ErrImagePullBackOff:
				klog.V(3).InfoS("Container start failed in pod", "containerType", typeName, "container", spec.container, "pod", klog.KObj(pod), "containerMessage", msg, "err", err)
			default:
				utilruntime.HandleError(fmt.Errorf("%v %+v start failed in pod %v: %v: %s", typeName, spec.container, format.Pod(pod), err, msg))
			}
			return err
		}

		return nil
	}

	for _, i := range initContainersToStart {
		container := pod.Spec.InitContainers[i]
		err := start(ctx, "init container", metrics.InitContainer, containerStartSpec(&container))
		if err != nil {
			klog.V(4).InfoS("Failed to start the restartable init container for the pod, skipping", "initContainerName", container.Name, "pod", klog.KObj(pod))
		}
	}

	for containerID, containerInfo := range containersToStop {
		klog.V(3).InfoS("Trying to terminate container for pod termination", "containerName", containerInfo.name, "containerID", containerID, "pod", klog.KObj(pod))
		killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, containerInfo.name)
		result.AddSyncResult(killContainerResult)
		m.containerTermination.Terminate(pod, containerID, containerInfo.name, containerInfo.canStop, gracePeriodOverride)
	}

	if len(containersToStop) != 0 || len(initContainersToStart) != 0 {
		return result, false
	}

	// stop sandbox, the sandbox will be removed in GarbageCollect
	killSandboxResult := kubecontainer.NewSyncResult(kubecontainer.KillPodSandbox, podStatus.ID)
	result.AddSyncResult(killSandboxResult)
	// Stop all sandboxes belongs to same pod
	for _, podSandbox := range podStatus.SandboxStatuses {
		if err := m.runtimeService.StopPodSandbox(ctx, podSandbox.Id); err != nil && !crierror.IsNotFound(err) {
			killSandboxResult.Fail(kubecontainer.ErrKillPodSandbox, err.Error())
			klog.ErrorS(nil, "Failed to stop sandbox", "podSandboxID", podSandbox.Id)
		}
	}

	m.containerTermination.CompletePod(pod.UID)

	return result, true
}

// containerTermination spawns a worker for each container to terminate the
// container. Each worker immediately executes the pre-stop hook, and then
// waits for the canStop channel to be closed before actually stopping the
// container.
type containerTermination struct {
	lock          sync.Mutex
	isTerminating map[kubetypes.UID]map[string]kubecontainer.ContainerID
	canStop       map[kubetypes.UID]map[string]chan struct{}

	recorder record.EventRecorder

	// gRPC service clients
	runtimeService internalapi.RuntimeService

	// Runner of lifecycle events.
	runner kubecontainer.HandlerRunner
}

func newContainerTermination(recorder record.EventRecorder, runtimeService internalapi.RuntimeService, runner kubecontainer.HandlerRunner) *containerTermination {
	return &containerTermination{
		isTerminating:  make(map[kubetypes.UID]map[string]kubecontainer.ContainerID),
		canStop:        make(map[kubetypes.UID]map[string]chan struct{}),
		recorder:       recorder,
		runtimeService: runtimeService,
		runner:         runner,
	}
}

// Terminate spawns a worker to terminate the container. If canStop is false,
// it will only spawn the worker, execute the pre-stop hook, but not stop the
// container.
func (ct *containerTermination) Terminate(pod *v1.Pod, containerID kubecontainer.ContainerID, containerName string, canStop bool, gracePeriodOverride *int64) {
	ct.lock.Lock()
	defer ct.lock.Unlock()

	uid := pod.UID
	if _, terminating := ct.isTerminating[uid]; !terminating {
		ct.isTerminating[uid] = make(map[string]kubecontainer.ContainerID)
		ct.canStop[uid] = make(map[string]chan struct{})
	}

	terminatingContainerID, terminating := ct.isTerminating[uid][containerName]
	if !terminating || (terminating && terminatingContainerID != containerID) {
		if terminating && terminatingContainerID != containerID {
			ch, ok := ct.canStop[uid][containerName]
			if !ok {
				return
			}

			klog.V(3).InfoS("Closed the channel to finish the previous container termination worker", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)
			close(ch)
			delete(ct.canStop[uid], containerName)
		}

		ct.isTerminating[uid][containerName] = containerID
		ch := make(chan struct{})
		ct.canStop[uid][containerName] = ch
		go func() {
			defer utilruntime.HandleCrash()
			klog.V(3).InfoS("Container termination worker started", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)
			defer klog.V(3).InfoS("Container termination worker stopped", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)
			ct.doTerminate(context.TODO(), pod, containerID, containerName, ch, gracePeriodOverride)
			ct.lock.Lock()
			defer ct.lock.Unlock()
			delete(ct.isTerminating[uid], containerName)
			delete(ct.canStop[uid], containerName)
		}()
	}

	if !canStop {
		return
	}

	ch, ok := ct.canStop[uid][containerName]
	if !ok {
		return
	}

	klog.V(3).InfoS("Closed the channel to stop container", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)
	close(ch)
	delete(ct.canStop[uid], containerName)
}

func (ct *containerTermination) CompletePod(uid kubetypes.UID) {
	ct.lock.Lock()
	defer ct.lock.Unlock()

	delete(ct.isTerminating, uid)
	delete(ct.canStop, uid)
}

func (ct *containerTermination) doTerminate(ctx context.Context, pod *v1.Pod, containerID kubecontainer.ContainerID, containerName string, canStop <-chan struct{}, gracePeriodOverride *int64) {
	var containerSpec *v1.Container
	if pod != nil {
		if containerSpec = kubecontainer.GetContainerSpec(pod, containerName); containerSpec == nil {
			klog.ErrorS(nil, "Failed to get container spec when terminating container", "pod", klog.KObj(pod), "containerID", containerID.String(), "containerName", containerName)
			return
		}
	} else {
		// Restore necessary information if one of the specs is nil.
		restoredPod, restoredContainer, err := ct.restoreSpecsFromContainerLabels(ctx, containerID)
		if err != nil {
			klog.ErrorS(err, "Failed to restore pod and container spec from container labels", "pod", klog.KObj(pod), "containerID", containerID.String(), "containerName", containerName)
			return
		}
		pod, containerSpec = restoredPod, restoredContainer
	}

	// From this point, pod and container must be non-nil.
	gracePeriod := setTerminationGracePeriod(pod, containerSpec, containerName, containerID, reasonUnknown)

	message := fmt.Sprintf("Stopping container %s", containerSpec.Name)
	ct.recordContainerEvent(pod, containerSpec, containerID.ID, v1.EventTypeNormal, events.KillingContainer, message)

	if gracePeriodOverride != nil {
		gracePeriod = *gracePeriodOverride
		klog.V(3).InfoS("Terminating container with a grace period override", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String(), "gracePeriod", gracePeriod)
	}

	// Run the pre-stop lifecycle hooks if applicable and if there is enough time to run it
	if containerSpec.Lifecycle != nil && containerSpec.Lifecycle.PreStop != nil && gracePeriod > 0 {
		gracePeriod -= ct.executePreStopHook(ctx, pod, containerID, containerSpec, gracePeriod)
	}

	start := time.Now()
	select {
	case <-canStop:
		klog.V(2).InfoS("Container can be stopped now", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String())
	case <-time.After(time.Duration(gracePeriod) * time.Second):
		// the termination grace period has expired, kill the container
		klog.V(2).InfoS("The termination grace period has expired, stopping the container", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String())
	}

	gracePeriod -= int64(time.Since(start).Seconds())

	// always give containers a minimal shutdown window to avoid unnecessary SIGKILLs
	if gracePeriod < minimumGracePeriodInSeconds {
		gracePeriod = minimumGracePeriodInSeconds
	}

	klog.V(2).InfoS("Stopping container with a grace period", "pod", klog.KObj(pod), "podUID", pod.UID,
		"containerName", containerName, "containerID", containerID.String(), "gracePeriod", gracePeriod)

	err := ct.runtimeService.StopContainer(ctx, containerID.ID, gracePeriod)
	if err != nil && !crierror.IsNotFound(err) {
		klog.ErrorS(err, "Container termination failed with gracePeriod", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerName, "containerID", containerID.String(), "gracePeriod", gracePeriod)
		return
	}
	klog.V(3).InfoS("Container exited normally", "pod", klog.KObj(pod), "podUID", pod.UID,
		"containerName", containerName, "containerID", containerID.String())
}

// restoreSpecsFromContainerLabels restores all information needed for killing
// a container. In some case we may not have pod and container spec when
// killing a container, e.g. pod is deleted during kubelet restart.
// To solve this problem, we've already written necessary information into
// container labels. Here we just need to retrieve them from container labels
// and restore the specs.
// TODO(random-liu): Add a node e2e test to test this behaviour.
// TODO(random-liu): Change the lifecycle handler to just accept information
// needed, so that we can just pass the needed function not create the fake
// object.
func (ct *containerTermination) restoreSpecsFromContainerLabels(ctx context.Context, containerID kubecontainer.ContainerID) (*v1.Pod, *v1.Container, error) {
	var pod *v1.Pod
	var container *v1.Container
	resp, err := ct.runtimeService.ContainerStatus(ctx, containerID.ID, false)
	if err != nil {
		return nil, nil, err
	}
	s := resp.GetStatus()
	if s == nil {
		return nil, nil, remote.ErrContainerStatusNil
	}

	l := getContainerInfoFromLabels(s.Labels)
	a := getContainerInfoFromAnnotations(s.Annotations)
	// Notice that the followings are not full spec. The container killing code should not use
	// un-restored fields.
	pod = &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:                        l.PodUID,
			Name:                       l.PodName,
			Namespace:                  l.PodNamespace,
			DeletionGracePeriodSeconds: a.PodDeletionGracePeriod,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: a.PodTerminationGracePeriod,
		},
	}
	container = &v1.Container{
		Name:                   l.ContainerName,
		Ports:                  a.ContainerPorts,
		TerminationMessagePath: a.TerminationMessagePath,
	}
	if a.PreStopHandler != nil {
		container.Lifecycle = &v1.Lifecycle{
			PreStop: a.PreStopHandler,
		}
	}
	return pod, container, nil
}

// executePreStopHook runs the pre-stop lifecycle hooks if applicable and
// returns the duration it takes.
func (ct *containerTermination) executePreStopHook(ctx context.Context, pod *v1.Pod, containerID kubecontainer.ContainerID, containerSpec *v1.Container, gracePeriod int64) int64 {
	klog.V(3).InfoS("Running preStop hook", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerSpec.Name, "containerID", containerID.String())

	start := metav1.Now()
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer utilruntime.HandleCrash()
		if _, err := ct.runner.Run(ctx, containerID, pod, containerSpec, containerSpec.Lifecycle.PreStop); err != nil {
			klog.ErrorS(err, "PreStop hook failed", "pod", klog.KObj(pod), "podUID", pod.UID,
				"containerName", containerSpec.Name, "containerID", containerID.String())
			// do not record the message in the event so that secrets won't leak from the server.
			ct.recordContainerEvent(pod, containerSpec, containerID.ID, v1.EventTypeWarning, events.FailedPreStopHook, "PreStopHook failed")
		}
	}()

	select {
	case <-time.After(time.Duration(gracePeriod) * time.Second):
		klog.V(2).InfoS("PreStop hook not completed in grace period", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerSpec.Name, "containerID", containerID.String(), "gracePeriod", gracePeriod)
	case <-done:
		klog.V(3).InfoS("PreStop hook completed", "pod", klog.KObj(pod), "podUID", pod.UID,
			"containerName", containerSpec.Name, "containerID", containerID.String())
	}

	return int64(metav1.Now().Sub(start.Time).Seconds())
}

// recordContainerEvent should be used by the runtime manager for all container
// related events. it has sanity checks to ensure that we do not write events
// that can abuse our masters. in particular, it ensures that a containerID
// never appears in an event message as that is prone to causing a lot of
// distinct events that do not count well. it replaces any reference to a
// containerID with the containerName which is stable, and is what users know.
func (ct *containerTermination) recordContainerEvent(pod *v1.Pod, container *v1.Container, containerID, eventType, reason, message string, args ...interface{}) {
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		klog.ErrorS(err, "Can't make a container ref", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name)
		return
	}
	eventMessage := message
	if len(args) > 0 {
		eventMessage = fmt.Sprintf(message, args...)
	}
	// this is a hack, but often the error from the runtime includes the containerID
	// which kills our ability to deduplicate events.  this protection makes a huge
	// difference in the number of unique events
	if containerID != "" {
		eventMessage = strings.ReplaceAll(eventMessage, containerID, container.Name)
	}
	ct.recorder.Event(ref, eventType, reason, eventMessage)
}
