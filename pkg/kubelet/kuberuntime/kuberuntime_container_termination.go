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
	internalapi "k8s.io/cri-api/pkg/apis"
	crierror "k8s.io/cri-api/pkg/errors"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/cri/remote"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

type terminationResult struct {
	containerName string
	containerID   kubecontainer.ContainerID
	err           error
}

// containerTermination spawns a worker for each container to terminate the
// container. Each worker immediately executes the pre-stop hook, and then
// waits for the its turn to terminate the container.
type containerTermination struct {
	lock       sync.Mutex
	containers map[kubetypes.UID]map[string]kubecontainer.ContainerID
	ordering   map[kubetypes.UID]*terminationOrdering

	recorder record.EventRecorder

	// gRPC service clients
	runtimeService internalapi.RuntimeService

	// Runner of lifecycle events.
	runner kubecontainer.HandlerRunner

	results map[kubetypes.UID][]terminationResult
}

func newContainerTermination(recorder record.EventRecorder, runtimeService internalapi.RuntimeService, runner kubecontainer.HandlerRunner) *containerTermination {
	return &containerTermination{
		containers:     make(map[kubetypes.UID]map[string]kubecontainer.ContainerID),
		ordering:       make(map[kubetypes.UID]*terminationOrdering),
		recorder:       recorder,
		runtimeService: runtimeService,
		runner:         runner,
		results:        make(map[kubetypes.UID][]terminationResult),
	}
}

// Terminate spawns a worker to terminate the container if it is not already
// scheduled.
func (ct *containerTermination) Terminate(pod *v1.Pod, containerID kubecontainer.ContainerID, containerName string, gracePeriodOverride *int64, ordering *terminationOrdering) {
	ct.lock.Lock()
	defer ct.lock.Unlock()

	podUID := pod.UID
	if _, scheduled := ct.containers[podUID]; !scheduled {
		ct.containers[podUID] = make(map[string]kubecontainer.ContainerID)
		ct.ordering[podUID] = ordering
	}

	_, scheduled := ct.containers[podUID][containerName]
	if scheduled {
		klog.V(3).InfoS("Container termination worker already scheduled", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)
		return
	}

	ct.containers[podUID][containerName] = containerID
	go func(ctx context.Context) {
		defer utilruntime.HandleCrash()
		klog.V(3).InfoS("Container termination worker started", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)
		defer klog.V(3).InfoS("Container termination worker stopped", "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", containerName)

		err := ct.doTerminate(ctx, pod, containerID, containerName, "", reasonUnknown, gracePeriodOverride, ordering)
		ct.lock.Lock()
		defer ct.lock.Unlock()
		ct.results[podUID] = append(ct.results[podUID], terminationResult{
			containerName: containerName,
			containerID:   containerID,
			err:           err,
		})
	}(context.TODO())
}

func (ct *containerTermination) IsTerminating(podUID kubetypes.UID, containerName string) bool {
	ct.lock.Lock()
	defer ct.lock.Unlock()

	if _, exists := ct.containers[podUID]; !exists {
		return false
	}

	_, scheduled := ct.containers[podUID][containerName]
	return scheduled
}

func (ct *containerTermination) DrainResults(podUID kubetypes.UID) []terminationResult {
	ct.lock.Lock()
	defer ct.lock.Unlock()

	results := ct.results[podUID]
	ct.results[podUID] = nil

	for _, result := range results {
		klog.V(3).InfoS("Drained container termination result", "podUID", podUID, "result", result)
		delete(ct.containers[podUID], result.containerName)
	}

	if len(ct.containers[podUID]) == 0 {
		klog.V(3).InfoS("All container terminations have completed", "podUID", podUID)
		delete(ct.containers, podUID)
		delete(ct.ordering, podUID)
		delete(ct.results, podUID)
	}

	return results
}

// TODO: Remove (*kubeGenericRuntimeManager).killContainer
func (ct *containerTermination) doTerminate(ctx context.Context, pod *v1.Pod, containerID kubecontainer.ContainerID, containerName string, message string, reason containerKillReason, gracePeriodOverride *int64, ordering *terminationOrdering) error {
	var containerSpec *v1.Container
	if pod != nil {
		if containerSpec = kubecontainer.GetContainerSpec(pod, containerName); containerSpec == nil {
			return fmt.Errorf("failed to get containerSpec %q (id=%q) in pod %q when killing container for reason %q",
				containerName, containerID.String(), format.Pod(pod), message)
		}
	} else {
		// Restore necessary information if one of the specs is nil.
		restoredPod, restoredContainer, err := ct.restoreSpecsFromContainerLabels(ctx, containerID)
		if err != nil {
			return err
		}
		pod, containerSpec = restoredPod, restoredContainer
	}

	// From this point, pod and container must be non-nil.
	gracePeriod := setTerminationGracePeriod(pod, containerSpec, containerName, containerID, reason)

	if len(message) == 0 {
		message = fmt.Sprintf("Stopping container %s", containerSpec.Name)
	}
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

	// if we care about termination ordering, then wait for this container's turn to exit if there is
	// time remaining
	if ordering != nil && gracePeriod > 0 {
		// grace period is only in seconds, so the time we've waited gets truncated downward
		gracePeriod -= int64(ordering.waitForTurn(containerName, gracePeriod))
	}

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
		return err
	}
	klog.V(3).InfoS("Container exited normally", "pod", klog.KObj(pod), "podUID", pod.UID,
		"containerName", containerName, "containerID", containerID.String())

	if ordering != nil {
		ordering.containerTerminated(containerName)
	}

	return nil
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
