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

package allocation

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/kubelet/sysctl"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
)

// podStatusManagerStateFile is the file name where status manager stores its state
const (
	allocatedPodsStateFile = "allocated_pods_state"
	actuatedPodsStateFile  = "actuated_pods_state"
)

// AllocationManager tracks pod resource allocations.
type Manager interface {
	// GetContainerResourceAllocation returns the AllocatedResources value for the container
	GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)

	// UpdatePodFromAllocation overwrites the pod spec with the allocation.
	// This function does a deep copy only if updates are needed.
	// Returns the updated (or original) pod, and whether there was an allocation stored.
	UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool)

	// SetAllocatedResources checkpoints the resources allocated to a pod's containers.
	SetAllocatedResources(allocatedPod *v1.Pod) error

	// SetActuatedResources records the actuated resources of the given container (or the entire
	// pod, if actuatedContainer is nil).
	SetActuatedResources(allocatedPod *v1.Pod, actuatedContainer *v1.Container) error

	// GetActuatedResources returns the stored actuated resources for the container, and whether they exist.
	GetActuatedResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)

	// AddPodAdmitHandlers adds the admit handlers to the allocation manager.
	AddPodAdmitHandlers(handlers lifecycle.PodAdmitHandlers)

	// AddPod checks if a pod can be admitted. If so, it admits the pod and updates the allocation.
	// The function returns a boolean value indicating whether the pod
	// can be admitted, a brief single-word reason and a message explaining why
	// the pod cannot be admitted.
	// allocatedPods should represent the pods that have already been admitted, along with their
	// admitted (allocated) resources.
	AddPod(allocatedPods []*v1.Pod, pod *v1.Pod) (ok bool, reason, message string)

	// RemovePod removes any stored state for the given pod UID.
	RemovePod(uid types.UID)

	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])

	// HandlePodResourcesResize returns the "allocated pod", which should be used for all resource
	// calculations after this function is called. It also updates the cached ResizeStatus according to
	// the allocation decision and pod status.
	HandlePodResourcesResize(allocatedPods []*v1.Pod, pod *v1.Pod, getNodeFunc func() (*v1.Node, error), podStatus *kubecontainer.PodStatus) (*v1.Pod, error)

	// IsPodResizingInProgress checks whether the actuated resizable resources differ from the allocated resources
	// for any running containers. Specifically, the following differences are ignored:
	// - Non-resizable containers: non-restartable init containers, ephemeral containers
	// - Non-resizable resources: only CPU & memory are resizable
	// - Non-running containers: they will be sized correctly when (re)started
	IsPodResizeInProgress(allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool
}

type manager struct {
	allocated state.State
	actuated  state.State

	admitHandlers    lifecycle.PodAdmitHandlers
	containerManager cm.ContainerManager
	statusManager    status.Manager

	allocationMutex sync.Mutex
}

func NewManager(checkpointDirectory string, containerManager cm.ContainerManager, statusManager status.Manager) Manager {
	return &manager{
		allocated: newStateImpl(checkpointDirectory, allocatedPodsStateFile),
		actuated:  newStateImpl(checkpointDirectory, actuatedPodsStateFile),

		containerManager: containerManager,
		statusManager:    statusManager,
		admitHandlers:    lifecycle.PodAdmitHandlers{},
	}
}

func newStateImpl(checkpointDirectory, checkpointName string) state.State {
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		return state.NewNoopStateCheckpoint()
	}

	stateImpl, err := state.NewStateCheckpoint(checkpointDirectory, checkpointName)
	if err != nil {
		// This is a critical, non-recoverable failure.
		klog.ErrorS(err, "Failed to initialize allocation checkpoint manager",
			"checkpointPath", filepath.Join(checkpointDirectory, checkpointName))
		panic(err)
	}

	return stateImpl
}

// NewInMemoryManager returns an allocation manager that doesn't persist state.
// For testing purposes only!
func NewInMemoryManager(containerManager cm.ContainerManager, statusManager status.Manager) Manager {
	return &manager{
		allocated: state.NewStateMemory(nil),
		actuated:  state.NewStateMemory(nil),

		containerManager: containerManager,
		statusManager:    statusManager,
		admitHandlers:    lifecycle.PodAdmitHandlers{},
	}
}

// GetContainerResourceAllocation returns the last checkpointed AllocatedResources values
// If checkpoint manager has not been initialized, it returns nil, false
func (m *manager) GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	return m.allocated.GetContainerResources(podUID, containerName)
}

// UpdatePodFromAllocation overwrites the pod spec with the allocation.
// This function does a deep copy only if updates are needed.
func (m *manager) UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool) {
	// TODO(tallclair): This clones the whole cache, but we only need 1 pod.
	allocs := m.allocated.GetPodResourceInfoMap()
	return updatePodFromAllocation(pod, allocs)
}

func updatePodFromAllocation(pod *v1.Pod, allocs state.PodResourceInfoMap) (*v1.Pod, bool) {
	allocated, found := allocs[pod.UID]
	if !found {
		return pod, false
	}

	updated := false
	containerAlloc := func(c v1.Container) (v1.ResourceRequirements, bool) {
		if cAlloc, ok := allocated.ContainerResources[c.Name]; ok {
			if !apiequality.Semantic.DeepEqual(c.Resources, cAlloc) {
				// Allocation differs from pod spec, retrieve the allocation
				if !updated {
					// If this is the first update to be performed, copy the pod
					pod = pod.DeepCopy()
					updated = true
				}
				return cAlloc, true
			}
		}
		return v1.ResourceRequirements{}, false
	}

	for i, c := range pod.Spec.Containers {
		if cAlloc, found := containerAlloc(c); found {
			// Allocation differs from pod spec, update
			pod.Spec.Containers[i].Resources = cAlloc
		}
	}
	for i, c := range pod.Spec.InitContainers {
		if cAlloc, found := containerAlloc(c); found {
			// Allocation differs from pod spec, update
			pod.Spec.InitContainers[i].Resources = cAlloc
		}
	}
	return pod, updated
}

// SetAllocatedResources checkpoints the resources allocated to a pod's containers
func (m *manager) SetAllocatedResources(pod *v1.Pod) error {
	return m.allocated.SetPodResourceInfo(pod.UID, allocationFromPod(pod))
}

func allocationFromPod(pod *v1.Pod) state.PodResourceInfo {
	var podAlloc state.PodResourceInfo
	podAlloc.ContainerResources = make(map[string]v1.ResourceRequirements)
	for _, container := range pod.Spec.Containers {
		alloc := *container.Resources.DeepCopy()
		podAlloc.ContainerResources[container.Name] = alloc
	}

	for _, container := range pod.Spec.InitContainers {
		if podutil.IsRestartableInitContainer(&container) {
			alloc := *container.Resources.DeepCopy()
			podAlloc.ContainerResources[container.Name] = alloc
		}
	}

	return podAlloc
}

func (m *manager) AddPodAdmitHandlers(handlers lifecycle.PodAdmitHandlers) {
	for _, a := range handlers {
		m.admitHandlers.AddPodAdmitHandler(a)
	}
}

func (m *manager) AddPod(allocatedPods []*v1.Pod, pod *v1.Pod) (bool, string, string) {
	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()

	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// To handle kubelet restarts, test pod admissibility using AllocatedResources values
		// (for cpu & memory) from checkpoint store. If found, that is the source of truth.
		pod, _ = m.UpdatePodFromAllocation(pod)
	}

	// Check if we can admit the pod; if so, update the allocation.
	ok, reason, message := m.canAdmitPod(allocatedPods, pod)

	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// Checkpoint the resource values at which the Pod has been admitted or resized.
		if err := m.SetAllocatedResources(pod); err != nil {
			//TODO(vinaykul,InPlacePodVerticalScaling): Can we recover from this in some way? Investigate
			klog.ErrorS(err, "SetPodAllocation failed", "pod", klog.KObj(pod))
		}
	}

	return ok, reason, message
}

func (m *manager) RemovePod(uid types.UID) {
	if err := m.allocated.RemovePod(uid); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		klog.V(3).ErrorS(err, "Failed to delete pod allocation", "podUID", uid)
	}

	if err := m.actuated.RemovePod(uid); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		klog.V(3).ErrorS(err, "Failed to delete pod allocation", "podUID", uid)
	}
}

func (m *manager) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	m.allocated.RemoveOrphanedPods(remainingPods)
	m.actuated.RemoveOrphanedPods(remainingPods)
}

func (m *manager) SetActuatedResources(allocatedPod *v1.Pod, actuatedContainer *v1.Container) error {
	if actuatedContainer == nil {
		alloc := allocationFromPod(allocatedPod)
		return m.actuated.SetPodResourceInfo(allocatedPod.UID, alloc)
	}

	return m.actuated.SetContainerResources(allocatedPod.UID, actuatedContainer.Name, actuatedContainer.Resources)
}

func (m *manager) GetActuatedResources(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	return m.actuated.GetContainerResources(podUID, containerName)
}

func (m *manager) HandlePodResourcesResize(
	allocatedPods []*v1.Pod,
	pod *v1.Pod,
	getNodeFunc func() (*v1.Node, error),
	podStatus *kubecontainer.PodStatus) (allocatedPod *v1.Pod, err error) {
	// Always check whether a resize is in progress so we can set the PodResizeInProgressCondition
	// accordingly.
	defer func() {
		if err != nil {
			return
		}
		if m.IsPodResizeInProgress(allocatedPod, podStatus) {
			// If a resize is in progress, make sure the cache has the correct state in case the Kubelet restarted.
			m.statusManager.SetPodResizeInProgressCondition(pod.UID, "", "", false)
		} else {
			// (Allocated == Actual) => clear the resize in-progress status.
			m.statusManager.ClearPodResizeInProgressCondition(pod.UID)
		}
	}()

	podFromAllocation, updated := m.UpdatePodFromAllocation(pod)
	if !updated {
		// Desired resources == allocated resources. Pod allocation does not need to be updated.
		m.statusManager.ClearPodResizePendingCondition(pod.UID)
		return podFromAllocation, nil

	} else if resizable, msg := IsInPlacePodVerticalScalingAllowed(pod); !resizable {
		// If there is a pending resize but the resize is not allowed, always use the allocated resources.
		m.statusManager.SetPodResizePendingCondition(pod.UID, v1.PodReasonInfeasible, msg)
		return podFromAllocation, nil
	}

	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()
	// Desired resources != allocated resources. Can we update the allocation to the desired resources?
	fit, reason, message := m.canResizePod(allocatedPods, pod, getNodeFunc)
	if fit {
		// Update pod resource allocation checkpoint
		if err := m.SetAllocatedResources(pod); err != nil {
			return nil, err
		}
		m.statusManager.ClearPodResizePendingCondition(pod.UID)

		// Clear any errors that may have been surfaced from a previous resize. The condition will be
		// added back as needed in the defer block, but this prevents old errors from being preserved.
		m.statusManager.ClearPodResizeInProgressCondition(pod.UID)

		return pod, nil
	}

	if reason != "" {
		m.statusManager.SetPodResizePendingCondition(pod.UID, reason, message)
	}

	return podFromAllocation, nil
}

// canAdmitPod determines if a pod can be admitted, and gives a reason if it
// cannot. "pod" is new pod, while "pods" are all admitted pods
// The function returns a boolean value indicating whether the pod
// can be admitted, a brief single-word reason and a message explaining why
// the pod cannot be admitted.
// allocatedPods should represent the pods that have already been admitted, along with their
// admitted (allocated) resources.
func (m *manager) canAdmitPod(allocatedPods []*v1.Pod, pod *v1.Pod) (bool, string, string) {
	// Filter out the pod being evaluated.
	allocatedPods = slices.DeleteFunc(allocatedPods, func(p *v1.Pod) bool { return p.UID == pod.UID })

	// the kubelet will invoke each pod admit handler in sequence
	// if any handler rejects, the pod is rejected.
	// TODO: move out of disk check into a pod admitter
	// TODO: out of resource eviction should have a pod admitter call-out
	attrs := &lifecycle.PodAdmitAttributes{Pod: pod, OtherPods: allocatedPods}
	for _, podAdmitHandler := range m.admitHandlers {
		if result := podAdmitHandler.Admit(attrs); !result.Admit {
			klog.InfoS("Pod admission denied", "podUID", attrs.Pod.UID, "pod", klog.KObj(attrs.Pod), "reason", result.Reason, "message", result.Message)

			recordAdmissionRejection(result.Reason)
			return false, result.Reason, result.Message
		}
	}

	return true, "", ""
}

// canResizePod determines if the requested resize is currently feasible.
// pod should hold the desired (pre-allocated) spec.
// Returns true if the resize can proceed; returns a reason and message
// otherwise.
func (m *manager) canResizePod(allocatedPods []*v1.Pod, pod *v1.Pod, getNodeFunc func() (*v1.Node, error)) (bool, string, string) {
	if v1qos.GetPodQOS(pod) == v1.PodQOSGuaranteed && !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) {
		if m.containerManager.GetNodeConfig().CPUManagerPolicy == "static" {
			msg := "Resize is infeasible for Guaranteed Pods alongside CPU Manager static policy"
			klog.V(3).InfoS(msg, "pod", format.Pod(pod))
			return false, v1.PodReasonInfeasible, msg
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.MemoryManager) {
			if m.containerManager.GetNodeConfig().MemoryManagerPolicy == "Static" {
				msg := "Resize is infeasible for Guaranteed Pods alongside Memory Manager static policy"
				klog.V(3).InfoS(msg, "pod", format.Pod(pod))
				return false, v1.PodReasonInfeasible, msg

			}
		}
	}

	node, err := getNodeFunc()
	if err != nil {
		klog.ErrorS(err, "getNode function failed")
		return false, "", ""
	}
	cpuAvailable := node.Status.Allocatable.Cpu().MilliValue()
	memAvailable := node.Status.Allocatable.Memory().Value()
	cpuRequests := resource.GetResourceRequest(pod, v1.ResourceCPU)
	memRequests := resource.GetResourceRequest(pod, v1.ResourceMemory)
	if cpuRequests > cpuAvailable || memRequests > memAvailable {
		var msg string
		if memRequests > memAvailable {
			msg = fmt.Sprintf("memory, requested: %d, capacity: %d", memRequests, memAvailable)
		} else {
			msg = fmt.Sprintf("cpu, requested: %d, capacity: %d", cpuRequests, cpuAvailable)
		}
		msg = "Node didn't have enough capacity: " + msg
		klog.V(3).InfoS(msg, "pod", klog.KObj(pod))
		return false, v1.PodReasonInfeasible, msg

	}

	if ok, failReason, failMessage := m.canAdmitPod(allocatedPods, pod); !ok {
		// Log reason and return. Let the next sync iteration retry the resize
		klog.V(3).InfoS("Resize cannot be accommodated", "pod", klog.KObj(pod), "reason", failReason, "message", failMessage)
		return false, v1.PodReasonDeferred, failMessage
	}

	return true, "", ""
}

func (m *manager) IsPodResizeInProgress(allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	return !podutil.VisitContainers(&allocatedPod.Spec, podutil.InitContainers|podutil.Containers,
		func(allocatedContainer *v1.Container, containerType podutil.ContainerType) (shouldContinue bool) {
			if !isResizableContainer(allocatedContainer, containerType) {
				return true
			}

			containerStatus := podStatus.FindContainerStatusByName(allocatedContainer.Name)
			if containerStatus == nil || containerStatus.State != kubecontainer.ContainerStateRunning {
				// If the container isn't running, it doesn't need to be resized.
				return true
			}

			actuatedResources, _ := m.GetActuatedResources(allocatedPod.UID, allocatedContainer.Name)
			allocatedResources := allocatedContainer.Resources

			return allocatedResources.Requests[v1.ResourceCPU].Equal(actuatedResources.Requests[v1.ResourceCPU]) &&
				allocatedResources.Limits[v1.ResourceCPU].Equal(actuatedResources.Limits[v1.ResourceCPU]) &&
				allocatedResources.Requests[v1.ResourceMemory].Equal(actuatedResources.Requests[v1.ResourceMemory]) &&
				allocatedResources.Limits[v1.ResourceMemory].Equal(actuatedResources.Limits[v1.ResourceMemory])
		})
}

var (
	admissionRejectionReasons = sets.New[string](
		lifecycle.AppArmorNotAdmittedReason,
		lifecycle.PodOSSelectorNodeLabelDoesNotMatch,
		lifecycle.PodOSNotSupported,
		lifecycle.InvalidNodeInfo,
		lifecycle.InitContainerRestartPolicyForbidden,
		lifecycle.SupplementalGroupsPolicyNotSupported,
		lifecycle.UnexpectedAdmissionError,
		lifecycle.UnknownReason,
		lifecycle.UnexpectedPredicateFailureType,
		lifecycle.OutOfCPU,
		lifecycle.OutOfMemory,
		lifecycle.OutOfEphemeralStorage,
		lifecycle.OutOfPods,
		tainttoleration.ErrReasonNotMatch,
		eviction.Reason,
		sysctl.ForbiddenReason,
		topologymanager.ErrorTopologyAffinity,
		nodeshutdown.NodeShutdownNotAdmittedReason,
	)
)

func recordAdmissionRejection(reason string) {
	// It is possible that the "reason" label can have high cardinality.
	// To avoid this metric from exploding, we create an allowlist of known
	// reasons, and only record reasons from this list. Use "Other" reason
	// for the rest.
	if admissionRejectionReasons.Has(reason) {
		metrics.AdmissionRejectionsTotal.WithLabelValues(reason).Inc()
	} else if strings.HasPrefix(reason, lifecycle.InsufficientResourcePrefix) {
		// non-extended resources (like cpu, memory, ephemeral-storage, pods)
		// are already included in admissionRejectionReasons.
		metrics.AdmissionRejectionsTotal.WithLabelValues("OutOfExtendedResources").Inc()
	} else {
		metrics.AdmissionRejectionsTotal.WithLabelValues("Other").Inc()
	}
}

func isResizableContainer(container *v1.Container, containerType podutil.ContainerType) bool {
	switch containerType {
	case podutil.InitContainers:
		return podutil.IsRestartableInitContainer(container)
	case podutil.Containers:
		return true
	default:
		return false
	}
}
