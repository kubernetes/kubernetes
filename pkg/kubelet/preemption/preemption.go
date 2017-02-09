/*
Copyright 2017 The Kubernetes Authors.

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

package preemption

import (
	"fmt"

	"github.com/golang/glog"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

const (
	reason  = "Preempted"
	message = "Preempted in order to admit critical pod"
)

// CriticalPodAdmissionFailureHandler is an AdmissionFailureHandler that handles admission failure for Critical Pods.
// If the ONLY admission failures are due to insufficient resources, then the CriticalPodAdmissionHandler evicts pods
// so that the critical pod can be admitted.  For evictions, the CriticalPodAdmissionHandler evicts a set of pods that
// frees up the required resource requests.  The set of pods is designed to minimize impact, and is prioritized according to the ordering:
// smallest number of Guaranteed pods > smallest number of Burstable pods > smallest Memory Requests > smallest CPU Request
type CriticalPodAdmissionHandler struct {
	getPodsFunc eviction.ActivePodsFunc
	killPodFunc eviction.KillPodFunc
	recorder    record.EventRecorder
}

var _ lifecycle.AdmissionFailureHandler = &CriticalPodAdmissionHandler{}

func NewCriticalPodAdmissionHandler(getPodsFunc eviction.ActivePodsFunc, killPodFunc eviction.KillPodFunc, recorder record.EventRecorder) *CriticalPodAdmissionHandler {
	return &CriticalPodAdmissionHandler{
		getPodsFunc: getPodsFunc,
		killPodFunc: killPodFunc,
		recorder:    recorder,
	}
}

type admissionRequirement struct {
	resourceName v1.ResourceName
	quantity     int64
}

type admissionRequirementList []*admissionRequirement

func (a admissionRequirementList) sub(pod *v1.Pod) admissionRequirementList {
	requests := predicates.GetResourceRequest(pod)
	newList := []*admissionRequirement{}
	for _, req := range a {
		value := getResourceRequest(requests, req.resourceName)
		newQuantity := req.quantity - value
		if newQuantity > 0 {
			newList = append(newList, &admissionRequirement{
				resourceName: req.resourceName,
				quantity:     newQuantity,
			})
		}
	}
	return newList
}

// HandleAdmissionFailure gracefully handles admission rejection, and, in some cases,
// to allow admission of the pod despite its previous failure.
func (c *CriticalPodAdmissionHandler) HandleAdmissionFailure(pod *v1.Pod, failureReasons []algorithm.PredicateFailureReason) (bool, []algorithm.PredicateFailureReason, error) {
	if !kubetypes.IsCriticalPod(pod) || !utilfeature.DefaultFeatureGate.Enabled(features.ExperimentalCriticalPodAnnotation) {
		return false, failureReasons, nil
	}
	// InsufficientResourceError is not a reason to reject a critical pod.
	// Instead of rejecting, we free up resources to admit it, if no other reasons for rejection exist.
	nonResourceReasons := []algorithm.PredicateFailureReason{}
	resourceReasons := []*admissionRequirement{}
	for _, reason := range failureReasons {
		if r, ok := reason.(*predicates.InsufficientResourceError); ok {
			resourceReasons = append(resourceReasons, &admissionRequirement{
				resourceName: r.ResourceName,
				quantity:     r.GetInsufficientAmount(),
			})
		} else {
			nonResourceReasons = append(nonResourceReasons, reason)
		}
	}
	if len(nonResourceReasons) > 0 {
		// Return only reasons that are not resource related, since critical pods cannot fail admission for resource reasons.
		return false, nonResourceReasons, nil
	}
	err := c.evictPodsToFreeRequests(admissionRequirementList(resourceReasons))
	// if no error is returned, preemption succeeded and the pod is safe to admit.
	return err == nil, nil, err
}

// freeRequests takes a list of insufficient resources, and attempts to free them by evicting pods
// based on requests.  For example, if the only insufficient resource is 200Mb of memory, this function could
// evict a pod with request=250Mb.
func (c *CriticalPodAdmissionHandler) evictPodsToFreeRequests(insufficientResources admissionRequirementList) error {
	podsToPreempt, err := getPodsToPreempt(c.getPodsFunc(), insufficientResources)
	if err != nil {
		return fmt.Errorf("Error finding a set of pods to preempt: %v", err)
	}
	glog.Infof("preemption: attempting to evict pods %v, in order to free up resources: %v", podsToPreempt, insufficientResources)
	for _, pod := range podsToPreempt {
		status := v1.PodStatus{
			Phase:   v1.PodFailed,
			Message: message,
			Reason:  reason,
		}
		// record that we are evicting the pod
		c.recorder.Eventf(pod, v1.EventTypeWarning, reason, message)
		// this is a blocking call and should only return when the pod and its containers are killed.
		err := c.killPodFunc(pod, status, nil)
		if err != nil {
			return fmt.Errorf("preemption: pod %s failed to evict %v", format.Pod(pod), err)
		}
		glog.Infof("preemption: pod %s evicted successfully", format.Pod(pod))
	}
	return nil
}

// getPodsToPreempt returns a list of pods that could be preempted to free requests >= insufficientResources
func getPodsToPreempt(possiblePods []*v1.Pod, insufficientResources admissionRequirementList) ([]*v1.Pod, error) {
	if len(insufficientResources) == 0 {
		// All requirements have been met! We do not need to preempt any pods.
		return []*v1.Pod{}, nil
	}
	if len(possiblePods) > 0 {
		withoutNextPod, errWithout := getPodsToPreempt(possiblePods[1:], insufficientResources)
		withNextPod, errWith := getPodsToPreempt(possiblePods[1:], insufficientResources.sub(possiblePods[0]))
		if errWith == nil {
			withNextPod = append(withNextPod, possiblePods[0])
			if errWithout == nil {
				return minCostPodList(withoutNextPod, withNextPod), nil
			}
			return withNextPod, nil
		}
		if errWithout == nil {
			return withoutNextPod, nil
		}
	}
	return nil, fmt.Errorf("No set of running pods found to reclaim resources: %v", insufficientResources)

}

func minCostPodList(list1 []*v1.Pod, list2 []*v1.Pod) []*v1.Pod {
	nGuaranteed1, nBurstable1, memory1, cpu1 := evaluatePodList(list1)
	nGuaranteed2, nBurstable2, memory2, cpu2 := evaluatePodList(list2)
	if nGuaranteed1 < nGuaranteed2 {
		return list1
	}
	if nGuaranteed1 > nGuaranteed2 {
		return list2
	}
	if nBurstable1 < nBurstable2 {
		return list1
	}
	if nBurstable1 > nBurstable2 {
		return list2
	}
	if memory1 < memory2 {
		return list1
	}
	if memory1 > memory2 {
		return list2
	}
	if cpu1 < cpu2 {
		return list1
	}
	if cpu1 > cpu2 {
		return list2
	}
	// they have the same cost!
	return list1
}

func evaluatePodList(pods []*v1.Pod) (int, int, int64, int64) {
	nGuaranteed := 0
	nBurstable := 0
	memory := int64(0)
	cpu := int64(0)
	for _, pod := range pods {
		if qos.GetPodQOS(pod) == v1.PodQOSGuaranteed {
			nGuaranteed += 1
		} else if qos.GetPodQOS(pod) == v1.PodQOSBurstable {
			nBurstable += 1
		}
		requests := predicates.GetResourceRequest(pod)
		memory += getResourceRequest(requests, v1.ResourceMemory)
		cpu += getResourceRequest(requests, v1.ResourceCPU)
	}
	return nGuaranteed, nBurstable, memory, cpu
}

func getResourceRequest(requests *schedulercache.Resource, resource v1.ResourceName) int64 {
	switch resource {
	case v1.ResourcePods:
		return 1
	case v1.ResourceCPU:
		return requests.MilliCPU
	case v1.ResourceMemory:
		return requests.Memory
	case v1.ResourceNvidiaGPU:
		return requests.NvidiaGPU
	default:
		req, ok := requests.OpaqueIntResources[resource]
		if ok {
			return req
		}
		return 0
	}
}

// 	activePods := c.getPodsFunc()
// 	burstablePods := filterPods(activePods, func(p *v1.Pod) bool {
// 		// find all burstable, non-critical pods
// 		return qos.GetPodQOS(p) == v1.PodQOSBurstable && !kubetypes.IsCriticalPod(pod)
// 	})
// 	guaranteedPods := filterPods(activePods, func(p *v1.Pod) bool {
// 		// find all guaranteed, non-critical pods
// 		return qos.GetPodQOS(p) == v1.PodQOSGuaranteed && !kubetypes.IsCriticalPod(pod)
// 	})
// 	burstablePods = sort(byResourceRequest(burstablePods))
// 	guaranteedPods = sort(byResourceRequest(guaranteedPods))
// 	// Our priority in preemption is to evict the fewest pods possible, while taking into account QOS ordering.
// 	// For example, it is better to evict 3 burstable pods than 1 guaranteed, but it is better to evict 1 guarateed and 1 burstable
// 	// than 1 guaranteed and 2 burstable pods.  We iterate through numbers in this order, looking for matches that evict those numbers of pods.
// 	// Return as soon as we find a match, since we are iterating through better combinations first.
// 	for nGuaranteed := 0; nGuaranteed < len(guaranteedPods); nGuaranteed++ {
// 		for nBurstable := 0; nBurstable < len(burstablePods); nBurstable++ {
// 			pods, found := getLowestCostMatch(insufficientResources, burstablePods, guaranteedPods, nBurstable, nGuaranteed)
// 			if found {
// 				return pods, nil
// 			}
// 		}
// 	}
// 	return nil, fmt.Errorf("No set of running pods found to reclaim requests: %v", insufficientResources)
// }

// func getLowestCostMatch(insufficientResources []predicates.InsufficientResourceError,
// 	sortedBurstable []*v1.Pod, sortedGuaranteed []*v1.Pod, nBurstable int, nGuaranteed int) ([]*v1.Pod, bool) {
// 	burstableIndicies := []int{}
// 	for i := 0; i < nBurstable; i++ {
// 		burstableIndicies = append(burstableIndicies, i)
// 	}
// 	guaranteedIndiciess := []int{}
// 	for i := 0; i < nGuaranteed; i++ {
// 		guaranteedIndiciess = append(guaranteedIndiciess, i)
// 	}
// 	for {
// 		pods := []*v1.Pod{}
// 		for _, i := range burstableIndicies {
// 			pods = append(pods, sortedBurstable[i])
// 		}
// 		for _, i := range guaranteedIndiciess {
// 			pods = append(pods, sortedGuaranteed[i])
// 		}
// 		if podsMeetResourceRequirements(pods, insufficientResources) {
// 			return pods, true
// 		}

// 	}
// }

// func podsMeetResourceRequirements(pods []*v1.Pods, resourceRequirements []predicates.InsufficientResourceError) bool {
// 	for _, resource := range resourceRequirements {
// 		totalResourcesFreed := int64(0)
// 		for _, pod := range pods {
// 			totalResourcesFreed += getResourceRequest(pod, resource.ResourceName)
// 		}
// 		if totalResourcesFreed < resource.GetInsufficientAmount() {
// 			return false
// 		}
// 	}
// 	return true
// }

// func incrementIndicies(indicies []int, maxIndex int) bool {
// 	for i := len(indicies) - 1; i >= 0; i-- {
// 		if
// 	}
// }

// // returns pods in the list of pods where filterFunc returns true.
// func filterPods(pods []*v1.Pods, filterFunc func(p *v1.Pod) bool) []*v1.Pod {
// 	filteredPods := []*v1.Pod{}
// 	for _, pod := range pods {
// 		if filterFunc(pod) {
// 			filteredPods = append(filteredPods, pod)
// 		}
// 	}
// 	return filteredPods
// }

// // byResourceRequest implements sort.Interface for []*v1.Pod, and sorts according to the requests on the given resource.
// type byResourceRequest []*v1.Pod

// func (a byResource) Len() int      { return len(a) }
// func (a byResource) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// // Less ranks based on memory requests.
// func (a byResource) Less(i, j int) bool {
// 	return getResourceRequestByResource(a[i], v1.ResourceMemory) < getResourceRequestByResource(a[j], v1.ResourceMemory)
// }


// // byPreemptionPriority implements sort.Interface for []predicates.InsufficientResourceError.
// type byPreemptionPriority []predicates.InsufficientResourceError

// func (a byPreemptionPriority) Len() int      { return len(a) }
// func (a byPreemptionPriority) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// // Less ranks based on the getPreemptionPriority function
// func (a byPreemptionPriority) Less(i, j int) bool {
// 	return getPreemptionPriority(a[i].ResourceName) < getPreemptionPriority(a[j].ResourceName)
// }

// // Lower return value indicates a higher priority
// // This determines which resouce preemption prioritizes utilization for.
// // For example, if ResourceMemory is chosen, then the preemption logic will find and evict the
// // set of pods that free the insufficientResources and consume the least amount of memory, leaving memory requests
// // as close to allocateable as possible, while successfully freeing insufficientResources.
// func getPreemptionPriority(resource v1.ResourceName) int {
// 	switch resource {
// 	case v1.ResourcePods:
// 		return 1
// 	case v1.ResourceMemory:
// 		return -2
// 	case v1.ResourceCPU:
// 		return -1
// 	default:
// 		return 0
// 	}
// }
