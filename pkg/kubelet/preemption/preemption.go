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
	pods := []*v1.Pod{}
	for _, pod := range c.getPodsFunc() {
		// only consider guaranteed or burstable pods that are not critical
		if !kubetypes.IsCriticalPod(pod) && (qos.GetPodQOS(pod) == v1.PodQOSGuaranteed || qos.GetPodQOS(pod) == v1.PodQOSBurstable) {
			pods = append(pods, pod)
		}
	}
	podsToPreempt, err := getPodsToPreempt(pods, insufficientResources)
	if err != nil {
		return fmt.Errorf("Error finding a set of pods to preempt: %v", err)
	}
	glog.Infof("preemption: attempting to evict pods %v, in order to free up resources: %s", podsToPreempt, insufficientResources.toString())
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
	return nil, fmt.Errorf("No set of running pods found to reclaim resources: %s", insufficientResources.toString())
}

// Return the list of pods with the smaller cost,
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

// given a list of pods, return the total:
// number of guaranteeed pods, number of burstable pods,
// memory requested, cpu requested
func evaluatePodList(pods []*v1.Pod) (nGuaranteed, nBurstable int, memory, cpu int64) {
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

// finds and returns the request for a specific resource.
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

func (a admissionRequirementList) toString() string {
	s := "["
	for _, req := range a {
		s += fmt.Sprintf("(res: %v, q: %d), ", req.resourceName, req.quantity)
	}
	return s + "]"
}
