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
	"context"
	"fmt"
	"math"
	"sort"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const message = "Preempted in order to admit critical pod"

// CriticalPodAdmissionHandler is an AdmissionFailureHandler that handles admission failure for Critical Pods.
// If the ONLY admission failures are due to insufficient resources, then CriticalPodAdmissionHandler evicts pods
// so that the critical pod can be admitted.  For evictions, the CriticalPodAdmissionHandler evicts a set of pods that
// frees up the required resource requests.  The set of pods is designed to minimize impact, and is prioritized according to the ordering:
// minimal impact for guaranteed pods > minimal impact for burstable pods > minimal impact for besteffort pods.
// minimal impact is defined as follows: fewest pods evicted > fewest total requests of pods.
// finding the fewest total requests of pods is considered besteffort.
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

// HandleAdmissionFailure gracefully handles admission rejection, and, in some cases,
// to allow admission of the pod despite its previous failure.
func (c *CriticalPodAdmissionHandler) HandleAdmissionFailure(ctx context.Context, admitPod *v1.Pod, failureReasons []lifecycle.PredicateFailureReason) ([]lifecycle.PredicateFailureReason, error) {
	if !kubetypes.IsCriticalPod(admitPod) {
		return failureReasons, nil
	}
	// InsufficientResourceError is not a reason to reject a critical pod.
	// Instead of rejecting, we free up resources to admit it, if no other reasons for rejection exist.
	nonResourceReasons := []lifecycle.PredicateFailureReason{}
	resourceReasons := []*admissionRequirement{}
	for _, reason := range failureReasons {
		if r, ok := reason.(*lifecycle.InsufficientResourceError); ok {
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
		return nonResourceReasons, nil
	}
	err := c.evictPodsToFreeRequests(ctx, admitPod, admissionRequirementList(resourceReasons))
	// if no error is returned, preemption succeeded and the pod is safe to admit.
	return nil, err
}

// evictPodsToFreeRequests takes a list of insufficient resources, and attempts to free them by evicting pods
// based on requests.  For example, if the only insufficient resource is 200Mb of memory, this function could
// evict a pod with request=250Mb.
func (c *CriticalPodAdmissionHandler) evictPodsToFreeRequests(ctx context.Context, admitPod *v1.Pod, insufficientResources admissionRequirementList) error {
	logger := klog.FromContext(ctx)
	podsToPreempt, err := getPodsToPreempt(admitPod, c.getPodsFunc(), insufficientResources)
	if err != nil {
		return fmt.Errorf("preemption: error finding a set of pods to preempt: %v", err)
	}
	for _, pod := range podsToPreempt {
		// record that we are evicting the pod
		c.recorder.Eventf(pod, v1.EventTypeWarning, events.PreemptContainer, message)
		// this is a blocking call and should only return when the pod and its containers are killed.
		logger.V(2).Info("Preempting pod to free up resources", "pod", klog.KObj(pod), "podUID", pod.UID, "insufficientResources", insufficientResources.toString(), "requestingPod", klog.KObj(admitPod))
		err := c.killPodFunc(pod, true, nil, func(status *v1.PodStatus) {
			status.Phase = v1.PodFailed
			status.Reason = events.PreemptContainer
			status.Message = message
			podutil.UpdatePodCondition(status, &v1.PodCondition{
				Type:               v1.DisruptionTarget,
				ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(status, pod.Generation, v1.DisruptionTarget),
				Status:             v1.ConditionTrue,
				Reason:             v1.PodReasonTerminationByKubelet,
				Message:            "Pod was preempted by Kubelet to accommodate a critical pod.",
			})
		})
		if err != nil {
			logger.Error(err, "Failed to evict pod", "pod", klog.KObj(pod))
			// In future syncPod loops, the kubelet will retry the pod deletion steps that it was stuck on.
			continue
		}
		if len(insufficientResources) > 0 {
			metrics.Preemptions.WithLabelValues(insufficientResources[0].resourceName.String()).Inc()
		} else {
			metrics.Preemptions.WithLabelValues("").Inc()
		}
		logger.Info("Pod evicted successfully", "pod", klog.KObj(pod))
	}
	return nil
}

// getPodsToPreempt returns a list of pods that could be preempted to free requests >= requirements
func getPodsToPreempt(pod *v1.Pod, pods []*v1.Pod, requirements admissionRequirementList) ([]*v1.Pod, error) {
	// podGroups is sorted from the most preemptable to least preemptable groups of pods.
	podGroups := groupPodsByPriorityAndQOS(pod, pods)
	n := len(podGroups)

	// build remaining requirements after subtracting from each podGroups group.
	// remainingReqs[i] represents the remaining resource requirements after
	// subtracting all pods in podGroups[0:i] (i.e., lower-priority groups).
	remainingReqs := make([]admissionRequirementList, n+1)
	remainingReqs[0] = requirements

	for i := 0; i < n; i++ {
		remainingReqs[i+1] = remainingReqs[i].subtract(podGroups[i]...)
	}

	// If even after subtracting all preemptable pods the requirements are not met,
	// then no feasible set of pods exists to reclaim resources.
	if len(remainingReqs[n]) > 0 {
		return nil, fmt.Errorf(
			"no set of running pods found to reclaim resources: %v",
			remainingReqs[n].toString(),
		)
	}

	var podsToPreempt []*v1.Pod

	// Process pod groups from the highest priority to lowest.
	// at each iteration, we assume that:
	//   1. All lower-priority groups have already been fully accounted for
	//      in remainingReqs[i].
	//   2. All higher-priority pods selected so far (podsToPreempt) will be evicted.
	// Based on the remaining resource requirements, we select the minimal set
	// of pods to evict from the current priority/QoS group using getPodsToPreemptByDistance.
	for i := n - 1; i >= 0; i-- {
		remainingReq := remainingReqs[i].subtract(podsToPreempt...)

		podsToEvict, err := getPodsToPreemptByDistance(podGroups[i], remainingReq)
		if err != nil {
			return nil, err
		}

		podsToPreempt = append(podsToPreempt, podsToEvict...)
	}

	return podsToPreempt, nil
}

// getPodsToPreemptByDistance finds the pods that have pod requests >= admission requirements.
// Chooses pods that minimize "distance" to the requirements.
// If more than one pod exists that fulfills the remaining requirements,
// it chooses the pod that has the "smaller resource request"
// This method, by repeatedly choosing the pod that fulfills as much of the requirements as possible,
// attempts to minimize the number of pods returned.
func getPodsToPreemptByDistance(pods []*v1.Pod, requirements admissionRequirementList) ([]*v1.Pod, error) {
	podsToEvict := []*v1.Pod{}
	// evict pods by shortest distance from remaining requirements, updating requirements every round.
	for len(requirements) > 0 {
		if len(pods) == 0 {
			return nil, fmt.Errorf("no set of running pods found to reclaim resources: %v", requirements.toString())
		}
		// all distances must be less than len(requirements), because the max distance for a single requirement is 1
		bestDistance := float64(len(requirements) + 1)
		bestPodIndex := 0
		// Find the pod with the smallest distance from requirements
		// Or, in the case of two equidistant pods, find the pod with "smaller" resource requests.
		for i, pod := range pods {
			dist := requirements.distance(pod)
			if dist < bestDistance || (bestDistance == dist && smallerResourceRequest(pod, pods[bestPodIndex])) {
				bestDistance = dist
				bestPodIndex = i
			}
		}
		// subtract the pod from requirements, and transfer the pod from input-pods to pods-to-evicted
		requirements = requirements.subtract(pods[bestPodIndex])
		podsToEvict = append(podsToEvict, pods[bestPodIndex])
		pods[bestPodIndex] = pods[len(pods)-1]
		pods = pods[:len(pods)-1]
	}
	return podsToEvict, nil
}

type admissionRequirement struct {
	resourceName v1.ResourceName
	quantity     int64
}

type admissionRequirementList []*admissionRequirement

// distance returns distance of the pods requests from the admissionRequirements.
// The distance is measured by the fraction of the requirement satisfied by the pod,
// so that each requirement is weighted equally, regardless of absolute magnitude.
func (a admissionRequirementList) distance(pod *v1.Pod) float64 {
	dist := float64(0)
	for _, req := range a {
		remainingRequest := float64(req.quantity - resource.GetResourceRequest(pod, req.resourceName))
		if remainingRequest > 0 {
			dist += math.Pow(remainingRequest/float64(req.quantity), 2)
		}
	}
	return dist
}

// subtract returns a new admissionRequirementList containing remaining requirements if the provided pod
// were to be preempted
func (a admissionRequirementList) subtract(pods ...*v1.Pod) admissionRequirementList {
	newList := []*admissionRequirement{}
	for _, req := range a {
		newQuantity := req.quantity
		for _, pod := range pods {
			newQuantity -= resource.GetResourceRequest(pod, req.resourceName)
			if newQuantity <= 0 {
				break
			}
		}
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

// groupPodsByPriorityAndQOS groups pods that the given preemptor pod can preempt.
//
//  1. Only pods that are preemptable by the preemptor are considered.
//  2. Pods are first sorted from lowest to highest based on:
//     a) Priority / criticality
//     b) QOS
//  3. Pods with the same priority and QoS are grouped together. Each group represents
//     a set of pods that are equally preemptable.
//  4. Higher priority / QoS pods appear at the end of the returned slice of groups.
func groupPodsByPriorityAndQOS(preemptor *v1.Pod, pods []*v1.Pod) (podGroups [][]*v1.Pod) {
	var preemptablePods []*v1.Pod

	// filter pods that the preemptor can preempt.
	for _, pod := range pods {
		if kubetypes.Preemptable(preemptor, pod) {
			podQOS := v1qos.GetPodQOS(pod)

			if podQOS != v1.PodQOSBestEffort && podQOS != v1.PodQOSGuaranteed && podQOS != v1.PodQOSBurstable {
				continue
			}

			preemptablePods = append(preemptablePods, pod)
		}
	}

	// sort from the least priority to the highest priority.
	sort.Slice(preemptablePods, func(i, j int) bool {
		firstPod := preemptablePods[i]
		secondPod := preemptablePods[j]

		// priority / criticality
		if kubetypes.Preemptable(firstPod, secondPod) {
			return false
		}
		if kubetypes.Preemptable(secondPod, firstPod) {
			return true
		}

		// QoS
		qosRank := map[v1.PodQOSClass]int{
			v1.PodQOSGuaranteed: 3,
			v1.PodQOSBurstable:  2,
			v1.PodQOSBestEffort: 1,
		}

		return qosRank[v1qos.GetPodQOS(firstPod)] < qosRank[v1qos.GetPodQOS(secondPod)]
	})

	// grouping phase
	for _, pod := range preemptablePods {
		// first pod always starts a group
		if len(podGroups) == 0 {
			podGroups = append(podGroups, []*v1.Pod{pod})
			continue
		}

		lastGroup := podGroups[len(podGroups)-1]
		lastPod := lastGroup[len(lastGroup)-1]

		// start a new group if priority or QoS differs
		if kubetypes.Preemptable(pod, lastPod) ||
			kubetypes.Preemptable(lastPod, pod) ||
			v1qos.GetPodQOS(pod) != v1qos.GetPodQOS(lastPod) {

			podGroups = append(podGroups, []*v1.Pod{pod})
			continue
		}

		// same priority and same QoS pods are in the same group
		podGroups[len(podGroups)-1] =
			append(lastGroup, pod)
	}

	return
}

// smallerResourceRequest returns true if pod1 has a smaller request than pod2
func smallerResourceRequest(pod1 *v1.Pod, pod2 *v1.Pod) bool {
	priorityList := []v1.ResourceName{
		v1.ResourceMemory,
		v1.ResourceCPU,
	}
	for _, res := range priorityList {
		req1 := resource.GetResourceRequest(pod1, res)
		req2 := resource.GetResourceRequest(pod2, res)
		if req1 < req2 {
			return true
		} else if req1 > req2 {
			return false
		}
	}
	return true
}
