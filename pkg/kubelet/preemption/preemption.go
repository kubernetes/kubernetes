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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
)

// CriticalPodAdmissionFailureHandler is an AdmissionFailureHandler that does not perform any handling of admission failure.
// It simply passes the failure on.
type CriticalPodAdmissionHandler struct{}

var _ lifecycle.AdmissionFailureHandler = &CriticalPodAdmissionHandler{}

func NewCriticalPodAdmissionHandler() *CriticalPodAdmissionHandler {
	return &CriticalPodAdmissionHandler{}
}

// HandleAdmissionFailure gracefully handles admission rejection, and, in some cases,
// to allow admission of the pod despite its previous failure.
func (c *CriticalPodAdmissionHandler) HandleAdmissionFailure(pod *v1.Pod, failureReasons []algorithm.PredicateFailureReason) (bool, []algorithm.PredicateFailureReason, error) {
	if !kubetypes.IsCriticalPod(pod) {
		return false, failureReasons, nil
	}
	// InsufficientResourceError is not a reason to reject a critical pod.
	// Instead of rejecting, we free up resources to admit it, if no other reasons for rejection exist.
	nonResourceReasons := []algorithm.PredicateFailureReason{}
	resourceReasons := []predicates.InsufficientResourceError{}
	for _, reason := range failureReasons {
		if r, ok := reason.(*predicates.InsufficientResourceError); ok {
			resourceReasons = append(resourceReasons, *r)
		} else {
			nonResourceReasons = append(nonResourceReasons, reason)
		}
	}
	if len(nonResourceReasons) == 0 {
		err := c.preemptPods(resourceReasons)
		// if no error is returned, preemption succeeded and the pod is safe to admit.
		return err == nil, nil, err
	}
	// Return only reasons that are not resource related, since critical pods cannot fail admission for resource reasons.
	return false, nonResourceReasons, nil
}

// PreemptyPods takes a list of insufficient resources, and attempts to free them by evicting pods
// based on requests.  For example, if the insufficient resource is 200Mb of memory, this function could
// evict a pod with request=250Mb.
func (m *CriticalPodAdmissionHandler) preemptPods(insufficientResources []predicates.InsufficientResourceError) error {
	return fmt.Errorf("NOT IMPLEMENTED ERROR")
}
