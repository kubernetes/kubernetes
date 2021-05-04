/*
Copyright 2016 The Kubernetes Authors.

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

package lifecycle

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	v1affinityhelper "k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
)

type getNodeAnyWayFuncType func() (*v1.Node, error)

type pluginResourceUpdateFuncType func(*schedulerframework.NodeInfo, *PodAdmitAttributes) error

// AdmissionFailureHandler is an interface which defines how to deal with a failure to admit a pod.
// This allows for the graceful handling of pod admission failure.
type AdmissionFailureHandler interface {
	HandleAdmissionFailure(admitPod *v1.Pod, failureReasons []PredicateFailureReason) ([]PredicateFailureReason, error)
}

type predicateAdmitHandler struct {
	getNodeAnyWayFunc        getNodeAnyWayFuncType
	pluginResourceUpdateFunc pluginResourceUpdateFuncType
	admissionFailureHandler  AdmissionFailureHandler
}

var _ PodAdmitHandler = &predicateAdmitHandler{}

// NewPredicateAdmitHandler returns a PodAdmitHandler which is used to evaluates
// if a pod can be admitted from the perspective of predicates.
func NewPredicateAdmitHandler(getNodeAnyWayFunc getNodeAnyWayFuncType, admissionFailureHandler AdmissionFailureHandler, pluginResourceUpdateFunc pluginResourceUpdateFuncType) PodAdmitHandler {
	return &predicateAdmitHandler{
		getNodeAnyWayFunc,
		pluginResourceUpdateFunc,
		admissionFailureHandler,
	}
}

func (w *predicateAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	var node *v1.Node
	var getNodeAnyWayErr error
	wait.PollImmediate(1*time.Second, 20*time.Minute, func() (bool, error) {
		node, getNodeAnyWayErr = w.getNodeAnyWayFunc()
		if allocatable := schedulerframework.NewResource(node.Status.Allocatable); allocatable.EphemeralStorage == 0 {
			klog.Infof("Node does not yet have any allocatable storage")
			return false, nil
		}
		return true, nil
	})
	if getNodeAnyWayErr != nil {
		klog.ErrorS(getNodeAnyWayErr, "Cannot get Node info")
		return PodAdmitResult{
			Admit:   false,
			Reason:  "InvalidNodeInfo",
			Message: "Kubelet cannot get node info.",
		}
	}
	admitPod := attrs.Pod
	pods := attrs.OtherPods
	nodeInfo := schedulerframework.NewNodeInfo(pods...)
	// The nomimal value of Requested.EphemeralStorage at the phase is 0, which makes sense as we're still simply scaffolding the node object type representation
	klog.Infof("Requested Ephemeral Storage after creating nodeInfo object for fit: %d", nodeInfo.Requested.EphemeralStorage)
	// The nomimal value of Allocatable.EphemeralStorage at the phase is 0, which makes sense as we're still simply scaffolding the node object type representation
	klog.Infof("Allocatable Ephemeral Storage after creating nodeInfo object: %d", nodeInfo.Allocatable.EphemeralStorage)
	nodeInfo.SetNode(node)
	// The nomimal value of Requested.EphemeralStorage at this phase may be 0, or it may be non-zero, depending on whether or not any other concurrent ephemeral-storage-requiring pods are in a state of being scheduled
	klog.Infof("Requested Ephemeral Storage after invoking nodeinfo.SetNode(node): %d", nodeInfo.Requested.EphemeralStorage)
	// The nomimal value of Allocatable.EphemeralStorage at this phase should be non-zero, and roughly equivalent to the available storage on the host OS
	// In the erratic case we see the value of this as 0! <-- That is going to fail admittance of any pods that have a non-zero ephemeral-storage resource requirements value
	klog.Infof("Allocatable Ephemeral Storage after invoking nodeinfo.SetNode(node): %d", nodeInfo.Allocatable.EphemeralStorage)
	// ensure the node has enough plugin resources for that required in pods
	if err = w.pluginResourceUpdateFunc(nodeInfo, attrs); err != nil {
		message := fmt.Sprintf("Update plugin resources failed due to %v, which is unexpected.", err)
		klog.InfoS("Failed to admit pod", "pod", klog.KObj(admitPod), "message", message)
		return PodAdmitResult{
			Admit:   false,
			Reason:  "UnexpectedAdmissionError",
			Message: message,
		}
	}
	// The nomimal value of Requested.EphemeralStorage should not have been changed by any side-effects of w.pluginResourceUpdateFunc
	klog.Infof("Requested Ephemeral Storage after invoking w.pluginResourceUpdateFunc(nodeInfo, attrs): %d", nodeInfo.Requested.EphemeralStorage)
	// The nomimal value of Allocatable.EphemeralStorage should not have been changed by any side-effects of w.pluginResourceUpdateFunc
	klog.Infof("Allocatable Ephemeral Storage after invoking w.pluginResourceUpdateFunc(nodeInfo, attrs): %d", nodeInfo.Allocatable.EphemeralStorage)

	// Remove the requests of the extended resources that are missing in the
	// node info. This is required to support cluster-level resources, which
	// are extended resources unknown to nodes.
	//
	// Caveat: If a pod was manually bound to a node (e.g., static pod) where a
	// node-level extended resource it requires is not found, then kubelet will
	// not fail admission while it should. This issue will be addressed with
	// the Resource Class API in the future.
	podWithoutMissingExtendedResources := removeMissingExtendedResources(admitPod, nodeInfo)

	reasons, err := GeneralPredicates(podWithoutMissingExtendedResources, nodeInfo)
	fit := len(reasons) == 0 && err == nil
	// The nomimal value of Requested.EphemeralStorage should not have been changed by any side-effects of removeMissingExtendedResources or GeneralPredicates
	klog.Infof("Requested Ephemeral Storage after checking for fit: %d", nodeInfo.Requested.EphemeralStorage)
	// The nomimal value of Allocatable.EphemeralStorage should not have been changed by any side-effects of removeMissingExtendedResources or GeneralPredicates
	klog.Infof("Allocatable Ephemeral Storage after checking for fit: %d", nodeInfo.Allocatable.EphemeralStorage)
	if err != nil {
		message := fmt.Sprintf("GeneralPredicates failed due to %v, which is unexpected.", err)
		klog.InfoS("Failed to admit pod, GeneralPredicates failed", "pod", klog.KObj(admitPod), "err", err)
		return PodAdmitResult{
			Admit:   fit,
			Reason:  "UnexpectedAdmissionError",
			Message: message,
		}
	}
	if !fit {
		reasons, err = w.admissionFailureHandler.HandleAdmissionFailure(admitPod, reasons)
		fit = len(reasons) == 0 && err == nil
		if err != nil {
			message := fmt.Sprintf("Unexpected error while attempting to recover from admission failure: %v", err)
			// This is the error outcome we are debugging, e.g.:
			/*
				May 03 21:51:51 capz-conf-ulw45w-control-plane-jfj9m kubelet[2453]: I0503 21:51:51.890526    2453 predicate.go:113] "Failed to admit pod, unexpected error while attempting to recover from admission failure" pod="kube-system/etcd-capz-conf-ulw45w-control-plane-jfj9m" err="preemption: error finding a set of pods to preempt: no set of running pods found to reclaim resources: [(res: ephemeral-storage, q: 104857600), ]"
			*/
			klog.InfoS("Failed to admit pod, unexpected error while attempting to recover from admission failure", "pod", klog.KObj(admitPod), "err", err)
			return PodAdmitResult{
				Admit:   fit,
				Reason:  "UnexpectedAdmissionError",
				Message: message,
			}
		}
	}
	if !fit {
		var reason string
		var message string
		if len(reasons) == 0 {
			message = fmt.Sprint("GeneralPredicates failed due to unknown reason, which is unexpected.")
			klog.InfoS("Failed to admit pod: GeneralPredicates failed due to unknown reason, which is unexpected", "pod", klog.KObj(admitPod))
			return PodAdmitResult{
				Admit:   fit,
				Reason:  "UnknownReason",
				Message: message,
			}
		}
		// If there are failed predicates, we only return the first one as a reason.
		r := reasons[0]
		switch re := r.(type) {
		case *PredicateFailureError:
			reason = re.PredicateName
			message = re.Error()
			klog.V(2).InfoS("Predicate failed on Pod", "pod", klog.KObj(admitPod), "err", message)
		case *InsufficientResourceError:
			reason = fmt.Sprintf("OutOf%s", re.ResourceName)
			message = re.Error()
			klog.V(2).InfoS("Predicate failed on Pod", "pod", klog.KObj(admitPod), "err", message)
		default:
			reason = "UnexpectedPredicateFailureType"
			message = fmt.Sprintf("GeneralPredicates failed due to %v, which is unexpected.", r)
			klog.InfoS("Failed to admit pod", "pod", klog.KObj(admitPod), "err", message)
		}
		return PodAdmitResult{
			Admit:   fit,
			Reason:  reason,
			Message: message,
		}
	}
	return PodAdmitResult{
		Admit: true,
	}
}

func removeMissingExtendedResources(pod *v1.Pod, nodeInfo *schedulerframework.NodeInfo) *v1.Pod {
	podCopy := pod.DeepCopy()
	for i, c := range pod.Spec.Containers {
		// We only handle requests in Requests but not Limits because the
		// PodFitsResources predicate, to which the result pod will be passed,
		// does not use Limits.
		podCopy.Spec.Containers[i].Resources.Requests = make(v1.ResourceList)
		for rName, rQuant := range c.Resources.Requests {
			if v1helper.IsExtendedResourceName(rName) {
				if _, found := nodeInfo.Allocatable.ScalarResources[rName]; !found {
					continue
				}
			}
			podCopy.Spec.Containers[i].Resources.Requests[rName] = rQuant
		}
	}
	return podCopy
}

// InsufficientResourceError is an error type that indicates what kind of resource limit is
// hit and caused the unfitting failure.
type InsufficientResourceError struct {
	ResourceName v1.ResourceName
	Requested    int64
	Used         int64
	Capacity     int64
}

func (e *InsufficientResourceError) Error() string {
	return fmt.Sprintf("Node didn't have enough resource: %s, requested: %d, used: %d, capacity: %d",
		e.ResourceName, e.Requested, e.Used, e.Capacity)
}

// PredicateFailureReason interface represents the failure reason of a predicate.
type PredicateFailureReason interface {
	GetReason() string
}

// GetReason returns the reason of the InsufficientResourceError.
func (e *InsufficientResourceError) GetReason() string {
	return fmt.Sprintf("Insufficient %v", e.ResourceName)
}

// GetInsufficientAmount returns the amount of the insufficient resource of the error.
func (e *InsufficientResourceError) GetInsufficientAmount() int64 {
	return e.Requested - (e.Capacity - e.Used)
}

// PredicateFailureError describes a failure error of predicate.
type PredicateFailureError struct {
	PredicateName string
	PredicateDesc string
}

func (e *PredicateFailureError) Error() string {
	return fmt.Sprintf("Predicate %s failed", e.PredicateName)
}

// GetReason returns the reason of the PredicateFailureError.
func (e *PredicateFailureError) GetReason() string {
	return e.PredicateDesc
}

// GeneralPredicates checks a group of predicates that the kubelet cares about.
func GeneralPredicates(pod *v1.Pod, nodeInfo *schedulerframework.NodeInfo) ([]PredicateFailureReason, error) {
	if nodeInfo.Node() == nil {
		return nil, fmt.Errorf("node not found")
	}

	var reasons []PredicateFailureReason
	for _, r := range noderesources.Fits(pod, nodeInfo) {
		reasons = append(reasons, &InsufficientResourceError{
			ResourceName: r.ResourceName,
			Requested:    r.Requested,
			Used:         r.Used,
			Capacity:     r.Capacity,
		})
	}

	// Ignore parsing errors for backwards compatibility.
	match, _ := v1affinityhelper.GetRequiredNodeAffinity(pod).Match(nodeInfo.Node())
	if !match {
		reasons = append(reasons, &PredicateFailureError{nodeaffinity.Name, nodeaffinity.ErrReasonPod})
	}
	if !nodename.Fits(pod, nodeInfo) {
		reasons = append(reasons, &PredicateFailureError{nodename.Name, nodename.ErrReason})
	}
	if !nodeports.Fits(pod, nodeInfo) {
		reasons = append(reasons, &PredicateFailureError{nodeports.Name, nodeports.ErrReason})
	}

	return reasons, nil
}
