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
	"context"
	"fmt"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	frameworkplugins "k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

type getNodeAnyWayFuncType func() (*v1.Node, error)

type pluginResourceUpdateFuncType func(*schedulernodeinfo.NodeInfo, *PodAdmitAttributes) error

// AdmissionFailureHandler is an interface which defines how to deal with a failure to admit a pod.
// This allows for the graceful handling of pod admission failure.
type AdmissionFailureHandler interface {
	HandleAdmissionFailure(admitPod *v1.Pod, failureStatuses schedulerframework.PluginToStatus) (bool, schedulerframework.PluginToStatus, error)
}

type predicateAdmitHandler struct {
	getNodeAnyWayFunc        getNodeAnyWayFuncType
	pluginResourceUpdateFunc pluginResourceUpdateFuncType
	admissionFailureHandler  AdmissionFailureHandler
	schedulerFramework       schedulerframework.Framework
}

var _ PodAdmitHandler = &predicateAdmitHandler{}

func NewPredicateAdmitHandler(getNodeAnyWayFunc getNodeAnyWayFuncType, admissionFailureHandler AdmissionFailureHandler, pluginResourceUpdateFunc pluginResourceUpdateFuncType) *predicateAdmitHandler {
	// New a scheduler algorithm registry, the argument `hardPodAffinityWeight` does not matter for us.
	providerRegistry := algorithmprovider.NewRegistry(0)
	config := providerRegistry[algorithmprovider.KubeletProvider]
	registry := frameworkplugins.NewInTreeRegistry(&frameworkplugins.RegistryArgs{
		VolumeBinder: nil, // Set it to nil because it will not be used in kubelet.
	})
	framework, err := schedulerframework.NewFramework(registry, config.FrameworkPlugins,
		config.FrameworkPluginConfig)
	if err != nil {
		return nil
	}

	return &predicateAdmitHandler{
		getNodeAnyWayFunc,
		pluginResourceUpdateFunc,
		admissionFailureHandler,
		framework,
	}
}

func (w *predicateAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	node, err := w.getNodeAnyWayFunc()
	if err != nil {
		klog.Errorf("Cannot get Node info: %v", err)
		return PodAdmitResult{
			Admit:   false,
			Reason:  "InvalidNodeInfo",
			Message: "Kubelet cannot get node info.",
		}
	}
	admitPod := attrs.Pod
	pods := attrs.OtherPods
	nodeInfo := schedulernodeinfo.NewNodeInfo(pods...)
	nodeInfo.SetNode(node)
	// ensure the node has enough plugin resources for that required in pods
	if err = w.pluginResourceUpdateFunc(nodeInfo, attrs); err != nil {
		message := fmt.Sprintf("Update plugin resources failed due to %v, which is unexpected.", err)
		klog.Warningf("Failed to admit pod %v - %s", format.Pod(admitPod), message)
		return PodAdmitResult{
			Admit:   false,
			Reason:  "UnexpectedAdmissionError",
			Message: message,
		}
	}

	// Remove the requests of the extended resources that are missing in the
	// node info. This is required to support cluster-level resources, which
	// are extended resources unknown to nodes.
	//
	// Caveat: If a pod was manually bound to a node (e.g., static pod) where a
	// node-level extended resource it requires is not found, then kubelet will
	// not fail admission while it should. This issue will be addressed with
	// the Resource Class API in the future.
	podWithoutMissingExtendedResources := removeMissingExtendedResources(admitPod, nodeInfo)

	fit, statuses, err := w.generalPredicates(podWithoutMissingExtendedResources, nodeInfo)
	if err != nil {
		message := fmt.Sprintf("GeneralPredicates failed due to %v, which is unexpected.", err)
		klog.Warningf("Failed to admit pod %v - %s", format.Pod(admitPod), message)
		return PodAdmitResult{
			Admit:   fit,
			Reason:  "UnexpectedAdmissionError",
			Message: message,
		}
	}
	if !fit {
		fit, statuses, err = w.admissionFailureHandler.HandleAdmissionFailure(admitPod, statuses)
		if err != nil {
			message := fmt.Sprintf("Unexpected error while attempting to recover from admission failure: %v", err)
			klog.Warningf("Failed to admit pod %v - %s", format.Pod(admitPod), message)
			return PodAdmitResult{
				Admit:   fit,
				Reason:  "UnexpectedAdmissionError",
				Message: message,
			}
		}
	}
	if !fit {
		var message string
		if len(statuses) == 0 {
			message = fmt.Sprint("GeneralPredicates failed due to unknown reason, which is unexpected.")
			klog.Warningf("Failed to admit pod %v - %s", format.Pod(admitPod), message)
			return PodAdmitResult{
				Admit:   fit,
				Reason:  "UnknownReason",
				Message: message,
			}
		}
		// If there are failed predicates, we only return the first one as a reason.
		message = statuses.Merge().Message()
		klog.V(2).Infof("Predicate failed on Pod: %v, for reason: %v", format.Pod(admitPod), message)
		return PodAdmitResult{
			Admit:   fit,
			Reason:  "PredicateFailure",
			Message: message,
		}
	}
	return PodAdmitResult{
		Admit: true,
	}
}

func (w *predicateAdmitHandler) generalPredicates(pod *v1.Pod,
	nodeInfo *schedulernodeinfo.NodeInfo) (bool, schedulerframework.PluginToStatus, error) {
	state := schedulerframework.NewCycleState()
	preFilterStatus := w.schedulerFramework.RunPreFilterPlugins(context.TODO(), state, pod)
	if !preFilterStatus.IsSuccess() {
		var err error
		if !preFilterStatus.IsUnschedulable() {
			err = preFilterStatus.AsError()
		}
		return false, schedulerframework.PluginToStatus{"prefilter": preFilterStatus}, err
	}

	filterStatus := w.schedulerFramework.RunFilterPlugins(context.TODO(), state, pod, nodeInfo)
	status := filterStatus.Merge()
	if !status.IsSuccess() && !status.IsUnschedulable() {
		return false, filterStatus, status.AsError()
	}

	return status.IsSuccess(), filterStatus, nil
}

func removeMissingExtendedResources(pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *v1.Pod {
	podCopy := pod.DeepCopy()
	for i, c := range pod.Spec.Containers {
		// We only handle requests in Requests but not Limits because the
		// PodFitsResources predicate, to which the result pod will be passed,
		// does not use Limits.
		podCopy.Spec.Containers[i].Resources.Requests = make(v1.ResourceList)
		for rName, rQuant := range c.Resources.Requests {
			if v1helper.IsExtendedResourceName(rName) {
				if _, found := nodeInfo.AllocatableResource().ScalarResources[rName]; !found {
					continue
				}
			}
			podCopy.Spec.Containers[i].Resources.Requests[rName] = rQuant
		}
	}
	return podCopy
}
