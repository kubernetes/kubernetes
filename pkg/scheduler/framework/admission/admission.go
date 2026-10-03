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

// Package admission holds the node admission checks the kubelet shares with the
// scheduler. It must stay free of the plugin registry's DRA and CEL dependencies.
package admission

import (
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1nodeaffinity "k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
)

// Check runs the node resources, node affinity, node name and node ports filters.
// It returns the first failure unless includeAllFailures is set.
func Check(pod *v1.Pod, nodeInfo *framework.NodeInfo, includeAllFailures bool) []Result {
	var admissionResults []Result
	insufficientResources := helper.Fits(pod, nodeInfo, nil, helper.ResourceRequestsOptions{
		EnablePodLevelResources:           utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources),
		EnableDRAExtendedResource:         utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource),
		EnableDRANodeAllocatableResources: utilfeature.DefaultFeatureGate.Enabled(features.DRANodeAllocatableResources),
	})
	if len(insufficientResources) != 0 {
		for i := range insufficientResources {
			admissionResults = append(admissionResults, Result{InsufficientResource: &insufficientResources[i]})
		}
		if !includeAllFailures {
			return admissionResults
		}
	}

	if matches, _ := corev1nodeaffinity.GetRequiredNodeAffinity(pod).Match(nodeInfo.Node()); !matches {
		admissionResults = append(admissionResults, Result{Name: nodeaffinity.Name, Reason: nodeaffinity.ErrReasonPod})
		if !includeAllFailures {
			return admissionResults
		}
	}
	if !nodename.Fits(pod, nodeInfo) {
		admissionResults = append(admissionResults, Result{Name: nodename.Name, Reason: nodename.ErrReason})
		if !includeAllFailures {
			return admissionResults
		}
	}
	if !nodeports.Fits(pod, nodeInfo) {
		admissionResults = append(admissionResults, Result{Name: nodeports.Name, Reason: nodeports.ErrReason})
		if !includeAllFailures {
			return admissionResults
		}
	}
	return admissionResults
}

// Result describes why the pod cannot be admitted to the node.
// If the reason is a resource fit one, InsufficientResource has the details.
type Result struct {
	Name                 string
	Reason               string
	InsufficientResource *helper.InsufficientResource
}
