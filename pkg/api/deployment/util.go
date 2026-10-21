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

package deployment

import (
	appsv1 "k8s.io/api/apps/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/validation/field"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	apps "k8s.io/kubernetes/pkg/apis/apps"
)

func WorkloadInputForDeploymentInternal(s *apps.DeploymentSchedulingConfiguration) workloadbuilder.WorkloadInput {
	if s == nil {
		s = &apps.DeploymentSchedulingConfiguration{}
	}
	return WorkloadInput(s.SchedulingPolicy, s.SchedulingConstraints, s.DisruptionMode, s.ResourceClaims)
}

func WorkloadInputForDeploymentV1(s *appsv1.DeploymentSchedulingConfiguration) workloadbuilder.WorkloadInput {
	if s == nil {
		s = &appsv1.DeploymentSchedulingConfiguration{}
	}
	return WorkloadInput(s.SchedulingPolicy, s.SchedulingConstraints, s.DisruptionMode, s.ResourceClaims)
}

// WorkloadInput maps Deployment scheduling building blocks into workloadbuilder input.
func WorkloadInput(
	policy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy,
	constraints *schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints,
	disruptionMode *schedulingv1alpha3.WorkloadPodGroupDisruptionMode,
	resourceClaims []schedulingv1alpha3.WorkloadPodGroupResourceClaim,
) workloadbuilder.WorkloadInput {
	return workloadbuilder.WorkloadInput{
		Policy: workloadbuilder.PolicyInput{
			PodGroupData: policy,
			PathElements: []string{"schedulingPolicy"},
		},
		Constraints: workloadbuilder.ConstraintsInput{
			PodGroupData: constraints,
			PathElements: []string{"schedulingConstraints"},
		},
		DisruptionMode: workloadbuilder.DisruptionModeInput{
			PodGroupData: disruptionMode,
			PathElements: []string{"disruptionMode"},
		},
		ResourceClaims: workloadbuilder.ResourceClaimsInput{
			PodGroupData: resourceClaims,
			PathElements: []string{"resourceClaims"},
		},
	}
}

// WorkloadItemForDeployment assembles a Deployment scheduling item.
func WorkloadItemForDeployment(name, priorityClassName string, input workloadbuilder.WorkloadInput, path *field.Path) *workloadbuilder.WorkloadItem {
	return &workloadbuilder.WorkloadItem{
		Name: name,
		Path: path,
		DefaultConfig: &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{
				Basic: &workloadbuilder.BasicSchedulingPolicy{},
			},
			PriorityClassName: priorityClassName,
		},
		Input: input,
	}
}
