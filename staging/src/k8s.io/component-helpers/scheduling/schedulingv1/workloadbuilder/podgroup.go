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

package workloadbuilder

import (
	"fmt"

	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// materializes a runtime PodGroup from the named PodGroupTemplate, which
// BuildWorkload does not create. The template's scheduling fields are
// deep-copied into the spec, workloadRef points back at the template,
// and the PodGroup lands in the workload's namespace. Owners are
// caller-supplied because they differ by case.
func newPodGroup(workload *schedulingv1beta1.Workload, templateName, podGroupName string, owners []metav1.OwnerReference) (*schedulingv1beta1.PodGroup, error) {
	if workload == nil {
		return nil, fmt.Errorf("workload must not be nil")
	}

	tmpl := findPodGroupTemplate(workload, templateName)
	if tmpl == nil {
		return nil, fmt.Errorf("podGroupTemplate %q not found in workload %q (have %v)", templateName, workload.Name, podGroupTemplateNames(workload))
	}

	spec := schedulingv1beta1.PodGroupSpec{
		WorkloadRef: &schedulingv1beta1.WorkloadReference{
			WorkloadName: workload.Name,
			TemplateName: tmpl.Name,
		},
		SchedulingPolicy:      *tmpl.SchedulingPolicy.DeepCopy(),
		SchedulingConstraints: tmpl.SchedulingConstraints.DeepCopy(),
		DisruptionMode:        tmpl.DisruptionMode.DeepCopy(),
		PriorityClassName:     tmpl.PriorityClassName,
	}
	if len(tmpl.ResourceClaims) > 0 {
		spec.ResourceClaims = make([]schedulingv1beta1.PodGroupResourceClaim, len(tmpl.ResourceClaims))
		for i := range tmpl.ResourceClaims {
			tmpl.ResourceClaims[i].DeepCopyInto(&spec.ResourceClaims[i])
		}
	}
	if tmpl.Priority != nil {
		p := *tmpl.Priority
		spec.Priority = &p
	}

	return &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:            podGroupName,
			Namespace:       workload.Namespace,
			OwnerReferences: owners,
		},
		Spec: spec,
	}, nil
}

func findPodGroupTemplate(workload *schedulingv1beta1.Workload, name string) *schedulingv1beta1.PodGroupTemplate {
	for i := range workload.Spec.PodGroupTemplates {
		if workload.Spec.PodGroupTemplates[i].Name == name {
			return &workload.Spec.PodGroupTemplates[i]
		}
	}
	return nil
}

func podGroupTemplateNames(workload *schedulingv1beta1.Workload) []string {
	names := make([]string, len(workload.Spec.PodGroupTemplates))
	for i := range workload.Spec.PodGroupTemplates {
		names[i] = workload.Spec.PodGroupTemplates[i].Name
	}
	return names
}
