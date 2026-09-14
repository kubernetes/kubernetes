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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// newPodGroup materializes a runtime PodGroup from the given PodGroupTemplate,
// which BuildWorkload does not create. The template's scheduling fields are
// deep-copied into the spec, workloadRef points back at the template, and the
// PodGroup lands in the workload's namespace. Template lookup is the caller's
// responsibility (the Builder resolves it through a cached name index).
func newPodGroup(workload *schedulingv1beta1.Workload, tmpl *schedulingv1beta1.PodGroupTemplate, podGroupName string, owners []metav1.OwnerReference) *schedulingv1beta1.PodGroup {
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
		spec.Priority = new(*tmpl.Priority)
	}

	return &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:            podGroupName,
			Namespace:       workload.Namespace,
			OwnerReferences: owners,
		},
		Spec: spec,
	}
}

// newCompositePodGroup materializes a runtime CompositePodGroup from the given
// CompositePodGroupTemplate, which BuildWorkload does not create.
func newCompositePodGroup(workload *schedulingv1beta1.Workload, tmpl *schedulingv1beta1.CompositePodGroupTemplate,
	compositePodGroupName string, owners []metav1.OwnerReference) (*schedulingv1alpha3.CompositePodGroup, error) {
	spec := schedulingv1alpha3.CompositePodGroupSpec{
		WorkloadRef: &schedulingv1alpha3.WorkloadReference{
			WorkloadName: workload.Name,
			TemplateName: tmpl.Name,
		},
		PriorityClassName: tmpl.PriorityClassName,
	}

	if err := Convert_v1beta1_CompositePodGroupSchedulingPolicy_To_v1alpha3_CompositePodGroupSchedulingPolicy(&tmpl.SchedulingPolicy, &spec.SchedulingPolicy, nil); err != nil {
		return nil, err
	}
	if tmpl.SchedulingConstraints != nil {
		spec.SchedulingConstraints = &schedulingv1alpha3.CompositePodGroupSchedulingConstraints{}
		if err := Convert_v1beta1_CompositePodGroupSchedulingConstraints_To_v1alpha3_CompositePodGroupSchedulingConstraints(tmpl.SchedulingConstraints, spec.SchedulingConstraints, nil); err != nil {
			return nil, err
		}
	}
	if tmpl.DisruptionMode != nil {
		spec.DisruptionMode = &schedulingv1alpha3.CompositeDisruptionMode{}
		if err := Convert_v1beta1_CompositeDisruptionMode_To_v1alpha3_CompositeDisruptionMode(tmpl.DisruptionMode, spec.DisruptionMode, nil); err != nil {
			return nil, err
		}
	}
	if tmpl.Priority != nil {
		spec.Priority = new(*tmpl.Priority)
	}

	cpg := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:            compositePodGroupName,
			Namespace:       workload.Namespace,
			OwnerReferences: owners,
		},
		Spec: spec,
	}
	// The generated Convert_* helpers are shallow (unsafe.Pointer) copies, so
	// the spec still aliases the template's Gang/Topology pointers. DeepCopy
	// before returning so the caller can mutate the CompositePodGroup without
	// corrupting the Workload template cached in the Builder.
	return cpg.DeepCopy(), nil
}

// indexPodGroupTemplates maps every leaf PodGroupTemplate by name for O(1)
// lookup. The Builder builds it once and reuses it across materializations
// rather than rescanning the slice on every NewPodGroup call.
func indexPodGroupTemplates(workload *schedulingv1beta1.Workload) map[string]*schedulingv1beta1.PodGroupTemplate {
	index := make(map[string]*schedulingv1beta1.PodGroupTemplate, len(workload.Spec.PodGroupTemplates))
	for i := range workload.Spec.PodGroupTemplates {
		index[workload.Spec.PodGroupTemplates[i].Name] = &workload.Spec.PodGroupTemplates[i]
	}
	return index
}

func podGroupTemplateNames(workload *schedulingv1beta1.Workload) []string {
	names := make([]string, len(workload.Spec.PodGroupTemplates))
	for i := range workload.Spec.PodGroupTemplates {
		names[i] = workload.Spec.PodGroupTemplates[i].Name
	}
	return names
}

// walkCompositePodGroupTemplates invokes visit for every template in the
// composite tree, in pre-order.
func walkCompositePodGroupTemplates(tmpls []schedulingv1beta1.CompositePodGroupTemplate,
	visit func(*schedulingv1beta1.CompositePodGroupTemplate)) {
	for i := range tmpls {
		visit(&tmpls[i])
		walkCompositePodGroupTemplates(tmpls[i].CompositePodGroupTemplates, visit)
	}
}

// indexCompositePodGroupTemplates maps every CompositePodGroupTemplate in the
// workload's composite tree by name. Template names are unique across the whole
// Workload, so the flat map is unambiguous. The Builder builds it once and
// reuses it across materializations rather than re-walking the tree on every
// NewCompositePodGroup call.
func indexCompositePodGroupTemplates(workload *schedulingv1beta1.Workload) map[string]*schedulingv1beta1.CompositePodGroupTemplate {
	index := map[string]*schedulingv1beta1.CompositePodGroupTemplate{}
	walkCompositePodGroupTemplates(workload.Spec.CompositePodGroupTemplates,
		func(tmpl *schedulingv1beta1.CompositePodGroupTemplate) {
			index[tmpl.Name] = tmpl
		})
	return index
}

// compositePodGroupTemplateNames lists every CompositePodGroupTemplate name in
// the workload's composite template tree, for error messages.
func compositePodGroupTemplateNames(workload *schedulingv1beta1.Workload) []string {
	var names []string
	walkCompositePodGroupTemplates(workload.Spec.CompositePodGroupTemplates,
		func(tmpl *schedulingv1beta1.CompositePodGroupTemplate) {
			names = append(names, tmpl.Name)
		})
	return names
}
