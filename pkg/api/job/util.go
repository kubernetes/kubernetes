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

package job

import (
	batchv1 "k8s.io/api/batch/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

// DropDisabledFields removes feature-gated fields from the JobSpec when their
// feature gate is disabled and they are not already in use in the old spec.
// oldJobSpec is nil on create. It is shared by the Job and CronJob registry
// strategies, which both embed a JobSpec.
func DropDisabledFields(jobSpec, oldJobSpec *batch.JobSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithJob) || schedulingInUse(oldJobSpec) {
		return
	}
	jobSpec.Scheduling = nil
}

func schedulingInUse(jobSpec *batch.JobSpec) bool {
	return jobSpec != nil && jobSpec.Scheduling != nil
}

func WorkloadInputForJobInternal(s *batch.JobSchedulingConfiguration) workloadbuilder.WorkloadInput {
	if s == nil {
		s = &batch.JobSchedulingConfiguration{}
	}
	return WorkloadInput(s.SchedulingPolicy, s.SchedulingConstraints, s.DisruptionMode, s.ResourceClaims)
}
func WorkloadInputForJobV1(s *batchv1.JobSchedulingConfiguration) workloadbuilder.WorkloadInput {
	if s == nil {
		s = &batchv1.JobSchedulingConfiguration{}
	}
	return WorkloadInput(s.SchedulingPolicy, s.SchedulingConstraints, s.DisruptionMode, s.ResourceClaims)
}

// WorkloadInput maps a Job's spec.scheduling building blocks into a
// workloadbuilder.WorkloadInput, pairing each block with the spec.scheduling
// sub-path where it lives so validation errors are reported at the right field.
// Both Job controller and the internal batch API embed the schedulingv1alpha3
// building blocks directly, so this single mapping serves both.
// Nil blocks fall back to the item's DefaultConfig.
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

// WorkloadItemForJob assembles the single-node logical workload tree for a Job.
// It is shared by the Job registry validation and the Job controller so the two
// never drift.
func WorkloadItemForJob(name, priorityClassName string, parallelism *int32, input workloadbuilder.WorkloadInput) *workloadbuilder.WorkloadItem {
	return &workloadbuilder.WorkloadItem{
		Name: name,
		DefaultConfig: &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{
				Basic: &workloadbuilder.BasicSchedulingPolicy{},
			},
			PriorityClassName: priorityClassName,
		},
		Input:     input,
		Callbacks: []workloadbuilder.SchedulingConfigFunc{defaultMinCountForJob(parallelism)},
	}
}

// defaultMinCountForJob returns a callback that defaults an unset gang MinCount
// to the Job's parallelism, clamped to a minimum of 1 (parallelism may be 0 for
// a suspended Job, but MinCount must be positive). It mutates only the resolved
// config, never writing the derived value back onto the Job's spec.
func defaultMinCountForJob(parallelism *int32) workloadbuilder.SchedulingConfigFunc {
	return func(cfg *workloadbuilder.SchedulingConfig) {
		if cfg == nil || cfg.Policy == nil || cfg.Policy.Gang == nil {
			return
		}
		if cfg.Policy.Gang.MinCount == nil {
			cfg.Policy.Gang.MinCount = new(max(int32(1), ptr.Deref(parallelism, 1)))
		}
	}
}
