/*
Copyright 2025 The Kubernetes Authors.

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

package cronjob

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/batch/cronjob"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidate(t, apiVersion)
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "batch",
		APIVersion:        apiVersion,
		Resource:          "cronjobs",
		IsResourceRequest: true,
		Verb:              "create",
	})
	schedulingPath := field.NewPath("spec", "jobTemplate", "spec", "scheduling")
	testCases := map[string]struct {
		input                 batch.CronJob
		enableWorkloadWithJob bool
		expectedErrs          field.ErrorList
	}{
		"valid": {
			input: mkCronJob(),
		},
		"schedule: empty": {
			input: mkCronJob(tweakSchedule("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "schedule"), "").MarkBeta(),
			},
		},
		"jobTemplate.spec.maxFailedIndexes set without backoffLimitPerIndex": {
			input: mkCronJob(tweakJobTemplateMaxFailedIndexes(ptr.To[int32](5))),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "jobTemplate", "spec", "backoffLimitPerIndex"), "").WithOrigin("dependentRequired").MarkAlpha(),
			},
		},
		"jobTemplate.spec.maxFailedIndexes and backoffLimitPerIndex both set": {
			input: mkCronJob(
				tweakJobTemplateMaxFailedIndexes(ptr.To[int32](5)),
				tweakJobTemplateBackoffLimitPerIndex(ptr.To[int32](1)),
			),
		},
		"valid with basic scheduling policy": {
			input:                 mkCronJob(tweakJobSchedulingBasic()),
			enableWorkloadWithJob: true,
		},
		"scheduling forbidden when feature gate disabled": {
			input: mkCronJob(tweakJobSchedulingBasic()),
			expectedErrs: field.ErrorList{
				field.Forbidden(schedulingPath, ""),
			},
		},
		"scheduling policy with neither basic nor gang": {
			input:                 mkCronJob(tweakJobSchedulingEmptyPolicy()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy"), nil, "").WithOrigin("union"),
			},
		},
		"scheduling policy with both basic and gang": {
			input:                 mkCronJob(tweakJobSchedulingBothPolicies()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy"), nil, "").WithOrigin("union"),
			},
		},
		"gang minCount zero": {
			input:                 mkCronJob(tweakJobSchedulingGang(0)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"gang minCount negative": {
			input:                 mkCronJob(tweakJobSchedulingGang(-1)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"topology constraints with too many entries": {
			input:                 mkCronJob(tweakJobSchedulingGang(4), tweakJobTopologyConstraint("foo"), tweakJobTopologyConstraint("bar")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.TooMany(schedulingPath.Child("constraints", "topology"), 2, 1).WithOrigin("maxItems"),
			},
		},
		"topology constraint with empty key": {
			input:                 mkCronJob(tweakJobSchedulingGang(4), tweakJobTopologyConstraint("")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Required(schedulingPath.Child("constraints", "topology").Index(0).Child("key"), ""),
			},
		},
		"topology constraint with invalid key": {
			input:                 mkCronJob(tweakJobSchedulingGang(4), tweakJobTopologyConstraint(strings.Repeat("a", 254)+"/foo")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("constraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key"),
			},
		},
		"disruption mode with neither single nor all": {
			input:                 mkCronJob(tweakJobSchedulingGang(4), tweakJobDisruptionModeNeither()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("disruptionMode"), nil, "").WithOrigin("union"),
			},
		},
		"too many resource claims": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "c1", ResourceClaimName: new("rc1")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c2", ResourceClaimName: new("rc2")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c3", ResourceClaimName: new("rc3")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c4", ResourceClaimName: new("rc4")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c5", ResourceClaimName: new("rc5")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.TooMany(schedulingPath.Child("resourceClaims"), 5, 4).WithOrigin("maxItems"),
			},
		},
		"resource claim with duplicate entries": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimName: new("rc-1")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimName: new("rc-2")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Duplicate(schedulingPath.Child("resourceClaims").Index(1), nil),
			},
		},
		"resource claim with neither name nor template": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim"},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"resource claim with invalid short name": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "../claim", ResourceClaimName: new("rc")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"resource claim with empty name": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "", ResourceClaimName: new("rc")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Required(schedulingPath.Child("resourceClaims").Index(0).Child("name"), ""),
			},
		},
		"resource claim with invalid resourceClaimName": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimName: new(".foo_bar")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"resource claim with invalid resourceClaimTemplateName": {
			input: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimTemplateName: new(".foo_bar")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"tolerations: valid key": {
			input: mkCronJob(tweakTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists})),
		},
		"tolerations: valid key without prefix": {
			input: mkCronJob(tweakTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists})),
		},
		"tolerations: invalid key format": {
			input: mkCronJob(tweakTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "jobTemplate", "spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tc.enableWorkloadWithJob,
				features.WorkloadWithJob: tc.enableWorkloadWithJob,
			})
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
	obj := mkCronJob()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())

}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateUpdate(t, apiVersion)
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "batch",
		APIVersion:        apiVersion,
		Resource:          "cronjobs",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})
	schedulingPath := field.NewPath("spec", "jobTemplate", "spec", "scheduling")
	testCases := map[string]struct {
		old                   batch.CronJob
		update                batch.CronJob
		enableWorkloadWithJob bool
		expectedErrs          field.ErrorList
	}{
		"valid (no changes)": {
			old:    mkCronJob(),
			update: mkCronJob(),
		},

		"schedule: updated to empty": {
			old:    mkCronJob(),
			update: mkCronJob(tweakSchedule("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "schedule"), "").MarkBeta(),
			},
		},
		"valid unchanged scheduling": {
			old:                   mkCronJob(tweakJobSchedulingBasic()),
			update:                mkCronJob(tweakJobSchedulingBasic()),
			enableWorkloadWithJob: true,
		},
		"gang minCount change is valid": {
			old:                   mkCronJob(tweakJobSchedulingGang(4)),
			update:                mkCronJob(tweakJobSchedulingGang(8)),
			enableWorkloadWithJob: true,
		},
		"gang minCount updated below minimum": {
			old:                   mkCronJob(tweakJobSchedulingGang(4)),
			update:                mkCronJob(tweakJobSchedulingGang(0)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"switching policy from basic to gang is allowed for cronjob": {
			// basic and gang policy variant immutability cannot be expressed via
			// DV and is only enforced imperatively on the Job update path. A
			// CronJob spawns a fresh Workload per Job, so its jobTemplate policy
			// may change; future Jobs simply use the new policy.
			old:                   mkCronJob(tweakJobSchedulingBasic()),
			update:                mkCronJob(tweakJobSchedulingGang(4)),
			enableWorkloadWithJob: true,
		},
		"adding scheduling after creation is immutable": {
			old:                   mkCronJob(),
			update:                mkCronJob(tweakJobSchedulingBasic()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath, nil, "").WithOrigin("update"),
			},
		},
		"removing scheduling after creation is immutable": {
			old:                   mkCronJob(tweakJobSchedulingBasic()),
			update:                mkCronJob(),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath, nil, "").WithOrigin("update"),
			},
		},
		"adding policy is immutable": {
			old:                   mkCronJob(tweakJobSchedulingNoPolicy()),
			update:                mkCronJob(tweakJobSchedulingBasic()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy"), nil, "").WithOrigin("update"),
			},
		},
		"removing policy is immutable": {
			old:                   mkCronJob(tweakJobSchedulingBasic()),
			update:                mkCronJob(tweakJobSchedulingNoPolicy()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("policy"), nil, "").WithOrigin("update"),
			},
		},
		"changing constraints is immutable": {
			old:                   mkCronJob(tweakJobSchedulingGang(4), tweakJobTopologyConstraint("topology.kubernetes.io/zone")),
			update:                mkCronJob(tweakJobSchedulingGang(4), tweakJobTopologyConstraint("topology.kubernetes.io/rack")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("constraints"), nil, "").WithOrigin("immutable"),
			},
		},
		"switching disruption mode is immutable": {
			old:                   mkCronJob(tweakJobSchedulingGang(4), tweakJobDisruptionModeSingle()),
			update:                mkCronJob(tweakJobSchedulingGang(4), tweakJobDisruptionModeAll()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"adding disruption mode is immutable": {
			old:                   mkCronJob(tweakJobSchedulingGang(4)),
			update:                mkCronJob(tweakJobSchedulingGang(4), tweakJobDisruptionModeSingle()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"removing disruption mode is immutable": {
			old:                   mkCronJob(tweakJobSchedulingGang(4), tweakJobDisruptionModeSingle()),
			update:                mkCronJob(tweakJobSchedulingGang(4)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"adding a resource claim is immutable": {
			old: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
			)),
			update: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-b", ResourceClaimName: new("rc-b")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"removing a resource claim is immutable": {
			old: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-b", ResourceClaimName: new("rc-b")},
			)),
			update: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"changing a resource claim name is immutable": {
			old: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
			)),
			update: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-b")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"changing a resource claim template name is immutable": {
			old: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimTemplateName: new("rct-a")},
			)),
			update: mkCronJob(tweakJobSchedulingGang(4), tweakJobResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimTemplateName: new("rct-b")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(schedulingPath.Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tc.enableWorkloadWithJob,
				features.WorkloadWithJob: tc.enableWorkloadWithJob,
			})
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
	updateObj := mkCronJob()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func tweakJobSchedulingBasic() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		cj.Spec.JobTemplate.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{
				Basic: &scheduling.WorkloadPodGroupBasicSchedulingPolicy{},
			},
		}
	}
}

func tweakJobSchedulingGang(minCount int32) func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		cj.Spec.JobTemplate.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{
				Gang: &scheduling.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(minCount)},
			},
		}
	}
}

func tweakJobSchedulingEmptyPolicy() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		cj.Spec.JobTemplate.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{},
		}
	}
}

func tweakJobSchedulingNoPolicy() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		cj.Spec.JobTemplate.Spec.Scheduling = &batch.JobSchedulingConfiguration{}
	}
}

func tweakJobSchedulingBothPolicies() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		cj.Spec.JobTemplate.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{
				Basic: &scheduling.WorkloadPodGroupBasicSchedulingPolicy{},
				Gang:  &scheduling.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(1))},
			},
		}
	}
}

func tweakJobTopologyConstraint(key string) func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		if cj.Spec.JobTemplate.Spec.Scheduling == nil {
			tweakJobSchedulingGang(4)(cj)
		}
		if cj.Spec.JobTemplate.Spec.Scheduling.Constraints == nil {
			cj.Spec.JobTemplate.Spec.Scheduling.Constraints = &scheduling.WorkloadPodGroupSchedulingConstraints{}
		}
		cj.Spec.JobTemplate.Spec.Scheduling.Constraints.Topology = append(
			cj.Spec.JobTemplate.Spec.Scheduling.Constraints.Topology,
			scheduling.TopologyConstraint{Key: key},
		)
	}
}

func tweakJobDisruptionModeNeither() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		if cj.Spec.JobTemplate.Spec.Scheduling == nil {
			tweakJobSchedulingGang(4)(cj)
		}
		cj.Spec.JobTemplate.Spec.Scheduling.DisruptionMode = &scheduling.WorkloadPodGroupDisruptionMode{}
	}
}

func tweakJobDisruptionModeSingle() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		if cj.Spec.JobTemplate.Spec.Scheduling == nil {
			tweakJobSchedulingGang(4)(cj)
		}
		cj.Spec.JobTemplate.Spec.Scheduling.DisruptionMode = &scheduling.WorkloadPodGroupDisruptionMode{
			Single: &scheduling.WorkloadPodGroupSingleDisruptionMode{},
		}
	}
}

func tweakJobDisruptionModeAll() func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		if cj.Spec.JobTemplate.Spec.Scheduling == nil {
			tweakJobSchedulingGang(4)(cj)
		}
		cj.Spec.JobTemplate.Spec.Scheduling.DisruptionMode = &scheduling.WorkloadPodGroupDisruptionMode{
			All: &scheduling.WorkloadPodGroupAllDisruptionMode{},
		}
	}
}

func tweakJobResourceClaims(claims ...scheduling.WorkloadPodGroupResourceClaim) func(*batch.CronJob) {
	return func(cj *batch.CronJob) {
		if cj.Spec.JobTemplate.Spec.Scheduling == nil {
			tweakJobSchedulingGang(4)(cj)
		}
		cj.Spec.JobTemplate.Spec.Scheduling.ResourceClaims = append(
			cj.Spec.JobTemplate.Spec.Scheduling.ResourceClaims, claims...,
		)
	}
}

func mkCronJob(tweaks ...func(*batch.CronJob)) batch.CronJob {
	job := batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-class",
			Namespace: "test-namespace",
		},
		Spec: validCronjobSpec,
	}

	for _, tweak := range tweaks {
		tweak(&job)
	}

	return job
}

func tweakSchedule(schedule string) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.Schedule = schedule
	}
}

func tweakJobTemplateMaxFailedIndexes(v *int32) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.JobTemplate.Spec.MaxFailedIndexes = v
	}
}

func tweakJobTemplateBackoffLimitPerIndex(v *int32) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.JobTemplate.Spec.BackoffLimitPerIndex = v
	}
}

func tweakTolerations(tolerations ...api.Toleration) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.JobTemplate.Spec.Template.Spec.Tolerations = tolerations
	}
}

var validCronjobSpec = batch.CronJobSpec{
	Schedule:          "5 5 * * ?",
	ConcurrencyPolicy: batch.AllowConcurrent,
	TimeZone:          ptr.To("Asia/Shanghai"),
	JobTemplate: batch.JobTemplateSpec{
		Spec: batch.JobSpec{
			Template: api.PodTemplateSpec{
				Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
			},
			CompletionMode: ptr.To(batch.IndexedCompletion),
			Completions:    ptr.To[int32](10),
			Parallelism:    ptr.To[int32](10),
		},
	},
}
