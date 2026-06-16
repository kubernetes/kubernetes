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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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
	registry "k8s.io/kubernetes/pkg/registry/batch/job"
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
		APIGroup:          "batch",
		APIVersion:        apiVersion,
		Resource:          "jobs",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input                 batch.Job
		enableWorkloadWithJob bool
		expectedErrs          field.ErrorList
	}{
		"valid": {
			input: mkJob(),
		},
		"maxFailedIndexes and backoffLimitPerIndex both set": {
			input: mkJob(
				tweakMaxFailedIndexes(ptr.To[int32](5)),
				tweakBackoffLimitPerIndex(ptr.To[int32](1)),
			),
		},
		"maxFailedIndexes set without backoffLimitPerIndex": {
			input: mkJob(tweakMaxFailedIndexes(ptr.To[int32](5))),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "backoffLimitPerIndex"), "").WithOrigin("dependentRequired").MarkAlpha(),
			},
		},
		"valid with basic scheduling policy": {
			input:                 mkJob(setBasicPolicy()),
			enableWorkloadWithJob: true,
		},
		"valid with gang scheduling policy": {
			input:                 mkJob(setGangPolicy(4)),
			enableWorkloadWithJob: true,
		},
		"scheduling forbidden when feature gate disabled": {
			input: mkJob(setBasicPolicy()),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "scheduling"), ""),
			},
		},
		"scheduling policy with neither basic nor gang": {
			input:                 mkJob(setEmptyPolicy()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy"), nil, "").WithOrigin("union"),
			},
		},
		"scheduling policy with both basic and gang": {
			input:                 mkJob(setBothPolicies()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy"), nil, "").WithOrigin("union"),
			},
		},
		"gang minCount zero": {
			input:                 mkJob(setGangPolicy(0)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"gang minCount negative": {
			input:                 mkJob(setGangPolicy(-1)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"valid with topology constraints": {
			input:                 mkJob(setGangPolicy(4), addTopologyConstraint("topology.kubernetes.io/zone")),
			enableWorkloadWithJob: true,
		},
		"topology constraints with too many entries": {
			input:                 mkJob(setGangPolicy(4), addTopologyConstraint("foo"), addTopologyConstraint("bar")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "scheduling", "constraints", "topology"), 2, 1).WithOrigin("maxItems"),
			},
		},
		"topology constraint with empty key": {
			input:                 mkJob(setGangPolicy(4), addTopologyConstraint("")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scheduling", "constraints", "topology").Index(0).Child("key"), ""),
			},
		},
		"topology constraint with invalid key": {
			input:                 mkJob(setGangPolicy(4), addTopologyConstraint(strings.Repeat("a", 254)+"/foo")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "constraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key"),
			},
		},
		"valid with disruption mode": {
			input:                 mkJob(setGangPolicy(4), setDisruptionModeSingle()),
			enableWorkloadWithJob: true,
		},
		"disruption mode with neither single nor all": {
			input:                 mkJob(setGangPolicy(4), setDisruptionModeNeither()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "disruptionMode"), nil, "").WithOrigin("union"),
			},
		},
		"valid resource claim with name reference": {
			input: mkJob(setGangPolicy(4), addResourceClaims(scheduling.WorkloadPodGroupResourceClaim{
				Name: "claim", ResourceClaimName: new("resource-claim"),
			})),
			enableWorkloadWithJob: true,
		},
		"too many resource claims": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "c1", ResourceClaimName: new("rc1")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c2", ResourceClaimName: new("rc2")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c3", ResourceClaimName: new("rc3")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c4", ResourceClaimName: new("rc4")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "c5", ResourceClaimName: new("rc5")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "scheduling", "resourceClaims"), 5, 4).WithOrigin("maxItems"),
			},
		},
		"resource claim with duplicate entries": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimName: new("rc-1")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimName: new("rc-2")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "scheduling", "resourceClaims").Index(1), nil),
			},
		},
		"resource claim with neither name nor template": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim"},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"resource claim with invalid short name": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "../my-claim", ResourceClaimName: new("rc")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"resource claim with empty name": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "", ResourceClaimName: new("rc")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("name"), ""),
			},
		},
		"resource claim with invalid resourceClaimName": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimName: new(".foo_bar")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"resource claim with invalid resourceClaimTemplateName": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim", ResourceClaimTemplateName: new(".foo_bar")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"valid toleration key": {
			input: mkJob(tweakTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists})),
		},
		"valid toleration key without prefix": {
			input: mkJob(tweakTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists})),
		},
		"invalid toleration key format": {
			input: mkJob(tweakTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
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
		job := mkJob()
		meta.RunObjectMetaTestCases(t, ctx, &job, registry.Strategy, meta.WithStringentFinalizerValidation())
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "batch",
		APIVersion:        apiVersion,
		Resource:          "jobs",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		old                   batch.Job
		update                batch.Job
		enableWorkloadWithJob bool
		expectedErrs          field.ErrorList
	}{
		"valid unchanged scheduling": {
			old:                   mkJob(setResourceVersion("1"), setBasicPolicy()),
			update:                mkJob(setResourceVersion("1"), setBasicPolicy()),
			enableWorkloadWithJob: true,
		},
		"gang minCount change is valid": {
			old:                   mkJob(setResourceVersion("1"), setGangPolicy(4)),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(8)),
			enableWorkloadWithJob: true,
		},
		"gang minCount updated below minimum": {
			old:                   mkJob(setResourceVersion("1"), setGangPolicy(4)),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(0)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"switching policy from basic to gang is immutable": {
			old:                   mkJob(setResourceVersion("1"), setBasicPolicy()),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(4)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				// Declarative immutability of the policy union: basic may not be
				// removed and gang may not be added.
				field.Invalid(field.NewPath("spec", "scheduling", "policy", "basic"), nil, "").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "scheduling", "policy", "gang"), nil, "").WithOrigin("update"),
			},
		},
		"adding scheduling after creation is immutable": {
			old:                   mkJob(setResourceVersion("1")),
			update:                mkJob(setResourceVersion("1"), setBasicPolicy()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				// Declarative NoSet constraint on the field itself.
				field.Invalid(field.NewPath("spec", "scheduling"), nil, "").WithOrigin("update"),
			},
		},
		"removing scheduling after creation is immutable": {
			old:                   mkJob(setResourceVersion("1"), setBasicPolicy()),
			update:                mkJob(setResourceVersion("1")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				// Declarative NoUnset constraint on the field itself.
				field.Invalid(field.NewPath("spec", "scheduling"), nil, "").WithOrigin("update"),
			},
		},
		"adding policy is immutable": {
			old:                   mkJob(setResourceVersion("1"), setSchedulingNoPolicy()),
			update:                mkJob(setResourceVersion("1"), setBasicPolicy()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy"), nil, "").WithOrigin("update"),
			},
		},
		"removing policy is immutable": {
			old:                   mkJob(setResourceVersion("1"), setBasicPolicy()),
			update:                mkJob(setResourceVersion("1"), setSchedulingNoPolicy()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "policy"), nil, "").WithOrigin("update"),
			},
		},
		"changing constraints is immutable": {
			old:                   mkJob(setResourceVersion("1"), setGangPolicy(4), addTopologyConstraint("topology.kubernetes.io/zone")),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(4), addTopologyConstraint("topology.kubernetes.io/rack")),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "constraints"), nil, "").WithOrigin("immutable"),
			},
		},
		"switching disruption mode is immutable": {
			old:                   mkJob(setResourceVersion("1"), setGangPolicy(4), setDisruptionModeSingle()),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(4), setDisruptionModeAll()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"adding disruption mode is immutable": {
			old:                   mkJob(setResourceVersion("1"), setGangPolicy(4)),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(4), setDisruptionModeSingle()),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"removing disruption mode is immutable": {
			old:                   mkJob(setResourceVersion("1"), setGangPolicy(4), setDisruptionModeSingle()),
			update:                mkJob(setResourceVersion("1"), setGangPolicy(4)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"adding a resource claim is immutable": {
			old: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
			)),
			update: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-b", ResourceClaimName: new("rc-b")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "scheduling", "resourceClaims").Index(1), "").WithOrigin("update"),
			},
		},
		"removing a resource claim is immutable": {
			old: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-b", ResourceClaimName: new("rc-b")},
			)),
			update: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "scheduling", "resourceClaims"), "").WithOrigin("update"),
			},
		},
		"changing a resource claim name is immutable": {
			old: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-a")},
			)),
			update: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimName: new("rc-b")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("immutable"),
			},
		},
		"changing a resource claim template name is immutable": {
			old: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimTemplateName: new("rct-a")},
			)),
			update: mkJob(setResourceVersion("1"), setGangPolicy(4), addResourceClaims(
				scheduling.WorkloadPodGroupResourceClaim{Name: "claim-a", ResourceClaimTemplateName: new("rct-b")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("immutable"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tc.enableWorkloadWithJob,
				features.WorkloadWithJob: tc.enableWorkloadWithJob,
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
	updateObj := mkJob()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkJob(mutators ...func(*batch.Job)) batch.Job {
	job := batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: validJobSpec,
	}
	manualSelector := true
	job.Spec.ManualSelector = &manualSelector
	for _, mutate := range mutators {
		mutate(&job)
	}
	return job
}

func setResourceVersion(rv string) func(*batch.Job) {
	return func(job *batch.Job) {
		job.ResourceVersion = rv
	}
}

func setBasicPolicy() func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{
				Basic: &scheduling.WorkloadPodGroupBasicSchedulingPolicy{},
			},
		}
	}
}

func setGangPolicy(minCount int32) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{
				Gang: &scheduling.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(minCount)},
			},
		}
	}
}

func setEmptyPolicy() func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{},
		}
	}
}

func setSchedulingNoPolicy() func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{}
	}
}

func setBothPolicies() func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.WorkloadPodGroupSchedulingPolicy{
				Basic: &scheduling.WorkloadPodGroupBasicSchedulingPolicy{},
				Gang:  &scheduling.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(1))},
			},
		}
	}
}

func addTopologyConstraint(key string) func(*batch.Job) {
	return func(job *batch.Job) {
		if job.Spec.Scheduling == nil {
			setGangPolicy(4)(job)
		}
		if job.Spec.Scheduling.Constraints == nil {
			job.Spec.Scheduling.Constraints = &scheduling.WorkloadPodGroupSchedulingConstraints{}
		}
		job.Spec.Scheduling.Constraints.Topology = append(
			job.Spec.Scheduling.Constraints.Topology,
			scheduling.TopologyConstraint{Key: key},
		)
	}
}

func setDisruptionModeSingle() func(*batch.Job) {
	return func(job *batch.Job) {
		if job.Spec.Scheduling == nil {
			setGangPolicy(4)(job)
		}
		job.Spec.Scheduling.DisruptionMode = &scheduling.WorkloadPodGroupDisruptionMode{
			Single: &scheduling.WorkloadPodGroupSingleDisruptionMode{},
		}
	}
}

func setDisruptionModeAll() func(*batch.Job) {
	return func(job *batch.Job) {
		if job.Spec.Scheduling == nil {
			setGangPolicy(4)(job)
		}
		job.Spec.Scheduling.DisruptionMode = &scheduling.WorkloadPodGroupDisruptionMode{
			All: &scheduling.WorkloadPodGroupAllDisruptionMode{},
		}
	}
}

func setDisruptionModeNeither() func(*batch.Job) {
	return func(job *batch.Job) {
		if job.Spec.Scheduling == nil {
			setGangPolicy(4)(job)
		}
		job.Spec.Scheduling.DisruptionMode = &scheduling.WorkloadPodGroupDisruptionMode{}
	}
}

func addResourceClaims(claims ...scheduling.WorkloadPodGroupResourceClaim) func(*batch.Job) {
	return func(job *batch.Job) {
		if job.Spec.Scheduling == nil {
			setGangPolicy(4)(job)
		}
		job.Spec.Scheduling.ResourceClaims = append(job.Spec.Scheduling.ResourceClaims, claims...)
	}
}

func tweakMaxFailedIndexes(v *int32) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.MaxFailedIndexes = v
	}
}

func tweakBackoffLimitPerIndex(v *int32) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.BackoffLimitPerIndex = v
	}
}

func tweakTolerations(tolerations ...api.Toleration) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Template.Spec.Tolerations = tolerations
	}
}

var validJobLabels = map[string]string{
	batch.ControllerUidLabel:       "1a2b3c",
	batch.LegacyControllerUidLabel: "1a2b3c",
	batch.JobNameLabel:             "myjob",
	batch.LegacyJobNameLabel:       "myjob",
}

var validJobSpec = batch.JobSpec{
	Selector: &metav1.LabelSelector{MatchLabels: validJobLabels},
	Template: api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: validJobLabels},
		Spec:       podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyNever)),
	},
	CompletionMode: ptr.To(batch.IndexedCompletion),
	Completions:    ptr.To[int32](10),
	Parallelism:    ptr.To[int32](10),
}
