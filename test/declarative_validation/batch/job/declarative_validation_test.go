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
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "batch",
		APIVersion: apiVersion,
	})

	testCases := map[string]struct {
		input                 batch.Job
		enableWorkloadWithJob bool
		expectedErrs          field.ErrorList
	}{
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
		"valid": {
			input: mkJob(),
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
				field.Required(field.NewPath("spec", "scheduling", "policy", "gang", "minCount"), ""),
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
			input: mkJob(setGangPolicy(4), addResourceClaims(scheduling.PodGroupResourceClaim{
				Name: "claim", ResourceClaimName: new("resource-claim"),
			})),
			enableWorkloadWithJob: true,
		},
		"resource claim with duplicate entries": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimName: new("rc-1")},
				scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimName: new("rc-2")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "scheduling", "resourceClaims").Index(1), nil),
			},
		},
		"resource claim with neither name nor template": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim"},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"resource claim with invalid short name": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "../my-claim", ResourceClaimName: new("rc")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"resource claim with empty name": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "", ResourceClaimName: new("rc")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("name"), ""),
			},
		},
		"resource claim with invalid resourceClaimName": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimName: new(".foo_bar")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"resource claim with invalid resourceClaimTemplateName": {
			input: mkJob(setGangPolicy(4), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimTemplateName: new(".foo_bar")},
			)),
			enableWorkloadWithJob: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scheduling", "resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
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
		APIGroup:   "batch",
		APIVersion: apiVersion,
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
				field.Required(field.NewPath("spec", "scheduling", "policy", "gang", "minCount"), ""),
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
			Policy: &scheduling.PodGroupSchedulingPolicy{
				Basic: &scheduling.BasicSchedulingPolicy{},
			},
		}
	}
}

func setGangPolicy(minCount int32) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{MinCount: minCount},
			},
		}
	}
}

func setEmptyPolicy() func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.PodGroupSchedulingPolicy{},
		}
	}
}

func setBothPolicies() func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Scheduling = &batch.JobSchedulingConfiguration{
			Policy: &scheduling.PodGroupSchedulingPolicy{
				Basic: &scheduling.BasicSchedulingPolicy{},
				Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
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
			job.Spec.Scheduling.Constraints = &scheduling.PodGroupSchedulingConstraints{}
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
		job.Spec.Scheduling.DisruptionMode = &scheduling.DisruptionMode{
			Single: &scheduling.SingleDisruptionMode{},
		}
	}
}

func setDisruptionModeNeither() func(*batch.Job) {
	return func(job *batch.Job) {
		if job.Spec.Scheduling == nil {
			setGangPolicy(4)(job)
		}
		job.Spec.Scheduling.DisruptionMode = &scheduling.DisruptionMode{}
	}
}

func addResourceClaims(claims ...scheduling.PodGroupResourceClaim) func(*batch.Job) {
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
