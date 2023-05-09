/*
Copyright 2015 The Kubernetes Authors.

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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

var ignoreErrValueDetail = cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")

// TestJobStrategy_PrepareForUpdate tests various scenearios for PrepareForUpdate
func TestJobStrategy_PrepareForUpdate(t *testing.T) {
	validSelector := getValidLabelSelector()
	validPodTemplateSpec := getValidPodTemplateSpecForSelector(validSelector)

	podFailurePolicy := &batch.PodFailurePolicy{
		Rules: []batch.PodFailurePolicyRule{
			{
				Action: batch.PodFailurePolicyActionFailJob,
				OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
					ContainerName: pointer.String("container-name"),
					Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
					Values:        []int32{1},
				},
			},
		},
	}
	updatedPodFailurePolicy := &batch.PodFailurePolicy{
		Rules: []batch.PodFailurePolicyRule{
			{
				Action: batch.PodFailurePolicyActionIgnore,
				OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
					ContainerName: pointer.String("updated-container-name"),
					Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
					Values:        []int32{2},
				},
			},
		},
	}

	cases := map[string]struct {
		enableJobPodFailurePolicy bool
		job                       batch.Job
		updatedJob                batch.Job
		wantJob                   batch.Job
	}{
		"update job with a new field; updated when JobPodFailurePolicy enabled": {
			enableJobPodFailurePolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
		},
		"update job with a new field; not updated when JobPodFailurePolicy disabled": {
			enableJobPodFailurePolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
		},
		"update pre-existing field; updated when JobPodFailurePolicy enabled": {
			enableJobPodFailurePolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
		},
		"update pre-existing field; updated when JobPodFailurePolicy disabled": {
			enableJobPodFailurePolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
		},
		"add tracking annotation back": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
		},
		"attempt status update and verify it doesn't change": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 2,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
		},
		"ensure generation doesn't change over non spec updates": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 2,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
		},
		"test updating suspend false->true": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  pointer.Bool(false),
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  pointer.Bool(true),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(1, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  pointer.Bool(true),
				},
			},
		},
		"test updating suspend nil -> true": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  pointer.Bool(true),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(1, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  pointer.Bool(true),
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodFailurePolicy, tc.enableJobPodFailurePolicy)()
			ctx := genericapirequest.NewDefaultContext()

			Strategy.PrepareForUpdate(ctx, &tc.updatedJob, &tc.job)

			if diff := cmp.Diff(tc.wantJob, tc.updatedJob); diff != "" {
				t.Errorf("Job update differences (-want,+got):\n%s", diff)
			}
		})
	}
}

// TestJobStrategy_PrepareForCreate tests various scenarios for PrepareForCreate
func TestJobStrategy_PrepareForCreate(t *testing.T) {
	validSelector := getValidLabelSelector()
	validPodTemplateSpec := getValidPodTemplateSpecForSelector(validSelector)

	podFailurePolicy := &batch.PodFailurePolicy{
		Rules: []batch.PodFailurePolicyRule{
			{
				Action: batch.PodFailurePolicyActionFailJob,
				OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
					ContainerName: pointer.String("container-name"),
					Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
					Values:        []int32{1},
				},
			},
		},
	}

	cases := map[string]struct {
		enableJobPodFailurePolicy bool
		job                       batch.Job
		wantJob                   batch.Job
	}{
		"create job with a new field; JobPodFailurePolicy enabled": {
			enableJobPodFailurePolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
		},
		"create job with a new field; JobPodFailurePolicy disabled": {
			enableJobPodFailurePolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
		},
		"job does not allow setting status on create": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodFailurePolicy, tc.enableJobPodFailurePolicy)()
			ctx := genericapirequest.NewDefaultContext()

			Strategy.PrepareForCreate(ctx, &tc.job)

			if diff := cmp.Diff(tc.wantJob, tc.job); diff != "" {
				t.Errorf("Job pod failure policy (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJobStrategy_GarbageCollectionPolicy(t *testing.T) {
	// Make sure we correctly implement the interface.
	// Otherwise a typo could silently change the default.
	var gcds rest.GarbageCollectionDeleteStrategy = Strategy
	if got, want := gcds.DefaultGarbageCollectionPolicy(genericapirequest.NewContext()), rest.DeleteDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}

	var (
		v1Ctx           = genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1", Resource: "jobs"})
		otherVersionCtx = genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v100", Resource: "jobs"})
	)
	if got, want := gcds.DefaultGarbageCollectionPolicy(v1Ctx), rest.OrphanDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}
	if got, want := gcds.DefaultGarbageCollectionPolicy(otherVersionCtx), rest.DeleteDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}
}

func TestJobStrategy_ValidateUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	now := metav1.Now()
	cases := map[string]struct {
		job      *batch.Job
		update   func(*batch.Job)
		wantErrs field.ErrorList
	}{
		"update parallelism": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Parallelism = pointer.Int32Ptr(2)
			},
		},
		"update completions disallowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
					Completions:    pointer.Int32Ptr(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32Ptr(2)
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.completions"},
			},
		},
		"preserving tracking annotation": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
					},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["foo"] = "bar"
			},
		},
		"deleting user annotation": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
						"foo":                      "bar",
					},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
			update: func(job *batch.Job) {
				delete(job.Annotations, "foo")
			},
		},
		"updating node selector for unsuspended job disallowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.template"},
			},
		},
		"updating node selector for suspended but previously started job disallowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
					Suspend:        pointer.BoolPtr(true),
				},
				Status: batch.JobStatus{
					StartTime: &now,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.template"},
			},
		},
		"updating node selector for suspended and not previously started job allowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
					Suspend:        pointer.BoolPtr(true),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
		},
		"invalid label selector": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"a": "b"},
						MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "key", Operator: metav1.LabelSelectorOpNotIn, Values: []string{"bad value"}}},
					},
					ManualSelector: pointer.BoolPtr(true),
					Template:       validPodTemplateSpec,
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
		"old job has no batch.kubernetes.io labels": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					UID:             "test",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
					Annotations:     map[string]string{"hello": "world"},
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "test"},
					},
					Parallelism: pointer.Int32(4),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob", batch.LegacyControllerUidLabel: "test"},
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
		"old job has all labels": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					UID:             "test",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.ControllerUidLabel: "test"},
					},
					Parallelism: pointer.Int32(4),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob", batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: "test", batch.ControllerUidLabel: "test"},
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			newJob := tc.job.DeepCopy()
			tc.update(newJob)
			errs := Strategy.ValidateUpdate(ctx, newJob, tc.job)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJobStrategy_WarningsOnUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	cases := map[string]struct {
		oldJob            *batch.Job
		job               *batch.Job
		wantWarningsCount int32
	}{
		"generation 0 for both": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},

			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
		},
		"generation 1 for new; force WarningsOnUpdate to check PodTemplate for updates": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      1,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},

			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
		},
		"force validation failure in pod template": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      1,
				},
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{Volumes: []api.Volume{{Name: "volume-name"}, {Name: "volume-name"}}},
					},
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},

			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
			wantWarningsCount: 1,
		},
	}
	for val, tc := range cases {
		t.Run(val, func(t *testing.T) {
			gotWarnings := Strategy.WarningsOnUpdate(ctx, tc.job, tc.oldJob)
			if len(gotWarnings) != int(tc.wantWarningsCount) {
				t.Errorf("got warning length of %d but expected %d", len(gotWarnings), tc.wantWarningsCount)
			}
		})
	}
}
func TestJobStrategy_WarningsOnCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	theUID := types.UID("1a2b3c4d5e6f7g8h9i0k")
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validSpec := batch.JobSpec{
		Selector: nil,
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector.MatchLabels,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyOnFailure,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
			},
		},
	}

	testcases := map[string]struct {
		job               *batch.Job
		wantWarningsCount int32
	}{
		"happy path job": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob2",
					Namespace: metav1.NamespaceDefault,
					UID:       theUID,
				},
				Spec: validSpec,
			},
		},
		"dns invalid name": {
			wantWarningsCount: 1,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my job2",
					Namespace: metav1.NamespaceDefault,
					UID:       theUID,
				},
				Spec: validSpec,
			},
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			gotWarnings := Strategy.WarningsOnCreate(ctx, tc.job)
			if len(gotWarnings) != int(tc.wantWarningsCount) {
				t.Errorf("got warning length of %d but expected %d", len(gotWarnings), tc.wantWarningsCount)
			}
		})
	}
}
func TestJobStrategy_Validate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	theUID := types.UID("1a2b3c4d5e6f7g8h9i0k")
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validLabels := map[string]string{batch.LegacyJobNameLabel: "myjob2", batch.JobNameLabel: "myjob2", batch.LegacyControllerUidLabel: string(theUID), batch.ControllerUidLabel: string(theUID)}
	labelsWithNonBatch := map[string]string{"a": "b", batch.LegacyJobNameLabel: "myjob2", batch.JobNameLabel: "myjob2", batch.LegacyControllerUidLabel: string(theUID), batch.ControllerUidLabel: string(theUID)}
	validPodSpec := api.PodSpec{
		RestartPolicy: api.RestartPolicyOnFailure,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
	}
	validObjectMeta := metav1.ObjectMeta{
		Name:      "myjob2",
		Namespace: metav1.NamespaceDefault,
		UID:       theUID,
	}
	testcases := map[string]struct {
		job              *batch.Job
		wantJob          *batch.Job
		wantWarningCount int32
	}{
		"valid job with labels in pod template": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: nil,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{batch.ControllerUidLabel: string(theUID)}},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					}},
			},
		},
		"no labels in job": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: nil,
					Template: api.PodTemplateSpec{
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{batch.ControllerUidLabel: string(theUID)}},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validLabels,
						},
						Spec: validPodSpec,
					}},
			},
		},
		"labels exist": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: nil,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{batch.ControllerUidLabel: string(theUID)}},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					}},
			},
		},
		"manual selector; do not generate labels": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					},
					Completions:    pointer.Int32Ptr(2),
					ManualSelector: pointer.BoolPtr(true),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					},
					Completions:    pointer.Int32Ptr(2),
					ManualSelector: pointer.BoolPtr(true),
				},
			},
		},
		"valid job with extended configuration": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: nil,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					},
					Completions:             pointer.Int32Ptr(2),
					Suspend:                 pointer.BoolPtr(true),
					TTLSecondsAfterFinished: pointer.Int32Ptr(0),
					CompletionMode:          completionModePtr(batch.IndexedCompletion),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{batch.ControllerUidLabel: string(theUID)}},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					},
					Completions:             pointer.Int32Ptr(2),
					Suspend:                 pointer.BoolPtr(true),
					TTLSecondsAfterFinished: pointer.Int32Ptr(0),
					CompletionMode:          completionModePtr(batch.IndexedCompletion),
				},
			},
		},
		"fail validation due to invalid volume spec": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: nil,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							Volumes:       []api.Volume{{Name: "volume-name"}},
						},
					},
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{batch.ControllerUidLabel: string(theUID)}},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							Volumes:       []api.Volume{{Name: "volume-name"}},
						},
					},
				},
			},
			wantWarningCount: 1,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			errs := Strategy.Validate(ctx, tc.job)
			if len(errs) != int(tc.wantWarningCount) {
				t.Errorf("want warnings %d but got %d", tc.wantWarningCount, len(errs))
			}
			if diff := cmp.Diff(tc.wantJob, tc.job); diff != "" {
				t.Errorf("Unexpected job (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestStrategy_ResetFields(t *testing.T) {
	resetFields := Strategy.GetResetFields()
	if len(resetFields) != 1 {
		t.Errorf("ResetFields should have 1 element, but have %d", len(resetFields))
	}
}

func TestJobStatusStrategy_ResetFields(t *testing.T) {
	resetFields := StatusStrategy.GetResetFields()
	if len(resetFields) != 1 {
		t.Errorf("ResetFields should have 1 element, but have %d", len(resetFields))
	}
}

func TestStatusStrategy_PrepareForUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	validObjectMeta := metav1.ObjectMeta{
		Name:            "myjob",
		Namespace:       metav1.NamespaceDefault,
		ResourceVersion: "10",
	}

	cases := map[string]struct {
		job     *batch.Job
		newJob  *batch.Job
		wantJob *batch.Job
	}{
		"job must allow status updates": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
				Status: batch.JobStatus{
					Active: 11,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
				Status: batch.JobStatus{
					Active: 12,
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
				Status: batch.JobStatus{
					Active: 12,
				},
			},
		},
		"parallelism changes not allowed": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(3),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(3),
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			StatusStrategy.PrepareForUpdate(ctx, tc.newJob, tc.job)
			if diff := cmp.Diff(tc.wantJob, tc.newJob); diff != "" {
				t.Errorf("Unexpected job (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestStatusStrategy_ValidateUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}

	cases := map[string]struct {
		job     *batch.Job
		newJob  *batch.Job
		wantJob *batch.Job
	}{
		"incoming resource version on update should not be mutated": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
				},
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
				},
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
				},
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: pointer.Int32(4),
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			errs := StatusStrategy.ValidateUpdate(ctx, tc.newJob, tc.job)
			if len(errs) != 0 {
				t.Errorf("Unexpected error %v", errs)
			}
			if diff := cmp.Diff(tc.wantJob, tc.newJob); diff != "" {
				t.Errorf("Unexpected job (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJobStrategy_GetAttrs(t *testing.T) {
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}

	cases := map[string]struct {
		job          *batch.Job
		wantErr      string
		nonJobObject *api.Pod
	}{
		"valid job with no labels": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
		},
		"valid job with a label": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Labels:          map[string]string{"a": "b"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: pointer.BoolPtr(true),
					Parallelism:    pointer.Int32Ptr(1),
				},
			},
		},
		"pod instead": {
			job:          nil,
			nonJobObject: &api.Pod{},
			wantErr:      "given object is not a job.",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if tc.job == nil {
				_, _, err := GetAttrs(tc.nonJobObject)
				if diff := cmp.Diff(tc.wantErr, err.Error()); diff != "" {
					t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
				}
			} else {
				gotLabels, _, err := GetAttrs(tc.job)
				if err != nil {
					t.Errorf("Error %s supposed to be nil", err.Error())
				}
				if diff := cmp.Diff(labels.Set(tc.job.ObjectMeta.Labels), gotLabels); diff != "" {
					t.Errorf("Unexpected attrs (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestJobToSelectiableFields(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"batch/v1",
		"Job",
		JobToSelectableFields(&batch.Job{}),
		nil,
	)
}

func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}

func getValidObjectMeta(generation int64) metav1.ObjectMeta {
	return getValidObjectMetaWithAnnotations(generation, nil)
}

func getValidObjectMetaWithAnnotations(generation int64, annotations map[string]string) metav1.ObjectMeta {
	return metav1.ObjectMeta{
		Name:        "myjob",
		Namespace:   metav1.NamespaceDefault,
		Generation:  generation,
		Annotations: annotations,
	}
}

func getValidLabelSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
}

func getValidPodTemplateSpecForSelector(validSelector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}
