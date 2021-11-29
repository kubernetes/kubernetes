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
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	batchv1 "k8s.io/api/batch/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func TestJobStrategy(t *testing.T) {
	cases := map[string]struct {
		indexedJobEnabled             bool
		suspendJobEnabled             bool
		trackingWithFinalizersEnabled bool
	}{
		"features disabled": {},
		"indexed job enabled": {
			indexedJobEnabled: true,
		},
		"suspend job enabled": {
			suspendJobEnabled: true,
		},
		"new job tracking enabled": {
			trackingWithFinalizersEnabled: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IndexedJob, tc.indexedJobEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SuspendJob, tc.suspendJobEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.trackingWithFinalizersEnabled)()
			testJobStrategy(t)
		})
	}
}

func testJobStrategy(t *testing.T) {
	indexedJobEnabled := utilfeature.DefaultFeatureGate.Enabled(features.IndexedJob)
	suspendJobEnabled := utilfeature.DefaultFeatureGate.Enabled(features.SuspendJob)
	trackingWithFinalizersEnabled := utilfeature.DefaultFeatureGate.Enabled(features.JobTrackingWithFinalizers)
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("Job must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Job should not allow create on update")
	}

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
	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myjob",
			Namespace: metav1.NamespaceDefault,
			Annotations: map[string]string{
				"foo": "bar",
			},
			ResourceVersion: "0",
		},
		Spec: batch.JobSpec{
			Selector:       validSelector,
			Template:       validPodTemplateSpec,
			ManualSelector: pointer.BoolPtr(true),
			Completions:    pointer.Int32Ptr(2),
			// Set gated values.
			Suspend:                 pointer.BoolPtr(true),
			TTLSecondsAfterFinished: pointer.Int32Ptr(0),
			CompletionMode:          completionModePtr(batch.IndexedCompletion),
		},
		Status: batch.JobStatus{
			Active: 11,
		},
	}

	Strategy.PrepareForCreate(ctx, job)
	if job.Status.Active != 0 {
		t.Errorf("Job does not allow setting status on create")
	}
	if job.Generation != 1 {
		t.Errorf("expected Generation=1, got %d", job.Generation)
	}
	errs := Strategy.Validate(ctx, job)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	if indexedJobEnabled != (job.Spec.CompletionMode != nil) {
		t.Errorf("Job should allow setting .spec.completionMode only when %v feature is enabled", features.IndexedJob)
	}
	if !suspendJobEnabled && (job.Spec.Suspend != nil) {
		t.Errorf("Job should allow setting .spec.suspend only when %v feature is enabled", features.SuspendJob)
	}
	wantAnnotations := map[string]string{"foo": "bar"}
	if trackingWithFinalizersEnabled {
		wantAnnotations[batchv1.JobTrackingFinalizer] = ""
	}
	if diff := cmp.Diff(wantAnnotations, job.Annotations); diff != "" {
		t.Errorf("Job has annotations (-want,+got):\n%s", diff)
	}

	parallelism := int32(10)

	// ensure we do not change generation for non-spec updates
	updatedLabelJob := job.DeepCopy()
	updatedLabelJob.Labels = map[string]string{"a": "true"}
	Strategy.PrepareForUpdate(ctx, updatedLabelJob, job)
	if updatedLabelJob.Generation != 1 {
		t.Errorf("expected Generation=1, got %d", updatedLabelJob.Generation)
	}
	errs = Strategy.ValidateUpdate(ctx, updatedLabelJob, job)
	if len(errs) != 0 {
		t.Errorf("Unexpected update validation error")
	}

	updatedJob := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "bar",
			ResourceVersion: "4",
			// remove one annotation and try to enforce the job tracking finalizer.
			Annotations: map[string]string{batchv1.JobTrackingFinalizer: ""},
		},
		Spec: batch.JobSpec{
			Parallelism: &parallelism,
			Completions: pointer.Int32Ptr(2),
			// Update gated features.
			TTLSecondsAfterFinished: pointer.Int32Ptr(1),
			CompletionMode:          completionModePtr(batch.IndexedCompletion), // No change because field is immutable.
		},
		Status: batch.JobStatus{
			Active: 11,
		},
	}
	// Ensure we do not change status
	job.Status.Active = 10
	Strategy.PrepareForUpdate(ctx, updatedJob, job)
	if updatedJob.Status.Active != 10 {
		t.Errorf("PrepareForUpdate should have preserved prior version status")
	}
	if updatedJob.Generation != 2 {
		t.Errorf("expected Generation=2, got %d", updatedJob.Generation)
	}
	wantAnnotations = make(map[string]string)
	if trackingWithFinalizersEnabled {
		wantAnnotations[batchv1.JobTrackingFinalizer] = ""
	}
	if diff := cmp.Diff(wantAnnotations, updatedJob.Annotations); diff != "" {
		t.Errorf("Job has annotations (-want,+got):\n%s", diff)
	}

	errs = Strategy.ValidateUpdate(ctx, updatedJob, job)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}

	// Ensure going from legacy tracking Job to tracking with finalizers is
	// disallowed.
	job = job.DeepCopy()
	job.Annotations = nil
	updatedJob = job.DeepCopy()
	updatedJob.Annotations = map[string]string{batch.JobTrackingFinalizer: ""}
	errs = Strategy.ValidateUpdate(ctx, updatedJob, job)
	if len(errs) != 1 {
		t.Errorf("Expected update validation error")
	}

	// Test updating suspend false->true and nil-> true when the feature gate is
	// disabled. We don't care about other combinations.
	job.Spec.Suspend, updatedJob.Spec.Suspend = pointer.BoolPtr(false), pointer.BoolPtr(true)
	Strategy.PrepareForUpdate(ctx, updatedJob, job)
	if !suspendJobEnabled && *updatedJob.Spec.Suspend {
		t.Errorf("[SuspendJob=%v] .spec.suspend should not be updated from false to true", suspendJobEnabled)
	}
	job.Spec.Suspend, updatedJob.Spec.Suspend = nil, pointer.BoolPtr(true)
	Strategy.PrepareForUpdate(ctx, updatedJob, job)
	if !suspendJobEnabled && updatedJob.Spec.Suspend != nil {
		t.Errorf("[SuspendJob=%v] .spec.suspend should not be updated from nil to non-nil", suspendJobEnabled)
	}

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

func TestJobStrategyValidateUpdate(t *testing.T) {
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
		job                                *batch.Job
		update                             func(*batch.Job)
		wantErrs                           field.ErrorList
		trackingWithFinalizersEnabled      bool
		mutableSchedulingDirectivesEnabled bool
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
		"adding tracking annotation disallowed, gate disabled": {
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
				job.Annotations[batch.JobTrackingFinalizer] = ""
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[batch.kubernetes.io/job-tracking]"},
			},
		},
		"adding tracking annotation disallowed, gate enabled": {
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
				job.Annotations[batch.JobTrackingFinalizer] = ""
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[batch.kubernetes.io/job-tracking]"},
			},
			trackingWithFinalizersEnabled: true,
		},
		"preserving tracking annotation, feature disabled": {
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
				// change something.
				job.Annotations["foo"] = "bar"
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
			mutableSchedulingDirectivesEnabled: true,
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
			mutableSchedulingDirectivesEnabled: true,
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
			mutableSchedulingDirectivesEnabled: true,
		},
		"updating node selector whilte gate disabled disallowed": {
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
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.template"},
			},
			mutableSchedulingDirectivesEnabled: false,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.trackingWithFinalizersEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobMutableNodeSchedulingDirectives, tc.mutableSchedulingDirectivesEnabled)()
			newJob := tc.job.DeepCopy()
			tc.update(newJob)
			errs := Strategy.ValidateUpdate(ctx, newJob, tc.job)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJobStrategyWithGeneration(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	theUID := types.UID("1a2b3c4d5e6f7g8h9i0k")

	validPodTemplateSpec := api.PodTemplateSpec{
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myjob2",
			Namespace: metav1.NamespaceDefault,
			UID:       theUID,
		},
		Spec: batch.JobSpec{
			Selector: nil,
			Template: validPodTemplateSpec,
		},
	}

	Strategy.PrepareForCreate(ctx, job)
	errs := Strategy.Validate(ctx, job)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	// Validate the stuff that validation should have validated.
	if job.Spec.Selector == nil {
		t.Errorf("Selector not generated")
	}
	expectedLabels := make(map[string]string)
	expectedLabels["controller-uid"] = string(theUID)
	if !reflect.DeepEqual(job.Spec.Selector.MatchLabels, expectedLabels) {
		t.Errorf("Expected label selector not generated")
	}
	if job.Spec.Template.ObjectMeta.Labels == nil {
		t.Errorf("Expected template labels not generated")
	}
	if v, ok := job.Spec.Template.ObjectMeta.Labels["job-name"]; !ok || v != "myjob2" {
		t.Errorf("Expected template labels not present")
	}
	if v, ok := job.Spec.Template.ObjectMeta.Labels["controller-uid"]; !ok || v != string(theUID) {
		t.Errorf("Expected template labels not present: ok: %v, v: %v", ok, v)
	}
}

func TestJobStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("Job must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Job should not allow create on update")
	}
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
	oldParallelism := int32(10)
	newParallelism := int32(11)
	oldJob := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "myjob",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "10",
		},
		Spec: batch.JobSpec{
			Selector:    validSelector,
			Template:    validPodTemplateSpec,
			Parallelism: &oldParallelism,
		},
		Status: batch.JobStatus{
			Active: 11,
		},
	}
	newJob := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "myjob",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "9",
		},
		Spec: batch.JobSpec{
			Selector:    validSelector,
			Template:    validPodTemplateSpec,
			Parallelism: &newParallelism,
		},
		Status: batch.JobStatus{
			Active: 12,
		},
	}

	StatusStrategy.PrepareForUpdate(ctx, newJob, oldJob)
	if newJob.Status.Active != 12 {
		t.Errorf("Job status updates must allow changes to job status")
	}
	if *newJob.Spec.Parallelism != 10 {
		t.Errorf("Job status updates must now allow changes to job spec")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newJob, oldJob)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if newJob.ResourceVersion != "9" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
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
