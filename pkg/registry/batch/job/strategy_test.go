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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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

func TestJobStrategy(t *testing.T) {
	cases := map[string]struct {
		ttlEnabled        bool
		indexedJobEnabled bool
		suspendJobEnabled bool
	}{
		"features disabled": {},
		"ttl enabled": {
			ttlEnabled: true,
		},
		"indexed job enabled": {
			indexedJobEnabled: true,
		},
		"suspend job enabled": {
			suspendJobEnabled: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TTLAfterFinished, tc.ttlEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IndexedJob, tc.indexedJobEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SuspendJob, tc.suspendJobEnabled)()
			testJobStrategy(t)
		})
	}
}

func testJobStrategy(t *testing.T) {
	ttlEnabled := utilfeature.DefaultFeatureGate.Enabled(features.TTLAfterFinished)
	indexedJobEnabled := utilfeature.DefaultFeatureGate.Enabled(features.IndexedJob)
	suspendJobEnabled := utilfeature.DefaultFeatureGate.Enabled(features.SuspendJob)
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
		},
		Spec: batch.JobSpec{
			Selector:       validSelector,
			Template:       validPodTemplateSpec,
			ManualSelector: pointer.BoolPtr(true),
			Completions:    pointer.Int32Ptr(2),
			// Set gated values.
			Suspend:                 pointer.BoolPtr(true),
			TTLSecondsAfterFinished: pointer.Int32Ptr(0),
			CompletionMode:          batch.IndexedCompletion,
		},
		Status: batch.JobStatus{
			Active: 11,
		},
	}

	Strategy.PrepareForCreate(ctx, job)
	if job.Status.Active != 0 {
		t.Errorf("Job does not allow setting status on create")
	}
	errs := Strategy.Validate(ctx, job)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	if ttlEnabled != (job.Spec.TTLSecondsAfterFinished != nil) {
		t.Errorf("Job should allow setting .spec.ttlSecondsAfterFinished only when %v feature is enabled", features.TTLAfterFinished)
	}
	if indexedJobEnabled != (job.Spec.CompletionMode != batch.NonIndexedCompletion) {
		t.Errorf("Job should allow setting .spec.completionMode=Indexed only when %v feature is enabled", features.IndexedJob)
	}
	if !suspendJobEnabled && *job.Spec.Suspend {
		t.Errorf("[SuspendJob=%v] .spec.suspend should be set to true", suspendJobEnabled)
	}

	parallelism := int32(10)
	updatedJob := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", ResourceVersion: "4"},
		Spec: batch.JobSpec{
			Parallelism: &parallelism,
			Completions: pointer.Int32Ptr(2),
			// Update gated features.
			TTLSecondsAfterFinished: pointer.Int32Ptr(1),
			CompletionMode:          batch.IndexedCompletion, // No change because field is immutable.
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
	if ttlEnabled != (updatedJob.Spec.TTLSecondsAfterFinished != nil) {
		t.Errorf("Job should only allow updating .spec.ttlSecondsAfterFinished when %v feature is enabled", features.TTLAfterFinished)
	}

	errs = Strategy.ValidateUpdate(ctx, updatedJob, job)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}

	// Existing gated fields should be preserved
	job.Spec.TTLSecondsAfterFinished = pointer.Int32Ptr(1)
	job.Spec.CompletionMode = batch.IndexedCompletion
	updatedJob.Spec.TTLSecondsAfterFinished = pointer.Int32Ptr(2)
	updatedJob.Spec.CompletionMode = batch.IndexedCompletion
	// Test updating suspend false->true and nil-> true when the feature gate is
	// disabled. We don't care about other combinations.
	job.Spec.Suspend, updatedJob.Spec.Suspend = pointer.BoolPtr(false), pointer.BoolPtr(true)
	Strategy.PrepareForUpdate(ctx, updatedJob, job)
	if job.Spec.TTLSecondsAfterFinished == nil || updatedJob.Spec.TTLSecondsAfterFinished == nil {
		t.Errorf("existing .spec.ttlSecondsAfterFinished should be preserved")
	}
	if job.Spec.CompletionMode == "" || updatedJob.Spec.CompletionMode == "" {
		t.Errorf("existing completionMode should be preserved")
	}
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
