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
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func newBool(a bool) *bool {
	return &a
}

func newInt32(i int32) *int32 {
	return &i
}

func TestJobStrategy(t *testing.T) {
	ttlEnabled := utilfeature.DefaultFeatureGate.Enabled(features.TTLAfterFinished)
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
			Selector:                validSelector,
			Template:                validPodTemplateSpec,
			TTLSecondsAfterFinished: newInt32(0), // Set TTL
			ManualSelector:          newBool(true),
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
	if ttlEnabled && job.Spec.TTLSecondsAfterFinished == nil {
		// When the TTL feature is enabled, the TTL field can be set
		t.Errorf("Job should allow setting .spec.ttlSecondsAfterFinished when %v feature is enabled", features.TTLAfterFinished)
	}
	if !ttlEnabled && job.Spec.TTLSecondsAfterFinished != nil {
		// When the TTL feature is disabled, the TTL field cannot be set
		t.Errorf("Job should not allow setting .spec.ttlSecondsAfterFinished when %v feature is disabled", features.TTLAfterFinished)
	}

	parallelism := int32(10)
	updatedJob := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", ResourceVersion: "4"},
		Spec: batch.JobSpec{
			Parallelism:             &parallelism,
			TTLSecondsAfterFinished: newInt32(1), // Update TTL
		},
		Status: batch.JobStatus{
			Active: 11,
		},
	}
	// ensure we do not change status
	job.Status.Active = 10
	Strategy.PrepareForUpdate(ctx, updatedJob, job)
	if updatedJob.Status.Active != 10 {
		t.Errorf("PrepareForUpdate should have preserved prior version status")
	}
	errs = Strategy.ValidateUpdate(ctx, updatedJob, job)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if ttlEnabled != (job.Spec.TTLSecondsAfterFinished != nil || updatedJob.Spec.TTLSecondsAfterFinished != nil) {
		t.Errorf("Job should only allow updating .spec.ttlSecondsAfterFinished when %v feature is enabled", features.TTLAfterFinished)
	}

	// Make sure we correctly implement the interface.
	// Otherwise a typo could silently change the default.
	var gcds rest.GarbageCollectionDeleteStrategy = Strategy
	if got, want := gcds.DefaultGarbageCollectionPolicy(genericapirequest.NewContext()), rest.OrphanDependents; got != want {
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
		testapi.Batch.GroupVersion().String(),
		"Job",
		JobToSelectableFields(&batch.Job{}),
		nil,
	)
}
