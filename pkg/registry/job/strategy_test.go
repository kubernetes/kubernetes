/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"
)

func TestJobStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("Job must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Job should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: validSelector,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	job := &experimental.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "myjob",
			Namespace: api.NamespaceDefault,
		},
		Spec: experimental.JobSpec{
			Selector: validSelector,
			Template: &validPodTemplateSpec,
		},
		Status: experimental.JobStatus{
			Active: 11,
		},
	}

	Strategy.PrepareForCreate(job)
	if job.Status.Active != 0 {
		t.Errorf("Job does not allow setting status on create")
	}
	errs := Strategy.Validate(ctx, job)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	parallelism := 10
	updatedJob := &experimental.Job{
		ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "4"},
		Spec: experimental.JobSpec{
			Parallelism: &parallelism,
		},
		Status: experimental.JobStatus{
			Active: 11,
		},
	}
	// ensure we do not change status
	job.Status.Active = 10
	Strategy.PrepareForUpdate(updatedJob, job)
	if updatedJob.Status.Active != 10 {
		t.Errorf("PrepareForUpdate should have preserved prior version status")
	}
	errs = Strategy.ValidateUpdate(ctx, updatedJob, job)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}

func TestJobStatusStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("Job must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Job should not allow create on update")
	}
	validSelector := map[string]string{"a": "b"}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: validSelector,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	oldParallelism := 10
	newParallelism := 11
	oldJob := &experimental.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "myjob",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "10",
		},
		Spec: experimental.JobSpec{
			Selector:    validSelector,
			Template:    &validPodTemplateSpec,
			Parallelism: &oldParallelism,
		},
		Status: experimental.JobStatus{
			Active: 11,
		},
	}
	newJob := &experimental.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "myjob",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "9",
		},
		Spec: experimental.JobSpec{
			Selector:    validSelector,
			Template:    &validPodTemplateSpec,
			Parallelism: &newParallelism,
		},
		Status: experimental.JobStatus{
			Active: 12,
		},
	}

	StatusStrategy.PrepareForUpdate(newJob, oldJob)
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
