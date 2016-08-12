/*
Copyright 2016 The Kubernetes Authors.

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

package scheduledjob

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
)

func newBool(a bool) *bool {
	r := new(bool)
	*r = a
	return r
}

func TestScheduledJobStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("ScheduledJob must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ScheduledJob should not allow create on update")
	}

	validPodTemplateSpec := api.PodTemplateSpec{
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	scheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:      "myscheduledjob",
			Namespace: api.NamespaceDefault,
		},
		Spec: batch.ScheduledJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
	}

	Strategy.PrepareForCreate(ctx, scheduledJob)
	if len(scheduledJob.Status.Active) != 0 {
		t.Errorf("ScheduledJob does not allow setting status on create")
	}
	errs := Strategy.Validate(ctx, scheduledJob)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	now := unversioned.Now()
	updatedScheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "4"},
		Spec: batch.ScheduledJobSpec{
			Schedule: "5 5 5 * ?",
		},
		Status: batch.ScheduledJobStatus{
			LastScheduleTime: &now,
		},
	}

	// ensure we do not change status
	Strategy.PrepareForUpdate(ctx, updatedScheduledJob, scheduledJob)
	if updatedScheduledJob.Status.Active != nil {
		t.Errorf("PrepareForUpdate should have preserved prior version status")
	}
	errs = Strategy.ValidateUpdate(ctx, updatedScheduledJob, scheduledJob)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}

func TestScheduledJobStatusStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("ScheduledJob must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("ScheduledJob should not allow create on update")
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	oldSchedule := "* * * * ?"
	oldScheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:            "myscheduledjob",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "10",
		},
		Spec: batch.ScheduledJobSpec{
			Schedule:          oldSchedule,
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
	}
	now := unversioned.Now()
	newScheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:            "myscheduledjob",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "9",
		},
		Spec: batch.ScheduledJobSpec{
			Schedule:          "5 5 * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
		Status: batch.ScheduledJobStatus{
			LastScheduleTime: &now,
		},
	}

	StatusStrategy.PrepareForUpdate(ctx, newScheduledJob, oldScheduledJob)
	if newScheduledJob.Status.LastScheduleTime == nil {
		t.Errorf("ScheduledJob status updates must allow changes to scheduledJob status")
	}
	if newScheduledJob.Spec.Schedule != oldSchedule {
		t.Errorf("ScheduledJob status updates must now allow changes to scheduledJob spec")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newScheduledJob, oldScheduledJob)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if newScheduledJob.ResourceVersion != "9" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

// FIXME: this is failing conversion.go
func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"batch/v2alpha1",
		"ScheduledJob",
		ScheduledJobToSelectableFields(&batch.ScheduledJob{}),
		nil,
	)
}
