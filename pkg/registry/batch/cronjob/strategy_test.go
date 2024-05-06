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

package cronjob

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

var (
	validPodTemplateSpec = api.PodTemplateSpec{
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	validCronjobSpec = batch.CronJobSpec{
		Schedule:          "5 5 * * ?",
		ConcurrencyPolicy: batch.AllowConcurrent,
		TimeZone:          ptr.To("Asia/Shanghai"),
		JobTemplate: batch.JobTemplateSpec{
			Spec: batch.JobSpec{
				Template:       validPodTemplateSpec,
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    ptr.To[int32](10),
				Parallelism:    ptr.To[int32](10),
			},
		},
	}
	cronjobSpecWithTZinSchedule = batch.CronJobSpec{
		Schedule:          "CRON_TZ=UTC 5 5 * * ?",
		ConcurrencyPolicy: batch.AllowConcurrent,
		TimeZone:          ptr.To("Asia/DoesNotExist"),
		JobTemplate: batch.JobTemplateSpec{
			Spec: batch.JobSpec{
				Template: validPodTemplateSpec,
			},
		},
	}
)

func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}

func TestCronJobStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("CronJob must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("CronJob should not allow create on update")
	}

	cronJob := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "mycronjob",
			Namespace:  metav1.NamespaceDefault,
			Generation: 999,
		},
		Spec: batch.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
	}

	Strategy.PrepareForCreate(ctx, cronJob)
	if len(cronJob.Status.Active) != 0 {
		t.Errorf("CronJob does not allow setting status on create")
	}
	if cronJob.Generation != 1 {
		t.Errorf("expected Generation=1, got %d", cronJob.Generation)
	}
	errs := Strategy.Validate(ctx, cronJob)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	now := metav1.Now()

	// ensure we do not change generation for non-spec updates
	updatedLabelCronJob := cronJob.DeepCopy()
	updatedLabelCronJob.Labels = map[string]string{"a": "true"}
	Strategy.PrepareForUpdate(ctx, updatedLabelCronJob, cronJob)
	if updatedLabelCronJob.Generation != 1 {
		t.Errorf("expected Generation=1, got %d", updatedLabelCronJob.Generation)
	}

	updatedCronJob := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", ResourceVersion: "4"},
		Spec: batch.CronJobSpec{
			Schedule: "5 5 5 * ?",
		},
		Status: batch.CronJobStatus{
			LastScheduleTime: &now,
		},
	}

	// ensure we do not change status
	Strategy.PrepareForUpdate(ctx, updatedCronJob, cronJob)
	if updatedCronJob.Status.Active != nil {
		t.Errorf("PrepareForUpdate should have preserved prior version status")
	}
	if updatedCronJob.Generation != 2 {
		t.Errorf("expected Generation=2, got %d", updatedCronJob.Generation)
	}
	errs = Strategy.ValidateUpdate(ctx, updatedCronJob, cronJob)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}

	// Make sure we correctly implement the interface.
	// Otherwise a typo could silently change the default.
	var gcds rest.GarbageCollectionDeleteStrategy = Strategy
	if got, want := gcds.DefaultGarbageCollectionPolicy(genericapirequest.NewContext()), rest.DeleteDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}

	var (
		v1beta1Ctx      = genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1beta1", Resource: "cronjobs"})
		otherVersionCtx = genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v100", Resource: "cronjobs"})
	)
	if got, want := gcds.DefaultGarbageCollectionPolicy(v1beta1Ctx), rest.OrphanDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}
	if got, want := gcds.DefaultGarbageCollectionPolicy(otherVersionCtx), rest.DeleteDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}
}

func TestCronJobStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("CronJob must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("CronJob should not allow create on update")
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	oldSchedule := "* * * * ?"
	oldCronJob := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "mycronjob",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "10",
		},
		Spec: batch.CronJobSpec{
			Schedule:          oldSchedule,
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
	}
	now := metav1.Now()
	newCronJob := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "mycronjob",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "9",
		},
		Spec: batch.CronJobSpec{
			Schedule:          "5 5 * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
		Status: batch.CronJobStatus{
			LastScheduleTime: &now,
		},
	}

	StatusStrategy.PrepareForUpdate(ctx, newCronJob, oldCronJob)
	if newCronJob.Status.LastScheduleTime == nil {
		t.Errorf("CronJob status updates must allow changes to cronJob status")
	}
	if newCronJob.Spec.Schedule != oldSchedule {
		t.Errorf("CronJob status updates must now allow changes to cronJob spec")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newCronJob, oldCronJob)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if newCronJob.ResourceVersion != "9" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestStrategy_ResetFields(t *testing.T) {
	resetFields := Strategy.GetResetFields()
	if len(resetFields) != 2 {
		t.Errorf("ResetFields should have 2 elements, but have %d", len(resetFields))
	}
}

func TestCronJobStatusStrategy_ResetFields(t *testing.T) {
	resetFields := StatusStrategy.GetResetFields()
	if len(resetFields) != 2 {
		t.Errorf("ResetFields should have 2 elements, but have %d", len(resetFields))
	}
}

func TestCronJobStrategy_WarningsOnCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	now := metav1.Now()

	testcases := map[string]struct {
		cronjob           *batch.CronJob
		wantWarningsCount int32
	}{
		"happy path cronjob": {
			wantWarningsCount: 0,
			cronjob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
		},
		"dns invalid name": {
			wantWarningsCount: 1,
			cronjob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "my cronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			gotWarnings := Strategy.WarningsOnCreate(ctx, tc.cronjob)
			if len(gotWarnings) != int(tc.wantWarningsCount) {
				t.Errorf("%s: got warning length of %d but expected %d", name, len(gotWarnings), tc.wantWarningsCount)
			}
		})
	}
}

func TestCronJobStrategy_WarningsOnUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	now := metav1.Now()

	cases := map[string]struct {
		oldCronJob        *batch.CronJob
		cronjob           *batch.CronJob
		wantWarningsCount int32
	}{
		"generation 0 for both": {
			wantWarningsCount: 0,
			oldCronJob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
					Generation:      0,
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
			cronjob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
					Generation:      0,
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
		},
		"generation 1 for new; force WarningsOnUpdate to check PodTemplate for updates": {
			wantWarningsCount: 0,
			oldCronJob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
					Generation:      1,
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
			cronjob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
					Generation:      0,
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
		},
		"force validation failure in pod template": {
			oldCronJob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      1,
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
			cronjob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.CronJobSpec{
					Schedule:          "5 5 * * ?",
					ConcurrencyPolicy: batch.AllowConcurrent,
					JobTemplate: batch.JobTemplateSpec{
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{
								Spec: api.PodSpec{ImagePullSecrets: []api.LocalObjectReference{{Name: ""}}},
							},
						},
					},
				},
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
			wantWarningsCount: 1,
		},
		"timezone invalid failure": {
			oldCronJob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      1,
				},
				Spec: validCronjobSpec,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
			cronjob: &batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "mycronjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: cronjobSpecWithTZinSchedule,
				Status: batch.CronJobStatus{
					LastScheduleTime: &now,
				},
			},
			wantWarningsCount: 1,
		},
	}
	for val, tc := range cases {
		t.Run(val, func(t *testing.T) {
			gotWarnings := Strategy.WarningsOnUpdate(ctx, tc.cronjob, tc.oldCronJob)
			if len(gotWarnings) != int(tc.wantWarningsCount) {
				t.Errorf("%s: got warning length of %d but expected %d", val, len(gotWarnings), tc.wantWarningsCount)
			}
		})
	}
}
