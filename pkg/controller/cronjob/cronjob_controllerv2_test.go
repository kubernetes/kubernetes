/*
Copyright 2020 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/robfig/cron/v3"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller"
)

func justASecondBeforeTheHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T09:59:59Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func TestControllerV2SyncCronJob(t *testing.T) {
	// Check expectations on deadline parameters
	if shortDead/60/60 >= 1 {
		t.Errorf("shortDead should be less than one hour")
	}

	if mediumDead/60/60 < 1 || mediumDead/60/60 >= 24 {
		t.Errorf("mediumDead should be between one hour and one day")
	}

	if longDead/60/60/24 < 10 {
		t.Errorf("longDead should be at least ten days")
	}

	testCases := map[string]struct {
		// cj spec
		concurrencyPolicy batchv1.ConcurrencyPolicy
		suspend           bool
		schedule          string
		deadline          int64

		// cj status
		ranPreviously bool
		stillActive   bool

		jobCreationTime time.Time

		// environment
		now time.Time

		// expectations
		expectCreate               bool
		expectDelete               bool
		expectActive               int
		expectedWarnings           int
		expectErr                  bool
		expectRequeueAfter         bool
		jobStillNotFoundInLister   bool
		jobPresentInCJActiveStatus bool
		jobCreateError             error
	}{
		"never ran, not valid schedule, A":      {A, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F, F, T, nil},
		"never ran, not valid schedule, F":      {f, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F, F, T, nil},
		"never ran, not valid schedule, R":      {f, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F, F, T, nil},
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterThePriorHour(), *justAfterTheHour(), F, F, 0, 0, F, F, F, T, nil},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterThePriorHour(), justAfterTheHour().Add(time.Minute * time.Duration(shortDead+1)), F, F, 0, 0, F, T, F, T, nil},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},

		"prev ran but done, not time, A":                {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"prev ran but done, not time, F":                {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"prev ran but done, not time, R":                {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"prev ran but done, is time, A":                 {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"prev ran but done, is time, F":                 {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"prev ran but done, is time, R":                 {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"prev ran but done, is time, suspended":         {A, T, onTheHour, noDead, T, F, justAfterThePriorHour(), *justAfterTheHour(), F, F, 0, 0, F, F, F, T, nil},
		"prev ran but done, is time, past deadline":     {A, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), *justAfterTheHour(), F, F, 0, 0, F, T, F, T, nil},
		"prev ran but done, is time, not past deadline": {A, F, onTheHour, longDead, T, F, justAfterThePriorHour(), *justAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},

		"still active, not time, A":                {A, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T, F, T, nil},
		"still active, not time, F":                {f, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T, F, T, nil},
		"still active, not time, R":                {R, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T, F, T, nil},
		"still active, is time, A":                 {A, F, onTheHour, noDead, T, T, justAfterThePriorHour(), *justAfterTheHour(), T, F, 2, 0, F, T, F, T, nil},
		"still active, is time, F":                 {f, F, onTheHour, noDead, T, T, justAfterThePriorHour(), *justAfterTheHour(), F, F, 1, 0, F, T, F, T, nil},
		"still active, is time, R":                 {R, F, onTheHour, noDead, T, T, justAfterThePriorHour(), *justAfterTheHour(), T, T, 1, 0, F, T, F, T, nil},
		"still active, is time, suspended":         {A, T, onTheHour, noDead, T, T, justAfterThePriorHour(), *justAfterTheHour(), F, F, 1, 0, F, F, F, T, nil},
		"still active, is time, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterThePriorHour(), *justAfterTheHour(), F, F, 1, 0, F, T, F, T, nil},
		"still active, is time, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterThePriorHour(), *justAfterTheHour(), T, F, 2, 0, F, T, F, T, nil},

		// Controller should fail to schedule these, as there are too many missed starting times
		// and either no deadline or a too long deadline.
		"prev ran but done, long overdue, not past deadline, A": {A, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T, nil},
		"prev ran but done, long overdue, not past deadline, R": {R, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T, nil},
		"prev ran but done, long overdue, not past deadline, F": {f, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T, nil},
		"prev ran but done, long overdue, no deadline, A":       {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T, nil},
		"prev ran but done, long overdue, no deadline, R":       {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T, nil},
		"prev ran but done, long overdue, no deadline, F":       {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T, nil},

		"prev ran but done, long overdue, past medium deadline, A": {A, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"prev ran but done, long overdue, past short deadline, A":  {A, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},

		"prev ran but done, long overdue, past medium deadline, R": {R, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"prev ran but done, long overdue, past short deadline, R":  {R, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},

		"prev ran but done, long overdue, past medium deadline, F": {f, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},
		"prev ran but done, long overdue, past short deadline, F":  {f, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T, nil},

		// Tests for time skews
		// the controller sees job is created, takes no actions
		"this ran but done, time drifted back, F": {f, F, onTheHour, noDead, T, F, *justAfterTheHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, F, errors.NewAlreadyExists(schema.GroupResource{Resource: "jobs", Group: "batch"}, "")},

		// Tests for slow job lister
		"this started but went missing, not past deadline, A": {A, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, T, T, nil},
		"this started but went missing, not past deadline, f": {f, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, T, T, nil},
		"this started but went missing, not past deadline, R": {R, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, T, T, nil},

		// Tests for slow cronjob list
		"this started but is not present in cronjob active list, not past deadline, A": {A, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, F, F, nil},
		"this started but is not present in cronjob active list, not past deadline, f": {f, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, F, F, nil},
		"this started but is not present in cronjob active list, not past deadline, R": {R, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, F, F, nil},
	}
	for name, tc := range testCases {
		name := name
		tc := tc
		t.Run(name, func(t *testing.T) {
			cj := cronJob()
			cj.Spec.ConcurrencyPolicy = tc.concurrencyPolicy
			cj.Spec.Suspend = &tc.suspend
			cj.Spec.Schedule = tc.schedule
			if tc.deadline != noDead {
				cj.Spec.StartingDeadlineSeconds = &tc.deadline
			}

			var (
				job *batchv1.Job
				err error
			)
			js := []*batchv1.Job{}
			realCJ := cj.DeepCopy()
			if tc.ranPreviously {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeThePriorHour()}
				cj.Status.LastScheduleTime = &metav1.Time{Time: justAfterThePriorHour()}
				job, err = getJobFromTemplate2(&cj, tc.jobCreationTime)
				if err != nil {
					t.Fatalf("%s: unexpected error creating a job from template: %v", name, err)
				}
				job.UID = "1234"
				job.Namespace = cj.Namespace
				if tc.stillActive {
					ref, err := getRef(job)
					if err != nil {
						t.Fatalf("%s: unexpected error getting the job object reference: %v", name, err)
					}
					if tc.jobPresentInCJActiveStatus {
						cj.Status.Active = []v1.ObjectReference{*ref}
					}
					realCJ.Status.Active = []v1.ObjectReference{*ref}
					if !tc.jobStillNotFoundInLister {
						js = append(js, job)
					}
				} else {
					job.Status.CompletionTime = &metav1.Time{Time: job.ObjectMeta.CreationTimestamp.Add(time.Second * 10)}
					job.Status.Conditions = append(job.Status.Conditions, batchv1.JobCondition{
						Type:   batchv1.JobComplete,
						Status: v1.ConditionTrue,
					})
					if !tc.jobStillNotFoundInLister {
						js = append(js, job)
					}
				}
			} else {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
				if tc.stillActive {
					t.Errorf("%s: test setup error: this case makes no sense", name)
				}
			}

			jc := &fakeJobControl{Job: job, CreateErr: tc.jobCreateError}
			cjc := &fakeCJControl{CronJob: realCJ}
			recorder := record.NewFakeRecorder(10)

			jm := ControllerV2{
				jobControl:     jc,
				cronJobControl: cjc,
				recorder:       recorder,
				now: func() time.Time {
					return tc.now
				},
			}
			cjCopy, requeueAfter, err := jm.syncCronJob(&cj, js)
			if tc.expectErr && err == nil {
				t.Errorf("%s: expected error got none with requeueAfter time: %#v", name, requeueAfter)
			}
			if tc.expectRequeueAfter {
				sched, err := cron.ParseStandard(tc.schedule)
				if err != nil {
					t.Errorf("%s: test setup error: the schedule %s is unparseable: %#v", name, tc.schedule, err)
				}
				expectedRequeueAfter := nextScheduledTimeDuration(sched, tc.now)
				if !reflect.DeepEqual(requeueAfter, expectedRequeueAfter) {
					t.Errorf("%s: expected requeueAfter: %+v, got requeueAfter time: %+v", name, expectedRequeueAfter, requeueAfter)
				}
			}
			expectedCreates := 0
			if tc.expectCreate {
				expectedCreates = 1
			}
			if tc.ranPreviously && !tc.stillActive {
				completionTime := tc.jobCreationTime.Add(10 * time.Second)
				if cjCopy.Status.LastSuccessfulTime == nil || !cjCopy.Status.LastSuccessfulTime.Time.Equal(completionTime) {
					t.Errorf("cj.status.lastSuccessfulTime: %s expected, got %#v", completionTime, cj.Status.LastSuccessfulTime)
				}
			}
			if len(jc.Jobs) != expectedCreates {
				t.Errorf("%s: expected %d job started, actually %v", name, expectedCreates, len(jc.Jobs))
			}
			for i := range jc.Jobs {
				job := &jc.Jobs[i]
				controllerRef := metav1.GetControllerOf(job)
				if controllerRef == nil {
					t.Errorf("%s: expected job to have ControllerRef: %#v", name, job)
				} else {
					if got, want := controllerRef.APIVersion, "batch/v1"; got != want {
						t.Errorf("%s: controllerRef.APIVersion = %q, want %q", name, got, want)
					}
					if got, want := controllerRef.Kind, "CronJob"; got != want {
						t.Errorf("%s: controllerRef.Kind = %q, want %q", name, got, want)
					}
					if got, want := controllerRef.Name, cj.Name; got != want {
						t.Errorf("%s: controllerRef.Name = %q, want %q", name, got, want)
					}
					if got, want := controllerRef.UID, cj.UID; got != want {
						t.Errorf("%s: controllerRef.UID = %q, want %q", name, got, want)
					}
					if controllerRef.Controller == nil || *controllerRef.Controller != true {
						t.Errorf("%s: controllerRef.Controller is not set to true", name)
					}
				}
			}

			expectedDeletes := 0
			if tc.expectDelete {
				expectedDeletes = 1
			}
			if len(jc.DeleteJobName) != expectedDeletes {
				t.Errorf("%s: expected %d job deleted, actually %v", name, expectedDeletes, len(jc.DeleteJobName))
			}

			// Status update happens once when ranging through job list, and another one if create jobs.
			expectUpdates := 1
			expectedEvents := 0
			if tc.expectCreate {
				expectedEvents++
				expectUpdates++
			}
			if tc.expectDelete {
				expectedEvents++
			}
			if name == "still active, is time, F" {
				// this is the only test case where we would raise an event for not scheduling
				expectedEvents++
			}
			expectedEvents += tc.expectedWarnings

			if len(recorder.Events) != expectedEvents {
				t.Errorf("%s: expected %d event, actually %v", name, expectedEvents, len(recorder.Events))
			}

			numWarnings := 0
			for i := 1; i <= len(recorder.Events); i++ {
				e := <-recorder.Events
				if strings.HasPrefix(e, v1.EventTypeWarning) {
					numWarnings++
				}
			}
			if numWarnings != tc.expectedWarnings {
				t.Errorf("%s: expected %d warnings, actually %v", name, tc.expectedWarnings, numWarnings)
			}

			if len(cjc.Updates) == expectUpdates && tc.expectActive != len(cjc.Updates[expectUpdates-1].Status.Active) {
				t.Errorf("%s: expected Active size %d, got %d", name, tc.expectActive, len(cjc.Updates[expectUpdates-1].Status.Active))
			}

			if &cj == cjCopy {
				t.Errorf("syncCronJob is not creating a copy of the original cronjob")
			}
		})
	}

}

type fakeQueue struct {
	workqueue.RateLimitingInterface
	delay time.Duration
	key   interface{}
}

func (f *fakeQueue) AddAfter(key interface{}, delay time.Duration) {
	f.delay = delay
	f.key = key
}

// this test will take around 61 seconds to complete
func TestControllerV2UpdateCronJob(t *testing.T) {
	tests := []struct {
		name          string
		oldCronJob    *batchv1.CronJob
		newCronJob    *batchv1.CronJob
		expectedDelay time.Duration
	}{
		{
			name: "spec.template changed",
			oldCronJob: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					JobTemplate: batchv1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels:      map[string]string{"a": "b"},
							Annotations: map[string]string{"x": "y"},
						},
						Spec: jobSpec(),
					},
				},
			},
			newCronJob: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					JobTemplate: batchv1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels:      map[string]string{"a": "foo"},
							Annotations: map[string]string{"x": "y"},
						},
						Spec: jobSpec(),
					},
				},
			},
			expectedDelay: 0 * time.Second,
		},
		{
			name: "spec.schedule changed",
			oldCronJob: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					Schedule: "30 * * * *",
					JobTemplate: batchv1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels:      map[string]string{"a": "b"},
							Annotations: map[string]string{"x": "y"},
						},
						Spec: jobSpec(),
					},
				},
			},
			newCronJob: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					Schedule: "*/1 * * * *",
					JobTemplate: batchv1.JobTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels:      map[string]string{"a": "foo"},
							Annotations: map[string]string{"x": "y"},
						},
						Spec: jobSpec(),
					},
				},
			},
			expectedDelay: 1*time.Second + nextScheduleDelta,
		},
		// TODO: Add more test cases for updating scheduling.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kubeClient := fake.NewSimpleClientset()
			sharedInformers := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
			jm, err := NewControllerV2(sharedInformers.Batch().V1().Jobs(), sharedInformers.Batch().V1().CronJobs(), kubeClient)
			if err != nil {
				t.Errorf("unexpected error %v", err)
				return
			}
			jm.now = justASecondBeforeTheHour
			queue := &fakeQueue{RateLimitingInterface: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "test-update-cronjob")}
			jm.queue = queue
			jm.jobControl = &fakeJobControl{}
			jm.cronJobControl = &fakeCJControl{}
			jm.recorder = record.NewFakeRecorder(10)

			jm.updateCronJob(tt.oldCronJob, tt.newCronJob)
			if queue.delay.Seconds() != tt.expectedDelay.Seconds() {
				t.Errorf("Expected delay %#v got %#v", tt.expectedDelay.Seconds(), queue.delay.Seconds())
			}
		})
	}
}

func TestControllerV2GetJobsToBeReconciled(t *testing.T) {
	trueRef := true
	tests := []struct {
		name     string
		cronJob  *batchv1.CronJob
		jobs     []runtime.Object
		expected []*batchv1.Job
	}{
		{
			name:    "test getting jobs in namespace without controller reference",
			cronJob: &batchv1.CronJob{ObjectMeta: metav1.ObjectMeta{Namespace: "foo-ns", Name: "fooer"}},
			jobs: []runtime.Object{
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns"}},
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "foo-ns"}},
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: "foo-ns"}},
			},
			expected: []*batchv1.Job{},
		},
		{
			name:    "test getting jobs in namespace with a controller reference",
			cronJob: &batchv1.CronJob{ObjectMeta: metav1.ObjectMeta{Namespace: "foo-ns", Name: "fooer"}},
			jobs: []runtime.Object{
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns"}},
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "foo-ns",
					OwnerReferences: []metav1.OwnerReference{{Name: "fooer", Controller: &trueRef}}}},
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: "foo-ns"}},
			},
			expected: []*batchv1.Job{
				{ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "foo-ns",
					OwnerReferences: []metav1.OwnerReference{{Name: "fooer", Controller: &trueRef}}}},
			},
		},
		{
			name:    "test getting jobs in other namespaces",
			cronJob: &batchv1.CronJob{ObjectMeta: metav1.ObjectMeta{Namespace: "foo-ns", Name: "fooer"}},
			jobs: []runtime.Object{
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar-ns"}},
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "bar-ns"}},
				&batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: "bar-ns"}},
			},
			expected: []*batchv1.Job{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kubeClient := fake.NewSimpleClientset()
			sharedInformers := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
			for _, job := range tt.jobs {
				sharedInformers.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			}
			jm, err := NewControllerV2(sharedInformers.Batch().V1().Jobs(), sharedInformers.Batch().V1().CronJobs(), kubeClient)
			if err != nil {
				t.Errorf("unexpected error %v", err)
				return
			}

			actual, err := jm.getJobsToBeReconciled(tt.cronJob)
			if err != nil {
				t.Errorf("unexpected error %v", err)
				return
			}
			if !reflect.DeepEqual(actual, tt.expected) {
				t.Errorf("\nExpected %#v,\nbut got %#v", tt.expected, actual)
			}
		})
	}
}
