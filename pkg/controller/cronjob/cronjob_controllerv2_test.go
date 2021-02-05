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
	"fmt"
	"github.com/robfig/cron"
	"k8s.io/apimachinery/pkg/labels"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	batchv1listers "k8s.io/client-go/listers/batch/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func justASecondBeforeTheHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T09:59:59Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func Test_syncOne2(t *testing.T) {
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
		concurrencyPolicy batchv1beta1.ConcurrencyPolicy
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
	}{
		"never ran, not valid schedule, A":      {A, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F, F, T},
		"never ran, not valid schedule, F":      {f, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F, F, T},
		"never ran, not valid schedule, R":      {f, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F, F, T},
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, F, F, T},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, T, F, T},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},

		"prev ran but done, not time, A":                {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},
		"prev ran but done, not time, F":                {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},
		"prev ran but done, not time, R":                {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},
		"prev ran but done, is time, A":                 {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"prev ran but done, is time, F":                 {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"prev ran but done, is time, R":                 {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"prev ran but done, is time, suspended":         {A, T, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, F, F, T},
		"prev ran but done, is time, past deadline":     {A, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, T, F, T},
		"prev ran but done, is time, not past deadline": {A, F, onTheHour, longDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T, F, T},

		"still active, not time, A":                {A, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T, F, T},
		"still active, not time, F":                {f, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T, F, T},
		"still active, not time, R":                {R, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T, F, T},
		"still active, is time, A":                 {A, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), T, F, 2, 0, F, T, F, T},
		"still active, is time, F":                 {f, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), F, F, 1, 0, F, T, F, T},
		"still active, is time, R":                 {R, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), T, T, 1, 0, F, T, F, T},
		"still active, is time, suspended":         {A, T, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), F, F, 1, 0, F, F, F, T},
		"still active, is time, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterThePriorHour(), justAfterTheHour(), F, F, 1, 0, F, T, F, T},
		"still active, is time, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterThePriorHour(), justAfterTheHour(), T, F, 2, 0, F, T, F, T},

		// Controller should fail to schedule these, as there are too many missed starting times
		// and either no deadline or a too long deadline.
		"prev ran but done, long overdue, not past deadline, A": {A, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T},
		"prev ran but done, long overdue, not past deadline, R": {R, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T},
		"prev ran but done, long overdue, not past deadline, F": {f, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T},
		"prev ran but done, long overdue, no deadline, A":       {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T},
		"prev ran but done, long overdue, no deadline, R":       {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T},
		"prev ran but done, long overdue, no deadline, F":       {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 1, F, T, F, T},

		"prev ran but done, long overdue, past medium deadline, A": {A, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"prev ran but done, long overdue, past short deadline, A":  {A, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T},

		"prev ran but done, long overdue, past medium deadline, R": {R, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"prev ran but done, long overdue, past short deadline, R":  {R, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T},

		"prev ran but done, long overdue, past medium deadline, F": {f, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T},
		"prev ran but done, long overdue, past short deadline, F":  {f, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T, F, T},

		// Tests for time skews
		"this ran but done, time drifted back, F": {f, F, onTheHour, noDead, T, F, justAfterTheHour(), justBeforeTheHour(), F, F, 0, 0, F, T, F, T},

		// Tests for slow job lister
		"this started but went missing, not past deadline, A": {A, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, T, T},
		"this started but went missing, not past deadline, f": {f, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, T, T},
		"this started but went missing, not past deadline, R": {R, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, T, T},

		// Tests for slow cronjob list
		"this started but is not present in cronjob active list, not past deadline, A": {A, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, F, F},
		"this started but is not present in cronjob active list, not past deadline, f": {f, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, F, F},
		"this started but is not present in cronjob active list, not past deadline, R": {R, F, onTheHour, longDead, T, T, topOfTheHour().Add(time.Millisecond * 100), justAfterTheHour().Add(time.Millisecond * 100), F, F, 1, 0, F, T, F, F},
	}
	for name, tc := range testCases {
		name := name
		tc := tc
		t.Run(name, func(t *testing.T) {
			if name == "this ran but done, time drifted back, F" {
				println("hello")
			}
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
				}
			} else {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
				if tc.stillActive {
					t.Errorf("%s: test setup error: this case makes no sense", name)
				}
			}

			jc := &fakeJobControl{Job: job}
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
			if len(jc.Jobs) != expectedCreates {
				t.Errorf("%s: expected %d job started, actually %v", name, expectedCreates, len(jc.Jobs))
			}
			for i := range jc.Jobs {
				job := &jc.Jobs[i]
				controllerRef := metav1.GetControllerOf(job)
				if controllerRef == nil {
					t.Errorf("%s: expected job to have ControllerRef: %#v", name, job)
				} else {
					if got, want := controllerRef.APIVersion, "batch/v1beta1"; got != want {
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

// this test will take around 61 seconds to complete
func TestController2_updateCronJob(t *testing.T) {
	cjc := &fakeCJControl{}
	jc := &fakeJobControl{}
	type fields struct {
		queue          workqueue.RateLimitingInterface
		recorder       record.EventRecorder
		jobControl     jobControlInterface
		cronJobControl cjControlInterface
	}
	type args struct {
		oldJobTemplate *batchv1beta1.JobTemplateSpec
		newJobTemplate *batchv1beta1.JobTemplateSpec
		oldJobSchedule string
		newJobSchedule string
	}
	tests := []struct {
		name                 string
		fields               fields
		args                 args
		deltaTimeForQueue    time.Duration
		roundOffTimeDuration time.Duration
	}{
		{
			name: "spec.template changed",
			fields: fields{
				queue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "test-update-cronjob"),
				recorder:       record.NewFakeRecorder(10),
				jobControl:     jc,
				cronJobControl: cjc,
			},
			args: args{
				oldJobTemplate: &batchv1beta1.JobTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels:      map[string]string{"a": "b"},
						Annotations: map[string]string{"x": "y"},
					},
					Spec: jobSpec(),
				},
				newJobTemplate: &batchv1beta1.JobTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels:      map[string]string{"a": "foo"},
						Annotations: map[string]string{"x": "y"},
					},
					Spec: jobSpec(),
				},
			},
			deltaTimeForQueue:    0 * time.Second,
			roundOffTimeDuration: 500*time.Millisecond + nextScheduleDelta,
		},
		{
			name: "spec.schedule changed",
			fields: fields{
				queue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "test-update-cronjob"),
				recorder:       record.NewFakeRecorder(10),
				jobControl:     jc,
				cronJobControl: cjc,
			},
			args: args{
				oldJobSchedule: "30 * * * *",
				newJobSchedule: "*/1 * * * *",
			},
			deltaTimeForQueue:    1*time.Second + nextScheduleDelta,
			roundOffTimeDuration: 750 * time.Millisecond,
		},
		// TODO: Add more test cases for updating scheduling.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cj := cronJob()
			newCj := cronJob()
			if tt.args.oldJobTemplate != nil {
				cj.Spec.JobTemplate = *tt.args.oldJobTemplate
			}
			if tt.args.newJobTemplate != nil {
				newCj.Spec.JobTemplate = *tt.args.newJobTemplate
			}
			if tt.args.oldJobSchedule != "" {
				cj.Spec.Schedule = tt.args.oldJobSchedule
			}
			if tt.args.newJobSchedule != "" {
				newCj.Spec.Schedule = tt.args.newJobSchedule
			}
			jm := &ControllerV2{
				queue:          tt.fields.queue,
				recorder:       tt.fields.recorder,
				jobControl:     tt.fields.jobControl,
				cronJobControl: tt.fields.cronJobControl,
			}
			jm.now = justASecondBeforeTheHour
			now := time.Now()
			then := time.Now()
			wg := sync.WaitGroup{}
			wg.Add(1)
			go func() {
				now = time.Now()
				jm.queue.Get()
				then = time.Now()
				wg.Done()
				return
			}()
			jm.updateCronJob(&cj, &newCj)
			wg.Wait()
			d := then.Sub(now)
			if d.Round(tt.roundOffTimeDuration).Seconds() != tt.deltaTimeForQueue.Round(tt.roundOffTimeDuration).Seconds() {
				t.Errorf("Expected %#v got %#v", tt.deltaTimeForQueue.Round(tt.roundOffTimeDuration).String(), d.Round(tt.roundOffTimeDuration).String())
			}
		})
	}
}

type FakeNamespacedJobLister struct {
	jobs      []*batchv1.Job
	namespace string
}

func (f *FakeNamespacedJobLister) Get(name string) (*batchv1.Job, error) {
	for _, j := range f.jobs {
		if j.Namespace == f.namespace && j.Namespace == name {
			return j, nil
		}
	}
	return nil, fmt.Errorf("Not Found")
}

func (f *FakeNamespacedJobLister) List(selector labels.Selector) ([]*batchv1.Job, error) {
	ret := []*batchv1.Job{}
	for _, j := range f.jobs {
		if f.namespace != "" && f.namespace != j.Namespace {
			continue
		}
		if selector.Matches(labels.Set(j.GetLabels())) {
			ret = append(ret, j)
		}
	}
	return ret, nil
}

func (f *FakeNamespacedJobLister) Jobs(namespace string) batchv1listers.JobNamespaceLister {
	f.namespace = namespace
	return f
}

func (f *FakeNamespacedJobLister) GetPodJobs(pod *v1.Pod) (jobs []batchv1.Job, err error) {
	panic("implement me")
}

func TestControllerV2_getJobList(t *testing.T) {
	trueRef := true
	type fields struct {
		jobLister batchv1listers.JobLister
	}
	type args struct {
		cronJob *batchv1beta1.CronJob
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    []*batchv1.Job
		wantErr bool
	}{
		{
			name: "test getting jobs in namespace without controller reference",
			fields: fields{
				&FakeNamespacedJobLister{jobs: []*batchv1.Job{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "foo-ns"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: "foo-ns"},
					},
				}}},
			args: args{cronJob: &batchv1beta1.CronJob{ObjectMeta: metav1.ObjectMeta{Namespace: "foo-ns", Name: "fooer"}}},
			want: []*batchv1.Job{},
		},
		{
			name: "test getting jobs in namespace with a controller reference",
			fields: fields{
				&FakeNamespacedJobLister{jobs: []*batchv1.Job{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "foo-ns",
							OwnerReferences: []metav1.OwnerReference{
								{
									Name:       "fooer",
									Controller: &trueRef,
								},
							}},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: "foo-ns"},
					},
				}}},
			args: args{cronJob: &batchv1beta1.CronJob{ObjectMeta: metav1.ObjectMeta{Namespace: "foo-ns", Name: "fooer"}}},
			want: []*batchv1.Job{{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "foo-ns",
					OwnerReferences: []metav1.OwnerReference{
						{
							Name:       "fooer",
							Controller: &trueRef,
						},
					}},
			}},
		},
		{
			name: "test getting jobs in other namespaces ",
			fields: fields{
				&FakeNamespacedJobLister{jobs: []*batchv1.Job{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar-ns"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: "bar-ns"},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "foo2", Namespace: "bar-ns"},
					},
				}}},
			args: args{cronJob: &batchv1beta1.CronJob{ObjectMeta: metav1.ObjectMeta{Namespace: "foo-ns", Name: "fooer"}}},
			want: []*batchv1.Job{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jm := &ControllerV2{
				jobLister: tt.fields.jobLister,
			}
			got, err := jm.getJobsToBeReconciled(tt.args.cronJob)
			if (err != nil) != tt.wantErr {
				t.Errorf("getJobsToBeReconciled() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getJobsToBeReconciled() got = %v, want %v", got, tt.want)
			}
		})
	}
}
