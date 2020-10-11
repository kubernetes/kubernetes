package cronjob

import (
	"fmt"
	"k8s.io/apimachinery/pkg/labels"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/robfig/cron"
	batchv1 "k8s.io/api/batch/v1"
	batchV1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/listers/batch/v1beta1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	// For the cronjob controller to do conversions.
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

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
		concurrencyPolicy batchV1beta1.ConcurrencyPolicy
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
		expectCreate       bool
		expectDelete       bool
		expectActive       int
		expectedWarnings   int
		expectErr          bool
		expectRequeueAfter bool
	}{
		"never ran, not valid schedule, A":      {A, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F},
		"never ran, not valid schedule, F":      {f, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F},
		"never ran, not valid schedule, R":      {f, F, errorSchedule, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 1, F, F},
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, F},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, T},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},

		"prev ran but done, not time, A":                {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
		"prev ran but done, not time, F":                {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
		"prev ran but done, not time, R":                {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
		"prev ran but done, is time, A":                 {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},
		"prev ran but done, is time, F":                 {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},
		"prev ran but done, is time, R":                 {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},
		"prev ran but done, is time, suspended":         {A, T, onTheHour, noDead, T, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, F},
		"prev ran but done, is time, past deadline":     {A, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), justAfterTheHour(), F, F, 0, 0, F, T},
		"prev ran but done, is time, not past deadline": {A, F, onTheHour, longDead, T, F, justAfterThePriorHour(), justAfterTheHour(), T, F, 1, 0, F, T},

		"still active, not time, A":                {A, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T},
		"still active, not time, F":                {f, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T},
		"still active, not time, R":                {R, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justBeforeTheHour(), F, F, 1, 0, F, T},
		"still active, is time, A":                 {A, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), T, F, 2, 0, F, T},
		"still active, is time, F":                 {f, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), F, F, 1, 0, F, T},
		"still active, is time, R":                 {R, F, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), T, T, 1, 0, F, T},
		"still active, is time, suspended":         {A, T, onTheHour, noDead, T, T, justAfterThePriorHour(), justAfterTheHour(), F, F, 1, 0, F, F},
		"still active, is time, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterThePriorHour(), justAfterTheHour(), F, F, 1, 0, F, T},
		"still active, is time, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterThePriorHour(), justAfterTheHour(), T, F, 2, 0, F, T},

		// Controller should fail to schedule these, as there are too many missed starting times
		// and either no deadline or a too long deadline.
		"prev ran but done, long overdue, not past deadline, A": {A, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), F, F, 0, 1, F, T},
		"prev ran but done, long overdue, not past deadline, R": {R, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), F, F, 0, 1, F, T},
		"prev ran but done, long overdue, not past deadline, F": {f, F, onTheHour, longDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), F, F, 0, 1, F, T},
		"prev ran but done, long overdue, no deadline, A":       {A, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), F, F, 0, 1, F, T},
		"prev ran but done, long overdue, no deadline, R":       {R, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), F, F, 0, 1, F, T},
		"prev ran but done, long overdue, no deadline, F":       {f, F, onTheHour, noDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), F, F, 0, 1, F, T},

		"prev ran but done, long overdue, past medium deadline, A": {A, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T},
		"prev ran but done, long overdue, past short deadline, A":  {A, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T},

		"prev ran but done, long overdue, past medium deadline, R": {R, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T},
		"prev ran but done, long overdue, past short deadline, R":  {R, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T},

		"prev ran but done, long overdue, past medium deadline, F": {f, F, onTheHour, mediumDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T},
		"prev ran but done, long overdue, past short deadline, F":  {f, F, onTheHour, shortDead, T, F, justAfterThePriorHour(), weekAfterTheHour(), T, F, 1, 0, F, T},

		// Tests for time skews
		"this ran but done, time drifted back, F": {f, F, onTheHour, noDead, T, F, justAfterTheHour(), justBeforeTheHour(), F, F, 0, 0, F, T},
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
			js := []batchv1.Job{}
			if tc.ranPreviously {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeThePriorHour()}
				cj.Status.LastScheduleTime = &metav1.Time{Time: justAfterThePriorHour()}
				job, err = getJobFromTemplate(&cj, tc.jobCreationTime)
				if err != nil {
					t.Fatalf("%s: unexpected error creating a job from template: %v", name, err)
				}
				job.UID = "1234"
				job.Namespace = ""
				if tc.stillActive {
					cj.Status.Active = []v1.ObjectReference{{UID: job.UID}}
					js = append(js, *job)
				}
			} else {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
				if tc.stillActive {
					t.Errorf("%s: test setup error: this case makes no sense", name)
				}
			}

			jc := &fakeJobControl{Job: job}
			cjc := &fakeCJControl{}
			recorder := record.NewFakeRecorder(10)

			err, requeueAfter := syncOne2(&cj, js, tc.now, jc, cjc, recorder)
			if tc.expectErr && err == nil {
				t.Errorf("%s: expected error got none with requeueAfter time: %#v", name, requeueAfter)
			}
			if tc.expectRequeueAfter {
				sched, err := cron.ParseStandard(tc.schedule)
				if err != nil {
					t.Errorf("%s: test setup error: the schedule %s is unparseable: %#v", name, tc.schedule, err)
				}
				expectedRequeueAfter := nextScheduledTimeDurationWithDelta(sched, tc.now)
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

			if tc.expectActive != len(cjc.Updates[expectUpdates-1].Status.Active) {
				t.Errorf("%s: expected Active size %d, got %d", name, tc.expectActive, len(cjc.Updates[expectUpdates-1].Status.Active))
			}
		})
	}

}

// this test will take around 61 seconds to complete
func TestController2_updateCronJob(t *testing.T) {
	cjc := &fakeCJControl{}
	jc := &fakeJobControl{}
	type fields struct {
		queue          workqueue.DelayingInterface
		recorder       record.EventRecorder
		jobControl     jobControlInterface
		cronJobControl cjControlInterface
	}
	type args struct {
		oldJobTemplate *batchV1beta1.JobTemplateSpec
		newJobTemplate *batchV1beta1.JobTemplateSpec
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
				queue:          workqueue.NewDelayingQueue(),
				recorder:       record.NewFakeRecorder(10),
				jobControl:     jc,
				cronJobControl: cjc,
			},
			args: args{
				oldJobTemplate: &batchV1beta1.JobTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels:      map[string]string{"a": "b"},
						Annotations: map[string]string{"x": "y"},
					},
					Spec: jobSpec(),
				},
				newJobTemplate: &batchV1beta1.JobTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels:      map[string]string{"a": "foo"},
						Annotations: map[string]string{"x": "y"},
					},
					Spec: jobSpec(),
				},
			},
			deltaTimeForQueue:    0 * time.Second,
			roundOffTimeDuration: 500*time.Millisecond + delta100ms,
		},
		{
			name: "spec.schedule changed",
			fields: fields{
				queue:          workqueue.NewDelayingQueue(),
				recorder:       record.NewFakeRecorder(10),
				jobControl:     jc,
				cronJobControl: cjc,
			},
			args: args{
				oldJobSchedule: "30 * * * *",
				newJobSchedule: "1 * * * *",
			},
			deltaTimeForQueue:    61*time.Second + delta100ms,
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

type FakeCronJobLister struct {
	cronjobs   []*batchV1beta1.CronJob
	errCount   int
	listCalled int
}

func (f FakeCronJobLister) List(selector labels.Selector) ([]*batchV1beta1.CronJob, error) {
	if f.listCalled <= f.errCount {
		return nil, fmt.Errorf("fake error")
	}
	ret := []*batchV1beta1.CronJob{}
	for _, cj := range f.cronjobs {
		if selector.Matches(labels.Set(cj.GetLabels())) {
			ret = append(ret, cj)
		}
	}
	return ret, nil
}

func (f FakeCronJobLister) CronJobs(namespace string) v1beta1.CronJobNamespaceLister {
	panic("implement me")
}

func TestControllerV2_enqueueAllExistingCronJobsWithRetries(t *testing.T) {
	type fields struct {
		queue         workqueue.DelayingInterface
		cronJobLister FakeCronJobLister
	}
	type args struct {
		retries int
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		error  bool
	}{
		{
			name: "test without errors",
			fields: fields{cronJobLister: FakeCronJobLister{
				cronjobs: []*batchV1beta1.CronJob{
					&batchV1beta1.CronJob{
						ObjectMeta: metav1.ObjectMeta{
							Namespace: "bar",
							Name:      "foo",
						},
					},
				},
				errCount:   -1,
				listCalled: 0,
			},
				queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "test-enqueue-all"),
			},
			args:  args{retries: 5},
			error: false,
		},
		{
			name: "test without errors",
			fields: fields{cronJobLister: FakeCronJobLister{
				cronjobs: []*batchV1beta1.CronJob{
					&batchV1beta1.CronJob{
						ObjectMeta: metav1.ObjectMeta{
							Namespace: "bar",
							Name:      "foo",
						},
					},
				},
				errCount:   6,
				listCalled: 0,
			},
				queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "test-enqueue-all"),
			},
			args:  args{retries: 5},
			error: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jm := &ControllerV2{
				queue:         tt.fields.queue,
				cronJobLister: tt.fields.cronJobLister,
			}
			jm.enqueueAllCronJobsWithRetries(tt.args.retries)
			if !tt.error && jm.queue.Len() != len(tt.fields.cronJobLister.cronjobs) {
				t.Errorf("expected %d, got %d\n", len(tt.fields.cronJobLister.cronjobs), jm.queue.Len())
			}
			if tt.error && jm.queue.Len() != 0 {
				t.Errorf("expected 0, got %d", jm.queue.Len())
			}
		})
	}
}
