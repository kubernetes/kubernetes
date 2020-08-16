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
	"strconv"
	"strings"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	batchV1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	// For the cronjob controller to do conversions.
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

var (
	// schedule is hourly on the hour
	onTheHour     = "0 * * * ?"
	errorSchedule = "obvious error schedule"
)

func justBeforeTheHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T09:59:00Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func topOfTheHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T10:00:00Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func justAfterTheHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T10:01:00Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func weekAfterTheHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-26T10:00:00Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func justBeforeThePriorHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T08:59:00Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func justAfterThePriorHour() time.Time {
	T1, err := time.Parse(time.RFC3339, "2016-05-19T09:01:00Z")
	if err != nil {
		panic("test setup error")
	}
	return T1
}

func startTimeStringToTime(startTime string) time.Time {
	T1, err := time.Parse(time.RFC3339, startTime)
	if err != nil {
		panic("test setup error")
	}
	return T1
}

// returns a cronJob with some fields filled in.
func cronJob() batchV1beta1.CronJob {
	return batchV1beta1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "mycronjob",
			Namespace:         "snazzycats",
			UID:               types.UID("1a2b3c"),
			SelfLink:          "/apis/batch/v1beta1/namespaces/snazzycats/cronjobs/mycronjob",
			CreationTimestamp: metav1.Time{Time: justBeforeTheHour()},
		},
		Spec: batchV1beta1.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batchV1beta1.AllowConcurrent,
			JobTemplate: batchV1beta1.JobTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{"a": "b"},
					Annotations: map[string]string{"x": "y"},
				},
				Spec: jobSpec(),
			},
		},
	}
}

func jobSpec() batchv1.JobSpec {
	one := int32(1)
	return batchv1.JobSpec{
		Parallelism: &one,
		Completions: &one,
		Template: v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Image: "foo/bar"},
				},
			},
		},
	}
}

func newJob(UID string) batchv1.Job {
	return batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(UID),
			Name:      "foobar",
			Namespace: metav1.NamespaceDefault,
			SelfLink:  "/apis/batch/v1/namespaces/snazzycats/jobs/myjob",
		},
		Spec: jobSpec(),
	}
}

var (
	shortDead  int64 = 10
	mediumDead int64 = 2 * 60 * 60
	longDead   int64 = 1000000
	noDead     int64 = -12345
	A                = batchV1beta1.AllowConcurrent
	f                = batchV1beta1.ForbidConcurrent
	R                = batchV1beta1.ReplaceConcurrent
	T                = true
	F                = false
)

func TestSyncOne_RunOrNot(t *testing.T) {
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

		// environment
		now time.Time

		// expectations
		expectCreate     bool
		expectDelete     bool
		expectActive     int
		expectedWarnings int
	}{
		"never ran, not valid schedule, A":      {A, F, errorSchedule, noDead, F, F, justBeforeTheHour(), F, F, 0, 1},
		"never ran, not valid schedule, F":      {f, F, errorSchedule, noDead, F, F, justBeforeTheHour(), F, F, 0, 1},
		"never ran, not valid schedule, R":      {f, F, errorSchedule, noDead, F, F, justBeforeTheHour(), F, F, 0, 1},
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, 0, 0},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, 0, 0},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, 0, 0},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterTheHour(), T, F, 1, 0},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterTheHour(), T, F, 1, 0},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterTheHour(), T, F, 1, 0},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterTheHour(), F, F, 0, 0},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterTheHour(), F, F, 0, 0},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterTheHour(), T, F, 1, 0},

		"prev ran but done, not time, A":                {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F, 0, 0},
		"prev ran but done, not time, F":                {f, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F, 0, 0},
		"prev ran but done, not time, R":                {R, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F, 0, 0},
		"prev ran but done, is time, A":                 {A, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, 1, 0},
		"prev ran but done, is time, F":                 {f, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, 1, 0},
		"prev ran but done, is time, R":                 {R, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, 1, 0},
		"prev ran but done, is time, suspended":         {A, T, onTheHour, noDead, T, F, justAfterTheHour(), F, F, 0, 0},
		"prev ran but done, is time, past deadline":     {A, F, onTheHour, shortDead, T, F, justAfterTheHour(), F, F, 0, 0},
		"prev ran but done, is time, not past deadline": {A, F, onTheHour, longDead, T, F, justAfterTheHour(), T, F, 1, 0},

		"still active, not time, A":                {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, 1, 0},
		"still active, not time, F":                {f, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, 1, 0},
		"still active, not time, R":                {R, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, 1, 0},
		"still active, is time, A":                 {A, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, 2, 0},
		"still active, is time, F":                 {f, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, 1, 0},
		"still active, is time, R":                 {R, F, onTheHour, noDead, T, T, justAfterTheHour(), T, T, 1, 0},
		"still active, is time, suspended":         {A, T, onTheHour, noDead, T, T, justAfterTheHour(), F, F, 1, 0},
		"still active, is time, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterTheHour(), F, F, 1, 0},
		"still active, is time, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterTheHour(), T, F, 2, 0},

		// Controller should fail to schedule these, as there are too many missed starting times
		// and either no deadline or a too long deadline.
		"prev ran but done, long overdue, not past deadline, A": {A, F, onTheHour, longDead, T, F, weekAfterTheHour(), F, F, 0, 1},
		"prev ran but done, long overdue, not past deadline, R": {R, F, onTheHour, longDead, T, F, weekAfterTheHour(), F, F, 0, 1},
		"prev ran but done, long overdue, not past deadline, F": {f, F, onTheHour, longDead, T, F, weekAfterTheHour(), F, F, 0, 1},
		"prev ran but done, long overdue, no deadline, A":       {A, F, onTheHour, noDead, T, F, weekAfterTheHour(), F, F, 0, 1},
		"prev ran but done, long overdue, no deadline, R":       {R, F, onTheHour, noDead, T, F, weekAfterTheHour(), F, F, 0, 1},
		"prev ran but done, long overdue, no deadline, F":       {f, F, onTheHour, noDead, T, F, weekAfterTheHour(), F, F, 0, 1},

		"prev ran but done, long overdue, past medium deadline, A": {A, F, onTheHour, mediumDead, T, F, weekAfterTheHour(), T, F, 1, 0},
		"prev ran but done, long overdue, past short deadline, A":  {A, F, onTheHour, shortDead, T, F, weekAfterTheHour(), T, F, 1, 0},

		"prev ran but done, long overdue, past medium deadline, R": {R, F, onTheHour, mediumDead, T, F, weekAfterTheHour(), T, F, 1, 0},
		"prev ran but done, long overdue, past short deadline, R":  {R, F, onTheHour, shortDead, T, F, weekAfterTheHour(), T, F, 1, 0},

		"prev ran but done, long overdue, past medium deadline, F": {f, F, onTheHour, mediumDead, T, F, weekAfterTheHour(), T, F, 1, 0},
		"prev ran but done, long overdue, past short deadline, F":  {f, F, onTheHour, shortDead, T, F, weekAfterTheHour(), T, F, 1, 0},
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
			js := []batchv1.Job{}
			if tc.ranPreviously {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeThePriorHour()}
				cj.Status.LastScheduleTime = &metav1.Time{Time: justAfterThePriorHour()}
				job, err = getJobFromTemplate(&cj, cj.Status.LastScheduleTime.Time)
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

			syncOne(&cj, js, tc.now, jc, cjc, recorder)
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

type CleanupJobSpec struct {
	StartTime           string
	IsFinished          bool
	IsSuccessful        bool
	ExpectDelete        bool
	IsStillInActiveList bool // only when IsFinished is set
}

func TestCleanupFinishedJobs_DeleteOrNot(t *testing.T) {
	limitThree := int32(3)
	limitTwo := int32(2)
	limitOne := int32(1)
	limitZero := int32(0)

	// Starting times are assumed to be sorted by increasing start time
	// in all the test cases
	testCases := map[string]struct {
		jobSpecs                   []CleanupJobSpec
		now                        time.Time
		successfulJobsHistoryLimit *int32
		failedJobsHistoryLimit     *int32
		expectActive               int
	}{
		"success. job limit reached": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, T, F},
				{"2016-05-19T05:00:00Z", T, T, T, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", F, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), &limitTwo, &limitOne, 1},

		"success. jobs not processed by Sync yet": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, T, F},
				{"2016-05-19T05:00:00Z", T, T, T, T},
				{"2016-05-19T06:00:00Z", T, T, F, T},
				{"2016-05-19T07:00:00Z", T, T, F, T},
				{"2016-05-19T08:00:00Z", F, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, T},
			}, justBeforeTheHour(), &limitTwo, &limitOne, 4},

		"failed job limit reached": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, F, T, F},
				{"2016-05-19T05:00:00Z", T, F, T, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", T, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), &limitTwo, &limitTwo, 0},

		"success. job limit set to zero": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, T, F},
				{"2016-05-19T05:00:00Z", T, F, T, F},
				{"2016-05-19T06:00:00Z", T, T, T, F},
				{"2016-05-19T07:00:00Z", T, T, T, F},
				{"2016-05-19T08:00:00Z", F, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), &limitZero, &limitOne, 1},

		"failed job limit set to zero": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, F, F},
				{"2016-05-19T05:00:00Z", T, F, T, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", F, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, T, F},
			}, justBeforeTheHour(), &limitThree, &limitZero, 1},

		"no limits reached": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, F, F},
				{"2016-05-19T05:00:00Z", T, F, F, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", T, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), &limitThree, &limitThree, 0},

		// This test case should trigger the short-circuit
		"limits disabled": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, F, F},
				{"2016-05-19T05:00:00Z", T, F, F, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", T, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), nil, nil, 0},

		"success limit disabled": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, F, F},
				{"2016-05-19T05:00:00Z", T, F, F, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", T, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), nil, &limitThree, 0},

		"failure limit disabled": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", T, T, F, F},
				{"2016-05-19T05:00:00Z", T, F, F, F},
				{"2016-05-19T06:00:00Z", T, T, F, F},
				{"2016-05-19T07:00:00Z", T, T, F, F},
				{"2016-05-19T08:00:00Z", T, F, F, F},
				{"2016-05-19T09:00:00Z", T, F, F, F},
			}, justBeforeTheHour(), &limitThree, nil, 0},

		"no limits reached because still active": {
			[]CleanupJobSpec{
				{"2016-05-19T04:00:00Z", F, F, F, F},
				{"2016-05-19T05:00:00Z", F, F, F, F},
				{"2016-05-19T06:00:00Z", F, F, F, F},
				{"2016-05-19T07:00:00Z", F, F, F, F},
				{"2016-05-19T08:00:00Z", F, F, F, F},
				{"2016-05-19T09:00:00Z", F, F, F, F},
			}, justBeforeTheHour(), &limitZero, &limitZero, 6},
	}

	for name, tc := range testCases {
		name := name
		tc := tc
		t.Run(name, func(t *testing.T) {
			cj := cronJob()
			suspend := false
			cj.Spec.ConcurrencyPolicy = f
			cj.Spec.Suspend = &suspend
			cj.Spec.Schedule = onTheHour

			cj.Spec.SuccessfulJobsHistoryLimit = tc.successfulJobsHistoryLimit
			cj.Spec.FailedJobsHistoryLimit = tc.failedJobsHistoryLimit

			var (
				job *batchv1.Job
				err error
			)

			// Set consistent timestamps for the CronJob
			if len(tc.jobSpecs) != 0 {
				firstTime := startTimeStringToTime(tc.jobSpecs[0].StartTime)
				lastTime := startTimeStringToTime(tc.jobSpecs[len(tc.jobSpecs)-1].StartTime)
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: firstTime}
				cj.Status.LastScheduleTime = &metav1.Time{Time: lastTime}
			} else {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
			}

			// Create jobs
			js := []batchv1.Job{}
			jobsToDelete := sets.NewString()
			cj.Status.Active = []v1.ObjectReference{}

			for i, spec := range tc.jobSpecs {
				job, err = getJobFromTemplate(&cj, startTimeStringToTime(spec.StartTime))
				if err != nil {
					t.Fatalf("%s: unexpected error creating a job from template: %v", name, err)
				}

				job.UID = types.UID(strconv.Itoa(i))
				job.Namespace = ""

				if spec.IsFinished {
					var conditionType batchv1.JobConditionType
					if spec.IsSuccessful {
						conditionType = batchv1.JobComplete
					} else {
						conditionType = batchv1.JobFailed
					}
					condition := batchv1.JobCondition{Type: conditionType, Status: v1.ConditionTrue}
					job.Status.Conditions = append(job.Status.Conditions, condition)

					if spec.IsStillInActiveList {
						cj.Status.Active = append(cj.Status.Active, v1.ObjectReference{UID: job.UID})
					}
				} else {
					if spec.IsSuccessful || spec.IsStillInActiveList {
						t.Errorf("%s: test setup error: this case makes no sense", name)
					}
					cj.Status.Active = append(cj.Status.Active, v1.ObjectReference{UID: job.UID})
				}

				js = append(js, *job)
				if spec.ExpectDelete {
					jobsToDelete.Insert(job.Name)
				}
			}

			jc := &fakeJobControl{Job: job}
			cjc := &fakeCJControl{}
			recorder := record.NewFakeRecorder(10)

			cleanupFinishedJobs(&cj, js, jc, cjc, recorder)

			// Check we have actually deleted the correct jobs
			if len(jc.DeleteJobName) != len(jobsToDelete) {
				t.Errorf("%s: expected %d job deleted, actually %d", name, len(jobsToDelete), len(jc.DeleteJobName))
			} else {
				jcDeleteJobName := sets.NewString(jc.DeleteJobName...)
				if !jcDeleteJobName.Equal(jobsToDelete) {
					t.Errorf("%s: expected jobs: %v deleted, actually: %v deleted", name, jobsToDelete, jcDeleteJobName)
				}
			}

			// Check for events
			expectedEvents := len(jobsToDelete)
			if name == "failed list pod err" {
				expectedEvents = len(tc.jobSpecs)
			}
			if len(recorder.Events) != expectedEvents {
				t.Errorf("%s: expected %d event, actually %v", name, expectedEvents, len(recorder.Events))
			}

			// Check for jobs still in active list
			numActive := 0
			if len(cjc.Updates) != 0 {
				numActive = len(cjc.Updates[len(cjc.Updates)-1].Status.Active)
			}
			if tc.expectActive != numActive {
				t.Errorf("%s: expected Active size %d, got %d", name, tc.expectActive, numActive)
			}
		})
	}
}

// TODO: simulation where the controller randomly doesn't run, and randomly has errors starting jobs or deleting jobs,
// but over time, all jobs run as expected (assuming Allow and no deadline).

// TestSyncOne_Status tests cj.UpdateStatus in syncOne
func TestSyncOne_Status(t *testing.T) {
	finishedJob := newJob("1")
	finishedJob.Status.Conditions = append(finishedJob.Status.Conditions, batchv1.JobCondition{Type: batchv1.JobComplete, Status: v1.ConditionTrue})
	unexpectedJob := newJob("2")
	missingJob := newJob("3")

	testCases := map[string]struct {
		// cj spec
		concurrencyPolicy batchV1beta1.ConcurrencyPolicy
		suspend           bool
		schedule          string
		deadline          int64

		// cj status
		ranPreviously  bool
		hasFinishedJob bool

		// environment
		now              time.Time
		hasUnexpectedJob bool
		hasMissingJob    bool
		beingDeleted     bool

		// expectations
		expectCreate bool
		expectDelete bool
	}{
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, F, F, F},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, F, F, F},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, F, F, F},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, F, T, F},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, F, T, F},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, F, T, F},
		"never ran, is time, deleting":          {A, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, T, F, F},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterTheHour(), F, F, F, F, F},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterTheHour(), F, F, F, F, F},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterTheHour(), F, F, F, T, F},

		"prev ran but done, not time, A":                                            {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F, F, F, F},
		"prev ran but done, not time, finished job, A":                              {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, F, F, F},
		"prev ran but done, not time, unexpected job, A":                            {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), T, F, F, F, F},
		"prev ran but done, not time, missing job, A":                               {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, T, F, F, F},
		"prev ran but done, not time, missing job, unexpected job, A":               {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), T, T, F, F, F},
		"prev ran but done, not time, finished job, unexpected job, A":              {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), T, F, F, F, F},
		"prev ran but done, not time, finished job, missing job, A":                 {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, T, F, F, F},
		"prev ran but done, not time, finished job, missing job, unexpected job, A": {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), T, T, F, F, F},
		"prev ran but done, not time, finished job, F":                              {f, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, F, F, F},
		"prev ran but done, not time, missing job, F":                               {f, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, T, F, F, F},
		"prev ran but done, not time, finished job, missing job, F":                 {f, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, T, F, F, F},
		"prev ran but done, not time, unexpected job, R":                            {R, F, onTheHour, noDead, T, F, justBeforeTheHour(), T, F, F, F, F},

		"prev ran but done, is time, A":                                               {A, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, finished job, A":                                 {A, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, unexpected job, A":                               {A, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, A":                 {A, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, F":                                               {f, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, finished job, F":                                 {f, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, unexpected job, F":                               {f, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, F":                 {f, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, R":                                               {R, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, finished job, R":                                 {R, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, unexpected job, R":                               {R, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, R":                 {R, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, deleting":                                        {A, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, T, F, F},
		"prev ran but done, is time, suspended":                                       {A, T, onTheHour, noDead, T, F, justAfterTheHour(), F, F, F, F, F},
		"prev ran but done, is time, finished job, suspended":                         {A, T, onTheHour, noDead, T, T, justAfterTheHour(), F, F, F, F, F},
		"prev ran but done, is time, unexpected job, suspended":                       {A, T, onTheHour, noDead, T, F, justAfterTheHour(), T, F, F, F, F},
		"prev ran but done, is time, finished job, unexpected job, suspended":         {A, T, onTheHour, noDead, T, T, justAfterTheHour(), T, F, F, F, F},
		"prev ran but done, is time, past deadline":                                   {A, F, onTheHour, shortDead, T, F, justAfterTheHour(), F, F, F, F, F},
		"prev ran but done, is time, finished job, past deadline":                     {A, F, onTheHour, shortDead, T, T, justAfterTheHour(), F, F, F, F, F},
		"prev ran but done, is time, unexpected job, past deadline":                   {A, F, onTheHour, shortDead, T, F, justAfterTheHour(), T, F, F, F, F},
		"prev ran but done, is time, finished job, unexpected job, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterTheHour(), T, F, F, F, F},
		"prev ran but done, is time, not past deadline":                               {A, F, onTheHour, longDead, T, F, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, finished job, not past deadline":                 {A, F, onTheHour, longDead, T, T, justAfterTheHour(), F, F, F, T, F},
		"prev ran but done, is time, unexpected job, not past deadline":               {A, F, onTheHour, longDead, T, F, justAfterTheHour(), T, F, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterTheHour(), T, F, F, T, F},
	}

	for name, tc := range testCases {
		name := name
		tc := tc
		t.Run(name, func(t *testing.T) {
			// Setup the test
			cj := cronJob()
			cj.Spec.ConcurrencyPolicy = tc.concurrencyPolicy
			cj.Spec.Suspend = &tc.suspend
			cj.Spec.Schedule = tc.schedule
			if tc.deadline != noDead {
				cj.Spec.StartingDeadlineSeconds = &tc.deadline
			}
			if tc.ranPreviously {
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeThePriorHour()}
				cj.Status.LastScheduleTime = &metav1.Time{Time: justAfterThePriorHour()}
			} else {
				if tc.hasFinishedJob || tc.hasUnexpectedJob || tc.hasMissingJob {
					t.Errorf("%s: test setup error: this case makes no sense", name)
				}
				cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
			}
			jobs := []batchv1.Job{}
			if tc.hasFinishedJob {
				ref, err := getRef(&finishedJob)
				if err != nil {
					t.Errorf("%s: test setup error: failed to get job's ref: %v.", name, err)
				}
				cj.Status.Active = []v1.ObjectReference{*ref}
				jobs = append(jobs, finishedJob)
			}
			if tc.hasUnexpectedJob {
				jobs = append(jobs, unexpectedJob)
			}
			if tc.hasMissingJob {
				ref, err := getRef(&missingJob)
				if err != nil {
					t.Errorf("%s: test setup error: failed to get job's ref: %v.", name, err)
				}
				cj.Status.Active = append(cj.Status.Active, *ref)
			}
			if tc.beingDeleted {
				timestamp := metav1.NewTime(tc.now)
				cj.DeletionTimestamp = &timestamp
			}

			jc := &fakeJobControl{}
			cjc := &fakeCJControl{}
			recorder := record.NewFakeRecorder(10)

			// Run the code
			syncOne(&cj, jobs, tc.now, jc, cjc, recorder)

			// Status update happens once when ranging through job list, and another one if create jobs.
			expectUpdates := 1
			// Events happens when there's unexpected / finished jobs, and upon job creation / deletion.
			expectedEvents := 0
			if tc.expectCreate {
				expectUpdates++
				expectedEvents++
			}
			if tc.expectDelete {
				expectedEvents++
			}
			if tc.hasFinishedJob {
				expectedEvents++
			}
			if tc.hasUnexpectedJob {
				expectedEvents++
			}
			if tc.hasMissingJob {
				expectedEvents++
			}

			if len(recorder.Events) != expectedEvents {
				t.Errorf("%s: expected %d event, actually %v: %#v", name, expectedEvents, len(recorder.Events), recorder.Events)
			}

			if expectUpdates != len(cjc.Updates) {
				t.Errorf("%s: expected %d status updates, actually %d", name, expectUpdates, len(cjc.Updates))
			}

			if tc.hasFinishedJob && inActiveList(cjc.Updates[0], finishedJob.UID) {
				t.Errorf("%s: expected finished job removed from active list, actually active list = %#v", name, cjc.Updates[0].Status.Active)
			}

			if tc.hasUnexpectedJob && inActiveList(cjc.Updates[0], unexpectedJob.UID) {
				t.Errorf("%s: expected unexpected job not added to active list, actually active list = %#v", name, cjc.Updates[0].Status.Active)
			}

			if tc.hasMissingJob && inActiveList(cjc.Updates[0], missingJob.UID) {
				t.Errorf("%s: expected missing job to be removed from active list, actually active list = %#v", name, cjc.Updates[0].Status.Active)
			}

			if tc.expectCreate && !cjc.Updates[1].Status.LastScheduleTime.Time.Equal(topOfTheHour()) {
				t.Errorf("%s: expected LastScheduleTime updated to %s, got %s", name, topOfTheHour(), cjc.Updates[1].Status.LastScheduleTime)
			}
		})
	}
}
