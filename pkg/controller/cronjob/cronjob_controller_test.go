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
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
)

// schedule is hourly on the hour
var (
	onTheHour string = "0 * * * ?"
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

// returns a cronJob with some fields filled in.
func cronJob() batch.CronJob {
	return batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "mycronjob",
			Namespace:         "snazzycats",
			UID:               types.UID("1a2b3c"),
			SelfLink:          "/apis/batch/v2alpha1/namespaces/snazzycats/cronjobs/mycronjob",
			CreationTimestamp: metav1.Time{Time: justBeforeTheHour()},
		},
		Spec: batch.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{"a": "b"},
					Annotations: map[string]string{"x": "y"},
				},
				Spec: jobSpec(),
			},
		},
	}
}

func jobSpec() batch.JobSpec {
	one := int32(1)
	return batch.JobSpec{
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

func newJob(UID string) batch.Job {
	return batch.Job{
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
	shortDead  int64                   = 10
	mediumDead int64                   = 2 * 60 * 60
	longDead   int64                   = 1000000
	noDead     int64                   = -12345
	A          batch.ConcurrencyPolicy = batch.AllowConcurrent
	f          batch.ConcurrencyPolicy = batch.ForbidConcurrent
	R          batch.ConcurrencyPolicy = batch.ReplaceConcurrent
	T          bool                    = true
	F          bool                    = false
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
		// sj spec
		concurrencyPolicy batch.ConcurrencyPolicy
		suspend           bool
		schedule          string
		deadline          int64

		// sj status
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
		sj := cronJob()
		sj.Spec.ConcurrencyPolicy = tc.concurrencyPolicy
		sj.Spec.Suspend = &tc.suspend
		sj.Spec.Schedule = tc.schedule
		if tc.deadline != noDead {
			sj.Spec.StartingDeadlineSeconds = &tc.deadline
		}

		var (
			job *batch.Job
			err error
		)
		js := []batch.Job{}
		if tc.ranPreviously {
			sj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeThePriorHour()}
			sj.Status.LastScheduleTime = &metav1.Time{Time: justAfterThePriorHour()}
			job, err = getJobFromTemplate(&sj, sj.Status.LastScheduleTime.Time)
			if err != nil {
				t.Fatalf("%s: nexpected error creating a job from template: %v", name, err)
			}
			job.UID = "1234"
			job.Namespace = ""
			if tc.stillActive {
				sj.Status.Active = []v1.ObjectReference{{UID: job.UID}}
				js = append(js, *job)
			}
		} else {
			sj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
			if tc.stillActive {
				t.Errorf("%s: test setup error: this case makes no sense", name)
			}
		}

		jc := &fakeJobControl{Job: job}
		sjc := &fakeSJControl{}
		pc := &fakePodControl{}
		recorder := record.NewFakeRecorder(10)

		SyncOne(sj, js, tc.now, jc, sjc, pc, recorder)
		expectedCreates := 0
		if tc.expectCreate {
			expectedCreates = 1
		}
		if len(jc.Jobs) != expectedCreates {
			t.Errorf("%s: expected %d job started, actually %v", name, expectedCreates, len(jc.Jobs))
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
				numWarnings += 1
			}
		}
		if numWarnings != tc.expectedWarnings {
			t.Errorf("%s: expected %d warnings, actually %v", name, tc.expectedWarnings, numWarnings)
		}

		if tc.expectActive != len(sjc.Updates[expectUpdates-1].Status.Active) {
			t.Errorf("%s: expected Active size %d, got %d", name, tc.expectActive, len(sjc.Updates[expectUpdates-1].Status.Active))
		}
	}
}

// TODO: simulation where the controller randomly doesn't run, and randomly has errors starting jobs or deleting jobs,
// but over time, all jobs run as expected (assuming Allow and no deadline).

// TestSyncOne_Status tests sj.UpdateStatus in SyncOne
func TestSyncOne_Status(t *testing.T) {
	finishedJob := newJob("1")
	finishedJob.Status.Conditions = append(finishedJob.Status.Conditions, batch.JobCondition{Type: batch.JobComplete, Status: v1.ConditionTrue})
	unexpectedJob := newJob("2")
	missingJob := newJob("3")

	testCases := map[string]struct {
		// sj spec
		concurrencyPolicy batch.ConcurrencyPolicy
		suspend           bool
		schedule          string
		deadline          int64

		// sj status
		ranPreviously  bool
		hasFinishedJob bool

		// environment
		now              time.Time
		hasUnexpectedJob bool
		hasMissingJob    bool

		// expectations
		expectCreate bool
		expectDelete bool
	}{
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, F, F},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, F, F},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F, F, F},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, T, F},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, T, F},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterTheHour(), F, F, T, F},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterTheHour(), F, F, F, F},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterTheHour(), F, F, F, F},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterTheHour(), F, F, T, F},

		"prev ran but done, not time, A":                                            {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F, F, F},
		"prev ran but done, not time, finished job, A":                              {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, F, F},
		"prev ran but done, not time, unexpected job, A":                            {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), T, F, F, F},
		"prev ran but done, not time, missing job, A":                               {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, T, F, F},
		"prev ran but done, not time, missing job, unexpected job, A":               {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), T, T, F, F},
		"prev ran but done, not time, finished job, unexpected job, A":              {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), T, F, F, F},
		"prev ran but done, not time, finished job, missing job, A":                 {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, T, F, F},
		"prev ran but done, not time, finished job, missing job, unexpected job, A": {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), T, T, F, F},
		"prev ran but done, not time, finished job, F":                              {f, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F, F, F},
		"prev ran but done, not time, missing job, F":                               {f, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, T, F, F},
		"prev ran but done, not time, finished job, missing job, F":                 {f, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, T, F, F},
		"prev ran but done, not time, unexpected job, R":                            {R, F, onTheHour, noDead, T, F, justBeforeTheHour(), T, F, F, F},

		"prev ran but done, is time, A":                                               {A, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, finished job, A":                                 {A, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, unexpected job, A":                               {A, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, A":                 {A, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, F":                                               {f, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, finished job, F":                                 {f, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, unexpected job, F":                               {f, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, F":                 {f, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, R":                                               {R, F, onTheHour, noDead, T, F, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, finished job, R":                                 {R, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, unexpected job, R":                               {R, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, R":                 {R, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, suspended":                                       {A, T, onTheHour, noDead, T, F, justAfterTheHour(), F, F, F, F},
		"prev ran but done, is time, finished job, suspended":                         {A, T, onTheHour, noDead, T, T, justAfterTheHour(), F, F, F, F},
		"prev ran but done, is time, unexpected job, suspended":                       {A, T, onTheHour, noDead, T, F, justAfterTheHour(), T, F, F, F},
		"prev ran but done, is time, finished job, unexpected job, suspended":         {A, T, onTheHour, noDead, T, T, justAfterTheHour(), T, F, F, F},
		"prev ran but done, is time, past deadline":                                   {A, F, onTheHour, shortDead, T, F, justAfterTheHour(), F, F, F, F},
		"prev ran but done, is time, finished job, past deadline":                     {A, F, onTheHour, shortDead, T, T, justAfterTheHour(), F, F, F, F},
		"prev ran but done, is time, unexpected job, past deadline":                   {A, F, onTheHour, shortDead, T, F, justAfterTheHour(), T, F, F, F},
		"prev ran but done, is time, finished job, unexpected job, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterTheHour(), T, F, F, F},
		"prev ran but done, is time, not past deadline":                               {A, F, onTheHour, longDead, T, F, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, finished job, not past deadline":                 {A, F, onTheHour, longDead, T, T, justAfterTheHour(), F, F, T, F},
		"prev ran but done, is time, unexpected job, not past deadline":               {A, F, onTheHour, longDead, T, F, justAfterTheHour(), T, F, T, F},
		"prev ran but done, is time, finished job, unexpected job, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterTheHour(), T, F, T, F},
	}

	for name, tc := range testCases {
		// Setup the test
		sj := cronJob()
		sj.Spec.ConcurrencyPolicy = tc.concurrencyPolicy
		sj.Spec.Suspend = &tc.suspend
		sj.Spec.Schedule = tc.schedule
		if tc.deadline != noDead {
			sj.Spec.StartingDeadlineSeconds = &tc.deadline
		}
		if tc.ranPreviously {
			sj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeThePriorHour()}
			sj.Status.LastScheduleTime = &metav1.Time{Time: justAfterThePriorHour()}
		} else {
			if tc.hasFinishedJob || tc.hasUnexpectedJob || tc.hasMissingJob {
				t.Errorf("%s: test setup error: this case makes no sense", name)
			}
			sj.ObjectMeta.CreationTimestamp = metav1.Time{Time: justBeforeTheHour()}
		}
		jobs := []batch.Job{}
		if tc.hasFinishedJob {
			ref, err := getRef(&finishedJob)
			if err != nil {
				t.Errorf("%s: test setup error: failed to get job's ref: %v.", name, err)
			}
			sj.Status.Active = []v1.ObjectReference{*ref}
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
			sj.Status.Active = append(sj.Status.Active, *ref)
		}

		jc := &fakeJobControl{}
		sjc := &fakeSJControl{}
		pc := &fakePodControl{}
		recorder := record.NewFakeRecorder(10)

		// Run the code
		SyncOne(sj, jobs, tc.now, jc, sjc, pc, recorder)

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

		if expectUpdates != len(sjc.Updates) {
			t.Errorf("%s: expected %d status updates, actually %d", name, expectUpdates, len(sjc.Updates))
		}

		if tc.hasFinishedJob && inActiveList(sjc.Updates[0], finishedJob.UID) {
			t.Errorf("%s: expected finished job removed from active list, actually active list = %#v", name, sjc.Updates[0].Status.Active)
		}

		if tc.hasUnexpectedJob && inActiveList(sjc.Updates[0], unexpectedJob.UID) {
			t.Errorf("%s: expected unexpected job not added to active list, actually active list = %#v", name, sjc.Updates[0].Status.Active)
		}

		if tc.hasMissingJob && inActiveList(sjc.Updates[0], missingJob.UID) {
			t.Errorf("%s: expected missing job to be removed from active list, actually active list = %#v", name, sjc.Updates[0].Status.Active)
		}

		if tc.expectCreate && !sjc.Updates[1].Status.LastScheduleTime.Time.Equal(topOfTheHour()) {
			t.Errorf("%s: expected LastScheduleTime updated to %s, got %s", name, topOfTheHour(), sjc.Updates[1].Status.LastScheduleTime)
		}
	}
}
