/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/types"
	"testing"
	"time"
)

// schedule is hourly on the hour
var (
	onTheHour string = "0 0 * * * ?"
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

// returns a scheduledJob with some fields filled in.
func scheduledJob() batch.ScheduledJob {
	return batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:              "myscheduledjob",
			Namespace:         "snazzycats",
			UID:               types.UID("1a2b3c"),
			SelfLink:          "/apis/extensions/v1beta1/namespaces/snazzycats/jobs/myscheduledjob",
			CreationTimestamp: unversioned.Time{justBeforeTheHour()},
		},
		Spec: batch.ScheduledJobSpec{
			Schedule:          "0 0 * * * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels:      map[string]string{"a": "b"},
					Annotations: map[string]string{"x": "y"},
				},
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{Image: "foo/bar"},
							},
						},
					},
				},
			},
		},
	}
}

var (
	shortDead int64                   = 10
	longDead  int64                   = 1000000
	noDead    int64                   = -12345
	A         batch.ConcurrencyPolicy = batch.AllowConcurrent
	f         batch.ConcurrencyPolicy = batch.ForbidConcurrent
	R         batch.ConcurrencyPolicy = batch.ReplaceConcurrent
	T         bool                    = true
	F         bool                    = false
)

func TestSyncOne_RunOrNot(t *testing.T) {

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
		expectCreate bool
		expectDelete bool
	}{
		"never ran, not time, A":                {A, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F},
		"never ran, not time, F":                {f, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F},
		"never ran, not time, R":                {R, F, onTheHour, noDead, F, F, justBeforeTheHour(), F, F},
		"never ran, is time, A":                 {A, F, onTheHour, noDead, F, F, justAfterTheHour(), T, F},
		"never ran, is time, F":                 {f, F, onTheHour, noDead, F, F, justAfterTheHour(), T, F},
		"never ran, is time, R":                 {R, F, onTheHour, noDead, F, F, justAfterTheHour(), T, F},
		"never ran, is time, suspended":         {A, T, onTheHour, noDead, F, F, justAfterTheHour(), F, F},
		"never ran, is time, past deadline":     {A, F, onTheHour, shortDead, F, F, justAfterTheHour(), F, F},
		"never ran, is time, not past deadline": {A, F, onTheHour, longDead, F, F, justAfterTheHour(), T, F},

		"prev ran but done, not time, A":                {A, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F},
		"prev ran but done, not time, F":                {f, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F},
		"prev ran but done, not time, R":                {R, F, onTheHour, noDead, T, F, justBeforeTheHour(), F, F},
		"prev ran but done, is time, A":                 {A, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F},
		"prev ran but done, is time, F":                 {f, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F},
		"prev ran but done, is time, R":                 {R, F, onTheHour, noDead, T, F, justAfterTheHour(), T, F},
		"prev ran but done, is time, suspended":         {A, T, onTheHour, noDead, T, F, justAfterTheHour(), F, F},
		"prev ran but done, is time, past deadline":     {A, F, onTheHour, shortDead, T, F, justAfterTheHour(), F, F},
		"prev ran but done, is time, not past deadline": {A, F, onTheHour, longDead, T, F, justAfterTheHour(), T, F},

		"still active, not time, A":                {A, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F},
		"still active, not time, F":                {f, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F},
		"still active, not time, R":                {R, F, onTheHour, noDead, T, T, justBeforeTheHour(), F, F},
		"still active, is time, A":                 {A, F, onTheHour, noDead, T, T, justAfterTheHour(), T, F},
		"still active, is time, F":                 {f, F, onTheHour, noDead, T, T, justAfterTheHour(), F, F},
		"still active, is time, R":                 {R, F, onTheHour, noDead, T, T, justAfterTheHour(), T, T},
		"still active, is time, suspended":         {A, T, onTheHour, noDead, T, T, justAfterTheHour(), F, F},
		"still active, is time, past deadline":     {A, F, onTheHour, shortDead, T, T, justAfterTheHour(), F, F},
		"still active, is time, not past deadline": {A, F, onTheHour, longDead, T, T, justAfterTheHour(), T, F},
	}
	for name, tc := range testCases {
		t.Log("Test case:", name)
		sj := scheduledJob()
		sj.Spec.ConcurrencyPolicy = tc.concurrencyPolicy
		sj.Spec.Suspend = tc.suspend
		sj.Spec.Schedule = tc.schedule
		if tc.deadline != noDead {
			sj.Spec.StartingDeadlineSeconds = &tc.deadline
		}

		if tc.ranPreviously {
			sj.ObjectMeta.CreationTimestamp = unversioned.Time{justBeforeThePriorHour()}
			sj.Status.LastScheduleTime = &unversioned.Time{justAfterThePriorHour()}
			if tc.stillActive {
				sj.Status.Active = []api.ObjectReference{{}}
			}
		} else {
			sj.ObjectMeta.CreationTimestamp = unversioned.Time{justBeforeTheHour()}
			if tc.stillActive {
				t.Errorf("Test setup error: this case makes no sense.")
			}
		}

		jc := &fakeJobControl{}
		sjc := &fakeSJControl{}
		recorder := record.NewFakeRecorder(10)

		SyncOne(sj, []batch.Job{}, tc.now, jc, sjc, recorder)
		expectedCreates := 0
		if tc.expectCreate {
			expectedCreates = 1
		}
		if len(jc.Jobs) != expectedCreates {
			t.Errorf("Expected %d job started, actually %v", expectedCreates, len(jc.Jobs))
		}

		expectedDeletes := 0
		if tc.expectDelete {
			expectedDeletes = 1
		}
		if len(jc.DeleteJobName) != expectedDeletes {
			t.Errorf("Expected %d job deleted, actually %v", expectedDeletes, len(jc.DeleteJobName))
		}

		expectedEvents := 0
		if tc.expectCreate {
			expectedEvents += 1
		}
		if tc.expectDelete {
			expectedEvents += 1
		}
		if len(recorder.Events) != expectedEvents {
			t.Errorf("Expected %d event, actually %v", expectedEvents, len(recorder.Events))
		}
	}
}

// TODO: simulation where the controller randomly doesn't run, and randomly has errors starting jobs or deleting jobs,
// but over time, all jobs run as expected (assuming Allow and no deadline).
