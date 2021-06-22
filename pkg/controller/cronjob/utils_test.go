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
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	cron "github.com/robfig/cron/v3"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	utilpointer "k8s.io/utils/pointer"
)

func TestGetJobFromTemplate(t *testing.T) {
	// getJobFromTemplate() needs to take the job template and copy the labels and annotations
	// and other fields, and add a created-by reference.

	var one int64 = 1
	var no bool

	cj := batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: "snazzycats",
			UID:       types.UID("1a2b3c"),
			SelfLink:  "/apis/batch/v1/namespaces/snazzycats/jobs/mycronjob",
		},
		Spec: batchv1.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batchv1.AllowConcurrent,
			JobTemplate: batchv1.JobTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{"a": "b"},
					Annotations: map[string]string{"x": "y"},
				},
				Spec: batchv1.JobSpec{
					ActiveDeadlineSeconds: &one,
					ManualSelector:        &no,
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
				},
			},
		},
	}

	var job *batchv1.Job
	job, err := getJobFromTemplate(&cj, time.Time{})
	if err != nil {
		t.Errorf("Did not expect error: %s", err)
	}
	if !strings.HasPrefix(job.ObjectMeta.Name, "mycronjob-") {
		t.Errorf("Wrong Name")
	}
	if len(job.ObjectMeta.Labels) != 1 {
		t.Errorf("Wrong number of labels")
	}
	if len(job.ObjectMeta.Annotations) != 1 {
		t.Errorf("Wrong number of annotations")
	}
}

func TestGetJobFromTemplate2(t *testing.T) {
	// getJobFromTemplate2() needs to take the job template and copy the labels and annotations
	// and other fields, and add a created-by reference.

	var one int64 = 1
	var no bool

	cj := batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: "snazzycats",
			UID:       types.UID("1a2b3c"),
			SelfLink:  "/apis/batch/v1/namespaces/snazzycats/jobs/mycronjob",
		},
		Spec: batchv1.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batchv1.AllowConcurrent,
			JobTemplate: batchv1.JobTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{"a": "b"},
					Annotations: map[string]string{"x": "y"},
				},
				Spec: batchv1.JobSpec{
					ActiveDeadlineSeconds: &one,
					ManualSelector:        &no,
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
				},
			},
		},
	}

	var job *batchv1.Job
	job, err := getJobFromTemplate2(&cj, time.Time{})
	if err != nil {
		t.Errorf("Did not expect error: %s", err)
	}
	if !strings.HasPrefix(job.ObjectMeta.Name, "mycronjob-") {
		t.Errorf("Wrong Name")
	}
	if len(job.ObjectMeta.Labels) != 1 {
		t.Errorf("Wrong number of labels")
	}
	if len(job.ObjectMeta.Annotations) != 1 {
		t.Errorf("Wrong number of annotations")
	}
}

func TestGetParentUIDFromJob(t *testing.T) {
	j := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foobar",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: batchv1.JobSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
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
		},
		Status: batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{{
				Type:   batchv1.JobComplete,
				Status: v1.ConditionTrue,
			}},
		},
	}
	{
		// Case 1: No ControllerRef
		_, found := getParentUIDFromJob(*j)

		if found {
			t.Errorf("Unexpectedly found uid")
		}
	}
	{
		// Case 2: Has ControllerRef
		j.ObjectMeta.SetOwnerReferences([]metav1.OwnerReference{
			{
				Kind:       "CronJob",
				UID:        types.UID("5ef034e0-1890-11e6-8935-42010af0003e"),
				Controller: utilpointer.BoolPtr(true),
			},
		})

		expectedUID := types.UID("5ef034e0-1890-11e6-8935-42010af0003e")

		uid, found := getParentUIDFromJob(*j)
		if !found {
			t.Errorf("Unexpectedly did not find uid")
		} else if uid != expectedUID {
			t.Errorf("Wrong UID: %v", uid)
		}
	}

}

func TestGroupJobsByParent(t *testing.T) {
	uid1 := types.UID("11111111-1111-1111-1111-111111111111")
	uid2 := types.UID("22222222-2222-2222-2222-222222222222")
	uid3 := types.UID("33333333-3333-3333-3333-333333333333")

	ownerReference1 := metav1.OwnerReference{
		Kind:       "CronJob",
		UID:        uid1,
		Controller: utilpointer.BoolPtr(true),
	}

	ownerReference2 := metav1.OwnerReference{
		Kind:       "CronJob",
		UID:        uid2,
		Controller: utilpointer.BoolPtr(true),
	}

	ownerReference3 := metav1.OwnerReference{
		Kind:       "CronJob",
		UID:        uid3,
		Controller: utilpointer.BoolPtr(true),
	}

	{
		// Case 1: There are no jobs and cronJobs
		js := []batchv1.Job{}
		jobsByCj := groupJobsByParent(js)
		if len(jobsByCj) != 0 {
			t.Errorf("Wrong number of items in map")
		}
	}

	{
		// Case 2: there is one controller with one job it created.
		js := []batchv1.Job{
			{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "x", OwnerReferences: []metav1.OwnerReference{ownerReference1}}},
		}
		jobsBySj := groupJobsByParent(js)

		if len(jobsBySj) != 1 {
			t.Errorf("Wrong number of items in map")
		}
		jobList1, found := jobsBySj[uid1]
		if !found {
			t.Errorf("Key not found")
		}
		if len(jobList1) != 1 {
			t.Errorf("Wrong number of items in map")
		}
	}

	{
		// Case 3: Two namespaces, one has two jobs from one controller, other has 3 jobs from two controllers.
		// There are also two jobs with no created-by annotation.
		js := []batchv1.Job{
			{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "x", OwnerReferences: []metav1.OwnerReference{ownerReference1}}},
			{ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "x", OwnerReferences: []metav1.OwnerReference{ownerReference2}}},
			{ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "x", OwnerReferences: []metav1.OwnerReference{ownerReference1}}},
			{ObjectMeta: metav1.ObjectMeta{Name: "d", Namespace: "x", OwnerReferences: []metav1.OwnerReference{}}},
			{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "y", OwnerReferences: []metav1.OwnerReference{ownerReference3}}},
			{ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "y", OwnerReferences: []metav1.OwnerReference{ownerReference3}}},
			{ObjectMeta: metav1.ObjectMeta{Name: "d", Namespace: "y", OwnerReferences: []metav1.OwnerReference{}}},
		}

		jobsBySj := groupJobsByParent(js)

		if len(jobsBySj) != 3 {
			t.Errorf("Wrong number of items in map")
		}
		jobList1, found := jobsBySj[uid1]
		if !found {
			t.Errorf("Key not found")
		}
		if len(jobList1) != 2 {
			t.Errorf("Wrong number of items in map")
		}
		jobList2, found := jobsBySj[uid2]
		if !found {
			t.Errorf("Key not found")
		}
		if len(jobList2) != 1 {
			t.Errorf("Wrong number of items in map")
		}
		jobList3, found := jobsBySj[uid3]
		if !found {
			t.Errorf("Key not found")
		}
		if len(jobList3) != 2 {
			t.Errorf("Wrong number of items in map")
		}
	}
}

func TestGetNextScheduleTime(t *testing.T) {
	// schedule is hourly on the hour
	schedule := "0 * * * ?"

	PraseSchedule := func(schedule string) cron.Schedule {
		sched, err := cron.ParseStandard(schedule)
		if err != nil {
			t.Errorf("Error parsing schedule: %#v", err)
			return nil
		}
		return sched
	}
	recorder := record.NewFakeRecorder(50)
	// T1 is a scheduled start time of that schedule
	T1, err := time.Parse(time.RFC3339, "2016-05-19T10:00:00Z")
	if err != nil {
		t.Errorf("test setup error: %v", err)
	}
	// T2 is a scheduled start time of that schedule after T1
	T2, err := time.Parse(time.RFC3339, "2016-05-19T11:00:00Z")
	if err != nil {
		t.Errorf("test setup error: %v", err)
	}

	cj := batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: batchv1.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: batchv1.AllowConcurrent,
			JobTemplate:       batchv1.JobTemplateSpec{},
		},
	}
	{
		// Case 1: no known start times, and none needed yet.
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Current time is more than creation time, but less than T1.
		now := T1.Add(-7 * time.Minute)
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule != nil {
			t.Errorf("expected no start time, got:  %v", schedule)
		}
	}
	{
		// Case 2: no known start times, and one needed.
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Current time is after T1
		now := T1.Add(2 * time.Second)
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule == nil {
			t.Errorf("expected 1 start time, got nil")
		} else if !schedule.Equal(T1) {
			t.Errorf("expected: %v, got: %v", T1, schedule)
		}
	}
	{
		// Case 3: known LastScheduleTime, no start needed.
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Status shows a start at the expected time.
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1}
		// Current time is after T1
		now := T1.Add(2 * time.Minute)
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule != nil {
			t.Errorf("expected 0 start times, got: %v", schedule)
		}
	}
	{
		// Case 4: known LastScheduleTime, a start needed
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Status shows a start at the expected time.
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1}
		// Current time is after T1 and after T2
		now := T2.Add(5 * time.Minute)
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule == nil {
			t.Errorf("expected 1 start times, got nil")
		} else if !schedule.Equal(T2) {
			t.Errorf("expected: %v, got: %v", T2, schedule)
		}
	}
	{
		// Case 5: known LastScheduleTime, two starts needed
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-2 * time.Hour)}
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1.Add(-1 * time.Hour)}
		// Current time is after T1 and after T2
		now := T2.Add(5 * time.Minute)
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule == nil {
			t.Errorf("expected 1 start times, got nil")
		} else if !schedule.Equal(T2) {
			t.Errorf("expected: %v, got: %v", T2, schedule)
		}
	}
	{
		// Case 6: now is way way ahead of last start time, and there is no deadline.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-2 * time.Hour)}
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1.Add(-1 * time.Hour)}
		now := T2.Add(10 * 24 * time.Hour)
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule == nil {
			t.Errorf("expected more than 0 missed times")
		}
	}
	{
		// Case 7: now is way way ahead of last start time, but there is a short deadline.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-2 * time.Hour)}
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1.Add(-1 * time.Hour)}
		now := T2.Add(10 * 24 * time.Hour)
		// Deadline is short
		deadline := int64(2 * 60 * 60)
		cj.Spec.StartingDeadlineSeconds = &deadline
		schedule, _ := getNextScheduleTime(cj, now, PraseSchedule(cj.Spec.Schedule), recorder)
		if schedule == nil {
			t.Errorf("expected more than 0 missed times")
		}
	}
}

func TestGetRecentUnmetScheduleTimes(t *testing.T) {
	// schedule is hourly on the hour
	schedule := "0 * * * ?"
	// T1 is a scheduled start time of that schedule
	T1, err := time.Parse(time.RFC3339, "2016-05-19T10:00:00Z")
	if err != nil {
		t.Errorf("test setup error: %v", err)
	}
	// T2 is a scheduled start time of that schedule after T1
	T2, err := time.Parse(time.RFC3339, "2016-05-19T11:00:00Z")
	if err != nil {
		t.Errorf("test setup error: %v", err)
	}

	cj := batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: batchv1.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: batchv1.AllowConcurrent,
			JobTemplate:       batchv1.JobTemplateSpec{},
		},
	}
	{
		// Case 1: no known start times, and none needed yet.
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Current time is more than creation time, but less than T1.
		now := T1.Add(-7 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(cj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 0 {
			t.Errorf("expected no start times, got:  %v", times)
		}
	}
	{
		// Case 2: no known start times, and one needed.
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Current time is after T1
		now := T1.Add(2 * time.Second)
		times, err := getRecentUnmetScheduleTimes(cj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 1 {
			t.Errorf("expected 1 start time, got: %v", times)
		} else if !times[0].Equal(T1) {
			t.Errorf("expected: %v, got: %v", T1, times[0])
		}
	}
	{
		// Case 3: known LastScheduleTime, no start needed.
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Status shows a start at the expected time.
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1}
		// Current time is after T1
		now := T1.Add(2 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(cj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 0 {
			t.Errorf("expected 0 start times, got: %v", times)
		}
	}
	{
		// Case 4: known LastScheduleTime, a start needed
		// Creation time is before T1.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-10 * time.Minute)}
		// Status shows a start at the expected time.
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1}
		// Current time is after T1 and after T2
		now := T2.Add(5 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(cj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 1 {
			t.Errorf("expected 1 start times, got: %v", times)
		} else if !times[0].Equal(T2) {
			t.Errorf("expected: %v, got: %v", T1, times[0])
		}
	}
	{
		// Case 5: known LastScheduleTime, two starts needed
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-2 * time.Hour)}
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1.Add(-1 * time.Hour)}
		// Current time is after T1 and after T2
		now := T2.Add(5 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(cj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 2 {
			t.Errorf("expected 2 start times, got: %v", times)
		} else {
			if !times[0].Equal(T1) {
				t.Errorf("expected: %v, got: %v", T1, times[0])
			}
			if !times[1].Equal(T2) {
				t.Errorf("expected: %v, got: %v", T2, times[1])
			}
		}
	}
	{
		// Case 6: now is way way ahead of last start time, and there is no deadline.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-2 * time.Hour)}
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1.Add(-1 * time.Hour)}
		now := T2.Add(10 * 24 * time.Hour)
		_, err := getRecentUnmetScheduleTimes(cj, now)
		if err == nil {
			t.Errorf("expected an error")
		}
	}
	{
		// Case 7: now is way way ahead of last start time, but there is a short deadline.
		cj.ObjectMeta.CreationTimestamp = metav1.Time{Time: T1.Add(-2 * time.Hour)}
		cj.Status.LastScheduleTime = &metav1.Time{Time: T1.Add(-1 * time.Hour)}
		now := T2.Add(10 * 24 * time.Hour)
		// Deadline is short
		deadline := int64(2 * 60 * 60)
		cj.Spec.StartingDeadlineSeconds = &deadline
		_, err := getRecentUnmetScheduleTimes(cj, now)
		if err != nil {
			t.Errorf("unexpected error")
		}
	}
}

func TestByJobStartTime(t *testing.T) {
	now := metav1.NewTime(time.Date(2018, time.January, 1, 2, 3, 4, 5, time.UTC))
	later := metav1.NewTime(time.Date(2019, time.January, 1, 2, 3, 4, 5, time.UTC))
	aNil := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "a"},
		Status:     batchv1.JobStatus{},
	}
	bNil := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "b"},
		Status:     batchv1.JobStatus{},
	}
	aSet := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "a"},
		Status:     batchv1.JobStatus{StartTime: &now},
	}
	bSet := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "b"},
		Status:     batchv1.JobStatus{StartTime: &now},
	}
	aSetLater := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "a"},
		Status:     batchv1.JobStatus{StartTime: &later},
	}

	testCases := []struct {
		name            string
		input, expected []batchv1.Job
	}{
		{
			name:     "both have nil start times",
			input:    []batchv1.Job{bNil, aNil},
			expected: []batchv1.Job{aNil, bNil},
		},
		{
			name:     "only the first has a nil start time",
			input:    []batchv1.Job{aNil, bSet},
			expected: []batchv1.Job{bSet, aNil},
		},
		{
			name:     "only the second has a nil start time",
			input:    []batchv1.Job{aSet, bNil},
			expected: []batchv1.Job{aSet, bNil},
		},
		{
			name:     "both have non-nil, equal start time",
			input:    []batchv1.Job{bSet, aSet},
			expected: []batchv1.Job{aSet, bSet},
		},
		{
			name:     "both have non-nil, different start time",
			input:    []batchv1.Job{aSetLater, bSet},
			expected: []batchv1.Job{bSet, aSetLater},
		},
	}

	for _, testCase := range testCases {
		sort.Sort(byJobStartTime(testCase.input))
		if !reflect.DeepEqual(testCase.input, testCase.expected) {
			t.Errorf("case: '%s', jobs not sorted as expected", testCase.name)
		}
	}
}

func TestGetMostRecentScheduleTime(t *testing.T) {
	type args struct {
		earliestTime *time.Time
		now          time.Time
		schedule     string
	}
	tests := []struct {
		name                   string
		args                   args
		expectedTime           *time.Time
		expectedNumberOfMisses int64
		wantErr                bool
	}{
		{
			name: "now before next schedule",
			args: args{
				earliestTime: topOfTheHour(),
				now:          topOfTheHour().Add(time.Second * 30),
				schedule:     "0 * * * *",
			},
			expectedTime: nil,
		},
		{
			name: "now just after next schedule",
			args: args{
				earliestTime: topOfTheHour(),
				now:          topOfTheHour().Add(time.Minute * 61),
				schedule:     "0 * * * *",
			},
			expectedTime:           deltaTimeAfterTopOfTheHour(time.Minute * 60),
			expectedNumberOfMisses: 1,
		},
		{
			name: "missed 5 schedules",
			args: args{
				earliestTime: deltaTimeAfterTopOfTheHour(time.Second * 10),
				now:          *deltaTimeAfterTopOfTheHour(time.Minute * 301),
				schedule:     "0 * * * *",
			},
			expectedTime:           deltaTimeAfterTopOfTheHour(time.Minute * 300),
			expectedNumberOfMisses: 5,
		},
		{
			name: "rogue cronjob",
			args: args{
				earliestTime: deltaTimeAfterTopOfTheHour(time.Second * 10),
				now:          *deltaTimeAfterTopOfTheHour(time.Hour * 1000000),
				schedule:     "59 23 31 2 *",
			},
			expectedTime:           nil,
			expectedNumberOfMisses: 0,
			wantErr:                true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sched, err := cron.ParseStandard(tt.args.schedule)
			if err != nil {
				t.Errorf("error setting up the test, %s", err)
			}
			gotTime, gotNumberOfMisses, err := getMostRecentScheduleTime(*tt.args.earliestTime, tt.args.now, sched)
			if tt.wantErr {
				if err == nil {
					t.Error("getMostRecentScheduleTime() got no error when expected one")
				}
				return
			}
			if !tt.wantErr && err != nil {
				t.Error("getMostRecentScheduleTime() got error when none expected")
			}
			if gotTime == nil && tt.expectedTime != nil {
				t.Errorf("getMostRecentScheduleTime() got nil, want %v", tt.expectedTime)
			}
			if gotTime != nil && tt.expectedTime != nil && !gotTime.Equal(*tt.expectedTime) {
				t.Errorf("getMostRecentScheduleTime() got = %v, want %v", gotTime, tt.expectedTime)
			}
			if gotNumberOfMisses != tt.expectedNumberOfMisses {
				t.Errorf("getMostRecentScheduleTime() got1 = %v, want %v", gotNumberOfMisses, tt.expectedNumberOfMisses)
			}
		})
	}
}
