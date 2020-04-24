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

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilpointer "k8s.io/utils/pointer"
)

func TestGetJobFromTemplate(t *testing.T) {
	// getJobFromTemplate() needs to take the job template and copy the labels and annotations
	// and other fields, and add a created-by reference.

	var one int64 = 1
	var no bool

	sj := batchv1beta1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: "snazzycats",
			UID:       types.UID("1a2b3c"),
			SelfLink:  "/apis/batch/v1/namespaces/snazzycats/jobs/mycronjob",
		},
		Spec: batchv1beta1.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batchv1beta1.AllowConcurrent,
			JobTemplate: batchv1beta1.JobTemplateSpec{
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
	job, err := getJobFromTemplate(&sj, time.Time{})
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
		// Case 1: There are no jobs and scheduledJobs
		js := []batchv1.Job{}
		jobsBySj := groupJobsByParent(js)
		if len(jobsBySj) != 0 {
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

	sj := batchv1beta1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: batchv1beta1.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: batchv1beta1.AllowConcurrent,
			JobTemplate:       batchv1beta1.JobTemplateSpec{},
		},
	}

	deadline := int64(2 * 60 * 60)

	testCases := []struct {
		caseName          string
		nowTime           time.Time
		creationTimestamp metav1.Time
		lastScheduleTime  *metav1.Time
		deadline          *int64
		expectErr         bool
		expectResult      []time.Time
	}{
		{
			caseName: "Case 1: no known start times, and none needed yet.",
			// Current time is more than creation time, but less than T1.
			nowTime: T1.Add(-7 * time.Minute),
			// Creation time is before T1.
			creationTimestamp: metav1.Time{Time: T1.Add(-10 * time.Minute)},
			lastScheduleTime:  nil,
			deadline:          nil,
			expectErr:         false,
			expectResult:      []time.Time{},
		},
		{
			caseName: "Case 2: no known start times, and one needed",
			// Current time is after T1
			nowTime: T1.Add(2 * time.Second),
			// Creation time is before T1.
			creationTimestamp: metav1.Time{Time: T1.Add(-10 * time.Minute)},
			lastScheduleTime:  nil,
			deadline:          nil,
			expectErr:         false,
			expectResult:      []time.Time{T1},
		},
		{
			caseName: "Case 3: known LastScheduleTime, no start needed.",
			// Current time is after T1
			nowTime: T1.Add(2 * time.Minute),
			// Creation time is before T1.
			creationTimestamp: metav1.Time{Time: T1.Add(-10 * time.Minute)},
			// Status shows a start at the expected time.
			lastScheduleTime: &metav1.Time{Time: T1},
			deadline:         nil,
			expectErr:        false,
			expectResult:     []time.Time{},
		},
		{
			caseName: "Case 4: known LastScheduleTime, a start needed.",
			// Current time is after T1 and after T2
			nowTime: T2.Add(5 * time.Minute),
			// Creation time is before T1.
			creationTimestamp: metav1.Time{Time: T1.Add(-10 * time.Minute)},
			// Status shows a start at the expected time.
			lastScheduleTime: &metav1.Time{Time: T1},
			deadline:         nil,
			expectErr:        false,
			expectResult:     []time.Time{T2},
		},
		{
			caseName: "Case 5: known LastScheduleTime, two starts needed.",
			// Current time is after T1 and after T2
			nowTime:           T2.Add(5 * time.Minute),
			creationTimestamp: metav1.Time{Time: T1.Add(-2 * time.Hour)},
			lastScheduleTime:  &metav1.Time{Time: T1.Add(-1 * time.Hour)},
			deadline:          nil,
			expectErr:         false,
			expectResult:      []time.Time{T1, T2},
		},
		{
			caseName:          "Case 6: now is way way ahead of last start time, and there is no deadline.",
			nowTime:           T2.Add(10 * 24 * time.Hour),
			creationTimestamp: metav1.Time{Time: T1.Add(-2 * time.Hour)},
			lastScheduleTime:  &metav1.Time{Time: T1.Add(-1 * time.Hour)},
			deadline:          nil,
			expectErr:         true,
			expectResult:      []time.Time{},
		},
		{
			caseName:          "Case 7: now is way way ahead of last start time, but there is a short deadline",
			nowTime:           T2.Add(10 * 24 * time.Hour),
			creationTimestamp: metav1.Time{Time: T1.Add(-2 * time.Hour)},
			lastScheduleTime:  &metav1.Time{Time: T1.Add(-1 * time.Hour)},
			// Deadline is short
			deadline:     &deadline,
			expectErr:    false,
			expectResult: []time.Time{T1.Add(240 * time.Hour), T1.Add(241 * time.Hour)},
		},
	}

	for _, testCase := range testCases {
		sj.ObjectMeta.CreationTimestamp = testCase.creationTimestamp
		sj.Status.LastScheduleTime = testCase.lastScheduleTime
		sj.Spec.StartingDeadlineSeconds = testCase.deadline

		times, err := getRecentUnmetScheduleTimes(sj, testCase.nowTime)
		if err != nil && !testCase.expectErr {
			t.Errorf("unexpect getRecentUnmetScheduleTimes err: %v, caseName: %v", err, testCase.caseName)
		}

		if err == nil && testCase.expectErr {
			t.Errorf("expect getRecentUnmetScheduleTimes err, but get err: nil , caseName: %v", testCase.caseName)
		}

		if !equal(times, testCase.expectResult) {
			t.Errorf("getRecentUnmetScheduleTimes expect: %v, get : %v, caseName: %v", testCase.expectResult, times, testCase.caseName)
		}
	}
}

func equal(slice1, slice2 []time.Time) bool {
	if len(slice1) != len(slice2) {
		return false
	}

	for i, item1 := range slice1 {
		if !item1.Equal(slice2[i]) {
			return false
		}
	}
	return true
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
