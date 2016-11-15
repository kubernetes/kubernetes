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
	//"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/types"
	//"k8s.io/kubernetes/pkg/controller"
	// "k8s.io/kubernetes/pkg/util/rand"
)

func TestGetJobFromTemplate(t *testing.T) {
	// getJobFromTemplate() needs to take the job template and copy the labels and annotations
	// and other fields, and add a created-by reference.

	var one int64 = 1
	var no bool = false

	sj := batch.CronJob{
		ObjectMeta: api.ObjectMeta{
			Name:      "mycronjob",
			Namespace: "snazzycats",
			UID:       types.UID("1a2b3c"),
			SelfLink:  "/apis/extensions/v1beta1/namespaces/snazzycats/jobs/mycronjob",
		},
		Spec: batch.CronJobSpec{
			Schedule:          "* * * * ?",
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels:      map[string]string{"a": "b"},
					Annotations: map[string]string{"x": "y"},
				},
				Spec: batch.JobSpec{
					ActiveDeadlineSeconds: &one,
					ManualSelector:        &no,
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

	var job *batch.Job
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
	if len(job.ObjectMeta.Annotations) != 2 {
		t.Errorf("Wrong number of annotations")
	}
	v, ok := job.ObjectMeta.Annotations[api.CreatedByAnnotation]
	if !ok {
		t.Errorf("Missing created-by annotation")
	}
	expectedCreatedBy := `{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"CronJob","namespace":"snazzycats","name":"mycronjob","uid":"1a2b3c","apiVersion":"extensions"}}
`
	if len(v) != len(expectedCreatedBy) {
		t.Errorf("Wrong length for created-by annotation, expected %v got %v", len(expectedCreatedBy), len(v))
	}
	if v != expectedCreatedBy {
		t.Errorf("Wrong value for created-by annotation, expected %v got %v", expectedCreatedBy, v)
	}
}

func TestGetParentUIDFromJob(t *testing.T) {
	j := &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foobar",
			Namespace: api.NamespaceDefault,
		},
		Spec: batch.JobSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
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
		Status: batch.JobStatus{
			Conditions: []batch.JobCondition{{
				Type:   batch.JobComplete,
				Status: api.ConditionTrue,
			}},
		},
	}
	{
		// Case 1: No UID annotation
		_, found := getParentUIDFromJob(*j)

		if found {
			t.Errorf("Unexpectedly found uid")
		}
	}
	{
		// Case 2: Has UID annotation
		j.ObjectMeta.Annotations = map[string]string{api.CreatedByAnnotation: `{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"CronJob","namespace":"default","name":"pi","uid":"5ef034e0-1890-11e6-8935-42010af0003e","apiVersion":"extensions","resourceVersion":"427339"}}`}

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
	createdBy1 := map[string]string{api.CreatedByAnnotation: `{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"CronJob","namespace":"x","name":"pi","uid":"11111111-1111-1111-1111-111111111111","apiVersion":"extensions","resourceVersion":"111111"}}`}
	createdBy2 := map[string]string{api.CreatedByAnnotation: `{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"CronJob","namespace":"x","name":"pi","uid":"22222222-2222-2222-2222-222222222222","apiVersion":"extensions","resourceVersion":"222222"}}`}
	createdBy3 := map[string]string{api.CreatedByAnnotation: `{"kind":"SerializedReference","apiVersion":"v1","reference":{"kind":"CronJob","namespace":"y","name":"pi","uid":"33333333-3333-3333-3333-333333333333","apiVersion":"extensions","resourceVersion":"333333"}}`}
	noCreatedBy := map[string]string{}

	{
		// Case 1: There are no jobs and scheduledJobs
		sjs := []batch.CronJob{}
		js := []batch.Job{}
		jobsBySj := groupJobsByParent(sjs, js)
		if len(jobsBySj) != 0 {
			t.Errorf("Wrong number of items in map")
		}
	}

	{
		// Case 2: there is one controller with no job.
		sjs := []batch.CronJob{
			{ObjectMeta: api.ObjectMeta{Name: "e", Namespace: "x", UID: uid1}},
		}
		js := []batch.Job{}
		jobsBySj := groupJobsByParent(sjs, js)
		if len(jobsBySj) != 0 {
			t.Errorf("Wrong number of items in map")
		}
	}

	{
		// Case 3: there is one controller with one job it created.
		sjs := []batch.CronJob{
			{ObjectMeta: api.ObjectMeta{Name: "e", Namespace: "x", UID: uid1}},
		}
		js := []batch.Job{
			{ObjectMeta: api.ObjectMeta{Name: "a", Namespace: "x", Annotations: createdBy1}},
		}
		jobsBySj := groupJobsByParent(sjs, js)

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
		// Case 4: Two namespaces, one has two jobs from one controller, other has 3 jobs from two controllers.
		// There are also two jobs with no created-by annotation.
		js := []batch.Job{
			{ObjectMeta: api.ObjectMeta{Name: "a", Namespace: "x", Annotations: createdBy1}},
			{ObjectMeta: api.ObjectMeta{Name: "b", Namespace: "x", Annotations: createdBy2}},
			{ObjectMeta: api.ObjectMeta{Name: "c", Namespace: "x", Annotations: createdBy1}},
			{ObjectMeta: api.ObjectMeta{Name: "d", Namespace: "x", Annotations: noCreatedBy}},
			{ObjectMeta: api.ObjectMeta{Name: "a", Namespace: "y", Annotations: createdBy3}},
			{ObjectMeta: api.ObjectMeta{Name: "b", Namespace: "y", Annotations: createdBy3}},
			{ObjectMeta: api.ObjectMeta{Name: "d", Namespace: "y", Annotations: noCreatedBy}},
		}
		sjs := []batch.CronJob{
			{ObjectMeta: api.ObjectMeta{Name: "e", Namespace: "x", UID: uid1}},
			{ObjectMeta: api.ObjectMeta{Name: "f", Namespace: "x", UID: uid2}},
			{ObjectMeta: api.ObjectMeta{Name: "g", Namespace: "y", UID: uid3}},
		}

		jobsBySj := groupJobsByParent(sjs, js)

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

	sj := batch.CronJob{
		ObjectMeta: api.ObjectMeta{
			Name:      "mycronjob",
			Namespace: api.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: batch.CronJobSpec{
			Schedule:          schedule,
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate:       batch.JobTemplateSpec{},
		},
	}
	{
		// Case 1: no known start times, and none needed yet.
		// Creation time is before T1.
		sj.ObjectMeta.CreationTimestamp = unversioned.Time{Time: T1.Add(-10 * time.Minute)}
		// Current time is more than creation time, but less than T1.
		now := T1.Add(-7 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(sj, now)
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
		sj.ObjectMeta.CreationTimestamp = unversioned.Time{Time: T1.Add(-10 * time.Minute)}
		// Current time is after T1
		now := T1.Add(2 * time.Second)
		times, err := getRecentUnmetScheduleTimes(sj, now)
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
		sj.ObjectMeta.CreationTimestamp = unversioned.Time{Time: T1.Add(-10 * time.Minute)}
		// Status shows a start at the expected time.
		sj.Status.LastScheduleTime = &unversioned.Time{Time: T1}
		// Current time is after T1
		now := T1.Add(2 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(sj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 0 {
			t.Errorf("expected 0 start times, got: , got: %v", times)
		}
	}
	{
		// Case 4: known LastScheduleTime, a start needed
		// Creation time is before T1.
		sj.ObjectMeta.CreationTimestamp = unversioned.Time{Time: T1.Add(-10 * time.Minute)}
		// Status shows a start at the expected time.
		sj.Status.LastScheduleTime = &unversioned.Time{Time: T1}
		// Current time is after T1 and after T2
		now := T2.Add(5 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(sj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 1 {
			t.Errorf("expected 2 start times, got: , got: %v", times)
		} else if !times[0].Equal(T2) {
			t.Errorf("expected: %v, got: %v", T1, times[0])
		}
	}
	{
		// Case 5: known LastScheduleTime, two starts needed
		sj.ObjectMeta.CreationTimestamp = unversioned.Time{Time: T1.Add(-2 * time.Hour)}
		sj.Status.LastScheduleTime = &unversioned.Time{Time: T1.Add(-1 * time.Hour)}
		// Current time is after T1 and after T2
		now := T2.Add(5 * time.Minute)
		times, err := getRecentUnmetScheduleTimes(sj, now)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(times) != 2 {
			t.Errorf("expected 2 start times, got: , got: %v", times)
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
		// Case 6: now is way way ahead of last start time.
		sj.ObjectMeta.CreationTimestamp = unversioned.Time{Time: T1.Add(-2 * time.Hour)}
		sj.Status.LastScheduleTime = &unversioned.Time{Time: T1.Add(-1 * time.Hour)}
		now := T2.Add(10 * 24 * time.Hour)
		_, err := getRecentUnmetScheduleTimes(sj, now)
		if err == nil {
			t.Errorf("unexpected lack of error")
		}
	}

}
