/*
Copyright 2018 The Kubernetes Authors.

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

package ttlafterfinished

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilpointer "k8s.io/utils/pointer"
)

func newJob(completionTime, failedTime metav1.Time, ttl *int32) *batch.Job {
	j := &batch.Job{
		TypeMeta: metav1.TypeMeta{Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foobar",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: batch.JobSpec{
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
	}

	if !completionTime.IsZero() {
		c := batch.JobCondition{Type: batch.JobComplete, Status: v1.ConditionTrue, LastTransitionTime: completionTime}
		j.Status.Conditions = append(j.Status.Conditions, c)
	}

	if !failedTime.IsZero() {
		c := batch.JobCondition{Type: batch.JobFailed, Status: v1.ConditionTrue, LastTransitionTime: failedTime}
		j.Status.Conditions = append(j.Status.Conditions, c)
	}

	if ttl != nil {
		j.Spec.TTLSecondsAfterFinished = ttl
	}

	return j
}

func durationPointer(n int) *time.Duration {
	s := time.Duration(n) * time.Second
	return &s
}

func TestTimeLeftForJob(t *testing.T) {
	now := metav1.Now()

	testCases := []struct {
		name             string
		completionTime   metav1.Time
		failedTime       metav1.Time
		ttl              *int32
		since            *time.Time
		expectErr        bool
		expectErrStr     string
		expectedTimeLeft *time.Duration
	}{
		{
			name:         "Error case: Job unfinished",
			ttl:          utilpointer.Int32Ptr(100),
			since:        &now.Time,
			expectErr:    true,
			expectErrStr: "should not be cleaned up",
		},
		{
			name:           "Error case: Job completed now, no TTL",
			completionTime: now,
			since:          &now.Time,
			expectErr:      true,
			expectErrStr:   "should not be cleaned up",
		},
		{
			name:             "Job completed now, 0s TTL",
			completionTime:   now,
			ttl:              utilpointer.Int32Ptr(0),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(0),
		},
		{
			name:             "Job completed now, 10s TTL",
			completionTime:   now,
			ttl:              utilpointer.Int32Ptr(10),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(10),
		},
		{
			name:             "Job completed 10s ago, 15s TTL",
			completionTime:   metav1.NewTime(now.Add(-10 * time.Second)),
			ttl:              utilpointer.Int32Ptr(15),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(5),
		},
		{
			name:         "Error case: Job failed now, no TTL",
			failedTime:   now,
			since:        &now.Time,
			expectErr:    true,
			expectErrStr: "should not be cleaned up",
		},
		{
			name:             "Job failed now, 0s TTL",
			failedTime:       now,
			ttl:              utilpointer.Int32Ptr(0),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(0),
		},
		{
			name:             "Job failed now, 10s TTL",
			failedTime:       now,
			ttl:              utilpointer.Int32Ptr(10),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(10),
		},
		{
			name:             "Job failed 10s ago, 15s TTL",
			failedTime:       metav1.NewTime(now.Add(-10 * time.Second)),
			ttl:              utilpointer.Int32Ptr(15),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(5),
		},
	}

	for _, tc := range testCases {
		job := newJob(tc.completionTime, tc.failedTime, tc.ttl)
		gotTimeLeft, gotErr := timeLeftForJob(job, tc.since)
		if tc.expectErr != (gotErr != nil) {
			t.Errorf("%s: expected error is %t, got %t, error: %v", tc.name, tc.expectErr, gotErr != nil, gotErr)
		}
		if tc.expectErr && len(tc.expectErrStr) == 0 {
			t.Errorf("%s: invalid test setup; error message must not be empty for error cases", tc.name)
		}
		if tc.expectErr && !strings.Contains(gotErr.Error(), tc.expectErrStr) {
			t.Errorf("%s: expected error message contains %q, got %v", tc.name, tc.expectErrStr, gotErr)
		}
		if !tc.expectErr {
			if *gotTimeLeft != *tc.expectedTimeLeft {
				t.Errorf("%s: expected time left %v, got %v", tc.name, tc.expectedTimeLeft, gotTimeLeft)
			}
		}
	}
}

func containerStatuses(numberOfContainers int, t metav1.Time) []v1.ContainerStatus {
	s := rand.NewSource(time.Now().UnixNano())
	r := rand.New(s)
	cs := []v1.ContainerStatus{}
	for i := 0; i <= numberOfContainers; i++ {
		c := v1.ContainerStatus{
			Name: fmt.Sprintf("foo-bar-%d", i),
			State: v1.ContainerState{
				Terminated: &v1.ContainerStateTerminated{
					FinishedAt: metav1.NewTime(t.Add(-1 * time.Duration(r.Intn(10)))),
				},
			},
		}
		cs = append(cs, c)
	}
	return cs
}

func newPod(completionTime, failedTime metav1.Time, ttl *int32, numberOfContainers int) *v1.Pod {

	var containers []v1.Container

	for i := 0; i <= numberOfContainers; i++ {
		c := v1.Container{
			Name:  fmt.Sprintf("foo-bar-%d", i),
			Image: "foo/bar",
		}
		containers = append(containers, c)
	}

	p := v1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foobar-pod",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			TTLSecondsAfterFinished: ttl,
		},
		Status: v1.PodStatus{},
	}

	p.Spec.Containers = containers

	s := rand.NewSource(time.Now().UnixNano())
	r := rand.New(s)

	if !completionTime.IsZero() {
		p.Status.Phase = v1.PodSucceeded
		containerStatuses := containerStatuses(numberOfContainers, completionTime)

		containerStatuses[r.Intn(numberOfContainers)].State.Terminated.FinishedAt = completionTime
		p.Status.ContainerStatuses = containerStatuses
	}

	if !failedTime.IsZero() {
		p.Status.Phase = v1.PodFailed
		containerStatuses := containerStatuses(numberOfContainers, failedTime)

		containerStatuses[r.Intn(numberOfContainers)].State.Terminated.FinishedAt = failedTime
		p.Status.ContainerStatuses = containerStatuses
	}

	return &p
}

func TestTimeLeftForPod(t *testing.T) {
	now := metav1.Now()

	numberOfContainers := 5

	testCases := []struct {
		name             string
		completionTime   metav1.Time
		failedTime       metav1.Time
		ttl              *int32
		since            *time.Time
		expectErr        bool
		expectErrStr     string
		expectedTimeLeft *time.Duration
	}{
		{
			name:         "Error case: Pod unfinished",
			ttl:          utilpointer.Int32Ptr(100),
			since:        &now.Time,
			expectErr:    true,
			expectErrStr: "should not be cleaned up",
		},
		{
			name:           "Error case: Pod completed now, no TTL",
			completionTime: now,
			since:          &now.Time,
			expectErr:      true,
			expectErrStr:   "should not be cleaned up",
		},
		{
			name:             "Pod completed now, 0s TTL",
			completionTime:   now,
			ttl:              utilpointer.Int32Ptr(0),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(0),
		},
		{
			name:             "Pod completed now, 10s TTL",
			completionTime:   now,
			ttl:              utilpointer.Int32Ptr(10),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(10),
		},
		{
			name:             "Pod completed 10s ago, 15s TTL",
			completionTime:   metav1.NewTime(now.Add(-10 * time.Second)),
			ttl:              utilpointer.Int32Ptr(15),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(5),
		},
		{
			name:         "Error case: Pod failed now, no TTL",
			failedTime:   now,
			since:        &now.Time,
			expectErr:    true,
			expectErrStr: "should not be cleaned up",
		},
		{
			name:             "Pod failed now, 0s TTL",
			failedTime:       now,
			ttl:              utilpointer.Int32Ptr(0),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(0),
		},
		{
			name:             "Pod failed now, 10s TTL",
			failedTime:       now,
			ttl:              utilpointer.Int32Ptr(10),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(10),
		},
		{
			name:             "Pod failed 10s ago, 15s TTL",
			failedTime:       metav1.NewTime(now.Add(-10 * time.Second)),
			ttl:              utilpointer.Int32Ptr(15),
			since:            &now.Time,
			expectedTimeLeft: durationPointer(5),
		},
	}

	for _, tc := range testCases {
		pod := newPod(tc.completionTime, tc.failedTime, tc.ttl, numberOfContainers)
		gotTimeLeft, gotErr := timeLeftForPod(pod, tc.since)
		if tc.expectErr != (gotErr != nil) {
			t.Errorf("%s: expected error is %t, got %t, error: %v", tc.name, tc.expectErr, gotErr != nil, gotErr)
		}
		if tc.expectErr && len(tc.expectErrStr) == 0 {
			t.Errorf("%s: invalid test setup; error message must not be empty for error cases", tc.name)
		}
		if tc.expectErr && !strings.Contains(gotErr.Error(), tc.expectErrStr) {
			t.Errorf("%s: expected error message contains %q, got %v", tc.name, tc.expectErrStr, gotErr)
		}
		if !tc.expectErr {
			if *gotTimeLeft != *tc.expectedTimeLeft {
				t.Errorf("%s: expected time left %v, got %v", tc.name, tc.expectedTimeLeft, gotTimeLeft)
			}
		}
	}
}
