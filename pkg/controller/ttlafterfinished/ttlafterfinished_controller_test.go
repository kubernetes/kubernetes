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
	"strings"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/pointer"
)

func newJob(completionTime, failedTime metav1.Time, ttl *int32) *batchv1.Job {
	j := &batchv1.Job{
		TypeMeta: metav1.TypeMeta{Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foobar",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: batchv1.JobSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Image: "foo/bar"},
					},
				},
			},
		},
	}

	if !completionTime.IsZero() {
		c := batchv1.JobCondition{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: completionTime}
		j.Status.Conditions = append(j.Status.Conditions, c)
	}

	if !failedTime.IsZero() {
		c := batchv1.JobCondition{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, LastTransitionTime: failedTime}
		j.Status.Conditions = append(j.Status.Conditions, c)
	}

	if ttl != nil {
		j.Spec.TTLSecondsAfterFinished = ttl
	}

	return j
}

func TestTimeLeft(t *testing.T) {
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
		expectedExpireAt time.Time
	}{
		{
			name:         "Error case: Job unfinished",
			ttl:          pointer.Int32(100),
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
			ttl:              pointer.Int32(0),
			since:            &now.Time,
			expectedTimeLeft: pointer.Duration(0 * time.Second),
			expectedExpireAt: now.Time,
		},
		{
			name:             "Job completed now, 10s TTL",
			completionTime:   now,
			ttl:              pointer.Int32(10),
			since:            &now.Time,
			expectedTimeLeft: pointer.Duration(10 * time.Second),
			expectedExpireAt: now.Add(10 * time.Second),
		},
		{
			name:             "Job completed 10s ago, 15s TTL",
			completionTime:   metav1.NewTime(now.Add(-10 * time.Second)),
			ttl:              pointer.Int32(15),
			since:            &now.Time,
			expectedTimeLeft: pointer.Duration(5 * time.Second),
			expectedExpireAt: now.Add(5 * time.Second),
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
			ttl:              pointer.Int32(0),
			since:            &now.Time,
			expectedTimeLeft: pointer.Duration(0 * time.Second),
			expectedExpireAt: now.Time,
		},
		{
			name:             "Job failed now, 10s TTL",
			failedTime:       now,
			ttl:              pointer.Int32(10),
			since:            &now.Time,
			expectedTimeLeft: pointer.Duration(10 * time.Second),
			expectedExpireAt: now.Add(10 * time.Second),
		},
		{
			name:             "Job failed 10s ago, 15s TTL",
			failedTime:       metav1.NewTime(now.Add(-10 * time.Second)),
			ttl:              pointer.Int32(15),
			since:            &now.Time,
			expectedTimeLeft: pointer.Duration(5 * time.Second),
			expectedExpireAt: now.Add(5 * time.Second),
		},
	}

	for _, tc := range testCases {

		job := newJob(tc.completionTime, tc.failedTime, tc.ttl)
		_, ctx := ktesting.NewTestContext(t)
		logger := klog.FromContext(ctx)
		gotTimeLeft, gotExpireAt, gotErr := timeLeft(logger, job, tc.since)
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
			if *gotExpireAt != tc.expectedExpireAt {
				t.Errorf("%s: expected expire at %v, got %v", tc.name, tc.expectedExpireAt, *gotExpireAt)
			}
		}
	}
}
