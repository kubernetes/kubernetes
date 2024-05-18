/*
Copyright 2023 The Kubernetes Authors.

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

package job

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	clocktesting "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestNewBackoffRecord(t *testing.T) {
	emptyStoreInitializer := func(*backoffStore) {}
	defaultTestTime := metav1.NewTime(time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC))
	testCases := map[string]struct {
		storeInitializer  func(*backoffStore)
		uncounted         uncountedTerminatedPods
		newSucceededPods  []metav1.Time
		newFailedPods     []metav1.Time
		wantBackoffRecord backoffRecord
	}{
		"Empty backoff store and one new failure": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{},
			newFailedPods: []metav1.Time{
				defaultTestTime,
			},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 1,
			},
		},
		"Empty backoff store and two new failures": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{},
			newFailedPods: []metav1.Time{
				defaultTestTime,
				metav1.NewTime(defaultTestTime.Add(-1 * time.Millisecond)),
			},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 2,
			},
		},
		"Empty backoff store, two failures followed by success": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{
				defaultTestTime,
			},
			newFailedPods: []metav1.Time{
				metav1.NewTime(defaultTestTime.Add(-2 * time.Millisecond)),
				metav1.NewTime(defaultTestTime.Add(-1 * time.Millisecond)),
			},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				failuresAfterLastSuccess: 0,
			},
		},
		"Empty backoff store, two failures, one success and two more failures": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{
				metav1.NewTime(defaultTestTime.Add(-2 * time.Millisecond)),
			},
			newFailedPods: []metav1.Time{
				defaultTestTime,
				metav1.NewTime(defaultTestTime.Add(-4 * time.Millisecond)),
				metav1.NewTime(defaultTestTime.Add(-3 * time.Millisecond)),
				metav1.NewTime(defaultTestTime.Add(-1 * time.Millisecond)),
			},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 2,
			},
		},
		"Backoff store having failure count 2 and one new failure": {
			storeInitializer: func(bis *backoffStore) {
				bis.updateBackoffRecord(backoffRecord{
					key:                      "key",
					failuresAfterLastSuccess: 2,
					lastFailureTime:          nil,
				})
			},
			newSucceededPods: []metav1.Time{},
			newFailedPods: []metav1.Time{
				defaultTestTime,
			},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 3,
			},
		},
		"Empty backoff store with success and failure at same timestamp": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{
				defaultTestTime,
			},
			newFailedPods: []metav1.Time{
				defaultTestTime,
			},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				failuresAfterLastSuccess: 0,
			},
		},
		"Empty backoff store with no success/failure": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{},
			newFailedPods:    []metav1.Time{},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				failuresAfterLastSuccess: 0,
			},
		},
		"Empty backoff store with one success": {
			storeInitializer: emptyStoreInitializer,
			newSucceededPods: []metav1.Time{
				defaultTestTime,
			},
			newFailedPods: []metav1.Time{},
			wantBackoffRecord: backoffRecord{
				key:                      "key",
				failuresAfterLastSuccess: 0,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			backoffRecordStore := newBackoffStore()
			tc.storeInitializer(backoffRecordStore)

			newSucceededPods := []*v1.Pod{}
			newFailedPods := []*v1.Pod{}

			for _, finishTime := range tc.newSucceededPods {
				newSucceededPods = append(newSucceededPods, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{},
					Status: v1.PodStatus{
						Phase: v1.PodSucceeded,
						ContainerStatuses: []v1.ContainerStatus{
							{
								State: v1.ContainerState{
									Terminated: &v1.ContainerStateTerminated{
										FinishedAt: finishTime,
									},
								},
							},
						},
					},
				})
			}

			for _, finishTime := range tc.newFailedPods {
				newFailedPods = append(newFailedPods, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{},
					Status: v1.PodStatus{
						Phase: v1.PodFailed,
						ContainerStatuses: []v1.ContainerStatus{
							{
								State: v1.ContainerState{
									Terminated: &v1.ContainerStateTerminated{
										FinishedAt: finishTime,
									},
								},
							},
						},
					},
				})
			}

			backoffRecord := backoffRecordStore.newBackoffRecord("key", newSucceededPods, newFailedPods)
			if diff := cmp.Diff(tc.wantBackoffRecord, backoffRecord, cmp.AllowUnexported(backoffRecord)); diff != "" {
				t.Errorf("backoffRecord not matching; (-want,+got): %v", diff)
			}
		})
	}
}

func TestGetFinishedTime(t *testing.T) {
	defaultTestTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	defaultTestTimeMinus30s := defaultTestTime.Add(-30 * time.Second)
	testCases := map[string]struct {
		pod            v1.Pod
		wantFinishTime time.Time
	}{
		"Pod with multiple containers and all containers terminated": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-1 * time.Second))},
							},
						},
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime)},
							},
						},
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-2 * time.Second))},
							},
						},
					},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		"Pod with multiple containers; two containers in terminated state and one in running state; fallback to deletionTimestamp": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-1 * time.Second))},
							},
						},
						{
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{},
							},
						},
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-2 * time.Second))},
							},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &metav1.Time{Time: defaultTestTime},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		"fallback to deletionTimestamp": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{},
							},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &metav1.Time{Time: defaultTestTime},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		"fallback to deletionTimestamp, decremented by grace period": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{},
							},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp:          &metav1.Time{Time: defaultTestTime},
					DeletionGracePeriodSeconds: ptr.To[int64](30),
				},
			},
			wantFinishTime: defaultTestTimeMinus30s,
		},
		"fallback to PodReady.LastTransitionTime when status of the condition is False": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{},
							},
						},
					},
					Conditions: []v1.PodCondition{
						{
							Type:               v1.PodReady,
							Status:             v1.ConditionFalse,
							Reason:             "PodFailed",
							LastTransitionTime: metav1.Time{Time: defaultTestTime},
						},
					},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		"skip fallback to PodReady.LastTransitionTime when status of the condition is True": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{},
							},
						},
					},
					Conditions: []v1.PodCondition{
						{
							Type:               v1.PodReady,
							Status:             v1.ConditionTrue,
							LastTransitionTime: metav1.Time{Time: defaultTestTimeMinus30s},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &metav1.Time{Time: defaultTestTime},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		"fallback to creationTimestamp": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{},
							},
						},
					},
				},
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.Time{Time: defaultTestTime},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		// In this case, init container is stopped after the regular containers.
		// This is because with the sidecar (restartable init) containers,
		// sidecar containers will always finish later than regular containers.
		"Pod with init container and all containers terminated": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-1 * time.Second))},
							},
						},
					},
					InitContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime)},
							},
						},
					},
				},
			},
			wantFinishTime: defaultTestTime,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			f := getFinishedTime(&tc.pod)
			if !f.Equal(tc.wantFinishTime) {
				t.Errorf("Expected value of finishedTime %v; got %v", tc.wantFinishTime, f)
			}
		})
	}
}

func TestGetRemainingBackoffTime(t *testing.T) {
	defaultTestTime := metav1.NewTime(time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC))
	testCases := map[string]struct {
		backoffRecord  backoffRecord
		currentTime    time.Time
		maxBackoff     time.Duration
		defaultBackoff time.Duration
		wantDuration   time.Duration
	}{
		"no failures": {
			backoffRecord: backoffRecord{
				lastFailureTime:          nil,
				failuresAfterLastSuccess: 0,
			},
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   0 * time.Second,
		},
		"one failure; current time and failure time are same": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 1,
			},
			currentTime:    defaultTestTime.Time,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   5 * time.Second,
		},
		"one failure; current time == 1 second + failure time": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 1,
			},
			currentTime:    defaultTestTime.Time.Add(time.Second),
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   4 * time.Second,
		},
		"one failure; current time == expected backoff time": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 1,
			},
			currentTime:    defaultTestTime.Time.Add(5 * time.Second),
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   0 * time.Second,
		},
		"one failure; current time == expected backoff time + 1 Second": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 1,
			},
			currentTime:    defaultTestTime.Time.Add(6 * time.Second),
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   0 * time.Second,
		},
		"three failures; current time and failure time are same": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 3,
			},
			currentTime:    defaultTestTime.Time,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   20 * time.Second,
		},
		"eight failures; current time and failure time are same; backoff not exceeding maxBackoff": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 8,
			},
			currentTime:    defaultTestTime.Time,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   640 * time.Second,
		},
		"nine failures; current time and failure time are same; backoff exceeding maxBackoff": {
			backoffRecord: backoffRecord{
				lastFailureTime:          &defaultTestTime.Time,
				failuresAfterLastSuccess: 9,
			},
			currentTime:    defaultTestTime.Time,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   700 * time.Second,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			fakeClock := clocktesting.NewFakeClock(tc.currentTime.Truncate(time.Second))
			d := tc.backoffRecord.getRemainingTime(fakeClock, tc.defaultBackoff, tc.maxBackoff)
			if d.Seconds() != tc.wantDuration.Seconds() {
				t.Errorf("Expected value of duration %v; got %v", tc.wantDuration, d)
			}
		})
	}
}

func TestGetRemainingBackoffTimePerIndex(t *testing.T) {
	defaultTestTime := metav1.NewTime(time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC))
	testCases := map[string]struct {
		currentTime    time.Time
		maxBackoff     time.Duration
		defaultBackoff time.Duration
		lastFailedPod  *v1.Pod
		wantDuration   time.Duration
	}{
		"no failures": {
			lastFailedPod:  nil,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   0 * time.Second,
		},
		"two prev failures; current time and failure time are same": {
			lastFailedPod:  buildPod().phase(v1.PodFailed).indexFailureCount("2").customDeletionTimestamp(defaultTestTime.Time).Pod,
			currentTime:    defaultTestTime.Time,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   20 * time.Second,
		},
		"one prev failure counted and one ignored; current time and failure time are same": {
			lastFailedPod:  buildPod().phase(v1.PodFailed).indexFailureCount("1").indexIgnoredFailureCount("1").customDeletionTimestamp(defaultTestTime.Time).Pod,
			currentTime:    defaultTestTime.Time,
			defaultBackoff: 5 * time.Second,
			maxBackoff:     700 * time.Second,
			wantDuration:   20 * time.Second,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			fakeClock := clocktesting.NewFakeClock(tc.currentTime.Truncate(time.Second))
			d := getRemainingTimePerIndex(logger, fakeClock, tc.defaultBackoff, tc.maxBackoff, tc.lastFailedPod)
			if d.Seconds() != tc.wantDuration.Seconds() {
				t.Errorf("Expected value of duration %v; got %v", tc.wantDuration, d)
			}
		})
	}
}
