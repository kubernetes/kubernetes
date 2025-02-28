/*
Copyright 2025 The Kubernetes Authors.

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

package queue

import (
	"math"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

func TestBackoffQueue_calculateBackoffDuration(t *testing.T) {
	tests := []struct {
		name                   string
		initialBackoffDuration time.Duration
		maxBackoffDuration     time.Duration
		podInfo                *framework.QueuedPodInfo
		want                   time.Duration
	}{
		{
			name:                   "no backoff",
			initialBackoffDuration: 1 * time.Nanosecond,
			maxBackoffDuration:     32 * time.Nanosecond,
			podInfo:                &framework.QueuedPodInfo{Attempts: 0},
			want:                   0,
		},
		{
			name:                   "normal",
			initialBackoffDuration: 1 * time.Nanosecond,
			maxBackoffDuration:     32 * time.Nanosecond,
			podInfo:                &framework.QueuedPodInfo{Attempts: 16},
			want:                   32 * time.Nanosecond,
		},
		{
			name:                   "overflow_32bit",
			initialBackoffDuration: 1 * time.Nanosecond,
			maxBackoffDuration:     math.MaxInt32 * time.Nanosecond,
			podInfo:                &framework.QueuedPodInfo{Attempts: 32},
			want:                   math.MaxInt32 * time.Nanosecond,
		},
		{
			name:                   "overflow_64bit",
			initialBackoffDuration: 1 * time.Nanosecond,
			maxBackoffDuration:     math.MaxInt64 * time.Nanosecond,
			podInfo:                &framework.QueuedPodInfo{Attempts: 64},
			want:                   math.MaxInt64 * time.Nanosecond,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bq := newBackoffQueue(clock.RealClock{}, tt.initialBackoffDuration, tt.maxBackoffDuration)
			if got := bq.calculateBackoffDuration(tt.podInfo); got != tt.want {
				t.Errorf("backoffQueue.calculateBackoffDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBackoffQueue_popEachBackoffCompleted(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	podInfos := map[string]*framework.QueuedPodInfo{
		"pod0": {
			PodInfo: &framework.PodInfo{
				Pod: st.MakePod().Name("pod0").Obj(),
			},
			Timestamp:            fakeClock.Now().Add(-time.Second),
			Attempts:             1,
			UnschedulablePlugins: sets.New("plugin"),
		},
		"pod1": {
			PodInfo: &framework.PodInfo{
				Pod: st.MakePod().Name("pod1").Obj(),
			},
			Timestamp:            fakeClock.Now().Add(time.Second),
			Attempts:             1,
			UnschedulablePlugins: sets.New("plugin"),
		},
		"pod2": {
			PodInfo: &framework.PodInfo{
				Pod: st.MakePod().Name("pod2").Obj(),
			},
			Timestamp: fakeClock.Now().Add(-time.Second),
			Attempts:  1,
		},
		"pod3": {
			PodInfo: &framework.PodInfo{
				Pod: st.MakePod().Name("pod3").Obj(),
			},
			Timestamp: fakeClock.Now().Add(time.Second),
			Attempts:  1,
		},
	}
	tests := []struct {
		name          string
		podsInBackoff []string
		wantPods      []string
	}{
		{
			name:          "Both queues empty, no pods moved to activeQ",
			podsInBackoff: []string{},
			wantPods:      nil,
		},
		{
			name:          "Pods only in backoffQ, some pods moved to activeQ",
			podsInBackoff: []string{"pod0", "pod1"},
			wantPods:      []string{"pod0"},
		},
		{
			name:          "Pods only in errorBackoffQ, some pods moved to activeQ",
			podsInBackoff: []string{"pod2", "pod3"},
			wantPods:      []string{"pod2"},
		},
		{
			name:          "Pods in both queues, some pods moved to activeQ",
			podsInBackoff: []string{"pod0", "pod1", "pod2", "pod3"},
			wantPods:      []string{"pod0", "pod2"},
		},
		{
			name:          "Pods in both queues, all pods moved to activeQ",
			podsInBackoff: []string{"pod0", "pod2"},
			wantPods:      []string{"pod0", "pod2"},
		},
		{
			name:          "Pods in both queues, no pods moved to activeQ",
			podsInBackoff: []string{"pod1", "pod3"},
			wantPods:      nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			bq := newBackoffQueue(fakeClock, DefaultPodInitialBackoffDuration, DefaultPodMaxBackoffDuration)
			for _, podName := range tt.podsInBackoff {
				bq.add(logger, podInfos[podName])
			}
			var gotPods []string
			bq.popEachBackoffCompleted(logger, func(pInfo *framework.QueuedPodInfo) {
				gotPods = append(gotPods, pInfo.Pod.Name)
			})
			if diff := cmp.Diff(tt.wantPods, gotPods); diff != "" {
				t.Errorf("Unexpected pods moved (-want, +got):\n%s", diff)
			}
			podsToStayInBackoff := len(tt.podsInBackoff) - len(tt.wantPods)
			if bq.len() != podsToStayInBackoff {
				t.Errorf("Expected %v pods to stay in backoffQ, but got: %v", podsToStayInBackoff, bq.len())
			}
		})
	}
}
