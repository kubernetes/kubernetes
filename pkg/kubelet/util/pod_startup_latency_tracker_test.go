/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	testingclock "k8s.io/utils/clock/testing"
)

var frozenTime = time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)

const (
	uid         = "3c1df8a9-11a8-4791-aeae-184c18cca686"
	metricsName = "kubelet_pod_start_sli_duration_seconds"
)

func TestNoEvents(t *testing.T) {
	t.Run("metrics registered; no incoming events", func(t *testing.T) {

		// expects no metrics in the output
		wants := ""

		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods: map[types.UID]*perPodState{},
		}

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func TestPodsRunningBeforeKubeletStarted(t *testing.T) {
	t.Run("pod was running for 10m before kubelet started", func(t *testing.T) {

		// expects no metrics in the output
		wants := ""

		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods: map[types.UID]*perPodState{},
		}

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		podStarted := &corev1.Pod{
			Status: corev1.PodStatus{
				StartTime: &metav1.Time{Time: frozenTime.Add(-10 * time.Minute)},
			},
		}
		tracker.ObservedPodOnWatch(podStarted, frozenTime)

		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func TestSinglePodOneImageDownloadRecorded(t *testing.T) {

	t.Run("single pod; started in 3s, image pulling 100ms", func(t *testing.T) {

		wants := `
# HELP kubelet_pod_start_sli_duration_seconds [ALPHA] Duration in seconds to start a pod, excluding time to pull images and run init containers, measured from pod creation timestamp to when all its containers are reported as started and observed via watch
# TYPE kubelet_pod_start_sli_duration_seconds histogram
kubelet_pod_start_sli_duration_seconds_bucket{le="0.5"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="1"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="2"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="3"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="4"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="5"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="6"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="8"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="10"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="20"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="30"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="45"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="60"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="120"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="180"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="240"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="300"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="360"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="480"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="600"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="900"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="1200"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="1800"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="2700"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="3600"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="+Inf"} 1
kubelet_pod_start_sli_duration_seconds_sum 2.9
kubelet_pod_start_sli_duration_seconds_count 1
		`

		fakeClock := testingclock.NewFakeClock(frozenTime)

		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods:  map[types.UID]*perPodState{},
			clock: fakeClock,
		}

		podInit := buildInitializingPod()
		tracker.ObservedPodOnWatch(podInit, frozenTime)

		// image pulling took 100ms
		tracker.RecordImageStartedPulling(podInit.UID)
		fakeClock.Step(time.Millisecond * 100)
		tracker.RecordImageFinishedPulling(podInit.UID)

		podState, ok := tracker.pods[podInit.UID]
		if !ok {
			t.Errorf("expected to track pod: %s, but pod not found", podInit.UID)
		}
		if !podState.lastFinishedPulling.Equal(podState.firstStartedPulling.Add(time.Millisecond * 100)) {
			t.Errorf("expected pod firstStartedPulling: %s and lastFinishedPulling: %s but got firstStartedPulling: %s and lastFinishedPulling: %s",
				podState.firstStartedPulling, podState.firstStartedPulling.Add(time.Millisecond*100), podState.firstStartedPulling, podState.lastFinishedPulling)
		}

		podStarted := buildRunningPod()
		tracker.RecordStatusUpdated(podStarted)

		// 3s later, observe the same pod on watch
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*3))

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		// cleanup
		tracker.DeletePodStartupState(podStarted.UID)

		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func TestSinglePodMultipleDownloadsAndRestartsRecorded(t *testing.T) {

	t.Run("single pod; started in 30s, image pulling between 10th and 20th seconds", func(t *testing.T) {

		wants := `
# HELP kubelet_pod_start_sli_duration_seconds [ALPHA] Duration in seconds to start a pod, excluding time to pull images and run init containers, measured from pod creation timestamp to when all its containers are reported as started and observed via watch
# TYPE kubelet_pod_start_sli_duration_seconds histogram
kubelet_pod_start_sli_duration_seconds_bucket{le="0.5"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="1"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="2"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="3"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="4"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="5"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="6"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="8"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="10"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="20"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="30"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="45"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="60"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="120"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="180"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="240"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="300"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="360"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="480"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="600"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="900"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="1200"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="1800"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="2700"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="3600"} 1
kubelet_pod_start_sli_duration_seconds_bucket{le="+Inf"} 1
kubelet_pod_start_sli_duration_seconds_sum 20
kubelet_pod_start_sli_duration_seconds_count 1
`

		fakeClock := testingclock.NewFakeClock(frozenTime)

		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods:  map[types.UID]*perPodState{},
			clock: fakeClock,
		}

		podInitializing := buildInitializingPod()
		tracker.ObservedPodOnWatch(podInitializing, frozenTime)

		// image pulling started at 10s and the last one finished at 30s
		// first image starts pulling at 10s
		fakeClock.SetTime(frozenTime.Add(time.Second * 10))
		tracker.RecordImageStartedPulling(podInitializing.UID)
		// second image starts pulling at 11s
		fakeClock.SetTime(frozenTime.Add(time.Second * 11))
		tracker.RecordImageStartedPulling(podInitializing.UID)
		// third image starts pulling at 14s
		fakeClock.SetTime(frozenTime.Add(time.Second * 14))
		tracker.RecordImageStartedPulling(podInitializing.UID)
		// first image finished pulling at 18s
		fakeClock.SetTime(frozenTime.Add(time.Second * 18))
		tracker.RecordImageFinishedPulling(podInitializing.UID)
		// second and third finished pulling at 20s
		fakeClock.SetTime(frozenTime.Add(time.Second * 20))
		tracker.RecordImageFinishedPulling(podInitializing.UID)

		podState, ok := tracker.pods[podInitializing.UID]
		if !ok {
			t.Errorf("expected to track pod: %s, but pod not found", podInitializing.UID)
		}
		if !podState.firstStartedPulling.Equal(frozenTime.Add(time.Second * 10)) { // second and third image start pulling should not affect pod firstStartedPulling
			t.Errorf("expected pod firstStartedPulling: %s but got firstStartedPulling: %s",
				podState.firstStartedPulling.Add(time.Second*10), podState.firstStartedPulling)
		}
		if !podState.lastFinishedPulling.Equal(frozenTime.Add(time.Second * 20)) { // should be updated when the pod's last image finished pulling
			t.Errorf("expected pod lastFinishedPulling: %s but got lastFinishedPulling: %s",
				podState.lastFinishedPulling.Add(time.Second*20), podState.lastFinishedPulling)
		}

		// pod started
		podStarted := buildRunningPod()
		tracker.RecordStatusUpdated(podStarted)

		// at 30s observe the same pod on watch
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*30))

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		// any new pod observations should not impact the metrics, as the pod should be recorder only once
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*150))
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*200))
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*250))

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		// cleanup
		tracker.DeletePodStartupState(podStarted.UID)

		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func buildInitializingPod() *corev1.Pod {
	return buildPodWithStatus([]corev1.ContainerStatus{
		{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}}},
	})
}

func buildRunningPod() *corev1.Pod {
	return buildPodWithStatus([]corev1.ContainerStatus{
		{State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{StartedAt: metav1.NewTime(frozenTime)}}},
	})
}

func buildPodWithStatus(cs []corev1.ContainerStatus) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:               types.UID(uid),
			CreationTimestamp: metav1.NewTime(frozenTime),
		},
		Status: corev1.PodStatus{
			ContainerStatuses: cs,
		},
	}
}
