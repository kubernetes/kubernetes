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
			pods:         map[types.UID]*perPodState{},
			excludedPods: map[types.UID]bool{},
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
			pods:         map[types.UID]*perPodState{},
			excludedPods: map[types.UID]bool{},
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
			pods: map[types.UID]*perPodState{}, excludedPods: map[types.UID]bool{},
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

		if len(podState.imagePullSessions) != 1 {
			t.Errorf("expected one image pull session to be recorded")
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

	t.Run("single pod; started in 30s, overlapping image pulling between 10th and 20th seconds", func(t *testing.T) {

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
			pods: map[types.UID]*perPodState{}, excludedPods: map[types.UID]bool{},
			clock: fakeClock,
		}

		podInitializing := buildInitializingPod()
		tracker.ObservedPodOnWatch(podInitializing, frozenTime)
		// Image 1: 10-16s
		fakeClock.SetTime(frozenTime.Add(time.Second * 10))
		tracker.RecordImageStartedPulling(podInitializing.UID)
		// Image 2: 11-18s
		fakeClock.SetTime(frozenTime.Add(time.Second * 11))
		tracker.RecordImageStartedPulling(podInitializing.UID)
		// Image 3: 14-20s
		fakeClock.SetTime(frozenTime.Add(time.Second * 14))
		tracker.RecordImageStartedPulling(podInitializing.UID)
		fakeClock.SetTime(frozenTime.Add(time.Second * 16))
		tracker.RecordImageFinishedPulling(podInitializing.UID)
		fakeClock.SetTime(frozenTime.Add(time.Second * 18))
		tracker.RecordImageFinishedPulling(podInitializing.UID)
		fakeClock.SetTime(frozenTime.Add(time.Second * 20))
		tracker.RecordImageFinishedPulling(podInitializing.UID)

		podState, ok := tracker.pods[podInitializing.UID]
		if !ok {
			t.Errorf("expected to track pod: %s, but pod not found", podInitializing.UID)
		}
		if len(podState.imagePullSessions) != 3 {
			t.Errorf("expected 3 image pull sessions but got %d", len(podState.imagePullSessions))
		}
		totalTime := calculateImagePullingTime(podState.imagePullSessions)
		expectedTime := time.Second * 10
		if totalTime != expectedTime {
			t.Errorf("expected total pulling time: %v but got %v", expectedTime, totalTime)
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

func TestPodWithInitContainersAndMainContainers(t *testing.T) {

	t.Run("single pod with multiple init containers and main containers", func(t *testing.T) {

		wants := `
# HELP kubelet_pod_start_sli_duration_seconds [ALPHA] Duration in seconds to start a pod, excluding time to pull images and run init containers, measured from pod creation timestamp to when all its containers are reported as started and observed via watch
# TYPE kubelet_pod_start_sli_duration_seconds histogram
kubelet_pod_start_sli_duration_seconds_bucket{le="0.5"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="1"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="2"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="3"} 0
kubelet_pod_start_sli_duration_seconds_bucket{le="4"} 0
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
kubelet_pod_start_sli_duration_seconds_sum 4.2
kubelet_pod_start_sli_duration_seconds_count 1
		`

		fakeClock := testingclock.NewFakeClock(frozenTime)

		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods: map[types.UID]*perPodState{}, excludedPods: map[types.UID]bool{},
			clock: fakeClock,
		}

		podInit := buildInitializingPod("init-1", "init-2")
		tracker.ObservedPodOnWatch(podInit, frozenTime)

		// Init container 1 image pull: 0.5s-1.5s
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 500))
		tracker.RecordImageStartedPulling(podInit.UID)
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 1500))
		tracker.RecordImageFinishedPulling(podInit.UID)
		// Init container 1 runtime: 2s-4s
		fakeClock.SetTime(frozenTime.Add(time.Second * 2))
		tracker.RecordInitContainerStarted(types.UID(uid), fakeClock.Now())
		fakeClock.SetTime(frozenTime.Add(time.Second * 4))
		tracker.RecordInitContainerFinished(types.UID(uid), fakeClock.Now())

		// Init container 2 image pull: 4.2s-5.2s
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 4200))
		tracker.RecordImageStartedPulling(podInit.UID)
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 5200))
		tracker.RecordImageFinishedPulling(podInit.UID)
		// Init container 2 runtime: 5.5s-7s
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 5500))
		tracker.RecordInitContainerStarted(types.UID(uid), fakeClock.Now())
		fakeClock.SetTime(frozenTime.Add(time.Second * 7))
		tracker.RecordInitContainerFinished(types.UID(uid), fakeClock.Now())

		// Main container 1 image pull: 7.2s-8.2s
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 7200))
		tracker.RecordImageStartedPulling(podInit.UID)
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 8200))
		tracker.RecordImageFinishedPulling(podInit.UID)
		// Main container 2 image pull: 7.3s-8.5s
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 7300))
		tracker.RecordImageStartedPulling(podInit.UID)
		fakeClock.SetTime(frozenTime.Add(time.Millisecond * 8500))
		tracker.RecordImageFinishedPulling(podInit.UID)

		// Pod becomes running at 11s
		podStarted := buildRunningPod()
		tracker.RecordStatusUpdated(podStarted)
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*11))

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		state := tracker.pods[types.UID(uid)]
		assert.NotNil(t, state, "Pod state should exist")

		expectedInitRuntime := 2*time.Second + 1500*time.Millisecond
		assert.Equal(t, expectedInitRuntime, state.totalInitContainerRuntime,
			"Total init container runtime should be 3.5s (2s + 1.5s)")

		totalImageTime := calculateImagePullingTime(state.imagePullSessions)
		expectedImageTime := 3300 * time.Millisecond // 1s + 1s + 1.3s concurrent overlap
		assert.Equal(t, expectedImageTime, totalImageTime,
			"Image pulling time should be 3.3s: init-1(1s) + init-2(1s) + concurrent-main(1.3s)")

		assert.Len(t, state.imagePullSessions, 4, "Should have 4 image pull sessions")

		// cleanup
		tracker.DeletePodStartupState(podStarted.UID)

		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func TestImmediatelySchedulablePods(t *testing.T) {
	t.Run("pods not immediately schedulable should be excluded from tracking", func(t *testing.T) {
		wants := ""

		fakeClock := testingclock.NewFakeClock(frozenTime)
		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods:         map[types.UID]*perPodState{},
			excludedPods: map[types.UID]bool{},
			clock:        fakeClock,
		}

		// pod that is not immediately schedulable (PodScheduled=False)
		podNotSchedulable := buildInitializingPod()
		podNotSchedulable.UID = "not-schedulable-pod"
		podNotSchedulable.Status.Conditions = []corev1.PodCondition{
			{
				Type:   corev1.PodScheduled,
				Status: corev1.ConditionFalse,
				Reason: "Unschedulable",
			},
		}
		tracker.ObservedPodOnWatch(podNotSchedulable, frozenTime)
		assert.Empty(t, tracker.pods, "Pod with PodScheduled=False should not be tracked")

		// pod that is immediately schedulable (PodScheduled=True)
		podSchedulable := buildInitializingPod()
		podSchedulable.UID = "schedulable-pod"
		podSchedulable.Status.Conditions = []corev1.PodCondition{
			{
				Type:   corev1.PodScheduled,
				Status: corev1.ConditionTrue,
				Reason: "Scheduled",
			},
		}
		tracker.ObservedPodOnWatch(podSchedulable, frozenTime)
		assert.Len(t, tracker.pods, 1, "Pod with PodScheduled=True should be tracked")

		// pod without PodScheduled condition
		podNoCondition := buildInitializingPod()
		podNoCondition.UID = "no-condition-pod"
		tracker.ObservedPodOnWatch(podNoCondition, frozenTime)
		assert.Len(t, tracker.pods, 2, "Pod without PodScheduled condition should be tracked by default")

		// Verify metrics are empty
		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		// cleanup
		tracker.DeletePodStartupState(podSchedulable.UID)
		tracker.DeletePodStartupState(podNoCondition.UID)
		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})

	t.Run("pod observed as schedulable first, then becomes unschedulable should not be tracked", func(t *testing.T) {
		wants := ""

		fakeClock := testingclock.NewFakeClock(frozenTime)
		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods: map[types.UID]*perPodState{}, excludedPods: map[types.UID]bool{},
			clock: fakeClock,
		}

		// First observe pod as schedulable
		podSchedulable := buildInitializingPod()
		podSchedulable.UID = "becomes-unschedulable"
		tracker.ObservedPodOnWatch(podSchedulable, frozenTime)
		assert.Len(t, tracker.pods, 1, "Pod should be tracked initially")

		// Later observe the same pod as unschedulable
		podUnschedulable := buildInitializingPod()
		podUnschedulable.UID = "becomes-unschedulable"
		podUnschedulable.Status.Conditions = []corev1.PodCondition{
			{
				Type:   corev1.PodScheduled,
				Status: corev1.ConditionFalse,
				Reason: "Unschedulable",
			},
		}

		tracker.ObservedPodOnWatch(podUnschedulable, frozenTime.Add(time.Second))
		assert.Empty(t, tracker.pods, "Pod should be removed when it becomes unschedulable")

		// Verify no metrics
		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		metrics.PodStartSLIDuration.Reset()
	})

	t.Run("pod observed as unschedulable first, then becomes schedulable should remain not tracked", func(t *testing.T) {
		wants := ""

		fakeClock := testingclock.NewFakeClock(frozenTime)
		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods:         map[types.UID]*perPodState{},
			excludedPods: map[types.UID]bool{},
			clock:        fakeClock,
		}

		// First observe pod as unschedulable
		podUnschedulable := buildInitializingPod()
		podUnschedulable.UID = "unschedulable-first"
		podUnschedulable.Status.Conditions = []corev1.PodCondition{
			{
				Type:   corev1.PodScheduled,
				Status: corev1.ConditionFalse,
				Reason: "Unschedulable",
			},
		}

		tracker.ObservedPodOnWatch(podUnschedulable, frozenTime)
		assert.Empty(t, tracker.pods, "Pod should not be tracked when first observed as unschedulable")
		assert.True(t, tracker.excludedPods[podUnschedulable.UID], "Pod should be in excludedPods map")

		// Later observe the same pod as schedulable (e.g., after cluster autoscaling)
		podSchedulable := buildInitializingPod()
		podSchedulable.UID = "unschedulable-first"
		podSchedulable.Status.Conditions = []corev1.PodCondition{
			{
				Type:   corev1.PodScheduled,
				Status: corev1.ConditionTrue,
				Reason: "Scheduled",
			},
		}

		tracker.ObservedPodOnWatch(podSchedulable, frozenTime.Add(time.Second*5))
		assert.Empty(t, tracker.pods, "Pod should remain excluded even after becoming schedulable")
		assert.True(t, tracker.excludedPods[podSchedulable.UID], "Pod should remain in excludedPods map")

		// Complete the startup process - should not record metrics
		podRunning := buildRunningPod()
		podRunning.UID = "unschedulable-first"
		tracker.RecordStatusUpdated(podRunning)
		tracker.ObservedPodOnWatch(podRunning, frozenTime.Add(time.Second*10))

		// Verify no SLI metrics recorded
		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), metricsName); err != nil {
			t.Fatal(err)
		}

		// cleanup
		tracker.DeletePodStartupState(podRunning.UID)
		assert.Empty(t, tracker.pods)
		assert.False(t, tracker.excludedPods[podRunning.UID], "Pod should be removed from excludedPods on cleanup")
		metrics.PodStartSLIDuration.Reset()
	})
}

func TestStatefulPodExclusion(t *testing.T) {
	t.Run("stateful pods should be excluded from SLI metrics", func(t *testing.T) {
		wantsSLI := ""

		fakeClock := testingclock.NewFakeClock(frozenTime)
		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods: map[types.UID]*perPodState{}, excludedPods: map[types.UID]bool{},
			clock: fakeClock,
		}

		// Create a stateful pod (with PVC volume)
		statefulPod := buildInitializingPod()
		statefulPod.UID = "stateful-pod"
		statefulPod.Spec.Volumes = []corev1.Volume{
			{
				Name: "persistent-storage",
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: "test-pvc",
					},
				},
			},
		}

		tracker.ObservedPodOnWatch(statefulPod, frozenTime)

		statefulPodRunning := buildRunningPod()
		statefulPodRunning.UID = "stateful-pod"
		statefulPodRunning.Spec.Volumes = statefulPod.Spec.Volumes
		tracker.RecordStatusUpdated(statefulPodRunning)

		// Observe pod as running after 3 seconds
		tracker.ObservedPodOnWatch(statefulPodRunning, frozenTime.Add(time.Second*3))

		// Verify no SLI metrics for stateful pod (
		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wantsSLI), metricsName); err != nil {
			t.Fatal(err)
		}

		// cleanup
		tracker.DeletePodStartupState(statefulPod.UID)
		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func TestFirstNetworkPodMetrics(t *testing.T) {

	t.Run("first network pod; started in 30s, image pulling between 10th and 20th seconds", func(t *testing.T) {

		wants := `
# HELP kubelet_first_network_pod_start_sli_duration_seconds [INTERNAL] Duration in seconds to start the first network pod, excluding time to pull images and run init containers, measured from pod creation timestamp to when all its containers are reported as started and observed via watch
# TYPE kubelet_first_network_pod_start_sli_duration_seconds gauge
kubelet_first_network_pod_start_sli_duration_seconds 30
`

		fakeClock := testingclock.NewFakeClock(frozenTime)

		metrics.Register()

		tracker := &basicPodStartupLatencyTracker{
			pods: map[types.UID]*perPodState{}, excludedPods: map[types.UID]bool{},
			clock: fakeClock,
		}

		// hostNetwork pods should not be tracked
		hostNetworkPodInitializing := buildInitializingPod()
		hostNetworkPodInitializing.UID = "11111-22222"
		hostNetworkPodInitializing.Spec.HostNetwork = true
		tracker.ObservedPodOnWatch(hostNetworkPodInitializing, frozenTime)

		hostNetworkPodStarted := buildRunningPod()
		hostNetworkPodStarted.UID = "11111-22222"
		hostNetworkPodStarted.Spec.HostNetwork = true
		tracker.RecordStatusUpdated(hostNetworkPodStarted)

		// track only the first pod with network
		podInitializing := buildInitializingPod()
		tracker.ObservedPodOnWatch(podInitializing, frozenTime)

		// pod started
		podStarted := buildRunningPod()
		tracker.RecordStatusUpdated(podStarted)

		// at 30s observe the same pod on watch
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*30))

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), "kubelet_first_network_pod_start_sli_duration_seconds", "kubelet_first_network_pod_start_total_duration_seconds"); err != nil {
			t.Fatal(err)
		}

		// any new pod observations should not impact the metrics, as the pod should be recorder only once
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*150))
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*200))
		tracker.ObservedPodOnWatch(podStarted, frozenTime.Add(time.Second*250))

		if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(wants), "kubelet_first_network_pod_start_sli_duration_seconds", "kubelet_first_network_pod_start_total_duration_seconds"); err != nil {
			t.Fatal(err)
		}

		// cleanup
		tracker.DeletePodStartupState(hostNetworkPodStarted.UID)
		tracker.DeletePodStartupState(podStarted.UID)

		assert.Empty(t, tracker.pods)
		metrics.PodStartSLIDuration.Reset()
	})
}

func buildInitializingPod(initContainerNames ...string) *corev1.Pod {
	pod := buildPodWithStatus([]corev1.ContainerStatus{
		{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}}},
	})

	// Add init containers if specified
	if len(initContainerNames) > 0 {
		pod.Spec.InitContainers = make([]corev1.Container, len(initContainerNames))
		for i, name := range initContainerNames {
			pod.Spec.InitContainers[i] = corev1.Container{Name: name}
		}
	}

	return pod
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
