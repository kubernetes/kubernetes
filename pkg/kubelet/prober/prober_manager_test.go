/*
Copyright 2015 The Kubernetes Authors.

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

package prober

import (
	"context"
	"fmt"
	"strconv"
	"testing"
	"testing/synctest"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func init() {
}

var defaultProbe = &v1.Probe{
	ProbeHandler: v1.ProbeHandler{
		Exec: &v1.ExecAction{},
	},
	TimeoutSeconds:   1,
	PeriodSeconds:    1,
	SuccessThreshold: 1,
	FailureThreshold: 3,
}

func TestAddRemovePods(t *testing.T) {
	ktesting.Init(t).SyncTest("", testAddRemovePods)
}

func testAddRemovePods(tCtx ktesting.TContext) {
	t := tCtx.TB()

	noProbePod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "no_probe_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "no_probe1",
			}, {
				Name: "no_probe2",
			}, {
				Name: "no_probe3",
			}},
		},
	}

	probePod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "probe_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "probe1",
			}, {
				Name:           "readiness",
				ReadinessProbe: defaultProbe,
			}, {
				Name: "probe2",
			}, {
				Name:          "liveness",
				LivenessProbe: defaultProbe,
			}, {
				Name: "probe3",
			}, {
				Name:         "startup",
				StartupProbe: defaultProbe,
			}},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	// Adding a pod with no probes should be a no-op.
	m.AddPod(tCtx, &noProbePod)
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	// Adding a pod with probes.
	m.AddPod(tCtx, &probePod)
	probePaths := []probeKey{
		{"probe_pod", "readiness", readiness},
		{"probe_pod", "liveness", liveness},
		{"probe_pod", "startup", startup},
	}
	if err := expectProbes(m, probePaths); err != nil {
		t.Error(err)
	}

	// Removing un-probed pod.
	m.RemovePod(&noProbePod)
	if err := expectProbes(m, probePaths); err != nil {
		t.Error(err)
	}

	// Removing probed pod.
	m.RemovePod(&probePod)
	if err := waitForWorkerExit(t, m, probePaths); err != nil {
		t.Fatal(err)
	}
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	// Removing already removed pods should be a no-op.
	m.RemovePod(&probePod)
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}
}

func TestAddPodContinuesAfterExistingWorker(t *testing.T) {
	ktesting.Init(t).SyncTest("", testAddPodContinuesAfterExistingWorker)
}

func testAddPodContinuesAfterExistingWorker(tCtx ktesting.TContext) {
	defer tCtx.Cancel("test completed")
	t := tCtx.TB()
	ctx := tCtx.Context

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "test_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           "container_a",
					ReadinessProbe: defaultProbe,
				},
				{
					Name:           "container_b",
					ReadinessProbe: defaultProbe,
				},
			},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)

	// First AddPod: registers workers for both containers.
	m.AddPod(ctx, &pod)
	if err := expectProbes(m, []probeKey{
		{"test_pod", "container_a", readiness},
		{"test_pod", "container_b", readiness},
	}); err != nil {
		t.Fatalf("after first AddPod: %v", err)
	}

	// Simulate container_b's worker being removed while container_a's is still present.
	m.workers[probeKey{"test_pod", "container_b", readiness}].stop()
	synctest.Wait()

	// Second AddPod: should re-register container_b's missing worker.
	// Previously, hitting container_a's existing worker caused an early return,
	// so container_b was never re-registered.
	m.AddPod(ctx, &pod)

	if err := expectProbes(m, []probeKey{
		{"test_pod", "container_a", readiness},
		{"test_pod", "container_b", readiness},
	}); err != nil {
		t.Errorf("container_b worker was not re-registered after second AddPod: %v", err)
	}
}

func TestAddRemovePodsWithRestartableInitContainer(t *testing.T) {
	tCtx := ktesting.Init(t)
	m := newTestManager()
	defer cleanup(t, m)
	if err := expectProbes(m, nil); err != nil {
		t.Error(err)
	}

	testCases := []struct {
		desc                        string
		probePaths                  []probeKey
		hasRestartableInitContainer bool
	}{
		{
			desc:                        "pod without sidecar",
			probePaths:                  nil,
			hasRestartableInitContainer: false,
		},
		{
			desc: "pod with sidecar",
			probePaths: []probeKey{
				{"restartable_init_container_pod", "restartable-init", liveness},
				{"restartable_init_container_pod", "restartable-init", readiness},
				{"restartable_init_container_pod", "restartable-init", startup},
			},
			hasRestartableInitContainer: true,
		},
	}

	containerRestartPolicy := func(hasRestartableInitContainer bool) *v1.ContainerRestartPolicy {
		if !hasRestartableInitContainer {
			return nil
		}
		restartPolicy := v1.ContainerRestartPolicyAlways
		return &restartPolicy
	}

	for _, tc := range testCases {
		tCtx.SyncTest(tc.desc, func(tCtx ktesting.TContext) {
			t := tCtx.TB()
			probePod := v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: "restartable_init_container_pod",
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{{
						Name: "init",
					}, {
						Name:           "restartable-init",
						LivenessProbe:  defaultProbe,
						ReadinessProbe: defaultProbe,
						StartupProbe:   defaultProbe,
						RestartPolicy:  containerRestartPolicy(tc.hasRestartableInitContainer),
					}},
					Containers: []v1.Container{{
						Name: "main",
					}},
				},
			}

			// Adding a pod with probes.
			m.AddPod(tCtx, &probePod)
			if err := expectProbes(m, tc.probePaths); err != nil {
				t.Error(err)
			}

			// Removing probed pod.
			m.RemovePod(&probePod)
			if err := waitForWorkerExit(t, m, tc.probePaths); err != nil {
				t.Fatal(err)
			}
			if err := expectProbes(m, nil); err != nil {
				t.Error(err)
			}

			// Removing already removed pods should be a no-op.
			m.RemovePod(&probePod)
			if err := expectProbes(m, nil); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestCleanupPods(t *testing.T) {
	ktesting.Init(t).SyncTest("", testCleanupPods)
}

func testCleanupPods(tCtx ktesting.TContext) {
	t := tCtx.TB()
	m := newTestManager()
	defer cleanup(t, m)
	podToCleanup := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "pod_cleanup",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:           "prober1",
				ReadinessProbe: defaultProbe,
			}, {
				Name:          "prober2",
				LivenessProbe: defaultProbe,
			}, {
				Name:         "prober3",
				StartupProbe: defaultProbe,
			}},
		},
	}
	podToKeep := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "pod_keep",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:           "prober1",
				ReadinessProbe: defaultProbe,
			}, {
				Name:          "prober2",
				LivenessProbe: defaultProbe,
			}, {
				Name:         "prober3",
				StartupProbe: defaultProbe,
			}},
		},
	}
	m.AddPod(tCtx, &podToCleanup)
	m.AddPod(tCtx, &podToKeep)

	desiredPods := map[types.UID]sets.Empty{}
	desiredPods[podToKeep.UID] = sets.Empty{}
	m.CleanupPods(desiredPods)

	removedProbes := []probeKey{
		{"pod_cleanup", "prober1", readiness},
		{"pod_cleanup", "prober2", liveness},
		{"pod_cleanup", "prober3", startup},
	}
	expectedProbes := []probeKey{
		{"pod_keep", "prober1", readiness},
		{"pod_keep", "prober2", liveness},
		{"pod_keep", "prober3", startup},
	}
	if err := waitForWorkerExit(t, m, removedProbes); err != nil {
		t.Fatal(err)
	}
	if err := expectProbes(m, expectedProbes); err != nil {
		t.Error(err)
	}
}

func TestCleanupRepeated(t *testing.T) {
	ktesting.Init(t).SyncTest("", testCleanupRepeated)
}

func testCleanupRepeated(tCtx ktesting.TContext) {
	t := tCtx.TB()
	m := newTestManager()
	defer cleanup(t, m)
	podTemplate := v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:           "prober1",
				ReadinessProbe: defaultProbe,
				LivenessProbe:  defaultProbe,
				StartupProbe:   defaultProbe,
			}},
		},
	}

	const numTestPods = 100
	for i := 0; i < numTestPods; i++ {
		pod := podTemplate
		pod.UID = types.UID(strconv.Itoa(i))
		m.AddPod(tCtx, &pod)
	}

	for i := 0; i < 10; i++ {
		m.CleanupPods(map[types.UID]sets.Empty{})
	}
}

func TestUpdatePodStatus(t *testing.T) {
	ctx := ktesting.Init(t)
	logger := ctx.Logger()
	unprobed := v1.ContainerStatus{
		Name:        "unprobed_container",
		ContainerID: "test://unprobed_container_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	probedReady := v1.ContainerStatus{
		Name:        "probed_container_ready",
		ContainerID: "test://probed_container_ready_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	probedPending := v1.ContainerStatus{
		Name:        "probed_container_pending",
		ContainerID: "test://probed_container_pending_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	probedUnready := v1.ContainerStatus{
		Name:        "probed_container_unready",
		ContainerID: "test://probed_container_unready_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	notStartedNoReadiness := v1.ContainerStatus{
		Name:        "not_started_container_no_readiness",
		ContainerID: "test://not_started_container_no_readiness_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	startedNoReadiness := v1.ContainerStatus{
		Name:        "started_container_no_readiness",
		ContainerID: "test://started_container_no_readiness_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	terminated := v1.ContainerStatus{
		Name:        "terminated_container",
		ContainerID: "test://terminated_container_id",
		State: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{},
		},
	}
	podStatus := v1.PodStatus{
		Phase: v1.PodRunning,
		ContainerStatuses: []v1.ContainerStatus{
			unprobed, probedReady, probedPending, probedUnready, notStartedNoReadiness, startedNoReadiness, terminated,
		},
	}

	m := newTestManager()
	// no cleanup: using fake workers.

	// Setup probe "workers" and cached results.
	m.workers = map[probeKey]*worker{
		{testPodUID, unprobed.Name, liveness}:             {},
		{testPodUID, probedReady.Name, readiness}:         {},
		{testPodUID, probedPending.Name, readiness}:       {},
		{testPodUID, probedUnready.Name, readiness}:       {},
		{testPodUID, notStartedNoReadiness.Name, startup}: {},
		{testPodUID, startedNoReadiness.Name, startup}:    {},
		{testPodUID, terminated.Name, readiness}:          {},
	}
	m.readinessManager.Set(kubecontainer.ParseContainerID(logger, probedReady.ContainerID), results.Success, &v1.Pod{})
	m.readinessManager.Set(kubecontainer.ParseContainerID(logger, probedUnready.ContainerID), results.Failure, &v1.Pod{})
	m.startupManager.Set(kubecontainer.ParseContainerID(logger, startedNoReadiness.ContainerID), results.Success, &v1.Pod{})
	m.readinessManager.Set(kubecontainer.ParseContainerID(logger, terminated.ContainerID), results.Success, &v1.Pod{})

	m.UpdatePodStatus(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: testPodUID,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: unprobed.Name},
				{Name: probedReady.Name},
				{Name: probedPending.Name},
				{Name: probedUnready.Name},
				{Name: notStartedNoReadiness.Name},
				{Name: startedNoReadiness.Name},
				{Name: terminated.Name},
			},
		},
	}, &podStatus)

	expectedReadiness := map[probeKey]bool{
		{testPodUID, unprobed.Name, readiness}:              true,
		{testPodUID, probedReady.Name, readiness}:           true,
		{testPodUID, probedPending.Name, readiness}:         false,
		{testPodUID, probedUnready.Name, readiness}:         false,
		{testPodUID, notStartedNoReadiness.Name, readiness}: false,
		{testPodUID, startedNoReadiness.Name, readiness}:    true,
		{testPodUID, terminated.Name, readiness}:            false,
	}
	for _, c := range podStatus.ContainerStatuses {
		expected, ok := expectedReadiness[probeKey{testPodUID, c.Name, readiness}]
		if !ok {
			t.Fatalf("Missing expectation for test case: %v", c.Name)
		}
		if expected != c.Ready {
			t.Errorf("Unexpected readiness for container %v: Expected %v but got %v",
				c.Name, expected, c.Ready)
		}
	}
}

func TestUpdatePodStatusWithInitContainers(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	notStarted := v1.ContainerStatus{
		Name:        "not_started_container",
		ContainerID: "test://not_started_container_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	started := v1.ContainerStatus{
		Name:        "started_container",
		ContainerID: "test://started_container_id",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
	terminated := v1.ContainerStatus{
		Name:        "terminated_container",
		ContainerID: "test://terminated_container_id",
		State: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{},
		},
	}

	m := newTestManager()
	// no cleanup: using fake workers.

	// Setup probe "workers" and cached results.
	m.workers = map[probeKey]*worker{
		{testPodUID, notStarted.Name, startup}: {},
		{testPodUID, started.Name, startup}:    {},
	}
	m.startupManager.Set(kubecontainer.ParseContainerID(logger, started.ContainerID), results.Success, &v1.Pod{})

	testCases := []struct {
		desc                        string
		expectedStartup             map[probeKey]bool
		expectedReadiness           map[probeKey]bool
		hasRestartableInitContainer bool
	}{
		{
			desc: "init containers",
			expectedStartup: map[probeKey]bool{
				{testPodUID, notStarted.Name, startup}: false,
				{testPodUID, started.Name, startup}:    true,
				{testPodUID, terminated.Name, startup}: false,
			},
			expectedReadiness: map[probeKey]bool{
				{testPodUID, notStarted.Name, readiness}: false,
				{testPodUID, started.Name, readiness}:    false,
				{testPodUID, terminated.Name, readiness}: true,
			},
			hasRestartableInitContainer: false,
		},
		{
			desc: "init container with Always restartPolicy",
			expectedStartup: map[probeKey]bool{
				{testPodUID, notStarted.Name, startup}: false,
				{testPodUID, started.Name, startup}:    true,
				{testPodUID, terminated.Name, startup}: false,
			},
			expectedReadiness: map[probeKey]bool{
				{testPodUID, notStarted.Name, readiness}: false,
				{testPodUID, started.Name, readiness}:    true,
				{testPodUID, terminated.Name, readiness}: false,
			},
			hasRestartableInitContainer: true,
		},
	}

	containerRestartPolicy := func(enableSidecarContainers bool) *v1.ContainerRestartPolicy {
		if !enableSidecarContainers {
			return nil
		}
		restartPolicy := v1.ContainerRestartPolicyAlways
		return &restartPolicy
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx := ktesting.Init(t)
			podStatus := v1.PodStatus{
				Phase: v1.PodRunning,
				InitContainerStatuses: []v1.ContainerStatus{
					notStarted, started, terminated,
				},
			}

			m.UpdatePodStatus(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID: testPodUID,
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          notStarted.Name,
							RestartPolicy: containerRestartPolicy(tc.hasRestartableInitContainer),
						},
						{
							Name:          started.Name,
							RestartPolicy: containerRestartPolicy(tc.hasRestartableInitContainer),
						},
						{
							Name:          terminated.Name,
							RestartPolicy: containerRestartPolicy(tc.hasRestartableInitContainer),
						},
					},
				},
			}, &podStatus)

			for _, c := range podStatus.InitContainerStatuses {
				{
					expected, ok := tc.expectedStartup[probeKey{testPodUID, c.Name, startup}]
					if !ok {
						t.Fatalf("Missing expectation for test case: %v", c.Name)
					}
					if expected != *c.Started {
						t.Errorf("Unexpected startup for container %v: Expected %v but got %v",
							c.Name, expected, *c.Started)
					}
				}
				{
					expected, ok := tc.expectedReadiness[probeKey{testPodUID, c.Name, readiness}]
					if !ok {
						t.Fatalf("Missing expectation for test case: %v", c.Name)
					}
					if expected != c.Ready {
						t.Errorf("Unexpected readiness for container %v: Expected %v but got %v",
							c.Name, expected, c.Ready)
					}
				}
			}
		})
	}
}

func (m *manager) extractedReadinessHandling(logger klog.Logger) {
	update := <-m.readinessManager.Updates()
	// This code corresponds to an extract from kubelet.syncLoopIteration()
	ready := update.Result == results.Success
	m.statusManager.SetContainerReadiness(logger, update.PodUID, update.ContainerID, ready)
}

func TestUpdateReadiness(t *testing.T) {
	ktesting.Init(t).SyncTest("", testUpdateReadiness)
}

func testUpdateReadiness(tCtx ktesting.TContext) {
	t := tCtx.TB()
	logger := tCtx.Logger()
	testPod := getTestPod()
	setTestProbe(testPod, readiness, v1.Probe{})
	m := newTestManager()
	defer cleanup(t, m)

	// Start syncing readiness without leaking goroutine.
	stopCh := make(chan struct{})
	go wait.Until(func() { m.extractedReadinessHandling(logger) }, 0, stopCh)
	defer func() {
		close(stopCh)
		// Send an update to exit extractedReadinessHandling()
		m.readinessManager.Set(kubecontainer.ContainerID{}, results.Success, &v1.Pod{})
	}()

	exec := syncExecProber{}
	exec.set(probe.Success, nil)
	m.prober.exec = &exec

	m.statusManager.SetPodStatus(logger, testPod, getTestRunningStatus())

	m.AddPod(tCtx, testPod)
	m.StartContainerProbes(tCtx.Context, testPod, &testPod.Spec.Containers[0], testContainerID, "", time.Now())
	probePaths := []probeKey{{testPodUID, testContainerName, readiness}}
	if err := expectProbes(m, probePaths); err != nil {
		t.Error(err)
	}

	// Wait for ready status.
	if err := waitForReadyStatus(t, m, true); err != nil {
		t.Error(err)
	}

	// Prober fails.
	exec.set(probe.Failure, nil)

	// Wait for failed status.
	if err := waitForReadyStatus(t, m, false); err != nil {
		t.Error(err)
	}
}

func expectProbes(m *manager, expectedProbes []probeKey) error {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	var unexpected []probeKey
	missing := make([]probeKey, len(expectedProbes))
	copy(missing, expectedProbes)

outer:
	for probePath := range m.workers {
		for i, expectedPath := range missing {
			if probePath == expectedPath {
				missing = append(missing[:i], missing[i+1:]...)
				continue outer
			}
		}
		unexpected = append(unexpected, probePath)
	}

	if len(missing) == 0 && len(unexpected) == 0 {
		return nil // Yay!
	}

	return fmt.Errorf("Unexpected probes: %v; Missing probes: %v;", unexpected, missing)
}

const interval = 1 * time.Second

// Wait for the given workers to exit & clean up.
func waitForWorkerExit(t ktesting.TB, m *manager, workerPaths []probeKey) error {
	for _, w := range workerPaths {
		condition := func() (bool, error) {
			_, exists := m.getWorker(w.podUID, w.containerName, w.probeType)
			return !exists, nil
		}
		if exited, _ := condition(); exited {
			continue // Already exited, no need to poll.
		}
		t.Logf("Polling %v", w)
		if err := wait.Poll(interval, wait.ForeverTestTimeout, condition); err != nil {
			return err
		}
	}

	return nil
}

// Wait for the given workers to exit & clean up.
func waitForReadyStatus(t ktesting.TB, m *manager, ready bool) error {
	condition := func() (bool, error) {
		status, ok := m.statusManager.GetPodStatus(testPodUID)
		if !ok {
			return false, fmt.Errorf("status not found: %q", testPodUID)
		}
		if len(status.ContainerStatuses) != 1 {
			return false, fmt.Errorf("expected single container, found %d", len(status.ContainerStatuses))
		}
		if status.ContainerStatuses[0].ContainerID != testContainerID.String() {
			return false, fmt.Errorf("expected container %q, found %q",
				testContainerID, status.ContainerStatuses[0].ContainerID)
		}
		return status.ContainerStatuses[0].Ready == ready, nil
	}
	t.Logf("Polling for ready state %v", ready)
	if err := wait.Poll(interval, wait.ForeverTestTimeout, condition); err != nil {
		return err
	}

	return nil
}

// cleanup running probes to avoid leaking goroutines.
func cleanup(t ktesting.TB, m *manager) {
	m.CleanupPods(nil)

	condition := func() (bool, error) {
		workerCount := m.workerCount()
		if workerCount > 0 {
			t.Logf("Waiting for %d workers to exit...", workerCount)
		}
		return workerCount == 0, nil
	}
	if exited, _ := condition(); exited {
		return // Already exited, no need to poll.
	}
	if err := wait.Poll(interval, wait.ForeverTestTimeout, condition); err != nil {
		t.Fatalf("Error during cleanup: %v", err)
	}
}

func TestStartContainerProbes(t *testing.T) {
	// Runs with the feature on and off. With the feature on, the probe workers are
	// activated by StartContainerProbes; with it off, they are launched by AddPod and
	// StartContainerProbes is a no-op. Either way the expected workers must be running.
	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("featureEnabled=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerLifecycleProber, enabled)
			ktesting.Init(t).SyncTest("", testStartContainerProbes)
		})
	}
}

func testStartContainerProbes(tCtx ktesting.TContext) {
	t := tCtx.TB()
	ctx := tCtx.Context

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "start_probe_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           "with_startup",
					StartupProbe:   defaultProbe,
					ReadinessProbe: defaultProbe,
					LivenessProbe:  defaultProbe,
				},
				{
					Name:           "without_startup",
					ReadinessProbe: defaultProbe,
					LivenessProbe:  defaultProbe,
				},
			},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)

	m.AddPod(ctx, &pod)

	c1ID := kubecontainer.ContainerID{Type: "docker", ID: "cid1"}
	c2ID := kubecontainer.ContainerID{Type: "docker", ID: "cid2"}
	now := time.Now()

	// Calling StartContainerProbes for container with startup probe starts startup probe worker.
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], c1ID, "10.0.0.1", now)

	w1Startup, ok := m.getWorker("start_probe_pod", "with_startup", startup)
	if !ok || !w1Startup.running.Load() {
		t.Errorf("expected startup worker to be running for with_startup container")
	}

	// Calling StartContainerProbes for container without startup probe starts readiness and liveness workers.
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[1], c2ID, "10.0.0.1", now)

	w2Readiness, ok := m.getWorker("start_probe_pod", "without_startup", readiness)
	if !ok || !w2Readiness.running.Load() {
		t.Errorf("expected readiness worker to be running for without_startup container")
	}
	w2Liveness, ok := m.getWorker("start_probe_pod", "without_startup", liveness)
	if !ok || !w2Liveness.running.Load() {
		t.Errorf("expected liveness worker to be running for without_startup container")
	}

	// StartContainerProbes is idempotent.
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[1], c2ID, "10.0.0.1", now)
}

func TestStopContainerProbes(t *testing.T) {
	// Runs with the feature on and off: StopContainerProbes must tear down the target
	// container's workers regardless of how they were started (StartContainerProbes when on,
	// AddPod when off).
	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("featureEnabled=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerLifecycleProber, enabled)
			ktesting.Init(t).SyncTest("", testStopContainerProbes)
		})
	}
}

func testStopContainerProbes(tCtx ktesting.TContext) {
	t := tCtx.TB()
	ctx := tCtx.Context

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "stop_probe_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           "container1",
					ReadinessProbe: defaultProbe,
				},
				{
					Name:           "container2",
					ReadinessProbe: defaultProbe,
				},
			},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)

	m.AddPod(ctx, &pod)

	c1ID := kubecontainer.ContainerID{Type: "docker", ID: "cid1"}
	c2ID := kubecontainer.ContainerID{Type: "docker", ID: "cid2"}
	now := time.Now()

	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], c1ID, "10.0.0.1", now)
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[1], c2ID, "10.0.0.1", now)

	// Stopping container1 should stop and remove container1's workers, but keep container2.
	m.StopContainerProbes("stop_probe_pod", "container1")

	if err := waitForWorkerExit(t, m, []probeKey{{"stop_probe_pod", "container1", readiness}}); err != nil {
		t.Fatalf("expected container1 worker to exit: %v", err)
	}

	if _, ok := m.getWorker("stop_probe_pod", "container2", readiness); !ok {
		t.Errorf("expected container2 worker to still exist")
	}
}

// TestStartContainerProbesCrashRestart verifies that calling StartContainerProbes with a
// different containerID for the same container (a crash-restart that did not go through
// StopContainerProbes) tears down the stale worker and starts a fresh one bound to the new
// instance.
func TestStartContainerProbesCrashRestart(t *testing.T) {
	ktesting.Init(t).SyncTest("", testStartContainerProbesCrashRestart)
}

func testStartContainerProbesCrashRestart(tCtx ktesting.TContext) {
	t := tCtx.TB()
	ctx := tCtx.Context

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "crash_restart_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:          "container1",
					LivenessProbe: defaultProbe,
				},
			},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)

	m.AddPod(ctx, &pod)

	oldID := kubecontainer.ContainerID{Type: "docker", ID: "cid-old"}
	newID := kubecontainer.ContainerID{Type: "docker", ID: "cid-new"}
	now := time.Now()

	// First instance.
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], oldID, "10.0.0.1", now)

	oldWorker, ok := m.getWorker("crash_restart_pod", "container1", liveness)
	if !ok || !oldWorker.running.Load() {
		t.Fatalf("expected liveness worker running for first instance")
	}
	if oldWorker.containerID != oldID {
		t.Fatalf("expected worker bound to %v, got %v", oldID, oldWorker.containerID)
	}

	// Second instance with a different containerID, without an intervening StopContainerProbes.
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], newID, "10.0.0.1", now)

	newWorker, ok := m.getWorker("crash_restart_pod", "container1", liveness)
	if !ok || !newWorker.running.Load() {
		t.Fatalf("expected liveness worker running for second instance")
	}
	if newWorker == oldWorker {
		t.Errorf("expected a fresh worker for the new container instance, got the stale one")
	}
	if newWorker.containerID != newID {
		t.Errorf("expected new worker bound to %v, got %v", newID, newWorker.containerID)
	}

	// The stale worker's goroutine must eventually exit.
	if err := wait.PollUntilContextTimeout(ctx, interval, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		return !oldWorker.running.Load(), nil
	}); err != nil {
		t.Errorf("stale worker did not stop: %v", err)
	}
}

// TestStartContainerProbesCrashRestartWithStartupProbe verifies that when a container with a startup
// probe and liveness probe crash-restarts without an intervening StopContainerProbes, the exiting
// stale liveness worker does not delete the replacement worker entry before the new startup probe
// completes and activates dependents.
func TestStartContainerProbesCrashRestartWithStartupProbe(t *testing.T) {
	ktesting.Init(t).SyncTest("", testStartContainerProbesCrashRestartWithStartupProbe)
}

func testStartContainerProbesCrashRestartWithStartupProbe(tCtx ktesting.TContext) {
	t := tCtx.TB()
	ctx := tCtx.Context

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "crash_restart_startup_pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:          "container1",
					StartupProbe:  defaultProbe,
					LivenessProbe: defaultProbe,
				},
			},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)

	m.AddPod(ctx, &pod)

	oldID := kubecontainer.ContainerID{Type: "docker", ID: "cid-old"}
	newID := kubecontainer.ContainerID{Type: "docker", ID: "cid-new"}
	now := time.Now()

	// First instance starts container probes (activates startup probe).
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], oldID, "10.0.0.1", now)

	// Simulate startup probe completing and activating liveness/readiness for the first instance.
	m.startLivenessAndReadiness(ctx, &pod, &pod.Spec.Containers[0], oldID, "10.0.0.1", now)

	oldLivenessWorker, ok := m.getWorker("crash_restart_startup_pod", "container1", liveness)
	if !ok || !oldLivenessWorker.running.Load() {
		t.Fatalf("expected active liveness worker for first instance")
	}

	// Crash-restart: StartContainerProbes called for newID without StopContainerProbes.
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], newID, "10.0.0.1", now)

	// Wait for the stale liveness worker goroutine from oldID to stop and exit.
	if err := wait.PollUntilContextTimeout(ctx, interval, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		return !oldLivenessWorker.running.Load(), nil
	}); err != nil {
		t.Fatalf("stale liveness worker did not stop: %v", err)
	}

	// Simulate startup probe succeeding for the new instance and activating dependents.
	m.startLivenessAndReadiness(ctx, &pod, &pod.Spec.Containers[0], newID, "10.0.0.1", now)

	newLivenessWorker, ok := m.getWorker("crash_restart_startup_pod", "container1", liveness)
	if !ok || !newLivenessWorker.running.Load() {
		t.Fatalf("expected active liveness worker for second instance after startup probe completed")
	}
	if newLivenessWorker == oldLivenessWorker {
		t.Errorf("expected a fresh worker for the new container instance, got the stale one")
	}
	if newLivenessWorker.containerID != newID {
		t.Errorf("expected new worker bound to %v, got %v", newID, newLivenessWorker.containerID)
	}
}

// TestUnstartedWorkerStopAndStart verifies that stopping an unstarted worker (e.g. before
// StartContainerProbes is called) and then starting probes for the container cleanly allocates
// and runs a fresh worker without map corruption or lost workers.
func TestUnstartedWorkerStopAndStart(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.ContainerLifecycleProber: true,
	})
	ktesting.Init(t).SyncTest("", testUnstartedWorkerStopAndStart)
}

func testUnstartedWorkerStopAndStart(tCtx ktesting.TContext) {
	t := tCtx.TB()
	ctx := tCtx.Context

	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "unstarted_stop_pod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:          "container1",
					LivenessProbe: defaultProbe,
				},
			},
		},
	}

	m := newTestManager()
	defer cleanup(t, m)

	// AddPod creates the unstarted placeholder worker.
	m.AddPod(ctx, &pod)

	placeholderWorker, ok := m.getWorker("unstarted_stop_pod", "container1", liveness)
	if !ok {
		t.Fatalf("expected placeholder worker after AddPod")
	}
	if placeholderWorker.running.Load() {
		t.Fatalf("expected worker not to be running before StartContainerProbes")
	}

	// Stop container probes while the worker is unstarted.
	m.StopContainerProbes("unstarted_stop_pod", "container1")

	// Start probes for the container instance.
	cID := kubecontainer.ContainerID{Type: "docker", ID: "cid-fresh"}
	now := time.Now()
	m.StartContainerProbes(ctx, &pod, &pod.Spec.Containers[0], cID, "10.0.0.1", now)

	activeWorker, ok := m.getWorker("unstarted_stop_pod", "container1", liveness)
	if !ok || !activeWorker.running.Load() {
		t.Fatalf("expected active running worker after StartContainerProbes")
	}
	if activeWorker == placeholderWorker {
		t.Errorf("expected fresh worker instance, got original placeholder")
	}
	if activeWorker.containerID != cID {
		t.Errorf("expected worker bound to %v, got %v", cID, activeWorker.containerID)
	}
}
