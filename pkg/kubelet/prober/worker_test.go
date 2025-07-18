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
	"fmt"
	"testing"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	probe "k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func init() {
}

func ptrBool(b bool) *bool {
	return &b
}

func TestDoProbe(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()

	ginkgo.BeforeEach(func() {
		m = newTestManager()
		m.statusManager = &fakeStatusManager{podStatuses: make(map[types.UID]v1.PodStatus)}
		m.prober.exec = fakeExecProber{result: probe.Success, err: nil}
	})

	for _, probeType := range [...]probeType{liveness, readiness, startup} {
		runningStatus := getTestRunningStatusWithStarted(probeType != startup)
		pendingStatus := getTestRunningStatusWithStarted(probeType != startup)
		pendingStatus.InitContainerStatuses[0].State.Running = nil
		terminatedStatus := getTestRunningStatusWithStarted(probeType != startup)
		terminatedStatus.InitContainerStatuses[0].State.Running = nil
		terminatedStatus.InitContainerStatuses[0].State.Terminated = &v1.ContainerStateTerminated{
			StartedAt: metav1.Now(),
		}
		otherStatus := getTestRunningStatusWithStarted(probeType != startup)
		otherStatus.InitContainerStatuses[0].Name = "otherContainer"
		failedStatus := getTestRunningStatusWithStarted(probeType != startup)
		failedStatus.Phase = v1.PodFailed

		tests := []struct {
			probe                v1.Probe
			podStatus            *v1.PodStatus
			expectContinue       map[string]bool
			expectSet            bool
			expectedResult       results.Result
			setDeletionTimestamp bool
		}{
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus: &runningStatus,
				expectContinue: map[string]bool{
					liveness.String():  true,
					readiness.String(): true,
					startup.String():   true,
				},
				expectSet:      true,
				expectedResult: results.Success,
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus: &failedStatus,
				expectContinue: map[string]bool{
					liveness.String():  false,
					readiness.String(): false,
					startup.String():   false,
				},
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus:            &runningStatus,
				setDeletionTimestamp: true,
				expectSet:            true,
				expectContinue: map[string]bool{
					liveness.String():  false, // Pod deleted, liveness stops
					readiness.String(): true,  // Readiness continues
					startup.String():   false, // Pod deleted, startup stops
				},
				expectedResult: results.Success, // Liveness/Startup set Success on deletion
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus: &otherStatus,
				expectContinue: map[string]bool{
					liveness.String():  true,
					readiness.String(): true,
					startup.String():   true,
				},
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus:      &pendingStatus,
				expectSet:      true,
				expectedResult: results.Failure,
				expectContinue: map[string]bool{
					liveness.String():  true,
					readiness.String(): false, // Non-running, readiness stops
					startup.String():   true,
				},
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus:      &terminatedStatus,
				expectSet:      true,
				expectedResult: results.Failure,
				expectContinue: map[string]bool{
					liveness.String():  false,
					readiness.String(): false,
					startup.String():   false,
				},
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
				},
				podStatus: &runningStatus,
				expectContinue: map[string]bool{
					liveness.String():  true,
					readiness.String(): true,
					startup.String():   true,
				},
				expectSet:      true,
				expectedResult: results.Success,
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
					InitialDelaySeconds: -100,
				},
				podStatus: &runningStatus,
				expectContinue: map[string]bool{
					liveness.String():  true,
					readiness.String(): true,
					startup.String():   true,
				},
				expectSet:      true,
				expectedResult: results.Success,
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
					InitialDelaySeconds: 0,
					PeriodSeconds:       1,
					FailureThreshold:    1,
				},
				podStatus: &v1.PodStatus{
					Phase: v1.PodRunning,
					InitContainerStatuses: []v1.ContainerStatus{
						{
							Name:        testContainerName,
							ContainerID: testContainerID.String(),
							State:       v1.ContainerState{Terminated: &v1.ContainerStateTerminated{StartedAt: metav1.Now()}},
						},
					},
				},
				expectContinue: map[string]bool{
					liveness.String():  false, // Terminated init container, stop probing
					readiness.String(): false, // Terminated init container, stop probing
					startup.String():   false, // Terminated init container, stop probing
				},
				expectSet:      true,
				expectedResult: results.Failure,
			},
			{
				probe: v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
					},
					InitialDelaySeconds: 0,
					PeriodSeconds:       1,
					FailureThreshold:    1,
				},
				podStatus: &v1.PodStatus{
					Phase: v1.PodRunning,
					InitContainerStatuses: []v1.ContainerStatus{
						{
							Name:        testContainerName,
							ContainerID: testContainerID.String(),
							State:       v1.ContainerState{Running: &v1.ContainerStateRunning{StartedAt: metav1.Now()}},
							Started:     ptrBool(true),
						},
					},
				},
				expectContinue: map[string]bool{
					liveness.String():  true,
					readiness.String(): true,
					startup.String():   true,
				},
				expectSet:      true,
				expectedResult: results.Success,
			},
		}

		for i, test := range tests {
			klog.V(4).InfoS("TestDoProbe", "probeType", probeType, "testIndex", i, "podStatus", test.podStatus)
			// Create pod with InitContainers to match newTestWorker
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testPodName,
					Namespace: testNamespace,
					UID:       testPodUID,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:          testContainerName,
							Image:         "foo",
							RestartPolicy: ptrContainerRestartPolicy(v1.ContainerRestartPolicyAlways),
						},
					},
					Containers: []v1.Container{
						{
							Name:  "main-container",
							Image: "bar",
						},
					},
				},
			}
			setTestProbe(pod, probeType, test.probe)
			w := newWorker(m, probeType, pod, pod.Spec.InitContainers[0])
			if test.podStatus != nil {
				m.statusManager.SetPodStatus(logger, w.pod, *test.podStatus)
			} else {
				// Default to running status if not specified
				m.statusManager.SetPodStatus(logger, w.pod, runningStatus)
			}
			if test.setDeletionTimestamp {
				now := metav1.Now()
				w.pod.DeletionTimestamp = &now
			}
			if c := w.doProbe(ctx); c != test.expectContinue[probeType.String()] {
				t.Errorf("[%s-%d] Expected continue to be %v but got %v", probeType, i, test.expectContinue[probeType.String()], c)
			}
			result, ok := resultsManager(m, probeType).Get(testContainerID)
			if ok != test.expectSet {
				t.Errorf("[%s-%d] Expected to have result: %v but got %v", probeType, i, test.expectSet, ok)
			}
			if result != test.expectedResult {
				t.Errorf("[%s-%d] Expected result: %v but got %v", probeType, i, test.expectedResult, result)
			}
			resultsManager(m, probeType).Remove(testContainerID)
		}
	}
}

func TestInitialDelay(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()

	for _, probeType := range [...]probeType{liveness, readiness, startup} {
		w := newTestWorker(m, probeType, v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
			},
			InitialDelaySeconds: 10,
		})
		m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatusWithStarted(probeType != startup))

		expectContinue(t, w, w.doProbe(ctx), "during initial delay")
		switch probeType {
		case liveness:
			expectResult(t, w, results.Success, "during initial delay")
		case readiness:
			expectResult(t, w, results.Failure, "during initial delay")
		case startup:
			expectResult(t, w, results.Unknown, "during initial delay")
		}

		laterStatus := getTestRunningStatusWithStarted(probeType != startup)
		laterStatus.InitContainerStatuses[0].State.Running.StartedAt.Time =
			time.Now().Add(-100 * time.Second)
		m.statusManager.SetPodStatus(logger, w.pod, laterStatus)

		expectContinue(t, w, w.doProbe(ctx), "after initial delay")
		expectResult(t, w, results.Success, "after initial delay")
	}
}

func TestFailureThreshold(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, readiness, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: 1,
		FailureThreshold: 3,
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())

	for i := 0; i < 2; i++ {
		m.prober.exec = fakeExecProber{result: probe.Success, err: nil}

		for j := 0; j < 3; j++ {
			msg := fmt.Sprintf("%d success (%d)", j+1, i)
			expectContinue(t, w, w.doProbe(ctx), msg)
			expectResult(t, w, results.Success, msg)
		}

		m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}

		for j := 0; j < 2; j++ {
			msg := fmt.Sprintf("%d failing (%d)", j+1, i)
			expectContinue(t, w, w.doProbe(ctx), msg)
			expectResult(t, w, results.Success, msg)
		}

		for j := 0; j < 3; j++ {
			msg := fmt.Sprintf("%d failure (%d)", j+3, i)
			expectContinue(t, w, w.doProbe(ctx), msg)
			expectResult(t, w, results.Failure, msg)
		}
	}
}

func TestSuccessThreshold(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, readiness, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: 3,
		FailureThreshold: 1,
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())

	w.resultsManager.Set(testContainerID, results.Failure, &v1.Pod{})

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			msg := fmt.Sprintf("%d success (%d)", j+1, i)
			expectContinue(t, w, w.doProbe(ctx), msg)
			expectResult(t, w, results.Failure, msg)
		}

		for j := 0; j < 3; j++ {
			msg := fmt.Sprintf("%d success (%d)", j+3, i)
			expectContinue(t, w, w.doProbe(ctx), msg)
			expectResult(t, w, results.Success, msg)
		}

		m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
		msg := fmt.Sprintf("1 failure (%d)", i)
		expectContinue(t, w, w.doProbe(ctx), msg)
		expectResult(t, w, results.Failure, msg)

		m.prober.exec = fakeExecProber{result: probe.Success, err: nil}
	}
}

func TestStartupProbeSuccessThreshold(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	successThreshold := 1
	failureThreshold := 3
	w := newTestWorker(m, startup, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: int32(successThreshold),
		FailureThreshold: int32(failureThreshold),
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestNotRunningStatus())
	m.prober.exec = fakeExecProber{result: probe.Success, err: nil}

	for i := 0; i < successThreshold+1; i++ {
		if i < successThreshold {
			if w.onHold {
				t.Errorf("Prober should not be on hold")
			}
			msg := fmt.Sprintf("%d success", i+1)
			expectContinue(t, w, w.doProbe(ctx), msg)
			expectResult(t, w, results.Success, msg)
		} else {
			if !w.onHold {
				t.Errorf("Prober should be on hold because successThreshold is exceeded")
			}
			if w.resultRun != 0 {
				t.Errorf("Prober resultRun should be 0, but %d", w.resultRun)
			}
		}
	}
}

func TestStartupProbeFailureThreshold(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	successThreshold := 1
	failureThreshold := 3
	w := newTestWorker(m, startup, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: int32(successThreshold),
		FailureThreshold: int32(failureThreshold),
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestNotRunningStatus())
	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}

	for i := 0; i < failureThreshold+1; i++ {
		if i < failureThreshold {
			if w.onHold {
				t.Errorf("Prober should not be on hold")
			}
			msg := fmt.Sprintf("%d failure", i+1)
			expectContinue(t, w, w.doProbe(ctx), msg)
			switch i {
			case 0, 1:
				expectResult(t, w, results.Unknown, msg)
				if w.resultRun != i+1 {
					t.Errorf("Prober resultRun should be %d, but %d", i+1, w.resultRun)
				}
			case 2:
				expectResult(t, w, results.Failure, msg)
				if w.resultRun != 0 {
					t.Errorf("Prober resultRun should be 0, but %d", w.resultRun)
				}
			}
		} else {
			if !w.onHold {
				t.Errorf("Prober should be on hold because failureThreshold is exceeded")
			}
			if w.resultRun != 0 {
				t.Errorf("Prober resultRun should be 0, but %d", w.resultRun)
			}
		}
	}
}

func TestCleanUp(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	m := newTestManager()

	for _, probeType := range [...]probeType{liveness, readiness, startup} {
		key := probeKey{testPodUID, testContainerName, probeType}
		w := newTestWorker(m, probeType, v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
			},
		})
		m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatusWithStarted(probeType != startup))
		go w.run()
		m.workers[key] = w

		condition := func() (bool, error) {
			ready, _ := resultsManager(m, probeType).Get(testContainerID)
			return ready == results.Success, nil
		}
		if ready, _ := condition(); !ready {
			if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, condition); err != nil {
				t.Fatalf("[%s] Error waiting for worker ready: %v", probeType, err)
			}
		}

		for i := 0; i < 10; i++ {
			w.stop()
		}
		if err := waitForWorkerExit(t, m, []probeKey{key}); err != nil {
			t.Fatalf("[%s] error waiting for worker exit: %v", probeType, err)
		}

		if _, ok := resultsManager(m, probeType).Get(testContainerID); ok {
			t.Errorf("[%s] Expected result to be cleared.", probeType)
		}
		if _, ok := m.workers[key]; ok {
			t.Errorf("[%s] Expected worker to be cleared.", probeType)
		}
	}
}

func expectResult(t *testing.T, w *worker, expectedResult results.Result, msg string) {
	result, ok := resultsManager(w.probeManager, w.probeType).Get(w.containerID)
	if !ok {
		t.Errorf("[%s - %s] Expected result to be set, but was not set", w.probeType, msg)
	} else if result != expectedResult {
		t.Errorf("[%s - %s] Expected result to be %v, but was %v",
			w.probeType, msg, expectedResult, result)
	}
}

func expectContinue(t *testing.T, w *worker, c bool, msg string) {
	if !c {
		t.Errorf("[%s - %s] Expected to continue, but did not", w.probeType, msg)
	}
}

func resultsManager(m *manager, probeType probeType) results.Manager {
	switch probeType {
	case readiness:
		return m.readinessManager
	case liveness:
		return m.livenessManager
	case startup:
		return m.startupManager
	}
	panic(fmt.Errorf("Unhandled case: %v", probeType))
}

func TestOnHoldOnLivenessOrStartupCheckFailure(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()

	for _, probeType := range [...]probeType{liveness, startup} {
		w := newTestWorker(m, probeType, v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
			},
			SuccessThreshold: 1,
			FailureThreshold: 1,
		})
		status := getTestRunningStatusWithStarted(probeType != startup)
		m.statusManager.SetPodStatus(logger, w.pod, status)

		m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
		msg := "first probe"
		expectContinue(t, w, w.doProbe(ctx), msg)
		expectResult(t, w, results.Failure, msg)
		if !w.onHold {
			t.Errorf("Prober should be on hold due to %s check failure", probeType)
		}
		m.prober.exec = fakeExecProber{result: probe.Success, err: nil}
		msg = "while on hold"
		expectContinue(t, w, w.doProbe(ctx), msg)
		expectResult(t, w, results.Failure, msg)
		if !w.onHold {
			t.Errorf("Prober should be on hold due to %s check failure", probeType)
		}

		status.InitContainerStatuses[0].ContainerID = "test://newCont_ID"
		m.statusManager.SetPodStatus(logger, w.pod, status)
		msg = "hold lifted"
		expectContinue(t, w, w.doProbe(ctx), msg)
		expectResult(t, w, results.Success, msg)
		if probeType == liveness && w.onHold {
			t.Errorf("Prober should not be on hold anymore")
		} else if probeType == startup && !w.onHold {
			t.Errorf("Prober should be on hold due to %s check success", probeType)
		}
	}
}

func TestResultRunOnLivenessCheckFailure(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, liveness, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: 1,
		FailureThreshold: 3,
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())

	m.prober.exec = fakeExecProber{result: probe.Success, err: nil}
	msg := "initial probe success"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Success, msg)
	if w.resultRun != 1 {
		t.Errorf("Prober resultRun should be 1")
	}

	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "probe failure, result success"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Success, msg)
	if w.resultRun != 1 {
		t.Errorf("Prober resultRun should be 1")
	}

	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "2nd probe failure, result success"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Success, msg)
	if w.resultRun != 2 {
		t.Errorf("Prober resultRun should be 2")
	}

	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "3rd probe failure, result failure"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Failure, msg)
	if w.resultRun != 0 {
		t.Errorf("Prober resultRun should be reset to 0")
	}
}

func TestResultRunOnStartupCheckFailure(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, startup, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: 1,
		FailureThreshold: 3,
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatusWithStarted(false))

	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg := "probe failure, result unknown"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Unknown, msg)
	if w.resultRun != 1 {
		t.Errorf("Prober resultRun should be 1")
	}

	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "2nd probe failure, result unknown"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Unknown, msg)
	if w.resultRun != 2 {
		t.Errorf("Prober resultRun should be 2")
	}

	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "3rd probe failure, result failure"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Failure, msg)
	if w.resultRun != 0 {
		t.Errorf("Prober resultRun should be reset to 0")
	}
}

func TestLivenessProbeDisabledByStarted(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, liveness, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: 1,
		FailureThreshold: 1,
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatusWithStarted(false))
	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg := "Not started, probe failure, result success"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Success, msg)
	m.statusManager.SetContainerStartup(logger, w.pod.UID, w.containerID, true)
	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "Started, probe failure, result failure"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Failure, msg)
}

func TestStartupProbeDisabledByStarted(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()
	w := newTestWorker(m, startup, v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
		},
		SuccessThreshold: 1,
		FailureThreshold: 2,
	})
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatusWithStarted(false))
	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg := "Not started, probe failure, result unknown"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Unknown, msg)
	m.prober.exec = fakeExecProber{result: probe.Success, err: nil}
	msg = "Not started, probe success, result success"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Success, msg)
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatusWithStarted(true))
	m.prober.exec = fakeExecProber{result: probe.Failure, err: nil}
	msg = "Started, probe failure, result success"
	expectContinue(t, w, w.doProbe(ctx), msg)
	expectResult(t, w, results.Success, msg)
}
