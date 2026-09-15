/*
Copyright The Kubernetes Authors.

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
	"testing/synctest"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"
	utilexec "k8s.io/utils/exec"
)

type timedExecProbe struct{ calls chan time.Time }

func (p *timedExecProbe) Probe(utilexec.Cmd) (probe.Result, string, error) {
	p.calls <- time.Now()
	time.Sleep(3 * time.Second)
	return probe.Success, "", nil
}

func TestProbeScheduleFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%v", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, enabled)

			synctest.Test(t, func(t *testing.T) {
				logger, ctx := ktesting.NewTestContext(t)
				m := newTestManager()
				m.start = time.Now().Add(-time.Hour)
				p := &timedExecProbe{calls: make(chan time.Time, 4)}
				m.prober.exec = p
				w := newTestWorker(m, readiness, v1.Probe{PeriodSeconds: 10})
				m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())
				done := make(chan struct{})
				go func() { defer close(done); w.run(ctx) }()
				defer func() { w.stop(); <-done }()

				first, second := <-p.calls, <-p.calls
				want := 10 * time.Second
				if enabled {
					want += 3 * time.Second
				}

				if got := second.Sub(first); got != want {
					t.Fatalf("probe interval=%v, want %v with MutableContainerProbes=%v", got, want, enabled)
				}
			})
		})
	}
}

func TestProbeStopFeatureGate(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%v", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, enabled)

			synctest.Test(t, func(t *testing.T) {
				logger, ctx := ktesting.NewTestContext(t)
				m := newTestManager()
				m.start = time.Now().Add(-time.Hour)
				runner := &heldProbeRunner{calls: make(chan heldProbeCall, 4), release: make(chan struct{})}
				m.prober = newProber(runner, &record.FakeRecorder{})
				w := newTestWorker(m, readiness, v1.Probe{PeriodSeconds: 10})
				w.spec.Exec.Command = []string{"check"}
				m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())
				key := probeKey{w.pod.UID, w.container.Name, readiness}
				m.workers[key] = w
				done := make(chan struct{})
				go func() { defer close(done); w.run(ctx) }()
				call := <-runner.calls
				defer func() { close(runner.release); <-done }()

				m.RemovePod(w.pod)
				canceled := false
				select {
				case <-call.canceled:
					canceled = true
				default:
				}

				if canceled != enabled {
					t.Errorf("in-flight cancellation=%v, want %v", canceled, enabled)
				}

				_, cached := m.readinessManager.Get(testContainerID)
				if cached == enabled {
					t.Errorf("result retained before call returns=%v, want %v", cached, !enabled)
				}

				// Legacy removal only signals the loop; the worker remains visible until it exits.
				_, visible := m.getWorker(w.pod.UID, w.container.Name, readiness)
				if visible == enabled {
					t.Errorf("worker visible before call returns=%v, want %v", visible, !enabled)
				}
			})
		})
	}
}

func TestAddPodRetainsLegacyLifecycle(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutableContainerProbes, false)

	synctest.Test(t, func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		m := newTestManager()
		m.start = time.Now().Add(-time.Hour)
		pod := getTestPod()
		setTestProbe(pod, readiness, v1.Probe{PeriodSeconds: 10})
		defer func() {
			m.RemovePod(pod)
			for m.workerCount() != 0 {
				time.Sleep(time.Second)
			}
		}()

		m.AddPod(ctx, pod)
		w, ok := m.getWorker(pod.UID, testContainerName, readiness)
		if !ok || w.mutable {
			t.Fatal("AddPod did not create a legacy worker")
		}

		changed := pod.DeepCopy()
		changed.Spec.Containers[0].ReadinessProbe = nil
		m.AddPod(ctx, changed)

		if current, ok := m.getWorker(pod.UID, testContainerName, readiness); !ok || current != w {
			t.Fatal("AddPod reconciled an existing probe")
		}

		m.RemovePod(pod)
		for m.workerCount() != 0 {
			time.Sleep(time.Second)
		}

		// The old AddPod path has no termination tombstone or terminal-pod admission filter.
		pod.Status.Phase = v1.PodSucceeded
		m.AddPod(ctx, pod)

		if _, ok := m.getWorker(pod.UID, testContainerName, readiness); !ok {
			t.Fatal("legacy AddPod rejected a pod based on mutable lifecycle state")
		}
		if m.IsResultCurrent(results.Update{ProbeID: 42}, readiness) != true {
			t.Fatal("legacy manager filtered a result")
		}
	})
}
