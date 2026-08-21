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
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/exec"
)

// countingExecProber returns a canned result and counts how many probes ran.
type countingExecProber struct {
	mu     sync.Mutex
	result probe.Result
	calls  int
}

func (p *countingExecProber) Probe(exec.Cmd) (probe.Result, string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	return p.result, "", nil
}

func (p *countingExecProber) set(result probe.Result) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.result = result
}

func (p *countingExecProber) count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

// instanceBoundWorkerFixture wires an instanceBoundWorker to a fake probe
// handler and records the callbacks the manager would normally receive.
type instanceBoundWorkerFixture struct {
	worker  *instanceBoundWorker
	exec    *countingExecProber
	results results.Manager

	exited           chan struct{}
	startupSucceeded chan struct{}
}

func newInstanceBoundWorkerFixture(tCtx ktesting.TContext, probeType probeType, spec v1.Probe, configure func(*instanceBoundWorkerOptions)) *instanceBoundWorkerFixture {
	pod := getTestPod()
	setTestProbe(pod, probeType, spec)

	f := &instanceBoundWorkerFixture{
		exec:             &countingExecProber{result: probe.Success},
		results:          results.NewManager(),
		exited:           make(chan struct{}),
		startupSucceeded: make(chan struct{}, 1),
	}

	pb := newProber(nil /* runner */, &record.FakeRecorder{})
	pb.exec = f.exec

	opts := instanceBoundWorkerOptions{
		probeType: probeType,
		target: probeTarget{
			pod:         pod,
			container:   pod.Spec.Containers[0],
			containerID: testContainerID,
			podIPs:      []string{"1.2.3.4"},
			startedAt:   time.Now(),
		},
		prober:         pb,
		resultsManager: f.results,
		onExit:         func(*instanceBoundWorker) { close(f.exited) },
		onStartupSucceeded: func(*instanceBoundWorker) {
			f.startupSucceeded <- struct{}{}
		},
	}
	if configure != nil {
		configure(&opts)
	}
	f.worker = newInstanceBoundWorker(tCtx, opts)
	return f
}

func (f *instanceBoundWorkerFixture) hasExited() bool {
	select {
	case <-f.exited:
		return true
	default:
		return false
	}
}

func (f *instanceBoundWorkerFixture) startupDidSucceed() bool {
	select {
	case <-f.startupSucceeded:
		return true
	default:
		return false
	}
}

// TestInstanceBoundWorkerInitialDelayAnchoredToContainerStart checks that
// InitialDelaySeconds is measured from when the container started rather than
// from when the worker was created, which is what makes probe timing
// independent of how long the pod sync took to reach the container.
// See https://github.com/kubernetes/kubernetes/issues/96614.
func TestInstanceBoundWorkerInitialDelayAnchoredToContainerStart(t *testing.T) {
	for _, tc := range []struct {
		name string
		// startedAgo is how long before worker creation the container started.
		startedAgo       time.Duration
		expectProbeAfter time.Duration
	}{{
		name:             "container just started: full delay applies",
		startedAgo:       0,
		expectProbeAfter: 10 * time.Second,
	}, {
		name:             "container started a while ago: delay already elapsed",
		startedAgo:       30 * time.Second,
		expectProbeAfter: 0,
	}, {
		name:             "container started mid-delay: only the remainder applies",
		startedAgo:       6 * time.Second,
		expectProbeAfter: 4 * time.Second,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				f := newInstanceBoundWorkerFixture(tCtx, readiness, v1.Probe{InitialDelaySeconds: 10}, func(o *instanceBoundWorkerOptions) {
					o.target.startedAt = time.Now().Add(-tc.startedAgo)
				})
				go f.worker.run()
				defer f.worker.stop()

				if tc.expectProbeAfter > 0 {
					// Stop just short of the deadline: still nothing.
					time.Sleep(tc.expectProbeAfter - time.Second)
					tCtx.Wait()
					if got := f.exec.count(); got != 0 {
						tCtx.Errorf("probed %d times before InitialDelaySeconds elapsed, want 0", got)
					}
				}

				time.Sleep(2 * time.Second)
				tCtx.Wait()
				if got := f.exec.count(); got == 0 {
					tCtx.Error("no probe ran after InitialDelaySeconds elapsed")
				}
			})
		})
	}
}

// TestInstanceBoundWorkerThresholds checks that a result is only published
// once it has repeated FailureThreshold/SuccessThreshold times, and that the
// run count restarts when the result flips.
func TestInstanceBoundWorkerThresholds(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		f := newInstanceBoundWorkerFixture(tCtx, readiness, v1.Probe{SuccessThreshold: 2, FailureThreshold: 3}, nil)
		f.exec.set(probe.Failure)
		go f.worker.run()
		defer f.worker.stop()

		// Two failures are below FailureThreshold, so nothing is published.
		// The first probe runs immediately; the rest are one period apart.
		for i := 1; i <= 2; i++ {
			tCtx.Wait()
			if _, ok := f.results.Get(testContainerID); ok {
				tCtx.Fatalf("result published after %d of 3 failures", i)
			}
			time.Sleep(time.Second)
		}
		tCtx.Wait()
		if result, ok := f.results.Get(testContainerID); !ok || result != results.Failure {
			tCtx.Fatalf("after FailureThreshold failures got (%v, %v), want (Failure, true)", result, ok)
		}

		// One success is below SuccessThreshold, so Failure still stands.
		f.exec.set(probe.Success)
		time.Sleep(time.Second)
		tCtx.Wait()
		if result, _ := f.results.Get(testContainerID); result != results.Failure {
			tCtx.Fatalf("after 1 of 2 successes got %v, want Failure", result)
		}
		time.Sleep(time.Second)
		tCtx.Wait()
		if result, _ := f.results.Get(testContainerID); result != results.Success {
			tCtx.Fatalf("after SuccessThreshold successes got %v, want Success", result)
		}
	})
}

// TestInstanceBoundWorkerTerminalTransitions checks which probe outcomes end
// the worker. A worker that ends is not a worker that failed: the container is
// about to be killed and restarted, and the restart brings new workers with it.
func TestInstanceBoundWorkerTerminalTransitions(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		probeType             probeType
		result                probe.Result
		expectExit            bool
		expectStartupCallback bool
		expectResult          results.Result
	}{{
		name:         "liveness success keeps probing",
		probeType:    liveness,
		result:       probe.Success,
		expectResult: results.Success,
	}, {
		name:         "liveness failure ends the worker",
		probeType:    liveness,
		result:       probe.Failure,
		expectExit:   true,
		expectResult: results.Failure,
	}, {
		name:         "readiness failure keeps probing",
		probeType:    readiness,
		result:       probe.Failure,
		expectResult: results.Failure,
	}, {
		name:                  "startup success ends the worker and hands off",
		probeType:             startup,
		result:                probe.Success,
		expectExit:            true,
		expectStartupCallback: true,
		expectResult:          results.Success,
	}, {
		name:         "startup failure ends the worker without handing off",
		probeType:    startup,
		result:       probe.Failure,
		expectExit:   true,
		expectResult: results.Failure,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
				f := newInstanceBoundWorkerFixture(tCtx, tc.probeType, v1.Probe{}, nil)
				f.exec.set(tc.result)
				go f.worker.run()
				defer f.worker.stop()

				time.Sleep(time.Second)
				tCtx.Wait()

				if result, ok := f.results.Get(testContainerID); !ok || result != tc.expectResult {
					tCtx.Errorf("got result (%v, %v), want (%v, true)", result, ok, tc.expectResult)
				}
				if got := f.hasExited(); got != tc.expectExit {
					tCtx.Errorf("worker exited = %v, want %v", got, tc.expectExit)
				}
				if got := f.startupDidSucceed(); got != tc.expectStartupCallback {
					tCtx.Errorf("startup callback fired = %v, want %v", got, tc.expectStartupCallback)
				}

				if tc.expectExit {
					// An exiting worker leaves its result behind: it is still
					// what the container's status says, and removing it is the
					// job of whoever tears the container's probes down.
					if _, ok := f.results.Get(testContainerID); !ok {
						tCtx.Error("exiting worker removed its cached result")
					}
					// And it really is gone, not just paused.
					before := f.exec.count()
					time.Sleep(5 * time.Second)
					tCtx.Wait()
					if after := f.exec.count(); after != before {
						tCtx.Errorf("worker probed %d more times after exiting", after-before)
					}
				}
			})
		})
	}
}

// TestInstanceBoundWorkerStopCancelsInFlightProbe checks that stop() aborts a
// probe that is already executing, and that the aborted probe is discarded
// rather than recorded as a failure.
func TestInstanceBoundWorkerStopCancelsInFlightProbe(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		runner := &hangingRunner{entered: make(chan struct{})}

		f := newInstanceBoundWorkerFixture(tCtx, liveness, v1.Probe{}, func(o *instanceBoundWorkerOptions) {
			// Probe through the real exec prober so the runner sees the
			// worker's context, the way the CRI client does.
			o.prober = newProber(runner, &record.FakeRecorder{})
		})
		go f.worker.run()

		<-runner.entered
		f.worker.stop()

		<-f.exited
		if _, ok := f.results.Get(testContainerID); ok {
			tCtx.Error("cancelled probe recorded a result; it should have been discarded")
		}
	})
}

// TestInstanceBoundWorkerAdoptionJitter checks that an adopted worker -- one
// created for a container that was already running, as after a kubelet restart
// -- spreads its first probe over a probe period instead of firing at once
// alongside every other adopted worker on the node.
func TestInstanceBoundWorkerAdoptionJitter(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		const period = 30

		f := newInstanceBoundWorkerFixture(tCtx, readiness, v1.Probe{PeriodSeconds: period}, func(o *instanceBoundWorkerOptions) {
			o.adopted = true
			// InitialDelaySeconds is long since elapsed, so only the jitter
			// stands between the worker and its first probe.
			o.target.startedAt = time.Now().Add(-time.Hour)
		})
		go f.worker.run()
		defer f.worker.stop()

		// The spread is random, so all that can be asserted deterministically
		// is that it stays inside one period.
		time.Sleep(period * time.Second)
		tCtx.Wait()
		if got := f.exec.count(); got != 1 {
			tCtx.Errorf("probed %d times in the first period, want exactly 1", got)
		}
	})
}
