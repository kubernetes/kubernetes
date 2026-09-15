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

// containerBoundWorkerFixture wires a containerBoundWorker to a fake probe
// handler and records the callbacks the manager would normally receive.
type containerBoundWorkerFixture struct {
	worker  *containerBoundWorker
	exec    *countingExecProber
	results results.Manager

	exited           chan struct{}
	startupSucceeded chan struct{}
}

func newContainerBoundWorkerFixture(tCtx ktesting.TContext, probeType probeType, spec v1.Probe, configure func(*containerBoundWorkerOptions)) *containerBoundWorkerFixture {
	pod := getTestPod()
	setTestProbe(pod, probeType, spec)

	f := &containerBoundWorkerFixture{
		exec:             &countingExecProber{result: probe.Success},
		results:          results.NewManager(),
		exited:           make(chan struct{}),
		startupSucceeded: make(chan struct{}, 1),
	}

	pb := newProber(nil /* runner */, &record.FakeRecorder{})
	pb.exec = f.exec

	opts := containerBoundWorkerOptions{
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
		onExit:         func(*containerBoundWorker) { close(f.exited) },
		onStartupSucceeded: func(*containerBoundWorker) {
			f.startupSucceeded <- struct{}{}
		},
	}
	if configure != nil {
		configure(&opts)
	}
	f.worker = newContainerBoundWorker(tCtx, opts)
	return f
}

func (f *containerBoundWorkerFixture) hasExited() bool {
	select {
	case <-f.exited:
		return true
	default:
		return false
	}
}

func (f *containerBoundWorkerFixture) startupDidSucceed() bool {
	select {
	case <-f.startupSucceeded:
		return true
	default:
		return false
	}
}

// TestContainerBoundWorkerInitialDelayAnchoredToContainerStart checks that
// InitialDelaySeconds is measured from when the container started rather than
// from when the worker was created, which is what makes probe timing
// independent of how long the pod sync took to reach the container.
func TestContainerBoundWorkerInitialDelayAnchoredToContainerStart(t *testing.T) {
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
				f := newContainerBoundWorkerFixture(tCtx, readiness, v1.Probe{InitialDelaySeconds: 10}, func(o *containerBoundWorkerOptions) {
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

// TestContainerBoundWorkerThresholds checks that a result is only published
// once it has repeated FailureThreshold/SuccessThreshold times, and that the
// run count restarts when the result flips.
func TestContainerBoundWorkerThresholds(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		f := newContainerBoundWorkerFixture(tCtx, readiness, v1.Probe{SuccessThreshold: 2, FailureThreshold: 3}, nil)
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

// TestContainerBoundWorkerTerminalTransitions checks which probe outcomes end
// the worker. A worker that ends is not a worker that failed: the container is
// about to be killed and restarted, and the restart brings new workers with it.
func TestContainerBoundWorkerTerminalTransitions(t *testing.T) {
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
				f := newContainerBoundWorkerFixture(tCtx, tc.probeType, v1.Probe{}, nil)
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

// TestContainerBoundWorkerStopCancelsInFlightProbe checks that stop() aborts a
// probe that is already executing, and that the aborted probe is discarded
// rather than recorded as a failure.
func TestContainerBoundWorkerStopCancelsInFlightProbe(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		runner := &hangingRunner{entered: make(chan struct{})}

		f := newContainerBoundWorkerFixture(tCtx, liveness, v1.Probe{}, func(o *containerBoundWorkerOptions) {
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

// TestContainerBoundWorkerAdoptionJitter checks that workers adopted after a kubelet
// restart jitter their initial probe across one period to prevent a thundering herd.
func TestContainerBoundWorkerAdoptionJitter(t *testing.T) {
	ktesting.Init(t).SyncTest("", func(tCtx ktesting.TContext) {
		const (
			period  = 30
			workers = 30
		)
		sharedExec := &countingExecProber{result: probe.Success}

		for range workers {
			f := newContainerBoundWorkerFixture(tCtx, readiness, v1.Probe{PeriodSeconds: period}, func(o *containerBoundWorkerOptions) {
				o.adopted = true
				// InitialDelaySeconds is long since elapsed, so only the jitter
				// stands between the worker and its first probe.
				o.target.startedAt = time.Now().Add(-time.Hour)
				o.prober.exec = sharedExec
			})
			go f.worker.run()
			defer f.worker.stop()
		}

		// At T=0, jitter prevents a thundering herd where all workers probe at once.
		tCtx.Wait()
		if got := sharedExec.count(); got >= workers {
			tCtx.Fatalf("all %d workers probed immediately at T=0 without jitter", workers)
		}

		// Halfway through the period, only a subset of workers should have fired.
		time.Sleep((period / 2) * time.Second)
		tCtx.Wait()
		if got := sharedExec.count(); got == 0 || got >= workers {
			tCtx.Fatalf("got %d probed workers at half-period, want between 1 and %d", got, workers-1)
		}

		// By the end of the full period, every worker should have fired its first probe exactly once.
		time.Sleep((period / 2) * time.Second)
		tCtx.Wait()
		if got := sharedExec.count(); got != workers {
			tCtx.Fatalf("probed %d times after full period, want exactly %d", got, workers)
		}
	})
}
