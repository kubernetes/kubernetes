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

package kuberuntime

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// recordingProbeLifecycle records the probe lifecycle calls the runtime makes,
// in order.
type recordingProbeLifecycle struct {
	mu          sync.Mutex
	calls       []string
	containerID kubecontainer.ContainerID
	podIPs      []string
	startedAt   time.Time
}

func (r *recordingProbeLifecycle) StartProbes(_ context.Context, _ *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIPs []string, startedAt time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, fmt.Sprintf("StartProbes(%s)", container.Name))
	r.containerID = containerID
	r.podIPs = podIPs
	r.startedAt = startedAt
}

func (r *recordingProbeLifecycle) StopProbes(containerID kubecontainer.ContainerID, probeTypes kubecontainer.ProbeType) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, fmt.Sprintf("StopProbes(%s, %s)", containerID.ID, probeTypes))
}

func (r *recordingProbeLifecycle) recorded() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.calls...)
}

func (r *recordingProbeLifecycle) startArgs() (kubecontainer.ContainerID, []string, time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.containerID, append([]string(nil), r.podIPs...), r.startedAt
}

func probeLifecycleTestPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "bar", Namespace: "default"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:            "foo",
				Image:           "busybox",
				ImagePullPolicy: v1.PullIfNotPresent,
			}},
		},
	}
}

// TestStartContainerStartsProbes verifies that successful container start triggers
// StartProbes with the container ID, pod IPs, and start timestamp.
func TestStartContainerStartsProbes(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatal(err)
	}
	probes := &recordingProbeLifecycle{}
	m.probeLifecycle = probes

	pod := probeLifecycleTestPod()
	sandbox, _ := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	sandboxConfig, err := m.generatePodSandboxConfig(tCtx, pod, 0)
	if err != nil {
		t.Fatal(err)
	}

	before := time.Now()
	if _, err := m.startContainer(tCtx, sandbox.Id, sandboxConfig, containerStartSpec(&pod.Spec.Containers[0]), pod, &kubecontainer.PodStatus{}, nil, "10.0.0.1", []string{"10.0.0.1"}, nil); err != nil {
		t.Fatalf("startContainer: %v", err)
	}

	want := []string{"StartProbes(foo)"}
	if got := probes.recorded(); !slices.Equal(got, want) {
		t.Errorf("probe lifecycle calls = %v, want %v", got, want)
	}
	gotID, gotIPs, gotStartedAt := probes.startArgs()
	if gotID.ID == "" || gotID.Type != m.runtimeName {
		t.Errorf("StartProbes containerID = %v, want non-empty %s ID", gotID, m.runtimeName)
	}
	if wantIPs := []string{"10.0.0.1"}; !slices.Equal(gotIPs, wantIPs) {
		t.Errorf("StartProbes podIPs = %v, want %v", gotIPs, wantIPs)
	}
	if gotStartedAt.Before(before) {
		t.Errorf("StartProbes startedAt = %v, want no earlier than %v", gotStartedAt, before)
	}
}

// TestStartContainerFailureDoesNotStartProbes verifies that if the runtime fails
// to start the container, StartProbes is not invoked.
func TestStartContainerFailureDoesNotStartProbes(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatal(err)
	}
	probes := &recordingProbeLifecycle{}
	m.probeLifecycle = probes

	pod := probeLifecycleTestPod()
	sandbox, _ := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	sandboxConfig, err := m.generatePodSandboxConfig(tCtx, pod, 0)
	if err != nil {
		t.Fatal(err)
	}

	fakeRuntime.InjectError("StartContainer", errors.New("start failed"))

	if _, err := m.startContainer(tCtx, sandbox.Id, sandboxConfig, containerStartSpec(&pod.Spec.Containers[0]), pod, &kubecontainer.PodStatus{}, nil, "10.0.0.1", []string{"10.0.0.1"}, nil); err == nil {
		t.Fatal("startContainer succeeded, want a StartContainer error")
	}

	if got := probes.recorded(); len(got) != 0 {
		t.Errorf("probe lifecycle calls = %v, want none when startContainer fails", got)
	}
}

// TestStartContainerStopsProbesOnPostStartFailure verifies probe cleanup when a PostStart hook fails.
// PostStart executes after container start and probe initialization, so a failure must abort the
// initialized probes before stopping the container.
func TestStartContainerStopsProbesOnPostStartFailure(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatal(err)
	}
	probes := &recordingProbeLifecycle{}
	m.probeLifecycle = probes

	pod := probeLifecycleTestPod()
	pod.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
		PostStart: &v1.LifecycleHandler{Exec: &v1.ExecAction{Command: []string{"PostStartCMD"}}},
	}
	m.runner = lifecycle.NewHandlerRunner(
		&fakeHTTP{},
		&containertest.FakeContainerCommandRunner{Err: errors.New("PostStart failed")},
		fakePodStatusProvider{pod: &kubecontainer.Pod{ID: pod.UID}, status: &kubecontainer.PodStatus{ID: pod.UID}},
		nil)

	sandbox, _ := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
	sandboxConfig, err := m.generatePodSandboxConfig(tCtx, pod, 0)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := m.startContainer(tCtx, sandbox.Id, sandboxConfig, containerStartSpec(&pod.Spec.Containers[0]), pod, &kubecontainer.PodStatus{}, nil, "", []string{}, nil); err == nil {
		t.Fatal("startContainer succeeded, want a PostStart hook failure")
	}

	gotID, _, _ := probes.startArgs()
	want := []string{
		"StartProbes(foo)",
		fmt.Sprintf("StopProbes(%s, Liveness|Startup)", gotID.ID),
		fmt.Sprintf("StopProbes(%s, All)", gotID.ID),
	}
	if got := probes.recorded(); !slices.Equal(got, want) {
		t.Errorf("probe lifecycle calls = %v, want %v", got, want)
	}
}

// TestKillContainerStopsProbes verifies probe shutdown ordering:
// liveness and startup probes stop before PreStop to prevent graceful shutdown
// from being cut short by a probe failure, while readiness probes run until the container
// exits so endpoints remain draining.
func TestKillContainerStopsProbes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		hasPreStop bool
		stopFails  bool
		// want is the expected call log, with %s standing in for the container
		// ID the fake runtime assigned.
		want []string
	}{{
		name:       "container with PreStop stops",
		hasPreStop: true,
		want:       []string{"StopProbes(%s, Liveness|Startup)", "PreStop", "StopProbes(%s, All)"},
	}, {
		name: "container without PreStop stops",
		want: []string{"StopProbes(%s, Liveness|Startup)", "StopProbes(%s, All)"},
	}, {
		name:       "container with PreStop fails to stop",
		hasPreStop: true,
		stopFails:  true,
		// Readiness keeps running and the results stay cached; the next sync
		// reconciles.
		want: []string{"StopProbes(%s, Liveness|Startup)", "PreStop"},
	}, {
		name:      "container without PreStop fails to stop",
		stopFails: true,
		want:      []string{"StopProbes(%s, Liveness|Startup)"},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			if err != nil {
				t.Fatal(err)
			}
			probes := &recordingProbeLifecycle{}
			m.probeLifecycle = probes

			pod := probeLifecycleTestPod()
			if tc.hasPreStop {
				pod.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
					PreStop: &v1.LifecycleHandler{Exec: &v1.ExecAction{Command: []string{"PreStopCMD"}}},
				}
				// The PreStop hook records itself in the same log, so its position
				// relative to the probe hooks is visible.
				m.runner = preStopRecorder{probes}
			}

			_, containers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
			containerID := kubecontainer.ContainerID{Type: "docker", ID: containers[0].Id}

			if tc.stopFails {
				fakeRuntime.InjectError("StopContainer", errors.New("stop failed"))
			}

			gracePeriod := int64(30)
			err = m.killContainer(tCtx, pod, containerID, "foo", "testKill", "", &gracePeriod, nil)
			if tc.stopFails != (err != nil) {
				t.Fatalf("killContainer error = %v, wanted an error: %v", err, tc.stopFails)
			}

			var want []string
			for _, call := range tc.want {
				if strings.Contains(call, "%s") {
					call = fmt.Sprintf(call, containerID.ID)
				}
				want = append(want, call)
			}
			if got := probes.recorded(); !slices.Equal(got, want) {
				t.Errorf("probe lifecycle calls = %v, want %v", got, want)
			}
		})
	}
}

// preStopRecorder is a lifecycle handler runner that records that the PreStop
// hook ran, interleaved with the probe lifecycle calls.
type preStopRecorder struct {
	probes *recordingProbeLifecycle
}

func (r preStopRecorder) Run(_ context.Context, _ kubecontainer.ContainerID, _ *v1.Pod, _ *v1.Container, _ *v1.LifecycleHandler) (string, error) {
	r.probes.mu.Lock()
	defer r.probes.mu.Unlock()
	r.probes.calls = append(r.probes.calls, "PreStop")
	return "", nil
}
