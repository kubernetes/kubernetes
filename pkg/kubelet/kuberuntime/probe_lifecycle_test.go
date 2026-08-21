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
	mu    sync.Mutex
	calls []string
	// podIPs and startedAt capture the arguments of the last StartProbes call.
	podIPs    []string
	startedAt time.Time
}

func (r *recordingProbeLifecycle) StartProbes(_ context.Context, _ *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIPs []string, startedAt time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, fmt.Sprintf("StartProbes(%s)", container.Name))
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

// TestStartContainerStartsProbes checks that a container that starts
// successfully has its probes started, with the pod IPs and a start time taken
// at the moment the container actually started.
func TestStartContainerStartsProbes(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatal(err)
	}
	probes := &recordingProbeLifecycle{}
	m.SetContainerProbeLifecycle(probes)

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
	if got := probes.recorded(); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("probe lifecycle calls = %v, want %v", got, want)
	}
	if got := fmt.Sprint(probes.podIPs); got != "[10.0.0.1]" {
		t.Errorf("StartProbes podIPs = %v, want [10.0.0.1]", got)
	}
	if probes.startedAt.Before(before) {
		t.Errorf("StartProbes startedAt = %v, want no earlier than %v", probes.startedAt, before)
	}
}

// TestStartContainerStopsProbesOnPostStartFailure checks that a container torn
// down because its PostStart hook failed does not leave probes behind.
func TestStartContainerStopsProbesOnPostStartFailure(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatal(err)
	}
	probes := &recordingProbeLifecycle{}
	m.SetContainerProbeLifecycle(probes)

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

	got := probes.recorded()
	if len(got) != 3 || got[0] != "StartProbes(foo)" ||
		!strings.HasPrefix(got[1], "StopProbes(") ||
		!strings.HasPrefix(got[2], "StopProbes(") {
		t.Errorf("probe lifecycle calls = %v, want StartProbes then the kill hooks", got)
	}
}

// TestKillContainerStopsProbes checks the ordering that lets a container drain:
// liveness and startup stop before the PreStop hook runs, and everything else
// stops only once the container has actually gone.
func TestKillContainerStopsProbes(t *testing.T) {
	for _, tc := range []struct {
		name      string
		stopFails bool
		// want is the expected call log, with %s standing in for the container
		// ID the fake runtime assigned.
		want []string
	}{{
		name: "container stops",
		want: []string{"StopProbes(%s, Liveness|Startup)", "PreStop", "StopProbes(%s, All)"},
	}, {
		name:      "container fails to stop",
		stopFails: true,
		// Readiness keeps running and the results stay cached; the next sync
		// reconciles.
		want: []string{"StopProbes(%s, Liveness|Startup)", "PreStop"},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			if err != nil {
				t.Fatal(err)
			}
			probes := &recordingProbeLifecycle{}
			m.SetContainerProbeLifecycle(probes)

			pod := probeLifecycleTestPod()
			pod.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{Exec: &v1.ExecAction{Command: []string{"PreStopCMD"}}},
			}
			// The PreStop hook records itself in the same log, so its position
			// relative to the probe hooks is visible.
			m.runner = preStopRecorder{probes}

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
			if got := probes.recorded(); fmt.Sprint(got) != fmt.Sprint(want) {
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
