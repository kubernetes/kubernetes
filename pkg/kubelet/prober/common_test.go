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
	"reflect"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/util/exec"
)

const (
	testContainerName = "cOnTaInEr_NaMe"
	testPodUID        = "pOd_UiD"
)

var testContainerID = kubecontainer.ContainerID{Type: "test", ID: "cOnTaInEr_Id"}

func getTestRunningStatus() api.PodStatus {
	containerStatus := api.ContainerStatus{
		Name:        testContainerName,
		ContainerID: testContainerID.String(),
	}
	containerStatus.State.Running = &api.ContainerStateRunning{StartedAt: unversioned.Now()}
	podStatus := api.PodStatus{
		Phase:             api.PodRunning,
		ContainerStatuses: []api.ContainerStatus{containerStatus},
	}
	return podStatus
}

func getTestPod() *api.Pod {
	container := api.Container{
		Name: testContainerName,
	}
	pod := api.Pod{
		Spec: api.PodSpec{
			Containers:    []api.Container{container},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	pod.Name = "testPod"
	pod.UID = testPodUID
	return &pod
}

func setTestProbe(pod *api.Pod, probeType probeType, probeSpec api.Probe) {
	// All tests rely on the fake exec prober.
	probeSpec.Handler = api.Handler{
		Exec: &api.ExecAction{},
	}

	// Apply test defaults, overwridden for test speed.
	defaults := map[string]int64{
		"TimeoutSeconds":   1,
		"PeriodSeconds":    1,
		"SuccessThreshold": 1,
		"FailureThreshold": 1,
	}
	for field, value := range defaults {
		f := reflect.ValueOf(&probeSpec).Elem().FieldByName(field)
		if f.Int() == 0 {
			f.SetInt(value)
		}
	}

	switch probeType {
	case readiness:
		pod.Spec.Containers[0].ReadinessProbe = &probeSpec
	case liveness:
		pod.Spec.Containers[0].LivenessProbe = &probeSpec
	}
}

func newTestManager() *manager {
	refManager := kubecontainer.NewRefManager()
	refManager.SetRef(testContainerID, &api.ObjectReference{}) // Suppress prober warnings.
	podManager := kubepod.NewBasicPodManager(nil)
	// Add test pod to pod manager, so that status manager can get the pod from pod manager if needed.
	podManager.AddPod(getTestPod())
	m := NewManager(
		status.NewManager(&fake.Clientset{}, podManager),
		results.NewManager(),
		nil, // runner
		refManager,
		&record.FakeRecorder{},
	).(*manager)
	// Don't actually execute probes.
	m.prober.exec = fakeExecProber{probe.Success, nil}
	return m
}

func newTestWorker(m *manager, probeType probeType, probeSpec api.Probe) *worker {
	pod := getTestPod()
	setTestProbe(pod, probeType, probeSpec)
	return newWorker(m, probeType, pod, pod.Spec.Containers[0])
}

type fakeExecProber struct {
	result probe.Result
	err    error
}

func (p fakeExecProber) Probe(_ exec.Cmd) (probe.Result, string, error) {
	return p.result, "", p.err
}

type syncExecProber struct {
	sync.RWMutex
	fakeExecProber
}

func (p *syncExecProber) set(result probe.Result, err error) {
	p.Lock()
	defer p.Unlock()
	p.result = result
	p.err = err
}

func (p *syncExecProber) Probe(cmd exec.Cmd) (probe.Result, string, error) {
	p.RLock()
	defer p.RUnlock()
	return p.fakeExecProber.Probe(cmd)
}
