/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
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

var testContainerID = kubecontainer.ContainerID{"test", "cOnTaInEr_Id"}

func getTestRunningStatus() api.PodStatus {
	containerStatus := api.ContainerStatus{
		Name:        testContainerName,
		ContainerID: testContainerID.String(),
	}
	containerStatus.State.Running = &api.ContainerStateRunning{unversioned.Now()}
	podStatus := api.PodStatus{
		Phase:             api.PodRunning,
		ContainerStatuses: []api.ContainerStatus{containerStatus},
	}
	return podStatus
}

func getTestPod(probeType probeType, probeSpec api.Probe) api.Pod {
	container := api.Container{
		Name: testContainerName,
	}

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
		container.ReadinessProbe = &probeSpec
	case liveness:
		container.LivenessProbe = &probeSpec
	}
	pod := api.Pod{
		Spec: api.PodSpec{
			Containers:    []api.Container{container},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	pod.Name = "testPod"
	pod.UID = testPodUID
	return pod
}

func newTestManager() *manager {
	refManager := kubecontainer.NewRefManager()
	refManager.SetRef(testContainerID, &api.ObjectReference{}) // Suppress prober warnings.
	m := NewManager(
		status.NewManager(&testclient.Fake{}, kubepod.NewBasicPodManager(nil)),
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
	pod := getTestPod(probeType, probeSpec)
	return newWorker(m, probeType, &pod, pod.Spec.Containers[0])
}

type fakeExecProber struct {
	result probe.Result
	err    error
}

func (p fakeExecProber) Probe(_ exec.Cmd) (probe.Result, string, error) {
	return p.result, "", p.err
}
