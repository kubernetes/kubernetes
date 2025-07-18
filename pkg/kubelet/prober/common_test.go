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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/status"
	probe "k8s.io/kubernetes/pkg/probe"
	"k8s.io/utils/exec"
)

const (
	testContainerName = "cOnTaInEr_NaMe"
	testPodUID        = "pOd_UiD"
	testPodName       = "test-pod"
	testNamespace     = "test-namespace"
)

var testContainerID = kubecontainer.ContainerID{Type: "test", ID: "cOnTaInEr_Id"}

func init() {
	// Register v1.Pod with the scheme to avoid "no kind is registered" errors
	legacyscheme.Scheme.AddKnownTypes(v1.SchemeGroupVersion, &v1.Pod{})
}

func getTestRunningStatus() v1.PodStatus {
	return getTestRunningStatusWithStarted(true)
}

func getTestNotRunningStatus() v1.PodStatus {
	return getTestRunningStatusWithStarted(false)
}

func getTestRunningStatusWithStarted(started bool) v1.PodStatus {
	containerStatus := v1.ContainerStatus{
		Name:        testContainerName,
		ContainerID: testContainerID.String(),
	}
	if started {
		containerStatus.State.Running = &v1.ContainerStateRunning{StartedAt: metav1.Now()}
	}
	containerStatus.Started = &started
	podStatus := v1.PodStatus{
		Phase:                 v1.PodRunning,
		InitContainerStatuses: []v1.ContainerStatus{containerStatus},
	}
	return podStatus
}

func getTestPod() *v1.Pod {
	container := v1.Container{
		Name: testContainerName,
	}
	pod := v1.Pod{
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{container},
			RestartPolicy:  v1.RestartPolicyNever,
		},
	}
	pod.Name = testPodName
	pod.UID = testPodUID
	pod.Namespace = testNamespace
	return &pod
}

func setTestProbe(pod *v1.Pod, probeType probeType, probeSpec v1.Probe) {
	probeSpec.ProbeHandler = v1.ProbeHandler{
		Exec: &v1.ExecAction{Command: []string{"/bin/true"}},
	}
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
	// Set probe on InitContainers[0] if it matches testContainerName
	for i, c := range pod.Spec.InitContainers {
		if c.Name == testContainerName {
			switch probeType {
			case readiness:
				pod.Spec.InitContainers[i].ReadinessProbe = &probeSpec
			case liveness:
				pod.Spec.InitContainers[i].LivenessProbe = &probeSpec
			case startup:
				pod.Spec.InitContainers[i].StartupProbe = &probeSpec
			}
			return
		}
	}
	// Fallback to Containers[0] if no matching init container
	if len(pod.Spec.Containers) > 0 {
		switch probeType {
		case readiness:
			pod.Spec.Containers[0].ReadinessProbe = &probeSpec
		case liveness:
			pod.Spec.Containers[0].LivenessProbe = &probeSpec
		case startup:
			pod.Spec.Containers[0].StartupProbe = &probeSpec
		}
	}
}

func ptrContainerRestartPolicy(r v1.ContainerRestartPolicy) *v1.ContainerRestartPolicy {
	return &r
}

func newTestManager() *manager {
	podManager := kubepod.NewBasicPodManager()
	podManager.AddPod(getTestPod())
	fakeStatus := &fakeStatusManager{podStatuses: make(map[types.UID]v1.PodStatus)}
	var _ status.Manager = fakeStatus // Ensure fakeStatusManager satisfies status.Manager
	m := &manager{
		livenessManager:  results.NewManager(),
		readinessManager: results.NewManager(),
		startupManager:   results.NewManager(),
		statusManager:    fakeStatus,
		prober:           newProber(nil, &record.FakeRecorder{}),
	}
	m.prober.exec = fakeExecProber{probe.Success, nil}
	return m
}

func newTestWorker(m *manager, probeType probeType, probeSpec v1.Probe) *worker {
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
					Name:           testContainerName,
					Image:          "foo",
					RestartPolicy:  ptrContainerRestartPolicy(v1.ContainerRestartPolicyAlways),
					StartupProbe:   &v1.Probe{},
					LivenessProbe:  &v1.Probe{},
					ReadinessProbe: &v1.Probe{},
				},
			},
			Containers: []v1.Container{
				{
					Name:           "main-container",
					Image:          "bar",
					ReadinessProbe: &v1.Probe{},
				},
			},
		},
	}
	setTestProbe(pod, probeType, probeSpec)
	return newWorker(m, probeType, pod, pod.Spec.InitContainers[0])
}

type fakeExecProber struct {
	result probe.Result
	err    error
}

func (p fakeExecProber) Probe(c exec.Cmd) (probe.Result, string, error) {
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
