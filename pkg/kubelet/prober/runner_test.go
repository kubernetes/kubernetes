/*
Copyright 2023 The Kubernetes Authors.

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
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilpointer "k8s.io/utils/pointer"
	"testing"
)

func newPod(handler v1.ProbeHandler) v1.Pod {
	pod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID("pod"),
			Name:      "pod",
			Namespace: "test",
		},
		Spec: v1.PodSpec{},
		Status: v1.PodStatus{
			Phase:  v1.PodPhase(v1.PodReady),
			PodIP:  "10.11.11.11",
			PodIPs: []v1.PodIP{{IP: "10.11.11.11"}},
		},
	}

	pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
		Name: "container",
		LivenessProbe: &v1.Probe{
			ProbeHandler: handler,
		},
	})

	pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses, v1.ContainerStatus{
		Name:        "container",
		ContainerID: "pod://container",
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{
				StartedAt: metav1.Now(),
			},
		},
		Started: utilpointer.Bool(true),
	})
	return pod
}

func TestNewProbeRunner(t *testing.T) {
	var pod v1.Pod
	var runner probeRunner

	// tcpProberRunner type
	pod = newPod(v1.ProbeHandler{TCPSocket: &v1.TCPSocketAction{
		Port: intstr.FromInt(5000),
	}})
	runner = newProbeRunner(pod.Spec.Containers[0], liveness)
	assert.IsType(t, newTCPProbeRunner(), runner)
	//

	// httpProberRunner type
	pod = newPod(v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
		Path:   "/test",
		Port:   intstr.FromInt(5000),
		Scheme: "http",
	}})
	runner = newProbeRunner(pod.Spec.Containers[0], liveness)
	assert.IsType(t, newHTTPProbeRunner(), runner)
	//

	// grpcProberRunner type
	pod = newPod(v1.ProbeHandler{GRPC: &v1.GRPCAction{
		Port:    int32(5000),
		Service: utilpointer.String(""),
	}})
	runner = newProbeRunner(pod.Spec.Containers[0], liveness)
	assert.IsType(t, newGRPCProbeRunner(), runner)
	//

	// execProberRunner type
	pod = newPod(v1.ProbeHandler{Exec: &v1.ExecAction{
		Command: []string{"test", "-f", "/app/run.lock"},
	}})
	runner = newProbeRunner(pod.Spec.Containers[0], liveness)
	assert.IsType(t, newExecProbeRunner(), runner)
	//

	// dummyProberRunner type
	var dummyRunner *dummyProbeRunner
	// case 1: unknown probe type
	pod = newPod(v1.ProbeHandler{})
	pod.Spec.Containers = []v1.Container{{
		Name: "container",
	}}
	const unknownProbeType probeType = 100
	runner = newProbeRunner(pod.Spec.Containers[0], unknownProbeType)
	assert.IsType(t, newDummyProbeRunner(false, true, false), runner)

	dummyRunner = any(runner).(*dummyProbeRunner)
	assert.Equal(t, dummyRunner.isProbeTypeUnknown, true)
	assert.Equal(t, dummyRunner.isProbeNil, false)
	assert.Equal(t, dummyRunner.isHandlerUnknown, false)
	//

	// case 2: nil probe spec
	pod = newPod(v1.ProbeHandler{})
	pod.Spec.Containers[0].LivenessProbe = nil
	runner = newProbeRunner(pod.Spec.Containers[0], liveness)
	assert.IsType(t, newDummyProbeRunner(false, true, false), runner)

	dummyRunner = any(runner).(*dummyProbeRunner)
	assert.Equal(t, dummyRunner.isProbeTypeUnknown, false)
	assert.Equal(t, dummyRunner.isProbeNil, true)
	assert.Equal(t, dummyRunner.isHandlerUnknown, false)
	//

	// case 3: unknown handler / empty probe
	pod = newPod(v1.ProbeHandler{})
	runner = newProbeRunner(pod.Spec.Containers[0], liveness)
	assert.IsType(t, newDummyProbeRunner(false, true, false), runner)

	dummyRunner = any(runner).(*dummyProbeRunner)
	assert.Equal(t, dummyRunner.isProbeTypeUnknown, false)
	assert.Equal(t, dummyRunner.isProbeNil, false)
	assert.Equal(t, dummyRunner.isHandlerUnknown, true)
	//

}
