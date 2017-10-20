/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

func TestRecordOperation(t *testing.T) {
	prometheus.MustRegister(metrics.RuntimeOperations)
	prometheus.MustRegister(metrics.RuntimeOperationsLatency)
	prometheus.MustRegister(metrics.RuntimeOperationsErrors)

	temporalServer := "127.0.0.1:1234"
	l, err := net.Listen("tcp", temporalServer)
	assert.NoError(t, err)
	defer l.Close()

	prometheusUrl := "http://" + temporalServer + "/metrics"
	mux := http.NewServeMux()
	mux.Handle("/metrics", prometheus.Handler())
	server := &http.Server{
		Addr:    temporalServer,
		Handler: mux,
	}
	go func() {
		server.Serve(l)
	}()

	recordOperation("create_container", time.Now())
	runtimeOperationsCounterExpected := "kubelet_runtime_operations{operation_type=\"create_container\"} 1"
	runtimeOperationsLatencyExpected := "kubelet_runtime_operations_latency_microseconds_count{operation_type=\"create_container\"} 1"

	assert.HTTPBodyContains(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mux.ServeHTTP(w, r)
	}), "GET", prometheusUrl, nil, runtimeOperationsCounterExpected)

	assert.HTTPBodyContains(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mux.ServeHTTP(w, r)
	}), "GET", prometheusUrl, nil, runtimeOperationsLatencyExpected)
}

func TestInstrumentedVersion(t *testing.T) {
	fakeRuntime, _, _, _ := createTestRuntimeManager()
	irs := newInstrumentedRuntimeService(fakeRuntime)
	vr, err := irs.Version("1")
	assert.NoError(t, err)
	assert.Equal(t, kubeRuntimeAPIVersion, vr.Version)
}

func TestStatus(t *testing.T) {
	fakeRuntime, _, _, _ := createTestRuntimeManager()
	fakeRuntime.FakeStatus = &runtimeapi.RuntimeStatus{
		Conditions: []*runtimeapi.RuntimeCondition{
			{Type: runtimeapi.RuntimeReady, Status: false},
			{Type: runtimeapi.NetworkReady, Status: true},
		},
	}
	irs := newInstrumentedRuntimeService(fakeRuntime)
	actural, err := irs.Status()
	assert.NoError(t, err)
	expected := &runtimeapi.RuntimeStatus{
		Conditions: []*runtimeapi.RuntimeCondition{
			{Type: runtimeapi.RuntimeReady, Status: false},
			{Type: runtimeapi.NetworkReady, Status: true},
		},
	}
	assert.Equal(t, expected, actural)
}

func TestInstrumentedListContainer(t *testing.T) {
	fakeRuntime, _, _, _ := createTestRuntimeManager()
	irs := newInstrumentedRuntimeService(fakeRuntime)
	podName, namespace := "foo", "bar"
	containerName, image := "sidecar", "logger"
	configs := []*runtimeapi.ContainerConfig{}
	sConfigs := []*runtimeapi.PodSandboxConfig{}
	for i := 0; i < 3; i++ {
		s := &runtimeapi.PodSandboxConfig{
			Metadata: &runtimeapi.PodSandboxMetadata{
				Name:      fmt.Sprintf("%s%d", podName, i),
				Namespace: fmt.Sprintf("%s%d", namespace, i),
				Uid:       fmt.Sprintf("%d", i),
				Attempt:   0,
			},
			Labels:      map[string]string{},
			Annotations: map[string]string{},
		}
		labels := map[string]string{"abc.xyz": fmt.Sprintf("label%d", i)}
		annotations := map[string]string{"foo.bar.baz": fmt.Sprintf("annotation%d", i)}
		c := &runtimeapi.ContainerConfig{
			Metadata: &runtimeapi.ContainerMetadata{
				Name:    fmt.Sprintf("%s%d", containerName, i),
				Attempt: uint32(i),
			},
			Image:       &runtimeapi.ImageSpec{Image: fmt.Sprintf("%s:v%d", image, i)},
			Labels:      labels,
			Annotations: annotations,
		}
		sConfigs = append(sConfigs, s)
		configs = append(configs, c)
	}

	expected := make(map[string]*runtimeapi.Container, 0)
	expectedStatus := make(map[string]*runtimeapi.ContainerStatus, 0)
	actual := make(map[string]*runtimeapi.Container, 0)
	state := runtimeapi.ContainerState_CONTAINER_RUNNING
	createdAt := int64(0)
	for i := range configs {
		// We don't care about the sandbox id; pass a bogus one.
		sandboxID := fmt.Sprintf("sandboxid%d", i)
		id, err := irs.CreateContainer(sandboxID, configs[i], sConfigs[i])
		assert.NoError(t, err)
		err = irs.StartContainer(id)
		assert.NoError(t, err)
		expected[id] = &runtimeapi.Container{
			Metadata:     configs[i].Metadata,
			Id:           id,
			PodSandboxId: sandboxID,
			State:        state,
			CreatedAt:    createdAt,
			Image:        configs[i].Image,
			ImageRef:     configs[i].Image.Image,
			Labels:       configs[i].Labels,
			Annotations:  configs[i].Annotations,
		}

		expectedStatus[id] = &runtimeapi.ContainerStatus{
			Metadata:    configs[i].Metadata,
			State:       runtimeapi.ContainerState_CONTAINER_EXITED,
			Id:          id,
			ExitCode:    0,
			Image:       configs[i].Image,
			ImageRef:    configs[i].Image.Image,
			Labels:      configs[i].Labels,
			Annotations: configs[i].Annotations,
		}
	}
	containers, err := irs.ListContainers(nil)
	assert.NoError(t, err)
	// CreateContainer of FakeRuntimeService will use time.Now().UnixNano() for createdAt.
	// We are not able to know a real time before run this method and setting CreatedAt to 0 for easily testing.
	for _, c := range containers {
		c.CreatedAt = createdAt
		actual[c.Id] = c
	}
	assert.Equal(t, actual, expected)
	for id := range expected {
		err = irs.StopContainer(id, 0)
		assert.NoError(t, err)
		status, err := irs.ContainerStatus(id)
		assert.NoError(t, err)
		// ContainerStatus of FakeRuntimeService will use time.Now().UnixNano() for CreatedAt and StartedAt.
		// We are not able to know a real time before run this method and setting this explicitly just for easily testing.
		expectedStatus[id].CreatedAt = status.CreatedAt
		expectedStatus[id].StartedAt = status.StartedAt
		expectedStatus[id].FinishedAt = status.FinishedAt
		assert.Equal(t, expectedStatus[id], status)
		err = irs.RemoveContainer(id)
		assert.NoError(t, err)
	}
	containers, err = irs.ListContainers(nil)
	assert.NoError(t, err)
	assert.Equal(t, 0, len(containers))
}
