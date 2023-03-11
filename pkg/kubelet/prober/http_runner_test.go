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
	"k8s.io/apimachinery/pkg/util/intstr"
	"testing"
)

func TestNewHTTPProbeRunner(t *testing.T) {
	runner := newHTTPProbeRunner()

	pod := newPod(v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
		Path:   "/test",
		Port:   intstr.FromInt(5000),
		Scheme: "http",
	}})

	_ = runner.sync(pod.Spec.Containers[0], pod.Status, liveness)
	assert.Equal(t, runner.host, pod.Status.PodIP)

	// test if http request object is not being recreated after
	// sync call as podIP hasn't changed.
	httpReqObject := runner.httpRequest
	for i := 0; i < 25; i++ {
		_ = runner.sync(pod.Spec.Containers[0], pod.Status, liveness)
		assert.Same(t, httpReqObject, runner.httpRequest)
	}

	// change ip address
	pod.Status.PodIP = "192.168.1.11"
	_ = runner.sync(pod.Spec.Containers[0], pod.Status, liveness)

	// test if http request object is recreated after
	// sync call as podIP is changed now.
	assert.NotSame(t, httpReqObject, runner.httpRequest)
	assert.Equal(t, runner.host, pod.Status.PodIP)
}
