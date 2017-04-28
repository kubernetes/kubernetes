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

package lifecycle

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

type FakeHandlerRunner struct {
	sync.Mutex
	HandlerRuns []string
	Err         error
}

func NewFakeHandlerRunner() *FakeHandlerRunner {
	return &FakeHandlerRunner{HandlerRuns: []string{}}
}

func (hr *FakeHandlerRunner) Run(containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.Handler) (string, error) {
	hr.Lock()
	defer hr.Unlock()

	if hr.Err != nil {
		return "", hr.Err
	}

	switch {
	case handler.Exec != nil:
		hr.HandlerRuns = append(hr.HandlerRuns, fmt.Sprintf("exec on pod: %v, container: %v: %v", format.Pod(pod), container.Name, containerID.String()))
	case handler.HTTPGet != nil:
		hr.HandlerRuns = append(hr.HandlerRuns, fmt.Sprintf("http-get on pod: %v, container: %v: %v", format.Pod(pod), container.Name, containerID.String()))
	case handler.TCPSocket != nil:
		hr.HandlerRuns = append(hr.HandlerRuns, fmt.Sprintf("tcp-socket on pod: %v, container: %v: %v", format.Pod(pod), container.Name, containerID.String()))
	default:
		return "", fmt.Errorf("Invalid handler: %v", handler)
	}
	return "", nil
}

func (hr *FakeHandlerRunner) Reset() {
	hr.HandlerRuns = []string{}
	hr.Err = nil
}
