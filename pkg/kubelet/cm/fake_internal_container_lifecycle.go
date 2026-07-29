/*
Copyright 2017 The Kubernetes Authors.

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

package cm

import (
	"k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
)

func NewFakeInternalContainerLifecycle() *fakeInternalContainerLifecycle {
	return &fakeInternalContainerLifecycle{}
}

type fakeInternalContainerLifecycle struct{}

func (f *fakeInternalContainerLifecycle) PreCreateContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerConfig *runtimeapi.ContainerConfig) error {
	return nil
}

func (f *fakeInternalContainerLifecycle) PreStartContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerID string) error {
	return nil
}

func (f *fakeInternalContainerLifecycle) PostStopContainer(logger klog.Logger, containerID string) error {
	return nil
}
