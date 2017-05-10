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

package noop

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type noopIsolator struct{ name string }

func (i *noopIsolator) Init() error {
	glog.Infof("noopIsolator[Init]")
	return nil
}

func New(name string) (*noopIsolator, error) {
	return &noopIsolator{name: name}, nil
}

func (i *noopIsolator) PreStartPod(podName string, containerName string, pod *v1.Pod, cgroupInfo *lifecycle.CgroupInfo) ([]*lifecycle.IsolationControl, error) {
	glog.Infof("noopIsolator[PreStartPod]:\npod: %s\ncgroupInfo: %v", pod, cgroupInfo)
	return []*lifecycle.IsolationControl{}, nil
}

func (i *noopIsolator) PostStopPod(podName string, containerName string, cgroupInfo *lifecycle.CgroupInfo) error {
	glog.Infof("noopIsolator[PostStopPod]:\npodName: %s\ncontainerName: %s\ncgroupInfo: %v", podName, containerName, cgroupInfo)
	return nil
}

func (i *noopIsolator) PreStartContainer(podName, containerName string) ([]*lifecycle.IsolationControl, error) {
	glog.Infof("noopIsolator[PreStartContainer]:\npodName: %s\ncontainerName: %v", podName, containerName)
	return []*lifecycle.IsolationControl{
		{
			Kind: lifecycle.IsolationControl_CONTAINER_ENV_VAR,
			MapValue: map[string]string{
				"ISOLATOR": i.Name(),
			},
		},
	}, nil
}

func (i *noopIsolator) PostStopContainer(podName, containerName string) error {
	glog.Infof("noopIsolator[PostStopContainer]:\npodName: %s\ncontainerName: %v", podName, containerName)
	return nil
}

func (i *noopIsolator) ShutDown() {
	glog.Infof("noopIsolator[ShutDown]")
}

func (i *noopIsolator) Name() string {
	return i.name
}
