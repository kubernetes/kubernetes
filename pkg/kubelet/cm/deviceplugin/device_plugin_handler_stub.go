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

package deviceplugin

import (
	"k8s.io/api/core/v1"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

// A simple stub implementation for DevicePluginHandler.
type DevicePluginHandlerStub struct{}

func NewDevicePluginHandlerStub() (*DevicePluginHandlerStub, error) {
	return &DevicePluginHandlerStub{}, nil
}

func (h *DevicePluginHandlerStub) Start() error {
	return nil
}

func (h *DevicePluginHandlerStub) Devices() map[string][]*pluginapi.Device {
	return make(map[string][]*pluginapi.Device)
}

func (h *DevicePluginHandlerStub) Allocate(pod *v1.Pod, container *v1.Container, activePods []*v1.Pod) ([]*pluginapi.AllocateResponse, error) {
	var ret []*pluginapi.AllocateResponse
	return ret, nil
}
