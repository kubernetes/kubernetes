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
	"fmt"

	"github.com/golang/glog"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/deviceplugin"
)

type DevicePluginHandlerImpl struct {
	devicePluginManager deviceplugin.Manager
}

// NewDevicePluginHandler create a DevicePluginHandler
func NewDevicePluginHandler() (*DevicePluginHandlerImpl, error) {
	glog.V(2).Infof("Starting Device Plugin Handler")

	mgr, err := deviceplugin.NewManagerImpl(pluginapi.DevicePluginPath,
		func(r string, a, u, d []*pluginapi.Device) {})

	if err != nil {
		return nil, fmt.Errorf("Failed to initialize device plugin: %+v", err)
	}

	if err := mgr.Start(); err != nil {
		return nil, err
	}

	return &DevicePluginHandlerImpl{
		devicePluginManager: mgr,
	}, nil
}

// TODO cache this
func (h *DevicePluginHandlerImpl) Devices() map[string][]*pluginapi.Device {
	return h.devicePluginManager.Devices()
}
