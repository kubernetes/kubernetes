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

package cm

import (
	//"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/container"

	v1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

func (d *DevicePluginHandler) shimAllocate(kind string, ndevices int, c *v1alpha1.ContainerConfig) ([]*pluginapi.Device, error) {

	devs, responses, err := d.devicePluginManager.Allocate(kind, ndevices)
	if err != nil {
		return nil, err
	}

	// TODO define merge strategy or error handling
	for _, response := range responses {
		for _, env := range response.Envs {
			c.Envs = append(c.Envs, &v1alpha1.KeyValue{
				Key:   env.Key,
				Value: env.Value,
			})
		}
	}

	for _, response := range responses {
		for _, mount := range response.Mounts {
			c.Mounts = append(c.Mounts, &v1alpha1.Mount{
				ContainerPath: mount.MountPath,
				HostPath:      mount.HostPath,
				Readonly:      mount.ReadOnly,
			})
		}
	}

	return devs, nil
}

/*
// FromPluginToAPIDevices converts []*plugin.Device to []v1.Device
func FromPluginToAPIDevices(pluginDevices []*pluginapi.Device) []v1.Device {
	var devs []v1.Device

	for _, dev := range pluginDevices {
		devs = append(devs, v1.Device{
			Kind:       dev.Kind,
			Vendor:     dev.Vendor,
			Name:       dev.Name,
			Properties: dev.Properties,
		})
	}

	return devs
}*/

// FromPluginToContainerDevices converts []*plugin.Device to []*container.Device
func FromPluginToContainerDevices(pluginDevices []*pluginapi.Device) []*container.Device {
	var devs []*container.Device

	for _, dev := range pluginDevices {
		devs = append(devs, &container.Device{
			Kind:       dev.Kind,
			Vendor:     dev.Vendor,
			Name:       dev.Name,
			Properties: dev.Properties,
		})
	}

	return devs
}

/*
// FromAPIToPluginDevices converts []v1.Device to []*pluginapi.Device
func FromAPIToPluginDevices(v1Devs []v1.Device) []*pluginapi.Device {
	var devs []*pluginapi.Device

	for _, dev := range v1Devs {
		devs = append(devs, &pluginapi.Device{
			Kind:       dev.Kind,
			Vendor:     dev.Vendor,
			Name:       dev.Name,
			Properties: dev.Properties,
		})
	}

	return devs
}

// FromPluginToAPI converts map[string][]*pluginapi.Device to map[string][]v1.Device
func FromPluginToAPI(pluginDevices map[string][]*pluginapi.Device) map[string][]v1.Device {
	devs := make(map[string][]v1.Device)

	for k, v := range pluginDevices {
		devs[k] = FromPluginToAPIDevices(v)
	}

	return devs
}

// FromAPIToPlugin converts map[string][]v1.Device to map[string][]*pluginapi.Device
func FromAPIToPlugin(pluginDevices map[string][]v1.Device) map[string][]*pluginapi.Device {
	devs := make(map[string][]*pluginapi.Device)

	for k, v := range pluginDevices {
		devs[k] = FromAPIToPluginDevices(v)
	}

	return devs
}*/
