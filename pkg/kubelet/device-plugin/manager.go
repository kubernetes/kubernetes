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

package deviceplugin

import (
	"fmt"

	"github.com/golang/glog"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

// NewManager creates a new manager on the socket `socketPath` and can
// rebuild state from devices and available []Device.
// f is the callback that is called when a device becomes unhealthy
// socketPath is present for testing purposes in production this is pluginapi.KubeletSocket
func NewManager(socketPath string, devices, available []*pluginapi.Device,
	f MonitorCallback) (*Manager, error) {

	registry, err := newRegistry(socketPath)
	if err != nil {
		return nil, err
	}

	m := &Manager{
		devices:   make(map[string][]*pluginapi.Device),
		available: make(map[string][]*pluginapi.Device),
		vendors:   make(map[string][]*pluginapi.Device),

		registry: registry,
		callback: f,
	}

	for _, d := range devices {
		m.devices[d.Kind] = append(m.devices[d.Kind], d)
		m.vendors[d.Vendor] = append(m.vendors[d.Vendor], d)
	}

	for _, d := range available {
		if d.Health == pluginapi.Unhealthy {
			continue
		}

		m.available[d.Kind] = append(m.available[d.Kind], d)
	}

	m.registry.Manager = m
	if err := m.startRegistry(); err != nil {
		return nil, err
	}

	return m, nil
}

// Devices is the map of devices that are known by the Device
// Plugin manager with the Kind of the devices as key
func (m *Manager) Devices() map[string][]*pluginapi.Device {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	devs := make(map[string][]*pluginapi.Device)
	for k, v := range m.devices {
		devs[k] = copyDevices(v)
	}

	return devs
}

// Available is the map of devices that are available to be
// consumed
func (m *Manager) Available() map[string][]*pluginapi.Device {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	devs := make(map[string][]*pluginapi.Device)
	for k, v := range m.available {
		devs[k] = copyDevices(v)
	}

	return devs
}

// Allocate is the call that you can use to allocate a set of Devices
func (m *Manager) Allocate(kind string, ndevices int) ([]*pluginapi.Device,
	[]*pluginapi.AllocateResponse, error) {

	m.mutex.Lock()
	defer m.mutex.Unlock()

	if len(m.available[kind]) < ndevices || ndevices < 0 {
		return nil, nil, fmt.Errorf("Not enough devices of type %s available", kind)
	}

	glog.Infof("Recieved request for %d devices of kind %s", ndevices, kind)

	devs := m.available[kind][:ndevices]
	m.available[kind] = m.available[kind][ndevices:]

	if len(devs) == 0 {
		return nil, nil, nil
	}

	var responses []*pluginapi.AllocateResponse
	group := make(map[string][]*pluginapi.Device)

	for _, d := range devs {
		group[d.Vendor] = append(group[d.Vendor], d)
	}

	for vendor, devs := range group {
		response, err := allocate(m.registry.Endpoints[vendor], devs)

		if err != nil {
			return nil, nil, err
		}

		responses = append(responses, response)
	}

	return devs, responses, nil
}

// Deallocate is the call that you can use to deallocate a set of allocated Device
func (m *Manager) Deallocate(devs []*pluginapi.Device) error {
	if len(devs) == 0 {
		return nil
	}

	group := make(map[string][]*pluginapi.Device)

	m.mutex.Lock()
	for _, d := range devs {
		// If we don't know the device
		i, ok := HasDevice(d, m.devices[d.Kind])
		if !ok {
			continue
		}

		group[d.Vendor] = append(group[d.Vendor], d)

		// If the device is unhealthy don't put it back in available
		if m.devices[d.Kind][i].Health == pluginapi.Unhealthy {
			continue
		}

		// If the device is already in available don't put it back in available
		if _, ok := HasDevice(d, m.available[d.Kind]); ok {
			continue
		}

		m.available[d.Kind] = append(m.available[d.Kind], d)
	}
	m.mutex.Unlock()

	return nil
}

// Stop is the function that can stop the gRPC server
func (m *Manager) Stop() {
	if m.registry != nil && m.registry.server != nil {
		m.registry.server.Stop()
	}
}

func (m *Manager) addDevice(d *pluginapi.Device) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.devices[d.Kind] = append(m.devices[d.Kind], d)
	m.available[d.Kind] = append(m.available[d.Kind], d)

	m.vendors[d.Vendor] = append(m.vendors[d.Vendor], d)
}

func (m *Manager) deleteDevices(vendor string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	devs, ok := m.vendors[vendor]
	if !ok {
		return
	}

	for _, d := range devs {
		m.available[d.Kind] = deleteDev(d, m.available[d.Kind])
		m.devices[d.Kind] = deleteDev(d, m.devices[d.Kind])
	}
}
