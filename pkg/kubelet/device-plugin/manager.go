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
	"sync"

	"github.com/golang/glog"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

type MonitorCallback func(*pluginapi.Device)

type Manager struct {
	registry *Registery

	// Key is Kind
	devices   map[string][]*pluginapi.Device
	available map[string][]*pluginapi.Device

	// Key is vendor
	vendors map[string][]*pluginapi.Device

	mutex sync.Mutex

	callback MonitorCallback
}

func NewManager(devices, available []*pluginapi.Device, f MonitorCallback) (*Manager, error) {
	m := &Manager{
		devices:   make(map[string][]*pluginapi.Device),
		available: make(map[string][]*pluginapi.Device),
		vendors:   make(map[string][]*pluginapi.Device),

		registry: newServer(),
		callback: f,
	}

	for _, d := range devices {
		m.devices[d.Kind] = append(m.devices[d.Kind], d)
		m.vendors[d.Vendor] = append(m.vendors[d.Vendor], d)
	}

	for _, d := range available {
		m.available[d.Kind] = append(m.available[d.Kind], d)
		m.vendors[d.Vendor] = append(m.vendors[d.Vendor], d)
	}

	m.registry.Manager = m
	if err := m.startRegistery(); err != nil {
		return nil, err
	}

	return m, nil
}

func (m *Manager) Devices() map[string][]*pluginapi.Device {
	return m.devices
}

func (m *Manager) Available() map[string][]*pluginapi.Device {
	return m.available
}

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

func (m *Manager) Deallocate(devs []*pluginapi.Device) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if len(devs) == 0 {
		return nil
	}

	group := make(map[string][]*pluginapi.Device)

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

	for vendor, devs := range group {
		e, ok := m.registry.Endpoints[vendor]
		if !ok {
			return fmt.Errorf("Endpoint for vendor %s does not exist", vendor)
		}

		_, err := deallocate(e, devs)
		if err != nil {
			return err
		}
	}

	return nil
}

func (m *Manager) Stop() {
	m.registry.server.Stop()
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
