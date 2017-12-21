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
	"sync"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

type deviceStore interface {
	Devices() []pluginapi.Device
	HealthyDevices() []pluginapi.Device
	Update(devs []*pluginapi.Device) (added, updated, deleted []pluginapi.Device)

	// Callback invokes the manager callback function with the sent parameters
	Callback(resourceName string, added, updated, deleted []pluginapi.Device)

	// Testing methods
	UpdateDevices(added, updated, deleted []pluginapi.Device)
	UpdateAndCallback(resourceName string, added, updated, deleted []pluginapi.Device)
}

type deviceStoreImpl struct {
	sync.Mutex
	devices map[string]pluginapi.Device

	callback managerCallback
}

type alwaysEmptyDeviceStore struct {
}

func newDeviceStoreImpl(callback managerCallback) *deviceStoreImpl {
	return &deviceStoreImpl{
		devices:  make(map[string]pluginapi.Device),
		callback: callback,
	}
}

func (s *deviceStoreImpl) Devices() []pluginapi.Device {
	s.Lock()
	defer s.Unlock()

	var devs []pluginapi.Device

	for _, d := range s.devices {
		devs = append(devs, d)
	}

	return devs
}

func (s *deviceStoreImpl) HealthyDevices() []pluginapi.Device {
	s.Lock()
	defer s.Unlock()

	var devs []pluginapi.Device

	for _, d := range s.devices {
		if d.Health != pluginapi.Healthy {
			continue
		}

		devs = append(devs, d)
	}

	return devs
}

func (s *deviceStoreImpl) Update(devs []*pluginapi.Device) (added, updated, deleted []pluginapi.Device) {
	newDevs := make(map[string]pluginapi.Device)

	s.Lock()
	defer s.Unlock()

	for _, d := range devs {
		dOld, ok := s.devices[d.ID]
		newDevs[d.ID] = *d

		if !ok {
			s.devices[d.ID] = *d
			added = append(added, *d)

			continue
		}

		if d.Health == dOld.Health {
			continue
		}

		s.devices[d.ID] = *d
		updated = append(updated, *d)
	}

	for id, d := range s.devices {
		if _, ok := newDevs[id]; ok {
			continue
		}

		deleted = append(deleted, d)
		delete(s.devices, id)
	}

	return added, updated, deleted
}

func (s *deviceStoreImpl) Callback(resourceName string, added, updated, deleted []pluginapi.Device) {
	s.callback(resourceName, added, updated, deleted)
}

func (s *deviceStoreImpl) UpdateDevices(added, updated, deleted []pluginapi.Device) {
	s.Lock()
	defer s.Unlock()

	for _, a := range added {
		s.devices[a.ID] = a
	}

	for _, u := range updated {
		s.devices[u.ID] = u
	}

	for _, r := range deleted {
		delete(s.devices, r.ID)
	}
}

func (s *deviceStoreImpl) UpdateAndCallback(resourceName string, added, updated, deleted []pluginapi.Device) {
	s.UpdateDevices(added, updated, deleted)
	s.Callback(resourceName, added, updated, deleted)
}

func newAlwaysEmptyDeviceStore() *alwaysEmptyDeviceStore {
	return &alwaysEmptyDeviceStore{}
}

func (s *alwaysEmptyDeviceStore) Devices() []pluginapi.Device {
	return nil
}

func (s *alwaysEmptyDeviceStore) HealthyDevices() []pluginapi.Device {
	return nil
}

func (s *alwaysEmptyDeviceStore) Update(devs []*pluginapi.Device) (added, updated, deleted []pluginapi.Device) {
	return nil, nil, nil
}

func (s *alwaysEmptyDeviceStore) Callback(resourceName string, added, updated, deleted []pluginapi.Device) {
}

func (s *alwaysEmptyDeviceStore) UpdateDevices(added, updated, deleted []pluginapi.Device) {
}

func (s *alwaysEmptyDeviceStore) UpdateAndCallback(resourceName string, added, updated, deleted []pluginapi.Device) {
}
