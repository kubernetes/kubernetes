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
	"fmt"
	"sync"
	"time"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

// endpointHandler handles all operations related to endpoints
// namely creating the endpoints, keeping a map of them and stoping them
type endpointHandler interface {
	NewEndpoint(resourceName, socketPath string) error

	Devices() map[string][]pluginapi.Device
	Endpoint(resourceName string) (endpoint, bool)

	Stop() error

	Store() endpointStore
	SetStore(endpointStore)

	SetCallback(f managerCallback)
}

type endpointStore interface {
	Endpoint(resourceName string) (endpoint, bool)
	SynchronizedEndpoint(resourceName string) (*synchronizedEndpoint, bool)

	SwapEndpoint(e endpoint) (endpoint, bool)
	DeleteEndpoint(resourceName string) error

	Range(func(string, *synchronizedEndpoint))
}

type synchronizedEndpoint struct {
	e    endpoint
	done chan string
}

type endpointStoreImpl struct {
	sync.Mutex
	endpoints map[string]*synchronizedEndpoint
}

type endpointHandlerImpl struct {
	store    endpointStore
	callback managerCallback
}

func newEndpointHandlerImpl(f managerCallback) *endpointHandlerImpl {
	return &endpointHandlerImpl{
		store:    newEndpointStoreImpl(),
		callback: f,
	}
}

func (h *endpointHandlerImpl) NewEndpoint(resourceName, socketPath string) error {
	e, err := newEndpoint(socketPath, resourceName)
	if err != nil {
		return err
	}

	// If an endpoint with this resourceName is already present
	var devStore deviceStore = newDeviceStoreImpl(h.callback)
	if old, ok := h.store.Endpoint(resourceName); ok && old != nil {
		devStore = old.Store()
	}

	// set a new store or the old store
	e.SetStore(devStore)

	// Add the endpoint to the store (and get the old one)
	old, ok := h.store.SwapEndpoint(e)
	go h.trackEnpoint(e)

	if ok && old != nil {
		// Prevent it to issue a Callback to the manager before stopping it
		old.SetStore(newAlwaysEmptyDeviceStore())

		if err = old.Stop(); err != nil {
			return err
		}
	}

	return nil
}

// This function has two code path:
//  - The endpoint was stopped to be removed
//     - It's still in the store and has the same value
//     - Remove it from the store
//     - Emit it's resource name to signal code has terminated
//       - If this is a test Stop was called and it can know when the
//         endpoint stopped
//       - If the endpoint terminated / was removed by the user then the
//         channel does not exist and no value is emited
//  - The endpoint was stopped to be replaced
//     - It's still in the store but was already replaced
//     - Do nothing and return
func (h *endpointHandlerImpl) trackEnpoint(e endpoint) {
	rName := e.ResourceName()
	e.Run()

	curr, ok := h.store.SynchronizedEndpoint(rName)
	if !ok || curr.e != e {
		return
	}

	h.store.DeleteEndpoint(rName)

	if curr.done != nil {
		curr.done <- rName
	}
}

func (h *endpointHandlerImpl) Devices() map[string][]pluginapi.Device {
	devs := make(map[string][]pluginapi.Device)

	h.store.Range(func(k string, s *synchronizedEndpoint) {
		devs[k] = s.e.Store().Devices()
	})

	return devs
}

func (h *endpointHandlerImpl) Endpoint(resourceName string) (endpoint, bool) {
	return h.store.Endpoint(resourceName)
}

func (h *endpointHandlerImpl) Stop() error {
	syncChan := make(chan string)

	var endpoints []*synchronizedEndpoint
	h.store.Range(func(k string, s *synchronizedEndpoint) {
		endpoints = append(endpoints, s)
	})

	for _, s := range endpoints {
		s.done = syncChan
		s.e.Stop()

		select {
		case <-syncChan:
			continue

		// This can only be called if a code change creates a code path
		// where the endpoint does not emit on the done chan upon death
		case <-time.After(time.Second):
			return fmt.Errorf("Could not stop endpoint %s", s.e.ResourceName())
		}
	}

	return nil
}

func (h *endpointHandlerImpl) Store() endpointStore {
	return h.store
}

func (h *endpointHandlerImpl) SetStore(s endpointStore) {
	h.store = s
}

func (h *endpointHandlerImpl) SetCallback(f managerCallback) {
	h.callback = f
}

func newEndpointStoreImpl() *endpointStoreImpl {
	return &endpointStoreImpl{
		endpoints: make(map[string]*synchronizedEndpoint),
	}
}

func (s *endpointStoreImpl) Endpoint(resourceName string) (endpoint, bool) {
	s.Lock()
	defer s.Unlock()

	se, ok := s.endpoints[resourceName]
	if ok && se != nil {
		return se.e, true
	}

	return nil, false
}

func (s *endpointStoreImpl) SynchronizedEndpoint(resourceName string) (*synchronizedEndpoint, bool) {
	s.Lock()
	defer s.Unlock()

	se, ok := s.endpoints[resourceName]
	return se, ok
}

func (s *endpointStoreImpl) SwapEndpoint(e endpoint) (endpoint, bool) {
	s.Lock()
	defer s.Unlock()

	old, ok := s.endpoints[e.ResourceName()]
	s.endpoints[e.ResourceName()] = &synchronizedEndpoint{e: e}

	if ok && old != nil {
		return old.e, ok
	}

	return nil, ok
}

func (s *endpointStoreImpl) DeleteEndpoint(resourceName string) error {
	s.Lock()
	defer s.Unlock()

	_, ok := s.endpoints[resourceName]
	if !ok {
		return fmt.Errorf("Endpoint %s does not exist", resourceName)
	}

	delete(s.endpoints, resourceName)
	return nil
}

func (s *endpointStoreImpl) Range(f func(string, *synchronizedEndpoint)) {
	s.Lock()
	defer s.Unlock()

	for k, v := range s.endpoints {
		f(k, v)
	}
}
