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

type endpointStoreShim struct {
	store endpointStore

	outChan      chan interface{}
	continueChan chan bool
}

type swapMessage struct {
	Old endpoint
	New endpoint
}

// outChan sends a message containing the parameters of the method called.
// After sending that message, execution blocks until continueChan is written on
func newInstrumentedEndpointStoreShim(outChan chan interface{}, continueChan chan bool) *endpointStoreShim {
	return &endpointStoreShim{
		store: newEndpointStoreImpl(),

		outChan:      outChan,
		continueChan: continueChan,
	}
}

func (s *endpointStoreShim) Endpoint(resourceName string) (endpoint, bool) {
	return s.store.Endpoint(resourceName)
}

func (s *endpointStoreShim) SynchronizedEndpoint(resourceName string) (*synchronizedEndpoint, bool) {
	return s.store.SynchronizedEndpoint(resourceName)
}

func (s *endpointStoreShim) DeleteEndpoint(resourceName string) error {
	return s.store.DeleteEndpoint(resourceName)
}

func (s *endpointStoreShim) SwapEndpoint(e endpoint) (endpoint, bool) {
	m := swapMessage{New: e}
	if old, ok := s.store.Endpoint(e.ResourceName()); ok {
		m.Old = old
	}

	s.outChan <- m
	<-s.continueChan

	return s.store.SwapEndpoint(e)
}

func (s *endpointStoreShim) Range(f func(k string, e *synchronizedEndpoint)) {
	s.store.Range(f)
}
