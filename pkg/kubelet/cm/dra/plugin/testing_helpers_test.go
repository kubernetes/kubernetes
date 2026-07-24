/*
Copyright 2025 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"sync"

	drahealthv1 "k8s.io/kubelet/pkg/apis/dra-health/v1"
)

type healthStreamLifecycleEvent struct {
	action     string
	driverName string
	generation uint64
}

// mockStreamHandler is a mock implementation of the StreamHandler interface,
// shared across all tests in this package.
type mockStreamHandler struct {
	mutex  sync.Mutex
	events []healthStreamLifecycleEvent
}

var _ StreamHandler = &mockStreamHandler{}

func (m *mockStreamHandler) ActivateHealthStream(_ context.Context, driverName string, generation uint64) {
	m.record("activate", driverName, generation)
}

func (m *mockStreamHandler) DeactivateHealthStream(_ context.Context, driverName string, generation uint64) {
	m.record("deactivate", driverName, generation)
}

func (m *mockStreamHandler) HandleWatchResourcesStream(ctx context.Context, stream drahealthv1.DRAResourceHealth_NodeWatchResourcesClient, driverName string, generation uint64) error {
	<-ctx.Done()
	return nil
}

func (m *mockStreamHandler) record(action, driverName string, generation uint64) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.events = append(m.events, healthStreamLifecycleEvent{action: action, driverName: driverName, generation: generation})
}

func (m *mockStreamHandler) lifecycleEvents() []healthStreamLifecycleEvent {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	return append([]healthStreamLifecycleEvent(nil), m.events...)
}
