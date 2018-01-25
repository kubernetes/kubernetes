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

package devicemanager

import (
	"path"
	"testing"

	"github.com/stretchr/testify/require"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

var (
	esocketName = "mock.sock"
)

func TestNewEndpoint(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}

	p, e := esetup(t, devs, socket, "mock", func(n string, a, u, r []pluginapi.Device) {})
	defer ecleanup(t, p, e)
}

func TestRun(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
		{ID: "AnotherDeviceId", Health: pluginapi.Healthy},
	}

	updated := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Unhealthy},
		{ID: "AThirdDeviceId", Health: pluginapi.Healthy},
	}

	callbackCount := 0
	callbackChan := make(chan int)
	callback := func(n string, a, u, r []pluginapi.Device) {
		// Should be called twice:
		// one for plugin registration, one for plugin update.
		if callbackCount > 2 {
			t.FailNow()
		}

		// Check plugin registration
		if callbackCount == 0 {
			require.Len(t, a, 2)
			require.Len(t, u, 0)
			require.Len(t, r, 0)
		}

		// Check plugin update
		if callbackCount == 1 {
			require.Len(t, a, 1)
			require.Len(t, u, 1)
			require.Len(t, r, 1)

			require.Equal(t, a[0].ID, updated[1].ID)
			require.Equal(t, u[0].ID, updated[0].ID)
			require.Equal(t, u[0].Health, updated[0].Health)
			require.Equal(t, r[0].ID, devs[1].ID)
		}

		callbackCount++
		callbackChan <- callbackCount
	}

	p, e := esetup(t, devs, socket, "mock", callback)
	defer ecleanup(t, p, e)

	go e.run()
	// Wait for the first callback to be issued.
	<-callbackChan

	p.Update(updated)

	// Wait for the second callback to be issued.
	<-callbackChan

	e.mutex.Lock()
	defer e.mutex.Unlock()

	require.Len(t, e.devices, 2)
	for _, dref := range updated {
		d, ok := e.devices[dref.ID]

		require.True(t, ok)
		require.Equal(t, d.ID, dref.ID)
		require.Equal(t, d.Health, dref.Health)
	}

}

func TestGetDevices(t *testing.T) {
	e := endpointImpl{
		devices: map[string]pluginapi.Device{
			"ADeviceId": {ID: "ADeviceId", Health: pluginapi.Healthy},
		},
	}
	devs := e.getDevices()
	require.Len(t, devs, 1)
}

func esetup(t *testing.T, devs []*pluginapi.Device, socket, resourceName string, callback monitorCallback) (*Stub, *endpointImpl) {
	p := NewDevicePluginStub(devs, socket)

	err := p.Start()
	require.NoError(t, err)

	e, err := newEndpointImpl(socket, resourceName, make(map[string]pluginapi.Device), callback)
	require.NoError(t, err)

	return p, e
}

func ecleanup(t *testing.T, p *Stub, e *endpointImpl) {
	p.Stop()
	e.stop()
}
