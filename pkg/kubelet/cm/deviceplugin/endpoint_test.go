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
	"path"
	"testing"
	"time"

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

	umap := make(map[string]*pluginapi.Device)
	for _, d := range updated {
		umap[d.ID] = d
	}

	callbackChan := make(chan int)
	callbackCount := 0

	p, e := esetup(t, devs, socket, "mock", func(n string, a, u, r []pluginapi.Device) {
		if callbackCount == 0 {
			require.Len(t, a, 2)
			require.Len(t, u, 0)
			require.Len(t, r, 0)
		} else if callbackCount == 1 {
			require.Len(t, a, 1)
			require.Len(t, u, 1)
			require.Len(t, r, 1)

			require.Equal(t, a[0].ID, updated[1].ID)

			require.Equal(t, u[0].ID, updated[0].ID)
			require.Equal(t, u[0].Health, updated[0].Health)

			require.Equal(t, r[0].ID, devs[1].ID)
		}

		callbackCount++
		if callbackChan != nil {
			callbackChan <- callbackCount
		}
	})

	defer ecleanup(t, p, e)

	go e.Run()

	select {
	case <-callbackChan:
		break
	case <-time.After(time.Second):
		t.FailNow()
	}

	// Ensure we have recieved the devices
	resp, err := e.Allocate([]string{e.Store().Devices()[0].ID})
	require.NotNil(t, resp)
	require.NoError(t, err)

	p.Update(updated)
	select {
	case <-callbackChan:
		break
	case <-time.After(time.Second):
		t.FailNow()
	}

	require.Len(t, e.Store().Devices(), 2)
	require.Len(t, e.Store().HealthyDevices(), 1)
	edevs := e.Store().Devices()
	for _, d := range edevs {
		dref, ok := umap[d.ID]

		require.True(t, ok)
		require.Equal(t, d.ID, dref.ID)
		require.Equal(t, d.Health, dref.Health)
	}

	close(callbackChan)
	callbackChan = nil
}

func esetup(t *testing.T, devs []*pluginapi.Device, socket, resourceName string, callback managerCallback) (*Stub, *endpointImpl) {
	p := NewDevicePluginStub(devs, socket)
	require.NoError(t, p.Start())

	dStore := newDeviceStoreImpl(callback)
	e, err := newEndpointWithStore(socket, resourceName, dStore)
	require.NoError(t, err)

	return p, e
}

func newTestEndpoint(resourceName string) endpoint {
	return &endpointImpl{
		resourceName: resourceName,
		devStore:     newDeviceStoreImpl(nil),
	}
}

func ecleanup(t *testing.T, p *Stub, e *endpointImpl) {
	e.Stop()
	p.Stop()
}
