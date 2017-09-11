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

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

var (
	esocketName = "mock.sock"
)

func TestNewEndpoint(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}

	p, e := esetup(t, devs, socket, "mock", func(n string, a, u, r []*pluginapi.Device) {})
	defer ecleanup(t, p, e)
}

func TestList(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}

	p, e := esetup(t, devs, socket, "mock", func(n string, a, u, r []*pluginapi.Device) {})
	defer ecleanup(t, p, e)

	_, err := e.list()
	require.NoError(t, err)

	e.mutex.Lock()
	defer e.mutex.Unlock()

	require.Len(t, e.devices, 1)

	d, ok := e.devices[devs[0].ID]
	require.True(t, ok)

	require.Equal(t, d.ID, devs[0].ID)
	require.Equal(t, d.Health, devs[0].Health)
}

func TestListAndWatch(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
		{ID: "AnotherDeviceId", Health: pluginapi.Healthy},
	}

	updated := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Unhealthy},
		{ID: "AThirdDeviceId", Health: pluginapi.Healthy},
	}

	p, e := esetup(t, devs, socket, "mock", func(n string, a, u, r []*pluginapi.Device) {
		require.Len(t, a, 1)
		require.Len(t, u, 1)
		require.Len(t, r, 1)

		require.Equal(t, a[0].ID, updated[1].ID)

		require.Equal(t, u[0].ID, updated[0].ID)
		require.Equal(t, u[0].Health, updated[0].Health)

		require.Equal(t, r[0].ID, devs[1].ID)
	})
	defer ecleanup(t, p, e)

	s, err := e.list()
	require.NoError(t, err)

	go e.listAndWatch(s)
	p.Update(updated)
	time.Sleep(time.Second)

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

func esetup(t *testing.T, devs []*pluginapi.Device, socket, resourceName string, callback MonitorCallback) (*Stub, *endpoint) {
	p := NewDevicePluginStub(devs, socket)

	err := p.Start()
	require.NoError(t, err)

	e, err := newEndpoint(socket, "mock", func(n string, a, u, r []*pluginapi.Device) {})
	require.NoError(t, err)

	return p, e
}

func ecleanup(t *testing.T, p *Stub, e *endpoint) {
	p.Stop()
	e.stop()
}
