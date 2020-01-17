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
	"time"

	"github.com/stretchr/testify/require"

	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

var (
	esocketName = "mock.sock"
)

func TestNewEndpoint(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}

	p, e := esetup(t, devs, socket, "mock", func(n string, d []pluginapi.Device) {})
	defer ecleanup(t, p, e)
}

func TestRun(t *testing.T) {
	socket := path.Join("/tmp", esocketName)

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
		{ID: "AnotherDeviceId", Health: pluginapi.Healthy},
		{ID: "AThirdDeviceId", Health: pluginapi.Unhealthy},
	}

	updated := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Unhealthy},
		{ID: "AThirdDeviceId", Health: pluginapi.Healthy},
		{ID: "AFourthDeviceId", Health: pluginapi.Healthy},
	}

	callbackCount := 0
	callbackChan := make(chan int)
	callback := func(n string, devices []pluginapi.Device) {
		// Should be called twice:
		// one for plugin registration, one for plugin update.
		if callbackCount > 2 {
			t.FailNow()
		}

		// Check plugin registration
		if callbackCount == 0 {
			require.Len(t, devices, 3)
			require.Equal(t, devices[0].ID, devs[0].ID)
			require.Equal(t, devices[1].ID, devs[1].ID)
			require.Equal(t, devices[2].ID, devs[2].ID)
			require.Equal(t, devices[0].Health, devs[0].Health)
			require.Equal(t, devices[1].Health, devs[1].Health)
			require.Equal(t, devices[2].Health, devs[2].Health)
		}

		// Check plugin update
		if callbackCount == 1 {
			require.Len(t, devices, 3)
			require.Equal(t, devices[0].ID, updated[0].ID)
			require.Equal(t, devices[1].ID, updated[1].ID)
			require.Equal(t, devices[2].ID, updated[2].ID)
			require.Equal(t, devices[0].Health, updated[0].Health)
			require.Equal(t, devices[1].Health, updated[1].Health)
			require.Equal(t, devices[2].Health, updated[2].Health)
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

	require.Equal(t, callbackCount, 2)
}

func TestAllocate(t *testing.T) {
	socket := path.Join("/tmp", esocketName)
	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}
	callbackCount := 0
	callbackChan := make(chan int)
	p, e := esetup(t, devs, socket, "mock", func(n string, d []pluginapi.Device) {
		callbackCount++
		callbackChan <- callbackCount
	})
	defer ecleanup(t, p, e)

	resp := new(pluginapi.AllocateResponse)
	contResp := new(pluginapi.ContainerAllocateResponse)
	contResp.Devices = append(contResp.Devices, &pluginapi.DeviceSpec{
		ContainerPath: "/dev/aaa",
		HostPath:      "/dev/aaa",
		Permissions:   "mrw",
	})

	contResp.Devices = append(contResp.Devices, &pluginapi.DeviceSpec{
		ContainerPath: "/dev/bbb",
		HostPath:      "/dev/bbb",
		Permissions:   "mrw",
	})

	contResp.Mounts = append(contResp.Mounts, &pluginapi.Mount{
		ContainerPath: "/container_dir1/file1",
		HostPath:      "host_dir1/file1",
		ReadOnly:      true,
	})

	resp.ContainerResponses = append(resp.ContainerResponses, contResp)

	p.SetAllocFunc(func(r *pluginapi.AllocateRequest, devs map[string]pluginapi.Device) (*pluginapi.AllocateResponse, error) {
		return resp, nil
	})

	go e.run()
	// Wait for the callback to be issued.
	select {
	case <-callbackChan:
		break
	case <-time.After(time.Second):
		t.FailNow()
	}

	respOut, err := e.allocate([]string{"ADeviceId"})
	require.NoError(t, err)
	require.Equal(t, resp, respOut)
}

func esetup(t *testing.T, devs []*pluginapi.Device, socket, resourceName string, callback monitorCallback) (*Stub, *endpointImpl) {
	p := NewDevicePluginStub(devs, socket, resourceName, false)

	err := p.Start()
	require.NoError(t, err)

	e, err := newEndpointImpl(socket, resourceName, callback)
	require.NoError(t, err)

	return p, e
}

func ecleanup(t *testing.T, p *Stub, e *endpointImpl) {
	p.Stop()
	e.stop()
}
