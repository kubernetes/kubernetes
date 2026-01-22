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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	plugin "k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/plugin/v1beta1"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// monitorCallback is the function called when a device's health state changes,
// or new devices are reported, or old devices are deleted.
// Updated contains the most recent state of the Device.
type monitorCallback func(logger klog.Logger, resourceName string, devices []*pluginapi.Device)

func newMockPluginManager() *mockPluginManager {
	return &mockPluginManager{
		func(string) error { return nil },
		func(string, plugin.DevicePlugin) error { return nil },
		func(klog.Logger, string) {},
		func(string, *pluginapi.ListAndWatchResponse) {},
	}
}

type mockPluginManager struct {
	cleanupPluginDirectory     func(string) error
	pluginConnected            func(string, plugin.DevicePlugin) error
	pluginDisconnected         func(klog.Logger, string)
	pluginListAndWatchReceiver func(string, *pluginapi.ListAndWatchResponse)
}

func (m *mockPluginManager) CleanupPluginDirectory(r string) error {
	return m.cleanupPluginDirectory(r)
}

func (m *mockPluginManager) PluginConnected(_ context.Context, r string, p plugin.DevicePlugin) error {
	return m.pluginConnected(r, p)
}

func (m *mockPluginManager) PluginDisconnected(logger klog.Logger, r string) {
	m.pluginDisconnected(logger, r)
}

func (m *mockPluginManager) PluginListAndWatchReceiver(_ klog.Logger, r string, lr *pluginapi.ListAndWatchResponse) {
	m.pluginListAndWatchReceiver(r, lr)
}

func esocketName() string {
	return fmt.Sprintf("mock%d.sock", time.Now().UnixNano())
}

func TestNewEndpoint(t *testing.T) {
	tCtx := ktesting.Init(t)

	socket := filepath.Join(os.TempDir(), esocketName())

	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}

	p, e := esetup(tCtx, t, devs, socket, "mock", func(logger klog.Logger, n string, d []*pluginapi.Device) {})
	defer func() {
		err := ecleanup(tCtx.Logger(), p, e)
		require.NoError(t, err)
	}()
}

func TestRun(t *testing.T) {
	tCtx := ktesting.Init(t)

	socket := filepath.Join(os.TempDir(), esocketName())

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
	callback := func(_ klog.Logger, n string, devices []*pluginapi.Device) {
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

	p, e := esetup(tCtx, t, devs, socket, "mock", callback)
	defer func() {
		err := ecleanup(tCtx.Logger(), p, e)
		require.NoError(t, err)
	}()

	go e.client.Run(tCtx)
	// Wait for the first callback to be issued.
	<-callbackChan

	p.Update(updated)

	// Wait for the second callback to be issued.
	<-callbackChan

	require.Equal(t, 2, callbackCount)
}

func TestAllocate(t *testing.T) {
	tCtx := ktesting.Init(t)

	socket := filepath.Join(os.TempDir(), esocketName())
	devs := []*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
	}
	callbackCount := 0
	callbackChan := make(chan int)
	p, e := esetup(tCtx, t, devs, socket, "mock", func(_ klog.Logger, n string, d []*pluginapi.Device) {
		callbackCount++
		callbackChan <- callbackCount
	})
	defer func() {
		err := ecleanup(tCtx.Logger(), p, e)
		require.NoError(t, err)
	}()

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

	p.SetAllocFunc(func(r *pluginapi.AllocateRequest, devs map[string]*pluginapi.Device) (*pluginapi.AllocateResponse, error) {
		return resp, nil
	})

	go e.client.Run(tCtx)
	// Wait for the callback to be issued.
	select {
	case <-callbackChan:
		break
	case <-time.After(time.Second):
		t.FailNow()
	}

	respOut, err := e.allocate(tCtx, []string{"ADeviceId"})
	require.NoError(t, err)
	require.True(t, proto.Equal(resp, respOut))
}

func TestGetPreferredAllocation(t *testing.T) {
	tCtx := ktesting.Init(t)

	socket := filepath.Join(os.TempDir(), esocketName())
	callbackCount := 0
	callbackChan := make(chan int)
	p, e := esetup(tCtx, t, []*pluginapi.Device{}, socket, "mock", func(_ klog.Logger, n string, d []*pluginapi.Device) {
		callbackCount++
		callbackChan <- callbackCount
	})
	defer func() {
		err := ecleanup(tCtx.Logger(), p, e)
		require.NoError(t, err)
	}()

	resp := &pluginapi.PreferredAllocationResponse{
		ContainerResponses: []*pluginapi.ContainerPreferredAllocationResponse{
			{DeviceIDs: []string{"device0", "device1", "device2"}},
		},
	}

	p.SetGetPreferredAllocFunc(func(r *pluginapi.PreferredAllocationRequest, devs map[string]*pluginapi.Device) (*pluginapi.PreferredAllocationResponse, error) {
		return resp, nil
	})

	go e.client.Run(tCtx)
	// Wait for the callback to be issued.
	select {
	case <-callbackChan:
		break
	case <-time.After(time.Second):
		t.FailNow()
	}

	respOut, err := e.getPreferredAllocation(tCtx, []string{}, []string{}, -1)
	require.NoError(t, err)
	require.True(t, proto.Equal(resp, respOut))
}

func esetup(ctx context.Context, t *testing.T, devs []*pluginapi.Device, socket, resourceName string, callback monitorCallback) (*plugin.Stub, *endpointImpl) {
	logger := klog.FromContext(ctx)
	m := newMockPluginManager()

	m.pluginListAndWatchReceiver = func(r string, resp *pluginapi.ListAndWatchResponse) {
		var newDevs []*pluginapi.Device
		for _, d := range resp.Devices {
			newDevs = append(newDevs, d)
		}
		callback(klog.FromContext(ctx), resourceName, newDevs)
	}

	var dp plugin.DevicePlugin
	var wg sync.WaitGroup
	wg.Add(1)
	m.pluginConnected = func(r string, c plugin.DevicePlugin) error {
		dp = c
		wg.Done()
		return nil
	}

	p := plugin.NewDevicePluginStub(logger, devs, socket, resourceName, false, false)
	err := p.Start(ctx)
	require.NoError(t, err)

	c := plugin.NewPluginClient(resourceName, socket, m)
	err = c.Connect(ctx)
	require.NoError(t, err)

	wg.Wait()

	e := newEndpointImpl(dp)
	e.client = c

	m.pluginDisconnected = func(logger klog.Logger, r string) {
		e.setStopTime(time.Now())
	}

	return p, e
}

func ecleanup(logger klog.Logger, p *plugin.Stub, e *endpointImpl) error {
	if err := p.Stop(logger); err != nil {
		return err
	}
	return e.client.Disconnect(logger)
}
