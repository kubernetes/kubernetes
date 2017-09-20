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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

const (
	socketName       = "/tmp/device_plugin/server.sock"
	pluginSocketName = "/tmp/device_plugin/device-plugin.sock"
	testResourceName = "fake-domain/resource"
)

func TestNewManagerImpl(t *testing.T) {
	_, err := NewManagerImpl("", func(n string, a, u, r []*pluginapi.Device) {})
	require.Error(t, err)

	_, err = NewManagerImpl(socketName, func(n string, a, u, r []*pluginapi.Device) {})
	require.NoError(t, err)
}

func TestNewManagerImplStart(t *testing.T) {
	m, p := setup(t, []*pluginapi.Device{}, func(n string, a, u, r []*pluginapi.Device) {})
	cleanup(t, m, p)
}

// Tests that the device plugin manager correctly handles registration and re-registration by
// making sure that after registration, devices are correctly updated and if a re-registration
// happens, we will NOT delete devices.
func TestDevicePluginReRegistration(t *testing.T) {
	devs := []*pluginapi.Device{
		{ID: "Dev1", Health: pluginapi.Healthy},
		{ID: "Dev2", Health: pluginapi.Healthy},
	}

	callbackCount := 0
	callbackChan := make(chan int)
	var stopping int32
	stopping = 0
	callback := func(n string, a, u, r []*pluginapi.Device) {
		// Should be called twice, one for each plugin registration, till we are stopping.
		if callbackCount > 1 && atomic.LoadInt32(&stopping) <= 0 {
			t.FailNow()
		}
		callbackCount++
		callbackChan <- callbackCount
	}
	m, p1 := setup(t, devs, callback)
	p1.Register(socketName, testResourceName)
	// Wait for the first callback to be issued.
	<-callbackChan
	devices := m.Devices()
	require.Equal(t, 2, len(devices[testResourceName]), "Devices are not updated.")

	p2 := NewDevicePluginStub(devs, pluginSocketName+".new")
	err := p2.Start()
	require.NoError(t, err)
	p2.Register(socketName, testResourceName)
	// Wait for the second callback to be issued.
	<-callbackChan

	devices2 := m.Devices()
	require.Equal(t, 2, len(devices2[testResourceName]), "Devices shouldn't change.")
	// Wait long enough to catch unexpected callbacks.
	time.Sleep(5 * time.Second)

	atomic.StoreInt32(&stopping, 1)
	cleanup(t, m, p1)
	p2.Stop()
}

func setup(t *testing.T, devs []*pluginapi.Device, callback MonitorCallback) (Manager, *Stub) {
	m, err := NewManagerImpl(socketName, callback)
	require.NoError(t, err)
	err = m.Start()
	require.NoError(t, err)

	p := NewDevicePluginStub(devs, pluginSocketName)
	err = p.Start()
	require.NoError(t, err)

	return m, p
}

func cleanup(t *testing.T, m Manager, p *Stub) {
	p.Stop()
	m.Stop()
}
