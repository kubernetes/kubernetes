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
	"testing"

	"github.com/stretchr/testify/require"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

func TestEndpointStore(t *testing.T) {
	p := NewDevicePluginStub(nil, pluginSocketName)
	require.NoError(t, p.Start())

	store := newEndpointStoreImpl()

	dStore := newDeviceStoreImpl(func(n string, a, u, r []pluginapi.Device) {})
	e, err := newEndpointWithStore(pluginSocketName, testResourceName, dStore)
	require.NoError(t, err)

	_, ok := store.Endpoint(e.ResourceName())
	require.False(t, ok)

	_, ok = store.SynchronizedEndpoint(e.ResourceName())
	require.False(t, ok)

	store.SwapEndpoint(e)

	curr, ok := store.Endpoint(e.ResourceName())
	require.True(t, ok)
	require.Equal(t, curr, e)

	se, ok := store.SynchronizedEndpoint(e.ResourceName())
	require.True(t, ok)
	require.Equal(t, se.e, e)

	devs := make(map[string][]pluginapi.Device)
	store.Range(func(k string, s *synchronizedEndpoint) {
		devs[k] = s.e.Store().Devices()
	})

	require.Len(t, devs, 1)

	require.NoError(t, e.Stop())
	p.Stop()
}

func TestSwapEndpoint(t *testing.T) {
	p1 := NewDevicePluginStub(nil, pluginSocketName)
	require.NoError(t, p1.Start())

	p2 := NewDevicePluginStub(nil, pluginSocketName+".new")
	require.NoError(t, p2.Start())

	store := newEndpointStoreImpl()

	f := func(n string, a, u, r []pluginapi.Device) {}
	dStore := newDeviceStoreImpl(f)
	e1, err := newEndpointWithStore(pluginSocketName, testResourceName, dStore)
	require.NoError(t, err)

	_, ok := store.SwapEndpoint(e1)
	require.False(t, ok)

	e2, err := newEndpointWithStore(pluginSocketName+".new", testResourceName, dStore)
	require.NoError(t, err)

	old, ok := store.SwapEndpoint(e2)
	require.True(t, ok)
	require.Equal(t, old, e1)

	curr, ok := store.Endpoint(e2.ResourceName())
	require.True(t, ok)
	require.Equal(t, curr, e2)

	se, ok := store.SynchronizedEndpoint(e2.ResourceName())
	require.True(t, ok)
	require.Equal(t, se.e, e2)

	require.NoError(t, e1.Stop())
	require.NoError(t, e2.Stop())
	p1.Stop()
	p2.Stop()
}

func TestDeleteEndpoint(t *testing.T) {
	p := NewDevicePluginStub(nil, pluginSocketName)
	require.NoError(t, p.Start())

	store := newEndpointStoreImpl()

	f := func(n string, a, u, r []pluginapi.Device) {}
	dStore := newDeviceStoreImpl(f)
	e, err := newEndpointWithStore(pluginSocketName, testResourceName, dStore)
	require.NoError(t, err)

	err = store.DeleteEndpoint(e.ResourceName())
	require.Error(t, err)

	_, ok := store.SwapEndpoint(e)
	require.False(t, ok)
	err = store.DeleteEndpoint(e.ResourceName())
	require.NoError(t, err)

	err = store.DeleteEndpoint(e.ResourceName())
	require.Error(t, err)

	require.NoError(t, e.Stop())
	p.Stop()
}
