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
	"time"

	"github.com/stretchr/testify/require"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
)

func TestDeviceStore(t *testing.T) {
	callbackChan := make(chan interface{})
	callbackFunc := func(n string, a, u, r []pluginapi.Device) {
		require.Equal(t, n, "foo")
		require.Len(t, a, 0)
		require.Len(t, u, 1)
		require.Len(t, r, 1)

		close(callbackChan)
	}

	d := newDeviceStoreImpl(callbackFunc)
	a, u, r := d.Update([]*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
		{ID: "AnotherDeviceId", Health: pluginapi.Healthy},
	})

	require.Len(t, a, 2)
	require.Len(t, u, 0)
	require.Len(t, r, 0)

	a, u, r = d.Update([]*pluginapi.Device{
		{ID: "AnotherDeviceId", Health: pluginapi.Unhealthy},
	})

	require.Len(t, a, 0)
	require.Len(t, u, 1)
	require.Len(t, r, 1)

	d.Callback("foo", a, u, r)
	select {
	case <-callbackChan:
		break
	case <-time.After(time.Second):
		t.FailNow()
	}

	require.Len(t, d.Devices(), 1)
	require.Len(t, d.HealthyDevices(), 0)
}

func TestAlwaysEmptyDeviceStore(t *testing.T) {
	d := newAlwaysEmptyDeviceStore()
	a, u, r := d.Update([]*pluginapi.Device{
		{ID: "ADeviceId", Health: pluginapi.Healthy},
		{ID: "AnotherDeviceId", Health: pluginapi.Healthy},
	})

	require.Len(t, a, 0)
	require.Len(t, u, 0)
	require.Len(t, r, 0)

	a, u, r = d.Update([]*pluginapi.Device{
		{ID: "AnotherDeviceId", Health: pluginapi.Unhealthy},
	})

	require.Len(t, a, 0)
	require.Len(t, u, 0)
	require.Len(t, r, 0)

	d.Callback("foo", a, u, r)
	require.Len(t, d.Devices(), 0)
	require.Len(t, d.HealthyDevices(), 0)
}
