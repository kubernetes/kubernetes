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

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

func TestCloneDevice(t *testing.T) {
	d := cloneDevice(&pluginapi.Device{ID: "ADeviceId", Health: pluginapi.Healthy})

	require.Equal(t, d.ID, "ADeviceId")
	require.Equal(t, d.Health, pluginapi.Healthy)
}

func TestCopyDevices(t *testing.T) {
	d := map[string]*pluginapi.Device{
		"ADeviceId": {ID: "ADeviceId", Health: pluginapi.Healthy},
	}

	devs := copyDevices(d)
	require.Len(t, devs, 1)
}

func TestIsResourceName(t *testing.T) {
	require.NotNil(t, IsResourceNameValid(""))
	require.NotNil(t, IsResourceNameValid("cpu"))
	require.NotNil(t, IsResourceNameValid("name1"))
	require.NotNil(t, IsResourceNameValid("alpha.kubernetes.io/name1"))
	require.NotNil(t, IsResourceNameValid("beta.kubernetes.io/name1"))
	require.NotNil(t, IsResourceNameValid("kubernetes.io/name1"))
	require.Nil(t, IsResourceNameValid("domain1.io/name1"))
	require.Nil(t, IsResourceNameValid("alpha.domain1.io/name1"))
	require.Nil(t, IsResourceNameValid("beta.domain1.io/name1"))
}
