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
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/require"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

const (
	msocketName = "/tmp/server.sock"
)

func TestNewManagerImpl(t *testing.T) {
	wd, _ := os.Getwd()
	socket := path.Join(wd, msocketName)

	_, err := NewManagerImpl("", func(n string, a, u, r []*pluginapi.Device) {})
	require.Error(t, err)

	_, err = NewManagerImpl(socket, func(n string, a, u, r []*pluginapi.Device) {})
	require.NoError(t, err)
}

func TestNewManagerImplStart(t *testing.T) {
	wd, _ := os.Getwd()
	socket := path.Join(wd, msocketName)

	_, err := NewManagerImpl(socket, func(n string, a, u, r []*pluginapi.Device) {})
	require.NoError(t, err)
}

func setup(t *testing.T, devs []*pluginapi.Device, pluginSocket, serverSocket string, callback MonitorCallback) (Manager, *Stub) {
	m, err := NewManagerImpl(serverSocket, callback)
	require.NoError(t, err)

	p := NewDevicePluginStub(devs, pluginSocket)
	err = p.Start()
	require.NoError(t, err)

	return m, p
}

func cleanup(t *testing.T, m Manager, p *Stub) {
	p.Stop()
	m.Stop()
}
