/*
Copyright 2018 The Kubernetes Authors.

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

package pluginwatcher

import (
	"errors"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/sets"
	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1alpha1"
)

// helper function
func waitTimeout(wg *sync.WaitGroup, timeout time.Duration) bool {
	c := make(chan struct{})
	go func() {
		defer close(c)
		wg.Wait()
	}()
	select {
	case <-c:
		return false // completed normally
	case <-time.After(timeout):
		return true // timed out
	}
}

func TestExamplePlugin(t *testing.T) {
	rootDir, err := ioutil.TempDir("", "plugin_test")
	require.NoError(t, err)
	w := NewWatcher(rootDir)
	h := NewExampleHandler()
	w.AddHandler(registerapi.DevicePlugin, h.Handler)

	require.NoError(t, w.Start())

	socketPath := filepath.Join(rootDir, "plugin.sock")
	PluginName := "example-plugin"

	// handler expecting plugin has a non-empty endpoint
	p := NewTestExamplePlugin(PluginName, registerapi.DevicePlugin, "")
	require.NoError(t, p.Serve(socketPath))
	require.False(t, waitForPluginRegistrationStatus(t, p.registrationStatus))
	require.NoError(t, p.Stop())

	p = NewTestExamplePlugin(PluginName, registerapi.DevicePlugin, "dummyEndpoint")
	require.NoError(t, p.Serve(socketPath))
	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	// Trying to start a plugin service at the same socket path should fail
	// with "bind: address already in use"
	require.NotNil(t, p.Serve(socketPath))

	// grpcServer.Stop() will remove the socket and starting plugin service
	// at the same path again should succeeds and trigger another callback.
	require.NoError(t, p.Stop())
	require.Nil(t, p.Serve(socketPath))
	require.False(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	// Starting another plugin with the same name got verification error.
	p2 := NewTestExamplePlugin(PluginName, registerapi.DevicePlugin, "dummyEndpoint")
	socketPath2 := filepath.Join(rootDir, "plugin2.sock")
	require.NoError(t, p2.Serve(socketPath2))
	require.False(t, waitForPluginRegistrationStatus(t, p2.registrationStatus))

	// Restarts plugin watcher should traverse the socket directory and issues a
	// callback for every existing socket.
	require.NoError(t, w.Stop())
	require.NoError(t, h.Cleanup())
	require.NoError(t, w.Start())

	var wg sync.WaitGroup
	wg.Add(2)
	var pStatus string
	var p2Status string
	go func() {
		pStatus = strconv.FormatBool(waitForPluginRegistrationStatus(t, p.registrationStatus))
		wg.Done()
	}()
	go func() {
		p2Status = strconv.FormatBool(waitForPluginRegistrationStatus(t, p2.registrationStatus))
		wg.Done()
	}()

	if waitTimeout(&wg, 2*time.Second) {
		t.Fatalf("Timed out waiting for wait group")
	}

	expectedSet := sets.NewString()
	expectedSet.Insert("true", "false")
	actualSet := sets.NewString()
	actualSet.Insert(pStatus, p2Status)

	require.Equal(t, expectedSet, actualSet)

	select {
	case err := <-h.chanForHandlerAckErrors:
		t.Fatalf("%v", err)
	case <-time.After(2 * time.Second):
	}

	require.NoError(t, w.Stop())
	require.NoError(t, w.Cleanup())
}

func TestPluginWithSubDir(t *testing.T) {
	rootDir, err := ioutil.TempDir("", "plugin_test")
	require.NoError(t, err)

	w := NewWatcher(rootDir)
	hcsi := NewExampleHandler()
	hdp := NewExampleHandler()

	w.AddHandler(registerapi.CSIPlugin, hcsi.Handler)
	w.AddHandler(registerapi.DevicePlugin, hdp.Handler)

	err = w.fs.MkdirAll(filepath.Join(rootDir, registerapi.DevicePlugin), 0755)
	require.NoError(t, err)
	err = w.fs.MkdirAll(filepath.Join(rootDir, registerapi.CSIPlugin), 0755)
	require.NoError(t, err)

	dpSocketPath := filepath.Join(rootDir, registerapi.DevicePlugin, "plugin.sock")
	csiSocketPath := filepath.Join(rootDir, registerapi.CSIPlugin, "plugin.sock")

	require.NoError(t, w.Start())

	// two plugins using the same name but with different type
	dp := NewTestExamplePlugin("exampleplugin", registerapi.DevicePlugin, "example-endpoint")
	require.NoError(t, dp.Serve(dpSocketPath))
	require.True(t, waitForPluginRegistrationStatus(t, dp.registrationStatus))

	csi := NewTestExamplePlugin("exampleplugin", registerapi.CSIPlugin, "example-endpoint")
	require.NoError(t, csi.Serve(csiSocketPath))
	require.True(t, waitForPluginRegistrationStatus(t, csi.registrationStatus))

	// Restarts plugin watcher should traverse the socket directory and issues a
	// callback for every existing socket.
	require.NoError(t, w.Stop())
	require.NoError(t, hcsi.Cleanup())
	require.NoError(t, hdp.Cleanup())
	require.NoError(t, w.Start())

	var wg sync.WaitGroup
	wg.Add(2)
	var dpStatus string
	var csiStatus string
	go func() {
		dpStatus = strconv.FormatBool(waitForPluginRegistrationStatus(t, dp.registrationStatus))
		wg.Done()
	}()
	go func() {
		csiStatus = strconv.FormatBool(waitForPluginRegistrationStatus(t, csi.registrationStatus))
		wg.Done()
	}()

	if waitTimeout(&wg, 4*time.Second) {
		require.NoError(t, errors.New("Timed out waiting for wait group"))
	}

	expectedSet := sets.NewString()
	expectedSet.Insert("true", "true")
	actualSet := sets.NewString()
	actualSet.Insert(dpStatus, csiStatus)

	require.Equal(t, expectedSet, actualSet)

	select {
	case err := <-hcsi.chanForHandlerAckErrors:
		t.Fatalf("%v", err)
	case err := <-hdp.chanForHandlerAckErrors:
		t.Fatalf("%v", err)
	case <-time.After(4 * time.Second):
	}

	require.NoError(t, w.Stop())
	require.NoError(t, w.Cleanup())
}

func waitForPluginRegistrationStatus(t *testing.T, statusCh chan registerapi.RegistrationStatus) bool {
	select {
	case status := <-statusCh:
		return status.PluginRegistered
	case <-time.After(10 * time.Second):
		t.Fatalf("Timed out while waiting for registration status")
	}
	return false
}
