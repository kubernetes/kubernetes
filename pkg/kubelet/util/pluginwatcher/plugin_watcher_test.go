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
	"fmt"
	"io/ioutil"
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/context"

	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1alpha1"
	v1beta1 "k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher/example_plugin_apis/v1beta1"
	v1beta2 "k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher/example_plugin_apis/v1beta2"
)

func TestExamplePlugin(t *testing.T) {
	socketDir, err := ioutil.TempDir("", "plugin_test")
	require.NoError(t, err)
	socketPath := socketDir + "/plugin.sock"
	w := NewWatcher(socketDir)

	testCases := []struct {
		description string
		returnErr   error
	}{
		{
			description: "Successfully register plugin through inotify",
			returnErr:   nil,
		},
		{
			description: "Successfully register plugin through inotify after plugin restarts",
			returnErr:   nil,
		},
		{
			description: "Fails registration with conflicting plugin name",
			returnErr:   fmt.Errorf("conflicting plugin name"),
		},
		{
			description: "Successfully register plugin during initial traverse after plugin watcher restarts",
			returnErr:   nil,
		},
		{
			description: "Fails registration with conflicting plugin name during initial traverse after plugin watcher restarts",
			returnErr:   fmt.Errorf("conflicting plugin name"),
		},
	}

	callbackCount := 0
	w.AddHandler(PluginType, func(name string, versions []string, sockPath string) error {
		require.True(t, callbackCount <= (len(testCases)-1))
		glog.Infof("receives plugin watcher callback for test: %s\n", testCases[callbackCount].description)
		require.Equal(t, PluginName, name, "Plugin name mismatched!!")
		require.Equal(t, []string{"v1beta1", "v1beta2"}, versions, "Plugin version mismatched!!")
		// Verifies the grpcServer is ready to serve services.
		_, conn, err := dial(socketPath)
		require.Nil(t, err)
		defer conn.Close()

		// The plugin handler should be able to use any listed service API version.
		v1beta1Client := v1beta1.NewExampleClient(conn)
		v1beta2Client := v1beta2.NewExampleClient(conn)

		// Tests v1beta1 GetExampleInfo
		_, err = v1beta1Client.GetExampleInfo(context.Background(), &v1beta1.ExampleRequest{})
		require.Nil(t, err)

		// Tests v1beta1 GetExampleInfo
		_, err = v1beta2Client.GetExampleInfo(context.Background(), &v1beta2.ExampleRequest{})
		ret := testCases[callbackCount].returnErr
		callbackCount++
		return ret
	})
	require.NoError(t, w.Start())

	p := NewTestExamplePlugin() //first cbk
	require.Nil(t, p.Serve(socketPath))
	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	// Trying to start a plugin service at the same socket path should fail
	// with "bind: address already in use"
	require.NotNil(t, p.Serve(socketPath))

	// grpcServer.Stop() will remove the socket and starting plugin service
	// at the same path again should succeeds and trigger another callback.
	require.Nil(t, p.Stop())
	p = NewTestExamplePlugin() // 2nd cbk
	go func() {
		require.Nil(t, p.Serve(socketPath))
	}()
	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	// Starting another plugin with the same name got verification error.
	p2 := NewTestExamplePlugin() //3rd cbk
	socketPath2 := socketDir + "/plugin2.sock"
	go func() {
		require.Nil(t, p2.Serve(socketPath2))
	}()
	require.False(t, waitForPluginRegistrationStatus(t, p2.registrationStatus))

	// Restarts plugin watcher should traverse the socket directory and issues a
	// callback for every existing socket.
	require.NoError(t, w.Stop())
	errCh := make(chan error)
	go func() {
		errCh <- w.Start()
	}()

	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))
	require.False(t, waitForPluginRegistrationStatus(t, p2.registrationStatus))

	select {
	case err = <-errCh:
		require.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatalf("Timed out while waiting for watcher start")

	}

	require.NoError(t, w.Stop())
	err = w.Cleanup()
	require.NoError(t, err)
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
