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
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/util/sets"
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
		description      string
		expectedEndpoint string
		returnErr        error
	}{
		{
			description:      "Successfully register plugin through inotify",
			expectedEndpoint: "",
			returnErr:        nil,
		},
		{
			description:      "Successfully register plugin through inotify and got expected optional endpoint",
			expectedEndpoint: "dummyEndpoint",
			returnErr:        nil,
		},
		{
			description:      "Fails registration because endpoint is expected to be non-empty",
			expectedEndpoint: "dummyEndpoint",
			returnErr:        fmt.Errorf("empty endpoint received"),
		},
		{
			description:      "Successfully register plugin through inotify after plugin restarts",
			expectedEndpoint: "",
			returnErr:        nil,
		},
		{
			description:      "Fails registration with conflicting plugin name",
			expectedEndpoint: "",
			returnErr:        fmt.Errorf("conflicting plugin name"),
		},
		{
			description:      "Successfully register plugin during initial traverse after plugin watcher restarts",
			expectedEndpoint: "",
			returnErr:        nil,
		},
		{
			description:      "Fails registration with conflicting plugin name during initial traverse after plugin watcher restarts",
			expectedEndpoint: "",
			returnErr:        fmt.Errorf("conflicting plugin name"),
		},
	}

	callbackCount := struct {
		mutex sync.Mutex
		count int32
	}{}
	w.AddHandler(PluginType, func(name string, endpoint string, versions []string, sockPath string) (error, chan bool) {
		callbackCount.mutex.Lock()
		localCount := callbackCount.count
		callbackCount.count = callbackCount.count + 1
		callbackCount.mutex.Unlock()

		require.True(t, localCount <= int32((len(testCases)-1)))
		require.Equal(t, PluginName, name, "Plugin name mismatched!!")
		retError := testCases[localCount].returnErr
		if retError == nil || retError.Error() != "empty endpoint received" {
			require.Equal(t, testCases[localCount].expectedEndpoint, endpoint, "Unexpected endpoint")
		} else {
			require.NotEqual(t, testCases[localCount].expectedEndpoint, endpoint, "Unexpected endpoint")
		}

		require.Equal(t, []string{"v1beta1", "v1beta2"}, versions, "Plugin version mismatched!!")
		// Verifies the grpcServer is ready to serve services.
		_, conn, err := dial(sockPath)
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
		//atomic.AddInt32(&callbackCount, 1)
		chanForAckOfNotification := make(chan bool)

		go func() {
			select {
			case <-chanForAckOfNotification:
				close(chanForAckOfNotification)
			case <-time.After(time.Second):
				t.Fatalf("Timed out while waiting for notification ack")
			}
		}()
		return retError, chanForAckOfNotification
	})
	require.NoError(t, w.Start())

	p := NewTestExamplePlugin("")
	require.NoError(t, p.Serve(socketPath))
	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	require.NoError(t, p.Stop())

	p = NewTestExamplePlugin("dummyEndpoint")
	require.NoError(t, p.Serve(socketPath))
	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	require.NoError(t, p.Stop())

	p = NewTestExamplePlugin("")
	require.NoError(t, p.Serve(socketPath))
	require.False(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	// Trying to start a plugin service at the same socket path should fail
	// with "bind: address already in use"
	require.NotNil(t, p.Serve(socketPath))

	// grpcServer.Stop() will remove the socket and starting plugin service
	// at the same path again should succeeds and trigger another callback.
	require.NoError(t, p.Stop())
	p = NewTestExamplePlugin("")
	go func() {
		require.Nil(t, p.Serve(socketPath))
	}()
	require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

	// Starting another plugin with the same name got verification error.
	p2 := NewTestExamplePlugin("")
	socketPath2 := socketDir + "/plugin2.sock"
	go func() {
		require.NoError(t, p2.Serve(socketPath2))
	}()
	require.False(t, waitForPluginRegistrationStatus(t, p2.registrationStatus))

	// Restarts plugin watcher should traverse the socket directory and issues a
	// callback for every existing socket.
	require.NoError(t, w.Stop())
	errCh := make(chan error)
	go func() {
		errCh <- w.Start()
	}()

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
	wg.Wait()
	expectedSet := sets.NewString()
	expectedSet.Insert("true", "false")
	actualSet := sets.NewString()
	actualSet.Insert(pStatus, p2Status)

	require.Equal(t, expectedSet, actualSet)

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
