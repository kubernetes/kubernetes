/*
Copyright 2023 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"fmt"
	"net"
	"os"
	"path"
	goruntime "runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	drapbv1alpha4 "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
	drapbv1beta1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type fakeGRPCServer struct {
}

var _ drapbv1beta1.DRAPluginServer = &fakeGRPCServer{}

func (f *fakeGRPCServer) NodePrepareResources(ctx context.Context, in *drapbv1beta1.NodePrepareResourcesRequest) (*drapbv1beta1.NodePrepareResourcesResponse, error) {
	return &drapbv1beta1.NodePrepareResourcesResponse{Claims: map[string]*drapbv1beta1.NodePrepareResourceResponse{"claim-uid": {
		Devices: []*drapbv1beta1.Device{
			{
				RequestNames: []string{"test-request"},
				CDIDeviceIDs: []string{"test-cdi-id"},
			},
		},
	}}}, nil
}

func (f *fakeGRPCServer) NodeUnprepareResources(ctx context.Context, in *drapbv1beta1.NodeUnprepareResourcesRequest) (*drapbv1beta1.NodeUnprepareResourcesResponse, error) {

	return &drapbv1beta1.NodeUnprepareResourcesResponse{}, nil
}

type tearDown func()

func setupFakeGRPCServer(service, endpoint string) (tearDown, error) {
	closeCh := make(chan struct{})
	teardown := func() {
		close(closeCh)
		if err := os.RemoveAll(endpoint); err != nil {
			panic(err)
		}
	}

	listener, err := net.Listen("unix", endpoint)
	if err != nil {
		teardown()
		return nil, err
	}

	s := grpc.NewServer()
	fakeGRPCServer := &fakeGRPCServer{}
	switch service {
	case drapbv1beta1.DRAPluginService:
		drapbv1beta1.RegisterDRAPluginServer(s, fakeGRPCServer)
	case drapbv1alpha4.NodeService:
		drapbv1alpha4.RegisterNodeServer(s, drapbv1alpha4.V1Beta1ServerWrapper{DRAPluginServer: fakeGRPCServer})
	default:
		return nil, fmt.Errorf("unsupported gRPC service: %s", service)
	}

	go func() {
		go func() {
			if err := s.Serve(listener); err != nil {
				panic(err)
			}
		}()
		<-closeCh
		s.GracefulStop()
	}()

	return teardown, nil
}

func TestGRPCConnIsReused(t *testing.T) {
	tCtx := ktesting.Init(t)

	endpoint := path.Join(t.TempDir(), "dra-plugin-test.sock")

	// Create a plugin gRPC server.
	service := drapbv1beta1.DRAPluginService
	teardown, err := setupFakeGRPCServer(service, endpoint)
	require.NoError(t, err)
	defer teardown()

	pluginName := "dummy-plugin"
	p := &Plugin{
		name:              pluginName,
		backgroundCtx:     tCtx,
		endpoint:          endpoint,
		chosenService:     service,
		clientCallTimeout: defaultClientCallTimeout,
	}

	conn, err := p.getOrCreateGRPCConn()
	defer func() {
		err := conn.Close()
		require.NoError(t, err, "failed to close gRPC connection")
	}()
	require.NoError(t, err, "failed to create gRPC connection for plugin %s", pluginName)

	// Add the plugin to the draPlugins store.
	draPlugins.add(p)
	defer draPlugins.remove(pluginName, endpoint)

	// Wait for the plugin to be connected.
	err = WaitForConnection(tCtx, pluginName, endpoint, ConnectionPollInterval, ConnectionTimeout)
	require.NoError(t, err, "failed to connect to plugin %s at %s", pluginName, endpoint)

	// Call NodePrepareResources to initiate a gRPC call.
	req := &drapbv1beta1.NodePrepareResourcesRequest{
		Claims: []*drapbv1beta1.Claim{
			{
				Namespace: "dummy-namespace",
				UID:       "dummy-uid",
				Name:      "dummy-claim",
			},
		},
	}

	cl, err := NewDRAPluginClient(pluginName)
	require.NoError(t, err, "failed to create DRA plugin client for plugin %s", pluginName)

	_, err = cl.NodePrepareResources(tCtx, req)
	require.NoError(t, err, "failed to call NodePrepareResources for plugin %s", pluginName)

	assert.Equal(t, conn, cl.conn, "expected the same gRPC connection to be reused")
}

func TestNewDRAPluginClient(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("DRA is not currently supported on Windows.")
	}
	// Create a plugin gRPC server.
	service := drapbv1beta1.DRAPluginService
	endpoint := path.Join(t.TempDir(), "dra-plugin-test.sock")
	teardown, err := setupFakeGRPCServer(service, endpoint)
	require.NoError(t, err)
	defer teardown()

	for _, test := range []struct {
		description string
		pluginName  string
		shouldError bool
	}{
		{
			description: "plugin name is empty",
			pluginName:  "",
			shouldError: true,
		},
		{
			description: "plugin name not found in the list",
			pluginName:  "plugin-name-not-found-in-the-list",
			shouldError: true,
		},
		{
			description: "plugin not connected",
			pluginName:  "dummy-plugin",
			shouldError: true,
		},
		{
			description: "plugin connected",
			pluginName:  "dummy-plugin",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			plugin := &Plugin{
				name:              test.pluginName,
				backgroundCtx:     tCtx,
				endpoint:          endpoint,
				chosenService:     service,
				clientCallTimeout: defaultClientCallTimeout,
			}

			// Add plugin to the store.
			require.NoError(t, draPlugins.add(plugin))
			defer draPlugins.remove(test.pluginName, endpoint)

			if test.shouldError {
				client, err := NewDRAPluginClient(test.pluginName)
				assert.Nil(t, client)
				assert.Error(t, err)
				return
			}
			// Connect to the plugin.
			conn, err := plugin.getOrCreateGRPCConn()
			defer func() {
				err := conn.Close()
				require.NoError(t, err, "failed to close gRPC connection")
			}()
			require.NoError(t, err, "failed to create gRPC connection for plugin %s", test.pluginName)

			err = WaitForConnection(tCtx, test.pluginName, endpoint, ConnectionPollInterval, ConnectionTimeout)
			require.NoError(t, err, "failed to connect to plugin %s at %s", test.pluginName, endpoint)

			client, err := NewDRAPluginClient(test.pluginName)
			assert.NotNil(t, client)
			assert.NoError(t, err)
		})
	}
}

func TestGRPCMethods(t *testing.T) {
	endpoint := path.Join(t.TempDir(), "dra-plugin-test.sock")
	for _, test := range []struct {
		description   string
		service       string
		chosenService string
		expectError   string
	}{
		{
			description:   "v1alpha4",
			service:       drapbv1alpha4.NodeService,
			chosenService: drapbv1alpha4.NodeService,
		},
		{
			description:   "v1beta1",
			service:       drapbv1beta1.DRAPluginService,
			chosenService: drapbv1beta1.DRAPluginService,
		},
		{
			// In practice, such a mismatch between plugin and kubelet should not happen.
			description:   "mismatch",
			service:       drapbv1beta1.DRAPluginService,
			chosenService: drapbv1alpha4.NodeService,
			expectError:   "unknown service v1alpha3.Node",
		},
		{
			// In practice, kubelet wouldn't choose an invalid service.
			description:   "internal-error",
			service:       drapbv1beta1.DRAPluginService,
			chosenService: "some-other-service",
			expectError:   "unsupported chosen service",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			teardown, err := setupFakeGRPCServer(test.service, endpoint)
			if err != nil {
				t.Fatal(err)
			}
			defer teardown()

			pluginName := "dummy-plugin"
			p := &Plugin{
				name:              pluginName,
				backgroundCtx:     tCtx,
				endpoint:          endpoint,
				chosenService:     test.chosenService,
				clientCallTimeout: defaultClientCallTimeout,
			}

			conn, err := p.getOrCreateGRPCConn()
			defer func() {
				err := conn.Close()
				require.NoError(t, err, "failed to close gRPC connection")
			}()
			require.NoError(t, err, "failed to create gRPC connection for plugin %s", pluginName)

			draPlugins.add(p)
			defer draPlugins.remove(pluginName, endpoint)

			// Wait for the plugin to be connected.
			err = WaitForConnection(tCtx, pluginName, endpoint, ConnectionPollInterval, ConnectionTimeout)
			require.NoError(t, err, "failed to connect to plugin %s at %s", pluginName, endpoint)

			client, err := NewDRAPluginClient(pluginName)
			if err != nil {
				t.Fatal(err)
			}

			_, err = client.NodePrepareResources(tCtx, &drapbv1beta1.NodePrepareResourcesRequest{})
			assertError(t, test.expectError, err)

			_, err = client.NodeUnprepareResources(tCtx, &drapbv1beta1.NodeUnprepareResourcesRequest{})
			assertError(t, test.expectError, err)
		})
	}
}

func assertError(t *testing.T, expectError string, err error) {
	t.Helper()
	switch {
	case err != nil && expectError == "":
		t.Errorf("Expected no error, got: %v", err)
	case err == nil && expectError != "":
		t.Errorf("Expected error %q, got none", expectError)
	case err != nil && !strings.Contains(err.Error(), expectError):
		t.Errorf("Expected error %q, got: %v", expectError, err)
	}
}
