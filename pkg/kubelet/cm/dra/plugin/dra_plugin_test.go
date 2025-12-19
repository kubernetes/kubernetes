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
	"errors"
	"io"
	"net"
	"path"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1"
	drapbv1beta1 "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	"k8s.io/kubernetes/test/utils/ktesting"

	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
)

// this interface satisfies what setupGRPCServerWithFake needs
type fakeGRPCServerInterface interface {
	drapbv1.DRAPluginServer
	drahealthv1alpha1.DRAResourceHealthServer
	drapbv1.UnsafeDRAPluginServer
}

type fakeGRPCServer struct {
	drapbv1beta1.UnimplementedDRAPluginServer
	drahealthv1alpha1.UnimplementedDRAResourceHealthServer
	drapbv1.UnsafeDRAPluginServer
}

var _ drapbv1.DRAPluginServer = &fakeGRPCServer{}
var _ fakeGRPCServerInterface = &fakeGRPCServer{}

func (f *fakeGRPCServer) NodePrepareResources(ctx context.Context, in *drapbv1.NodePrepareResourcesRequest) (*drapbv1.NodePrepareResourcesResponse, error) {
	return &drapbv1.NodePrepareResourcesResponse{Claims: map[string]*drapbv1.NodePrepareResourceResponse{"claim-uid": {
		Devices: []*drapbv1.Device{
			{
				RequestNames: []string{"test-request"},
				CdiDeviceIds: []string{"test-cdi-id"},
			},
		},
	}}}, nil
}

func (f *fakeGRPCServer) NodeUnprepareResources(ctx context.Context, in *drapbv1.NodeUnprepareResourcesRequest) (*drapbv1.NodeUnprepareResourcesResponse, error) {

	return &drapbv1.NodeUnprepareResourcesResponse{}, nil
}

func (f *fakeGRPCServer) NodeWatchResources(in *drahealthv1alpha1.NodeWatchResourcesRequest, srv drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesServer) error {
	resp := &drahealthv1alpha1.NodeWatchResourcesResponse{
		Devices: []*drahealthv1alpha1.DeviceHealth{
			{
				Device: &drahealthv1alpha1.DeviceIdentifier{
					PoolName:   "pool1",
					DeviceName: "dev1",
				},
				Health: drahealthv1alpha1.HealthStatus_HEALTHY,
			},
		},
	}
	if err := srv.Send(resp); err != nil {
		return err
	}
	return nil
}

// tearDown is an idempotent cleanup function.
type tearDown func()

func setupGRPCServerWithFake(service, addr string, fakeGRPCServer fakeGRPCServerInterface) (tearDown, error) {
	ctx, cancel := context.WithCancel(context.Background())

	listener, err := net.Listen("unix", addr)
	if err != nil {
		cancel()
		return nil, err
	}

	s := grpc.NewServer()

	switch service {
	case drapbv1.DRAPluginService:
		drapbv1.RegisterDRAPluginServer(s, fakeGRPCServer)
	case drapbv1beta1.DRAPluginService:
		drapbv1beta1.RegisterDRAPluginServer(s, drapbv1beta1.V1ServerWrapper{DRAPluginServer: fakeGRPCServer})
	case drahealthv1alpha1.DRAResourceHealth_ServiceDesc.ServiceName:
		drahealthv1alpha1.RegisterDRAResourceHealthServer(s, fakeGRPCServer)
	default:
		if service == "" {
			drapbv1.RegisterDRAPluginServer(s, fakeGRPCServer)
			drapbv1beta1.RegisterDRAPluginServer(s, drapbv1beta1.V1ServerWrapper{DRAPluginServer: fakeGRPCServer})
			drahealthv1alpha1.RegisterDRAResourceHealthServer(s, fakeGRPCServer)
		} else {
			cancel()
			return nil, err
		}
	}

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		if err := s.Serve(listener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			panic(err)
		}
	}()

	go func() {
		<-ctx.Done()
		s.GracefulStop()
	}()

	teardown := func() {
		cancel()
		wg.Wait() // wait for Serve() to finish
	}

	return teardown, nil
}

func setupFakeGRPCServer(service, addr string) (tearDown, error) {
	return setupGRPCServerWithFake(service, addr, &fakeGRPCServer{})
}

func TestGRPCConnIsReused(t *testing.T) {
	tCtx := ktesting.Init(t)
	addr := path.Join(t.TempDir(), "dra.sock")
	teardown, err := setupFakeGRPCServer("", addr)
	require.NoError(t, err)
	defer teardown()

	reusedConns := make(map[*grpc.ClientConn]int)
	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	driverName := "dummy-driver"

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0)
	tCtx.ExpectNoError(draPlugins.add(driverName, addr, drapbv1.DRAPluginService, defaultClientCallTimeout), "add plugin")
	plugin, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get plugin")
	conn := plugin.conn

	// we call `NodePrepareResource` 2 times and check whether a new connection is created or the same is reused
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			plugin, err := draPlugins.GetPlugin(driverName)
			if err != nil {
				t.Error(err)
				return
			}

			req := &drapbv1.NodePrepareResourcesRequest{
				Claims: []*drapbv1.Claim{
					{
						Namespace: "dummy-namespace",
						Uid:       "dummy-uid",
						Name:      "dummy-claim",
					},
				},
			}

			_, err = plugin.NodePrepareResources(tCtx, req)
			assert.NoError(t, err)

			conn := plugin.conn

			m.Lock()
			defer m.Unlock()
			reusedConns[conn]++
		}()
	}

	wg.Wait()
	// We should have only one entry otherwise it means another gRPC connection has been created
	require.Len(t, reusedConns, 1, "expected length to be 1 but got %d", len(reusedConns))
	require.Equal(t, 2, reusedConns[conn], "expected counter to be 2 but got %d", reusedConns[conn])

	tCtx.Run("health_api_reuses_connection", func(tCtx ktesting.TContext) {
		ctx, cancel := context.WithTimeout(tCtx, 5*time.Second)
		defer cancel()

		originalConn := plugin.conn

		stream, err := plugin.NodeWatchResources(ctx)
		require.NoError(tCtx, err, "Health stream should work")
		require.NotNil(tCtx, stream)

		require.Equal(tCtx, originalConn, plugin.conn, "Health API should reuse the same connection")

		resp, err := stream.Recv()
		require.NoError(tCtx, err, "Should receive health data")
		require.NotNil(tCtx, resp)
		require.Len(tCtx, resp.Devices, 1)
		assert.Equal(tCtx, "pool1", resp.Devices[0].GetDevice().GetPoolName())
		assert.Equal(tCtx, "dev1", resp.Devices[0].GetDevice().GetDeviceName())
		assert.Equal(tCtx, drahealthv1alpha1.HealthStatus_HEALTHY, resp.Devices[0].GetHealth())

		require.Equal(tCtx, originalConn, plugin.conn, "Connection should remain unchanged after health operations")

		prepareReq := &drapbv1.NodePrepareResourcesRequest{
			Claims: []*drapbv1.Claim{
				{
					Namespace: "dummy-namespace",
					Uid:       "dummy-uid",
					Name:      "dummy-claim",
				},
			},
		}

		prepareResp, err := plugin.NodePrepareResources(ctx, prepareReq)
		require.NoError(tCtx, err, "NodePrepareResources should work")
		require.NotNil(tCtx, prepareResp)
		require.NotNil(tCtx, prepareResp.Claims["claim-uid"])

		require.Equal(tCtx, originalConn, plugin.conn, "Connection should remain unchanged after NodePrepareResources")

		unprepareReq := &drapbv1.NodeUnprepareResourcesRequest{
			Claims: []*drapbv1.Claim{
				{
					Namespace: "dummy-namespace",
					Uid:       "dummy-uid",
					Name:      "dummy-claim",
				},
			},
		}

		unprepareResp, err := plugin.NodeUnprepareResources(ctx, unprepareReq)
		require.NoError(tCtx, err, "NodeUnprepareResources should work")
		require.NotNil(tCtx, unprepareResp)

		require.Equal(tCtx, originalConn, plugin.conn, "Connection should remain unchanged after all API calls")
	})
}

func TestGRPCConnUsableAfterIdle(t *testing.T) {
	tCtx := ktesting.Init(t)
	service := drapbv1.DRAPluginService
	addr := path.Join(t.TempDir(), "dra.sock")
	teardown, err := setupFakeGRPCServer(service, addr)
	require.NoError(t, err)
	defer teardown()

	driverName := "dummy-driver"

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0)
	draPlugins.withIdleTimeout = 5 * time.Second
	tCtx.ExpectNoError(draPlugins.add(driverName, addr, service, defaultClientCallTimeout), "add plugin")
	plugin, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get plugin")

	// The connection doesn't really become idle because HandleConn
	// kicks it back to ready by calling Connect. Just sleep long
	// enough here, the code should be reached...
	tCtx.Log("Waiting for idle timeout...")
	time.Sleep(2 * draPlugins.withIdleTimeout)

	req := &drapbv1.NodePrepareResourcesRequest{
		Claims: []*drapbv1.Claim{
			{
				Namespace: "dummy-namespace",
				Uid:       "dummy-uid",
				Name:      "dummy-claim",
			},
		},
	}

	callCtx := ktesting.WithTimeout(tCtx, 10*time.Second, "call timed out")
	_, err = plugin.NodePrepareResources(callCtx, req)
	tCtx.ExpectNoError(err, "NodePrepareResources")
}

func TestGetDRAPlugin(t *testing.T) {
	for _, test := range []struct {
		description string
		setup       func(*DRAPluginManager) error
		driverName  string
		shouldError bool
	}{
		{
			description: "driver-name is empty",
			shouldError: true,
		},
		{
			description: "driver name not found in the list",
			driverName:  "driver-name-not-found-in-the-list",
			shouldError: true,
		},
		{
			description: "plugin exists",
			setup: func(draPlugins *DRAPluginManager) error {
				return draPlugins.add("dummy-driver", "/tmp/dra.sock", "", defaultClientCallTimeout)
			},
			driverName: "dummy-driver",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0)
			if test.setup != nil {
				require.NoError(t, test.setup(draPlugins), "setup plugin")
			}
			plugin, err := draPlugins.GetPlugin(test.driverName)
			if test.shouldError {
				assert.Nil(t, plugin)
				assert.Error(t, err)
			} else {
				assert.NotNil(t, plugin)
				assert.NoError(t, err)
			}
		})
	}
}

func TestGRPCMethods(t *testing.T) {
	for _, test := range []struct {
		description   string
		service       string
		chosenService string
		expectError   string
	}{
		{
			description:   "v1beta1",
			service:       drapbv1beta1.DRAPluginService,
			chosenService: drapbv1beta1.DRAPluginService,
		},
		{
			description:   "v1",
			service:       drapbv1.DRAPluginService,
			chosenService: drapbv1.DRAPluginService,
		},
		{
			// In practice, such a mismatch between plugin and kubelet should not happen.
			description:   "mismatch",
			service:       drapbv1.DRAPluginService,
			chosenService: drapbv1beta1.DRAPluginService,
			expectError:   "unknown service k8s.io.kubelet.pkg.apis.dra.v1beta1.DRAPlugin",
		},
		{
			// In practice, kubelet wouldn't choose an invalid service.
			description:   "internal-error",
			service:       drapbv1.DRAPluginService,
			chosenService: "some-other-service",
			expectError:   "unsupported chosen service",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			addr := path.Join(t.TempDir(), "dra.sock")
			teardown, err := setupFakeGRPCServer(test.service, addr)
			if err != nil {
				t.Fatal(err)
			}
			defer teardown()

			driverName := "dummy-driver"
			draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0)
			tCtx.ExpectNoError(draPlugins.add(driverName, addr, test.chosenService, defaultClientCallTimeout))

			plugin, err := draPlugins.GetPlugin(driverName)
			if err != nil {
				t.Fatal(err)
			}

			_, err = plugin.NodePrepareResources(tCtx, &drapbv1.NodePrepareResourcesRequest{})
			assertError(t, test.expectError, err)

			_, err = plugin.NodeUnprepareResources(tCtx, &drapbv1.NodeUnprepareResourcesRequest{})
			assertError(t, test.expectError, err)
		})
	}
}

func TestGRPCWithTimeoutEnforced(t *testing.T) {
	tCtx := ktesting.Init(t)

	service := drapbv1.DRAPluginService
	addr := path.Join(t.TempDir(), "dra.sock")

	// we force NodePrepareResources to block, this will cause the timeout
	// to elapse, as a consequence the client should abort.
	// once the client aborts, we unblock NodePrepareResources
	blocked := make(chan struct{})
	server := &timeoutFakeGRPCServer{
		t:              t,
		fakeGRPCServer: &fakeGRPCServer{},
		blocked:        blocked,
		done:           make(chan struct{}),
	}
	teardown, err := setupGRPCServerWithFake(service, addr, server)
	require.NoError(t, err, "failed to setup grpc server")
	defer teardown()

	driverName := "dummy-driver"
	timeout := time.Second
	manager := NewDRAPluginManager(tCtx, nil, nil, nil, 0)
	err = manager.add(driverName, addr, service, timeout)
	require.NoError(t, err, "unexpected error while adding the plugin")

	plugin, err := manager.GetPlugin(driverName)
	require.NoError(t, err, "unexpected error while retrieving the plugin")

	// we will invoke the method on a new gorouinte, in case there
	// is no timeout enforced it might block forever
	errCh := make(chan error, 1)
	go func() {
		defer close(errCh)
		req := &drapbv1.NodePrepareResourcesRequest{
			Claims: []*drapbv1.Claim{
				{
					Namespace: "dummy-namespace",
					Uid:       "dummy-uid",
					Name:      "dummy-claim",
				},
			},
		}
		_, err := plugin.NodePrepareResources(tCtx, req)
		errCh <- err
	}()

	// wait for the grpc caller to timeout
	select {
	// not using wait.ForeverTestTimeout, we will wait at most
	// 3*timeout to account for flakes in CI
	case <-time.After(3 * timeout):
		t.Errorf("expected the grpc caller to return after the timeout had elapsed")
	case err = <-errCh:
		// the grpc call returned
	}

	// unblock the request handler on the server, and wait for it to
	// return, this ensures that the server method was invoked.
	// if the timeout is not enforced, we don't leak any goroutine
	close(blocked)
	select {
	case <-server.done:
	// not using wait.ForeverTestTimeout, we will wait at most
	// 3*timeout to account for flakes in CI
	case <-time.After(3 * timeout):
		t.Errorf("expected the grpc method to have been invoked")
	}

	require.Error(t, err, "expected the grpc method to return an error")
	status, ok := grpcstatus.FromError(err)
	// if it is not a gRPC error then the operation may have failed before
	// the gRPC method was called, otherwise we would get gRPC error.
	require.True(t, ok, "expected an error of type: %T, but got: %T", &grpcstatus.Status{}, err)

	// DeadlineExceeded means operation expired before completion. The gRPC
	// framework will generate this error code when the deadline is exceeded.
	// More here: https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
	assert.Equal(t, grpccodes.DeadlineExceeded, status.Code(), "expected status code to match")
}

// it embeds a fakeGRPCServer instance and overrides the NodePrepareResources
// method to simulate a timeout scenario.
// NodePrepareResources will block until the test explicitly closes the blocked
// channel to unblock it, and it will close the done channel when it completes
// so the test can wait for it to finish.
type timeoutFakeGRPCServer struct {
	*fakeGRPCServer

	t       *testing.T
	blocked <-chan struct{}
	done    chan struct{}
}

func (f *timeoutFakeGRPCServer) NodePrepareResources(ctx context.Context, in *drapbv1.NodePrepareResourcesRequest) (*drapbv1.NodePrepareResourcesResponse, error) {
	defer close(f.done)
	now := time.Now()
	deadline, ok := ctx.Deadline()
	f.t.Logf("request context has deadline: %t, after: %s", ok, deadline.Sub(now))
	if !ok {
		f.t.Errorf("expected the request context to have a deadline")
	}
	f.t.Logf("NodePrepareResources: blocking so the client times out before the server sends a reply")

	<-f.blocked

	f.t.Logf("NodePrepareResources: blocked for: %s", time.Since(now))
	return &drapbv1.NodePrepareResourcesResponse{}, nil
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

func TestPlugin_WatchResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()

	driverName := "test-driver"
	addr := path.Join(t.TempDir(), "dra.sock")

	teardown, err := setupFakeGRPCServer("", addr)
	require.NoError(t, err)
	defer teardown()

	draPlugins := NewDRAPluginManager(tCtx, nil, nil, &mockStreamHandler{}, 0)
	err = draPlugins.add(driverName, addr, drapbv1beta1.DRAPluginService, 5*time.Second)
	require.NoError(t, err)
	defer draPlugins.remove(driverName, addr)

	p, err := draPlugins.GetPlugin(driverName)
	require.NoError(t, err)

	stream, err := p.NodeWatchResources(ctx)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// 1. Receive the first message that our fake server sends.
	resp, err := stream.Recv()
	require.NoError(t, err, "The first Recv() should succeed with the message from the server")
	require.NotNil(t, resp)
	require.Len(t, resp.Devices, 1)
	assert.Equal(t, "pool1", resp.Devices[0].GetDevice().GetPoolName())
	assert.Equal(t, drahealthv1alpha1.HealthStatus_HEALTHY, resp.Devices[0].GetHealth())

	// 2. The second receive should fail with io.EOF because the server
	//    closed the stream by returning nil. This confirms the stream ended cleanly.
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF, "The second Recv() should return an io.EOF error to signal a clean stream closure")
}
