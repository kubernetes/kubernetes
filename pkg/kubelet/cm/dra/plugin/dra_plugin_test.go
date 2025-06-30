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
	"path"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

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

// tearDown is an idempotent cleanup function.
type tearDown func()

func setupFakeGRPCServer(service, addr string) (tearDown, error) {
	ctx, cancel := context.WithCancel(context.Background())
	teardown := func() {
		cancel()
	}

	listener, err := net.Listen("unix", addr)
	if err != nil {
		teardown()
		return nil, err
	}

	s := grpc.NewServer()
	fakeGRPCServer := &fakeGRPCServer{}
	switch service {
	case drapbv1beta1.DRAPluginService:
		drapbv1beta1.RegisterDRAPluginServer(s, fakeGRPCServer)
	default:
		return nil, fmt.Errorf("unsupported gRPC service: %s", service)
	}

	go func() {
		go func() {
			if err := s.Serve(listener); err != nil {
				panic(err)
			}
		}()
		<-ctx.Done()
		s.GracefulStop()
	}()

	return teardown, nil
}

func TestGRPCConnIsReused(t *testing.T) {
	tCtx := ktesting.Init(t)
	service := drapbv1beta1.DRAPluginService
	addr := path.Join(t.TempDir(), "dra.sock")
	teardown, err := setupFakeGRPCServer(service, addr)
	if err != nil {
		t.Fatal(err)
	}
	defer teardown()

	reusedConns := make(map[*grpc.ClientConn]int)
	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	driverName := "dummy-driver"

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, 0)
	tCtx.ExpectNoError(draPlugins.add(driverName, addr, service, defaultClientCallTimeout), "add plugin")
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

			req := &drapbv1beta1.NodePrepareResourcesRequest{
				Claims: []*drapbv1beta1.Claim{
					{
						Namespace: "dummy-namespace",
						UID:       "dummy-uid",
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
	if len(reusedConns) != 1 {
		t.Errorf("expected length to be 1 but got %d", len(reusedConns))
	}
	if counter, ok := reusedConns[conn]; ok && counter != 2 {
		t.Errorf("expected counter to be 2 but got %d", counter)
	}
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
			draPlugins := NewDRAPluginManager(tCtx, nil, nil, 0)
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
			// In practice, kubelet wouldn't choose an invalid service.
			description:   "internal-error",
			service:       drapbv1beta1.DRAPluginService,
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
			draPlugins := NewDRAPluginManager(tCtx, nil, nil, 0)
			tCtx.ExpectNoError(draPlugins.add(driverName, addr, test.chosenService, defaultClientCallTimeout))
			plugin, err := draPlugins.GetPlugin(driverName)
			if err != nil {
				t.Fatal(err)
			}

			_, err = plugin.NodePrepareResources(tCtx, &drapbv1beta1.NodePrepareResourcesRequest{})
			assertError(t, test.expectError, err)

			_, err = plugin.NodeUnprepareResources(tCtx, &drapbv1beta1.NodeUnprepareResourcesRequest{})
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
