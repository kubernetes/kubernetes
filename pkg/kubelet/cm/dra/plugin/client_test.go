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
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	v1alpha4Version = "v1alpha4"
)

type fakeV1alpha4GRPCServer struct {
	drapb.UnimplementedNodeServer
}

var _ drapb.NodeServer = &fakeV1alpha4GRPCServer{}

func (f *fakeV1alpha4GRPCServer) NodePrepareResources(ctx context.Context, in *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error) {
	return &drapb.NodePrepareResourcesResponse{Claims: map[string]*drapb.NodePrepareResourceResponse{"claim-uid": {
		Devices: []*drapb.Device{
			{
				RequestNames: []string{"test-request"},
				CDIDeviceIDs: []string{"test-cdi-id"},
			},
		},
	}}}, nil
}

func (f *fakeV1alpha4GRPCServer) NodeUnprepareResources(ctx context.Context, in *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error) {

	return &drapb.NodeUnprepareResourcesResponse{}, nil
}

type tearDown func()

func setupFakeGRPCServer(version string) (string, tearDown, error) {
	p, err := os.MkdirTemp("", "dra_plugin")
	if err != nil {
		return "", nil, err
	}

	closeCh := make(chan struct{})
	addr := filepath.Join(p, "server.sock")
	teardown := func() {
		close(closeCh)
		os.RemoveAll(addr)
	}

	listener, err := net.Listen("unix", addr)
	if err != nil {
		teardown()
		return "", nil, err
	}

	s := grpc.NewServer()
	switch version {
	case v1alpha4Version:
		fakeGRPCServer := &fakeV1alpha4GRPCServer{}
		drapb.RegisterNodeServer(s, fakeGRPCServer)
	default:
		return "", nil, fmt.Errorf("unsupported version: %s", version)
	}

	go func() {
		go s.Serve(listener)
		<-closeCh
		s.GracefulStop()
	}()

	return addr, teardown, nil
}

func TestGRPCConnIsReused(t *testing.T) {
	ctx := ktesting.Init(t)
	addr, teardown, err := setupFakeGRPCServer(v1alpha4Version)
	if err != nil {
		t.Fatal(err)
	}
	defer teardown()

	reusedConns := make(map[*grpc.ClientConn]int)
	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	p := &Plugin{
		backgroundCtx: ctx,
		endpoint:      addr,
	}

	conn, err := p.getOrCreateGRPCConn()
	defer func() {
		err := conn.Close()
		if err != nil {
			t.Error(err)
		}
	}()
	if err != nil {
		t.Fatal(err)
	}

	// ensure the plugin we are using is registered
	draPlugins.add("dummy-plugin", p)
	defer draPlugins.delete("dummy-plugin")

	// we call `NodePrepareResource` 2 times and check whether a new connection is created or the same is reused
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client, err := NewDRAPluginClient("dummy-plugin")
			if err != nil {
				t.Error(err)
				return
			}

			req := &drapb.NodePrepareResourcesRequest{
				Claims: []*drapb.Claim{
					{
						Namespace: "dummy-namespace",
						UID:       "dummy-uid",
						Name:      "dummy-claim",
					},
				},
			}
			client.NodePrepareResources(context.TODO(), req)

			client.mutex.Lock()
			conn := client.conn
			client.mutex.Unlock()

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

func TestNewDRAPluginClient(t *testing.T) {
	for _, test := range []struct {
		description string
		setup       func(string) tearDown
		pluginName  string
		shouldError bool
	}{
		{
			description: "plugin name is empty",
			setup: func(_ string) tearDown {
				return func() {}
			},
			pluginName:  "",
			shouldError: true,
		},
		{
			description: "plugin name not found in the list",
			setup: func(_ string) tearDown {
				return func() {}
			},
			pluginName:  "plugin-name-not-found-in-the-list",
			shouldError: true,
		},
		{
			description: "plugin exists",
			setup: func(name string) tearDown {
				draPlugins.add(name, &Plugin{})
				return func() {
					draPlugins.delete(name)
				}
			},
			pluginName: "dummy-plugin",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			teardown := test.setup(test.pluginName)
			defer teardown()

			client, err := NewDRAPluginClient(test.pluginName)
			if test.shouldError {
				assert.Nil(t, client)
				assert.Error(t, err)
			} else {
				assert.NotNil(t, client)
				assert.Nil(t, err)
			}
		})
	}
}

func TestNodeUnprepareResources(t *testing.T) {
	for _, test := range []struct {
		description   string
		serverSetup   func(string) (string, tearDown, error)
		serverVersion string
		request       *drapb.NodeUnprepareResourcesRequest
	}{
		{
			description:   "server supports v1alpha4",
			serverSetup:   setupFakeGRPCServer,
			serverVersion: v1alpha4Version,
			request:       &drapb.NodeUnprepareResourcesRequest{},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			ctx := ktesting.Init(t)
			addr, teardown, err := setupFakeGRPCServer(test.serverVersion)
			if err != nil {
				t.Fatal(err)
			}
			defer teardown()

			p := &Plugin{
				backgroundCtx: ctx,
				endpoint:      addr,
				clientTimeout: PluginClientTimeout,
			}

			conn, err := p.getOrCreateGRPCConn()
			defer func() {
				err := conn.Close()
				if err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}

			draPlugins.add("dummy-plugin", p)
			defer draPlugins.delete("dummy-plugin")

			client, err := NewDRAPluginClient("dummy-plugin")
			if err != nil {
				t.Fatal(err)
			}

			_, err = client.NodeUnprepareResources(context.TODO(), test.request)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
