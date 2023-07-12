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
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"google.golang.org/grpc"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
)

type fakeGRPCServer struct {
	drapbv1.UnimplementedNodeServer
}

func (f *fakeGRPCServer) NodePrepareResource(ctx context.Context, in *drapbv1.NodePrepareResourcesRequest) (*drapbv1.NodePrepareResourcesResponse, error) {
	return &drapbv1.NodePrepareResourcesResponse{Claims: map[string]*drapbv1.NodePrepareResourceResponse{"dummy": {CDIDevices: []string{"dummy"}}}}, nil
}

func (f *fakeGRPCServer) NodeUnprepareResource(ctx context.Context, in *drapbv1.NodeUnprepareResourcesRequest) (*drapbv1.NodeUnprepareResourcesResponse, error) {
	return &drapbv1.NodeUnprepareResourcesResponse{}, nil
}

type tearDown func()

func setupFakeGRPCServer() (string, tearDown, error) {
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
	fakeGRPCServer := &fakeGRPCServer{}
	drapbv1.RegisterNodeServer(s, fakeGRPCServer)

	go func() {
		go s.Serve(listener)
		<-closeCh
		s.GracefulStop()
	}()

	return addr, teardown, nil
}

func TestGRPCConnIsReused(t *testing.T) {
	addr, teardown, err := setupFakeGRPCServer()
	if err != nil {
		t.Fatal(err)
	}
	defer teardown()

	reusedConns := make(map[*grpc.ClientConn]int)
	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	plugin := &Plugin{
		endpoint: addr,
	}

	conn, err := plugin.getOrCreateGRPCConn()
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
	draPlugins.Set("dummy-plugin", plugin)

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

			req := &drapbv1.NodePrepareResourcesRequest{
				Claims: []*drapbv1.Claim{
					{
						Namespace:      "dummy-namespace",
						Uid:            "dummy-uid",
						Name:           "dummy-claim",
						ResourceHandle: "dummy-resource",
					},
				},
			}
			client.NodePrepareResources(context.TODO(), req)

			client.(*draPluginClient).plugin.Lock()
			conn := client.(*draPluginClient).plugin.conn
			client.(*draPluginClient).plugin.Unlock()

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

	draPlugins.Delete("dummy-plugin")
}
