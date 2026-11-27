/*
Copyright The Kubernetes Authors.

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

package operationexecutor

import (
	"context"
	"net"
	"path/filepath"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"k8s.io/ktesting"
	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

// authorityRecorder is a plugin registration server which records the
// :authority pseudo-header of each GetInfo call.
type authorityRecorder struct {
	registerapi.UnimplementedRegistrationServer
	authorities chan []string
}

func (r *authorityRecorder) GetInfo(ctx context.Context, _ *registerapi.InfoRequest) (*registerapi.PluginInfo, error) {
	r.authorities <- metadata.ValueFromIncomingContext(ctx, ":authority")
	return &registerapi.PluginInfo{}, nil
}

// grpc-go derives :authority from the dial target when none is set, which for
// a bare socket path is the percent-encoded path. grpc-go servers accept that,
// but HTTP/2 servers which validate the header reject the stream.
func TestDialAuthority(t *testing.T) {
	tCtx := ktesting.Init(t)

	socketPath := filepath.Join(t.TempDir(), "plugin.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("Failed to listen on %s: %v", socketPath, err)
	}

	recorder := &authorityRecorder{authorities: make(chan []string, 1)}
	server := grpc.NewServer()
	registerapi.RegisterRegistrationServer(server, recorder)
	go func() {
		if err := server.Serve(listener); err != nil {
			t.Errorf("Failed to serve on %s: %v", socketPath, err)
		}
	}()
	t.Cleanup(server.Stop)

	client, conn, err := dial(tCtx, socketPath, dialTimeoutDuration)
	if err != nil {
		t.Fatalf("Failed to dial %s: %v", socketPath, err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	if _, err := client.GetInfo(tCtx, &registerapi.InfoRequest{}); err != nil {
		t.Fatalf("GetInfo failed: %v", err)
	}

	if got := <-recorder.authorities; len(got) != 1 || got[0] != "localhost" {
		t.Errorf("Expected :authority [\"localhost\"], got %q", got)
	}
}
