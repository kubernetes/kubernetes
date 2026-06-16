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

package v1beta1

import (
	"context"
	"fmt"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	api "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

// fakeRegistrationHandler implements RegistrationHandler for testing purposes.
type fakeRegistrationHandler struct{}

func (h *fakeRegistrationHandler) CleanupPluginDirectory(klog.Logger, string) error {
	return nil
}

// fakeSuccessfulClientHandler implements ClientHandler and succeeds on PluginConnected,
// counting the number of times PluginConnected is called.
type fakeSuccessfulClientHandler struct {
	mu           sync.Mutex
	connectCount int
}

func (h *fakeSuccessfulClientHandler) PluginConnected(context.Context, string, DevicePlugin) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.connectCount++
	return nil
}

func (h *fakeSuccessfulClientHandler) PluginDisconnected(klog.Logger, string, string) {}

func (h *fakeSuccessfulClientHandler) PluginListAndWatchReceiver(klog.Logger, string, *api.ListAndWatchResponse) {
}

// fakeFailingClientHandler implements ClientHandler and fails on PluginConnected.
type fakeFailingClientHandler struct{}

func (h *fakeFailingClientHandler) PluginConnected(_ context.Context, resourceName string, _ DevicePlugin) error {
	return fmt.Errorf("simulated PluginConnected failure for %s", resourceName)
}

func (h *fakeFailingClientHandler) PluginDisconnected(klog.Logger, string, string) {}

func (h *fakeFailingClientHandler) PluginListAndWatchReceiver(klog.Logger, string, *api.ListAndWatchResponse) {
}

// lightStub is a minimal device plugin gRPC server for handler tests.
// It deliberately avoids fsnotify so the test is not affected by inotify limits.
type lightStub struct {
	socket string
	server *grpc.Server
	stopCh chan struct{}
	wg     sync.WaitGroup

	api.UnimplementedDevicePluginServer
}

func newLightStub(socket string) *lightStub {
	return &lightStub{
		socket: socket,
		stopCh: make(chan struct{}),
	}
}

func (s *lightStub) start(ctx context.Context) error {
	if err := os.Remove(s.socket); err != nil && !os.IsNotExist(err) {
		return err
	}

	sock, err := net.Listen("unix", s.socket)
	if err != nil {
		return err
	}

	s.server = grpc.NewServer()
	api.RegisterDevicePluginServer(s.server, s)

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.server.Serve(sock)
	}()

	var lastDialErr error
	wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		var conn *grpc.ClientConn
		_, conn, lastDialErr = dial(ctx, s.socket)
		if lastDialErr != nil {
			return false, nil
		}
		conn.Close()
		return true, nil
	})
	return lastDialErr
}

func (s *lightStub) stop() {
	if s.server != nil {
		s.server.Stop()
		s.wg.Wait()
		s.server = nil
	}
	close(s.stopCh)
}

// ListAndWatch is the only RPC the handler tests exercise (via client.Run).
// It blocks until stop to keep the gRPC stream alive for the test duration.
func (s *lightStub) ListAndWatch(*api.Empty, api.DevicePlugin_ListAndWatchServer) error {
	<-s.stopCh
	return nil
}

// setupTestServer provisions a temp directory, a running lightStub plugin, and a
// *server for handler-level testing. Cleanup is registered via t.Cleanup.
func setupTestServer(t *testing.T, ch ClientHandler) (ctx context.Context, srv *server, pluginSocketPath string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)

	tmpDir := t.TempDir()
	pluginSocketPath = tmpDir + "/device-plugin.sock"

	p := newLightStub(pluginSocketPath)
	if err := p.start(ctx); err != nil {
		t.Fatalf("failed to start plugin stub: %v", err)
	}
	t.Cleanup(p.stop)

	srv = newTestServer(t, tmpDir, &fakeRegistrationHandler{}, ch)
	return ctx, srv, pluginSocketPath
}

// TestDuplicateRegistration verifies that connectClient deduplicates when a client
// with the same resource name and socket path is already registered, preventing
// duplicate gRPC connections and goroutine leaks (issue #124716).
func TestDuplicateRegistration(t *testing.T) {
	handler := &fakeSuccessfulClientHandler{}
	ctx, srv, pluginSocketPath := setupTestServer(t, handler)

	const resourceName = "fake-domain/resource"
	const iterations = 3

	for i := 0; i < iterations; i++ {
		if err := srv.connectClient(ctx, resourceName, pluginSocketPath); err != nil {
			t.Fatalf("connectClient iteration %d: expected no error, got: %v", i, err)
		}
	}

	if got := len(srv.clients[resourceName]); got != 1 {
		t.Fatalf("expected 1 client after %d connectClient calls, got %d", iterations, got)
	}

	handler.mu.Lock()
	defer handler.mu.Unlock()
	if handler.connectCount != 1 {
		t.Fatalf("expected PluginConnected to be called once, got %d", handler.connectCount)
	}
}

// TestConnectFailureCleanup verifies that when Connect fails after establishing the
// gRPC connection (e.g. PluginConnected returns an error), the gRPC connection is
// closed and the client is deregistered, preventing memory and goroutine leaks.
func TestConnectFailureCleanup(t *testing.T) {
	handler := &fakeFailingClientHandler{}
	ctx, srv, pluginSocketPath := setupTestServer(t, handler)

	const resourceName = "fake-domain/resource"

	// Handler-level path: connectClient must deregister on failure.
	if err := srv.connectClient(ctx, resourceName, pluginSocketPath); err == nil {
		t.Fatal("connectClient: expected error when PluginConnected fails")
	}
	if len(srv.clients) != 0 {
		t.Fatalf("expected no registered clients after connect failure, got %d", len(srv.clients))
	}

	// Client-level path: Connect must close the gRPC connection on failure.
	c := NewPluginClient(resourceName, pluginSocketPath, handler)
	if err := c.Connect(ctx); err == nil {
		t.Fatal("Connect: expected error when PluginConnected fails")
	}

	impl, ok := c.(*client)
	if !ok {
		t.Fatal("failed to cast Client to *client for internal state verification")
	}
	impl.mutex.Lock()
	defer impl.mutex.Unlock()
	if impl.grpc != nil {
		t.Fatal("expected gRPC connection to be nil after Connect failure; connection leaked")
	}
}

// newTestServer creates a *server for testing with the given handlers.
func newTestServer(t *testing.T, socketDir string, rh RegistrationHandler, ch ClientHandler) *server {
	t.Helper()
	s, err := NewServer(klog.Background(), socketDir+"/kubelet.sock", rh, ch)
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	return s.(*server)
}
