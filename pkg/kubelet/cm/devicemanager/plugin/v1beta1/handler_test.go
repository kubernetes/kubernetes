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
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	api "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// --- Handler-level (v1beta1 server) endpoint lifecycle tests ---
//
// These tests exercise connectClient / Connect / deregisterClient at the
// v1beta1 registration-server layer. They are the handler-level counterpart of
// the socket-level manager tests in manager_test.go (PR #139338), which cover
// ManagerImpl.PluginConnected / PluginDisconnected.
//
// The two layers cooperate to deduplicate registrations for the same
// (resourceName, socketPath):
//   - connectClient deduplicates silently: if a client for that pair is already
//     tracked, it returns nil without dialing or calling PluginConnected.
//   - PluginConnected (the manager) rejects a duplicate with
//     "device plugin already connected: <socket>" when a registration slips past
//     the handler-level dedup (e.g. the old client was deregistered at the
//     handler layer but the manager's endpointStore entry is not gone yet — the
//     fast-takeover race described in handler.go).
//
// Because connectClient / Connect dial a real unix socket, these tests use
// lightStub (a minimal device-plugin gRPC server) rather than the purely
// in-memory fakes used by the manager-level tests.

// fakeRegistrationHandler implements RegistrationHandler for testing purposes.
type fakeRegistrationHandler struct{}

func (h *fakeRegistrationHandler) CleanupPluginDirectory(klog.Logger, string) error {
	return nil
}

// fakeClientHandler is a configurable ClientHandler for the handler-level
// tests. connectErr, if set, is returned from PluginConnected to simulate the
// manager rejecting a registration — most importantly the
// "device plugin already connected" error from ManagerImpl.PluginConnected
// (see manager.go). It records PluginConnected invocations so tests can assert
// on connect counts and dedup behavior.
type fakeClientHandler struct {
	mu           sync.Mutex
	connectCount int
	connectErr   error
}

func (h *fakeClientHandler) PluginConnected(_ context.Context, _ string, _ DevicePlugin) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.connectCount++
	return h.connectErr
}

func (h *fakeClientHandler) PluginDisconnected(klog.Logger, string, string) {}

func (h *fakeClientHandler) PluginListAndWatchReceiver(klog.Logger, string, *api.ListAndWatchResponse) {
}

// getConnectCount returns the number of PluginConnected calls under the handler's lock.
func (h *fakeClientHandler) getConnectCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.connectCount
}

// lightStub is a minimal device-plugin gRPC server for the handler tests.
// Unlike the purely in-memory fakes used by the manager-level tests, it serves
// a real unix socket because connectClient / Connect dial the plugin. It
// deliberately avoids fsnotify (and therefore inotify limits) by not embedding
// the full Stub.
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

// startStub starts a lightStub device-plugin gRPC server at socketPath and
// registers its cleanup with t.
func startStub(t *testing.T, ctx context.Context, socketPath string) {
	t.Helper()
	p := newLightStub(socketPath)
	if err := p.start(ctx); err != nil {
		t.Fatalf("failed to start plugin stub at %s: %v", socketPath, err)
	}
	t.Cleanup(p.stop)
}

// setupTestServer provisions a short-lived temp directory, a running lightStub
// plugin at a default socket, and a *server for handler-level testing. Cleanup
// is via t.Cleanup. The ctx is used to dial the stub while starting it.
//
// A short os.MkdirTemp directory is used instead of t.TempDir(): on Windows,
// AF_UNIX bind() rejects paths that are too long, and t.TempDir() embeds the
// (long) test name and exceeds the limit. This mirrors tmpSocketDir() in
// manager_test.go.
func setupTestServer(t *testing.T, ctx context.Context, ch ClientHandler) (srv *server, pluginSocketPath string) {
	t.Helper()
	socketDir, err := os.MkdirTemp("", "device_plugin")
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })
	pluginSocketPath = filepath.Join(socketDir, "device-plugin.sock")
	startStub(t, ctx, pluginSocketPath)
	srv = newTestServer(t, socketDir, &fakeRegistrationHandler{}, ch)
	return srv, pluginSocketPath
}

// newTestServer creates a *server for testing with the given handlers.
func newTestServer(t *testing.T, socketDir string, rh RegistrationHandler, ch ClientHandler) *server {
	t.Helper()
	s, err := NewServer(klog.Background(), filepath.Join(socketDir, "kubelet.sock"), rh, ch)
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	return s.(*server)
}

// TestConnectClient_SameResourceSameSocketDeduplicated verifies that
// connectClient deduplicates registrations for the same (resourceName,
// socketPath): repeated calls do not create additional clients and do not call
// PluginConnected again. This is the handler-level counterpart of the manager's
// TestPluginConnected_SameResourceSameSocketRejected (manager_test.go): where
// the manager *rejects* a duplicate with "device plugin already connected", the
// handler *deduplicates silently* because getClient short-circuits before dial.
func TestConnectClient_SameResourceSameSocketDeduplicated(t *testing.T) {
	handler := &fakeClientHandler{}
	_, tCtx := ktesting.NewTestContext(t)
	srv, pluginSocketPath := setupTestServer(t, tCtx, handler)

	const resourceName = "fake-domain/resource"
	const iterations = 3

	for i := 0; i < iterations; i++ {
		require.NoError(t, srv.connectClient(tCtx, resourceName, pluginSocketPath),
			"connectClient iteration %d: dedup must not return an error", i)
	}

	require.Len(t, srv.clients[resourceName], 1,
		"expected 1 client after %d connectClient calls", iterations)

	require.Equal(t, 1, handler.getConnectCount(),
		"PluginConnected must be called once; dedup must not reach Connect")
}

// TestConnectClient_SameResourceDifferentSocketsCoexist verifies that two
// endpoints for the same resource at different socket paths can coexist in
// server.clients. This is the handler-level counterpart of the manager's
// TestPluginConnected_SameResourceDifferentSocketsCoexist (manager_test.go).
func TestConnectClient_SameResourceDifferentSocketsCoexist(t *testing.T) {
	handler := &fakeClientHandler{}
	_, tCtx := ktesting.NewTestContext(t)
	srv, socketA := setupTestServer(t, tCtx, handler)

	const resourceName = "fake-domain/resource"
	socketB := filepath.Join(filepath.Dir(socketA), "device-plugin-B.sock")
	startStub(t, tCtx, socketB)

	require.NoError(t, srv.connectClient(tCtx, resourceName, socketA))
	require.NoError(t, srv.connectClient(tCtx, resourceName, socketB),
		"two endpoints at different socket paths for the same resource must coexist")

	require.Len(t, srv.clients[resourceName], 2,
		"server.clients must track both endpoints")
	require.NotNil(t, srv.getClient(resourceName, socketA), "socketA client must be tracked")
	require.NotNil(t, srv.getClient(resourceName, socketB), "socketB client must be tracked")
	require.Equal(t, 2, handler.getConnectCount(), "PluginConnected must be called once per endpoint")
}

// TestConnectClient_PluginConnectedFailureDeregisters verifies that when
// PluginConnected rejects the registration — as ManagerImpl.PluginConnected does
// with "device plugin already connected" for a duplicate socket (see manager.go
// and the handler.go comment added in #139338) — connectClient deregisters the
// client and surfaces the error. This is the handler-level view of the
// fast-takeover race: a registration that slips past the handler dedup is
// rejected by the manager, and the handler must clean up so the plugin can retry.
func TestConnectClient_PluginConnectedFailureDeregisters(t *testing.T) {
	const resourceName = "fake-domain/resource"
	// Simulate the manager rejecting a duplicate (resource, socket).
	rejectErr := fmt.Errorf("device plugin already connected: %s", "/var/lib/kubelet/plugins_registry/dup.sock")
	handler := &fakeClientHandler{connectErr: rejectErr}
	_, tCtx := ktesting.NewTestContext(t)
	srv, pluginSocketPath := setupTestServer(t, tCtx, handler)

	err := srv.connectClient(tCtx, resourceName, pluginSocketPath)
	require.Error(t, err, "connectClient must surface the PluginConnected failure")
	require.Contains(t, err.Error(), "device plugin already connected",
		"the manager's rejection must propagate through connectClient")

	require.NotContains(t, srv.clients, resourceName,
		"connectClient must deregister the client when Connect fails, preventing a leak")
	require.Equal(t, 1, handler.getConnectCount(),
		"PluginConnected must have been called once before failing")
}

// TestConnect_PluginConnectedFailureClosesConnection verifies the client-level
// counterpart of the above: when PluginConnected fails after dial succeeded,
// client.Connect closes the gRPC connection and resets its internal state, so
// no file descriptor or goroutine is leaked (issue #124716).
func TestConnect_PluginConnectedFailureClosesConnection(t *testing.T) {
	const resourceName = "fake-domain/resource"
	rejectErr := fmt.Errorf("device plugin already connected: %s", "/var/lib/kubelet/plugins_registry/dup.sock")
	handler := &fakeClientHandler{connectErr: rejectErr}
	_, tCtx := ktesting.NewTestContext(t)
	_, pluginSocketPath := setupTestServer(t, tCtx, handler)

	c := NewPluginClient(resourceName, pluginSocketPath, handler)
	err := c.Connect(tCtx)
	require.Error(t, err, "Connect: expected error when PluginConnected fails")
	require.Contains(t, err.Error(), "device plugin already connected")

	impl, ok := c.(*client)
	require.True(t, ok, "failed to cast Client to *client for internal state verification")
	impl.mutex.Lock()
	defer impl.mutex.Unlock()
	require.Nil(t, impl.grpc, "gRPC connection must be nil after Connect failure; connection leaked")
	require.Nil(t, impl.client, "plugin API client must be nil after Connect failure")
}

// TestDeregisterClient_WrongSocketIsNoop verifies that deregisterClient only
// removes the client whose socket path matches, and is a no-op for an unknown
// socket. This is the handler-level counterpart of the manager's
// TestPluginDisconnected_WrongSocketIsNoop (manager_test.go). No real plugin
// socket is needed because deregisterClient only inspects SocketPath().
func TestDeregisterClient_WrongSocketIsNoop(t *testing.T) {
	handler := &fakeClientHandler{}
	_, tCtx := ktesting.NewTestContext(t)
	srv := newTestServer(t, t.TempDir(), &fakeRegistrationHandler{}, handler)
	logger := klog.FromContext(tCtx)

	const resourceName = "fake-domain/resource"
	const socketA = "/var/lib/kubelet/plugins_registry/socketA.sock"
	const socketB = "/var/lib/kubelet/plugins_registry/socketB.sock"
	const socketC = "/var/lib/kubelet/plugins_registry/socketC.sock"

	srv.registerClient(logger, resourceName, NewPluginClient(resourceName, socketA, handler))
	srv.registerClient(logger, resourceName, NewPluginClient(resourceName, socketB, handler))

	// deregistering an unknown socket is a no-op.
	srv.deregisterClient(logger, resourceName, socketC)
	require.Len(t, srv.clients[resourceName], 2,
		"deregistering an unknown socket must not remove any client")

	// deregistering a matching socket removes only that client.
	srv.deregisterClient(logger, resourceName, socketA)
	require.Len(t, srv.clients[resourceName], 1,
		"deregistering a matching socket must remove only that client")
	require.Equal(t, socketB, srv.clients[resourceName][0].SocketPath(),
		"the surviving client must be the one at socketB")

	// deregistering the last client deletes the resource key entirely.
	srv.deregisterClient(logger, resourceName, socketB)
	require.NotContains(t, srv.clients, resourceName,
		"deregistering the last client must delete the resource key")
}

// TestSameSocketRace_ReconnectAfterCleanDisconnect verifies the handler-level
// slot lifecycle that gates connectClient's dedup: once a client for a
// (resource, socket) is deregistered, the slot is free and the same socket can
// be reused. This is the handler-level counterpart of the manager's
// TestSameSocketRace_DisconnectBeforeReconnectAttempt (manager_test.go). It
// operates on the tracked-clients map directly (no dial) to keep the assertion
// deterministic — the manager-level test covers the full reject→disconnect→
// reconnect sequence.
func TestSameSocketRace_ReconnectAfterCleanDisconnect(t *testing.T) {
	handler := &fakeClientHandler{}
	_, tCtx := ktesting.NewTestContext(t)
	srv := newTestServer(t, t.TempDir(), &fakeRegistrationHandler{}, handler)
	logger := klog.FromContext(tCtx)

	const resourceName = "fake-domain/resource"
	const socketA = "/var/lib/kubelet/plugins_registry/socketA.sock"

	// Occupy the slot with a tracked client. connectClient deduplicates via
	// getClient, so a tracked client gates a subsequent connectClient.
	srv.registerClient(logger, resourceName, NewPluginClient(resourceName, socketA, handler))
	require.NotNil(t, srv.getClient(resourceName, socketA), "the slot must be occupied")

	// After deregisterClient frees the slot, the same socket can be reused.
	srv.deregisterClient(logger, resourceName, socketA)
	require.Nil(t, srv.getClient(resourceName, socketA),
		"after a clean disconnect, the slot must be free for reuse")

	srv.registerClient(logger, resourceName, NewPluginClient(resourceName, socketA, handler))
	require.NotNil(t, srv.getClient(resourceName, socketA),
		"the freed slot must be reusable for the same socket")
}
