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

package localnodeportproxy

import (
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"strconv"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxymetrics "k8s.io/kubernetes/pkg/proxy/metrics"
	testingclock "k8s.io/utils/clock/testing"
)

func makeServicePortName(ns, name, port string) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: types.NamespacedName{Namespace: ns, Name: name},
		Port:           port,
		Protocol:       v1.ProtocolTCP,
	}
}

func randomNodePort() int {
	return 30000 + rand.Intn(2767)
}

// startTCPEchoServer starts a TCP server that echoes back everything it receives.
func startTCPEchoServer(t *testing.T, network, addr string) net.Listener {
	t.Helper()
	l, err := net.Listen(network, addr)
	if err != nil {
		t.Fatalf("Failed to start echo server: %v", err)
	}
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close() //nolint:errcheck
				_, _ = io.Copy(c, c)
			}(conn)
		}
	}()
	return l
}

// assertEchoThroughProxy dials the proxy at network/addr, sends msg, and fails
// unless the connection echoes msg back unchanged.
func assertEchoThroughProxy(t *testing.T, network, addr, msg string) {
	t.Helper()
	conn, err := net.DialTimeout(network, addr, 2*time.Second)
	if err != nil {
		t.Fatalf("Failed to connect to nodeport proxy: %v", err)
	}
	defer conn.Close() //nolint:errcheck
	_, _ = fmt.Fprint(conn, msg)
	_ = conn.(*net.TCPConn).CloseWrite()
	buf, err := io.ReadAll(conn)
	if err != nil {
		t.Fatalf("Failed to read from proxy: %v", err)
	}
	if string(buf) != msg {
		t.Errorf("Expected %q, got %q", msg, string(buf))
	}
}

func TestSyncNodePorts_AddAndRemove(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	svcName := makeServicePortName("default", "test-svc", "http")

	// Start a backend echo server
	backend := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend.Close() //nolint:errcheck
	backendPort := backend.Addr().(*net.TCPAddr).Port

	ep := net.JoinHostPort("127.0.0.1", strconv.Itoa(backendPort))

	nodePort := randomNodePort()

	desired := []NodePortSpec{{
		ServicePortName: svcName,
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       []string{ep},
	}}
	key := desired[0].nodePortKey()

	p.SyncNodePorts(desired)

	if len(p.active) != 1 {
		t.Fatalf("Expected 1 active listener, got %d", len(p.active))
	}
	if _, ok := p.active[key]; !ok {
		t.Fatalf("Expected listener for key %s", key)
	}

	// Verify we can connect through the proxy
	assertEchoThroughProxy(t, "tcp4", fmt.Sprintf("127.0.0.1:%d", nodePort), "hello nodeport")

	// Remove the service
	p.SyncNodePorts(nil)
	if len(p.active) != 0 {
		t.Fatalf("Expected 0 active listeners after removal, got %d", len(p.active))
	}

	// Verify port is closed
	_, err := net.DialTimeout("tcp4", fmt.Sprintf("127.0.0.1:%d", nodePort), 500*time.Millisecond)
	if err == nil {
		t.Fatal("Expected connection to fail after listener removal")
	}
}

func TestSyncNodePorts_UpdateEndpoints(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	// Start two backend servers
	backend1 := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend1.Close() //nolint:errcheck
	backend2 := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend2.Close() //nolint:errcheck

	ep1 := net.JoinHostPort("127.0.0.1", strconv.Itoa(backend1.Addr().(*net.TCPAddr).Port))
	ep2 := net.JoinHostPort("127.0.0.1", strconv.Itoa(backend2.Addr().(*net.TCPAddr).Port))

	svcName := makeServicePortName("default", "test-svc", "http")

	nodePort := randomNodePort()

	// Start with ep1 only
	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: svcName,
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       []string{ep1},
	}})

	if len(p.active) != 1 {
		t.Fatalf("Expected 1 active listener, got %d", len(p.active))
	}

	// Update to ep2
	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: svcName,
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       []string{ep2},
	}})

	// Should still have exactly 1 listener (same one, updated endpoints)
	if len(p.active) != 1 {
		t.Fatalf("Expected 1 active listener after update, got %d", len(p.active))
	}

	// Verify connectivity still works
	assertEchoThroughProxy(t, "tcp4", fmt.Sprintf("127.0.0.1:%d", nodePort), "after update")
}

func TestNoEndpoints(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	backend := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend.Close() //nolint:errcheck
	ep := net.JoinHostPort("127.0.0.1", strconv.Itoa(backend.Addr().(*net.TCPAddr).Port))

	nodePort := randomNodePort()

	spec := NodePortSpec{
		ServicePortName: makeServicePortName("default", "svc", "http"),
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       nil,
	}

	// A spec that never has endpoints is never listened on.
	p.SyncNodePorts([]NodePortSpec{spec})
	if len(p.active) != 0 {
		t.Fatalf("Expected no listener for spec with no endpoints, got %d active", len(p.active))
	}

	// Adding endpoints brings the listener up.
	spec.Endpoints = []string{ep}
	p.SyncNodePorts([]NodePortSpec{spec})
	if len(p.active) != 1 {
		t.Fatalf("Expected 1 active listener, got %d", len(p.active))
	}

	// Draining endpoints tears it back down and the port stops accepting.
	spec.Endpoints = nil
	p.SyncNodePorts([]NodePortSpec{spec})
	if len(p.active) != 0 {
		t.Fatalf("Expected listener to be torn down when endpoints drain, got %d active", len(p.active))
	}
	if _, err := net.DialTimeout("tcp4", fmt.Sprintf("127.0.0.1:%d", nodePort), 500*time.Millisecond); err == nil {
		t.Fatal("Expected dial to fail after endpoints drain")
	}
}

func TestSyncNodePorts_SkipUDP(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	svcName := makeServicePortName("default", "udp-svc", "dns")

	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: svcName,
		Protocol:        v1.ProtocolUDP,
		NodePort:        30053,
		Endpoints:       []string{"10.0.0.1:53"},
	}})

	if len(p.active) != 0 {
		t.Fatalf("Expected 0 active listeners for UDP, got %d", len(p.active))
	}
}

func TestPickEndpoint_NoEndpointsReturnsEmpty(t *testing.T) {
	l := &nodePortListener{}
	if got := l.pickEndpoint(); got != "" {
		t.Errorf("pickEndpoint() with no endpoints = %q, want empty", got)
	}
}

func TestBackendConnectionFailure(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	// Use an endpoint that isn't listening
	ep := "127.0.0.1:1" // port 1 is almost certainly not listening

	nodePort := randomNodePort()

	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: makeServicePortName("default", "fail-svc", "http"),
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       []string{ep},
	}})

	// Connect — the proxy should accept but then close the connection
	// when the backend dial fails
	conn, err := net.DialTimeout("tcp4", fmt.Sprintf("127.0.0.1:%d", nodePort), 2*time.Second)
	if err != nil {
		t.Fatalf("Failed to connect to proxy: %v", err)
	}
	buf, _ := io.ReadAll(conn)
	_ = conn.Close()

	if len(buf) != 0 {
		t.Errorf("Expected empty response on backend failure, got %q", string(buf))
	}
}

func TestShutdown(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)

	// Create multiple listeners
	var ports []int
	var desired []NodePortSpec
	for range 3 {
		port := randomNodePort()
		ports = append(ports, port)

		desired = append(desired, NodePortSpec{
			ServicePortName: makeServicePortName("default", fmt.Sprintf("svc-%d", port), "http"),
			Protocol:        v1.ProtocolTCP,
			NodePort:        port,
			Endpoints:       []string{"127.0.0.1:1"},
		})
	}

	p.SyncNodePorts(desired)
	if len(p.active) != 3 {
		t.Fatalf("Expected 3 active listeners, got %d", len(p.active))
	}

	p.Shutdown()
	if len(p.active) != 0 {
		t.Fatalf("Expected 0 active listeners after shutdown, got %d", len(p.active))
	}

	// Verify all ports are closed
	for _, port := range ports {
		_, err := net.DialTimeout("tcp4", fmt.Sprintf("127.0.0.1:%d", port), 500*time.Millisecond)
		if err == nil {
			t.Errorf("Port %d still accepting connections after shutdown", port)
		}
	}
}

func TestShutdownClosesInFlightConnections(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)

	backend := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend.Close() //nolint:errcheck
	backendPort := backend.Addr().(*net.TCPAddr).Port

	ep := net.JoinHostPort("127.0.0.1", strconv.Itoa(backendPort))

	nodePort := randomNodePort()

	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: makeServicePortName("default", "inflight-svc", "http"),
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       []string{ep},
	}})

	// Open an idle in-flight connection through the proxy.
	conn, err := net.DialTimeout("tcp4", fmt.Sprintf("127.0.0.1:%d", nodePort), 2*time.Second)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close() //nolint:errcheck

	// A successful echo roundtrip read proves the handleTCPConn connection is established before we shut down.
	if _, err := conn.Write([]byte("x")); err != nil {
		t.Fatalf("Failed to write to proxy: %v", err)
	}
	buf := make([]byte, 1)
	if _, err := io.ReadFull(conn, buf); err != nil {
		t.Fatalf("Failed to read echo through proxy: %v", err)
	}

	p.Shutdown()

	// After shutdown the in-flight connection must be torn down; a blocking
	// Read should return promptly (EOF / use of closed / reset), not block.
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, err := conn.Read(buf); err == nil {
		t.Fatal("Expected connection to be closed after Shutdown, got nil error")
	} else {
		if ne, ok := errors.AsType[net.Error](err); ok && ne.Timeout() {
			t.Fatalf("Expected connection close after Shutdown, got read timeout: %v", err)
		}
	}
}

func TestIPv6(t *testing.T) {
	// Check if IPv6 loopback is available
	l, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Skipf("IPv6 loopback not available: %v", err)
	}
	_ = l.Close()

	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv6Protocol)
	defer p.Shutdown()

	if p.listenIP != "::1" {
		t.Errorf("Expected listenIP '::1', got %q", p.listenIP)
	}

	// Start an IPv6 backend
	backend := startTCPEchoServer(t, "tcp6", "[::1]:0")
	defer backend.Close() //nolint:errcheck
	backendPort := backend.Addr().(*net.TCPAddr).Port

	ep := net.JoinHostPort("::1", strconv.Itoa(backendPort))

	nodePort := randomNodePort()

	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: makeServicePortName("default", "v6-svc", "http"),
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       []string{ep},
	}})

	assertEchoThroughProxy(t, "tcp6", fmt.Sprintf("[::1]:%d", nodePort), "hello ipv6")
}

// startPortReportingServer starts a TCP server that sends its own port number
// to each connecting client.
func startPortReportingServer(t *testing.T, network string) (net.Listener, int, string) {
	t.Helper()
	addr := "127.0.0.1:0"
	if network == "tcp6" {
		addr = "[::1]:0"
	}
	l, err := net.Listen(network, addr)
	if err != nil {
		t.Fatalf("Failed to start backend: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close() //nolint:errcheck
				_, _ = fmt.Fprintf(c, "%d", port)
			}(conn)
		}
	}()
	ip := "127.0.0.1"
	if network == "tcp6" {
		ip = "::1"
	}
	return l, port, net.JoinHostPort(ip, strconv.Itoa(port))
}

// readBackendID dials network/addr and returns the port the proxy routed the connection to.
func readBackendID(t *testing.T, network, addr string) int {
	t.Helper()
	conn, err := net.DialTimeout(network, addr, 2*time.Second)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	buf, err := io.ReadAll(conn)
	_ = conn.Close()
	if err != nil {
		t.Fatalf("Failed to read: %v", err)
	}
	port, err := strconv.Atoi(string(buf))
	if err != nil {
		t.Fatalf("Failed to parse backend port %q: %v", buf, err)
	}
	return port
}

func TestSessionAffinity_PinsToSingleEndpoint(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := newLocalNodePortProxy(ctx, v1.IPv4Protocol, testingclock.NewFakeClock(time.Now()))
	defer p.Shutdown()

	// Start 3 backends; each reports its own port
	var backends []net.Listener
	var endpoints []string
	for range 3 {
		b, _, ep := startPortReportingServer(t, "tcp4")
		backends = append(backends, b)
		endpoints = append(endpoints, ep)
	}
	defer func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}()

	nodePort := randomNodePort()

	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName:     makeServicePortName("default", "sticky-svc", "http"),
		Protocol:            v1.ProtocolTCP,
		NodePort:            nodePort,
		Endpoints:           endpoints,
		SessionAffinityType: v1.ServiceAffinityClientIP,
		StickyMaxAgeSeconds: 10800,
	}})

	addr := fmt.Sprintf("127.0.0.1:%d", nodePort)
	first := readBackendID(t, "tcp4", addr)
	for range 10 {
		got := readBackendID(t, "tcp4", addr)
		if got != first {
			t.Fatalf("SessionAffinity ClientIP: expected all requests to hit %d, got %d", first, got)
		}
	}
}

func TestSessionAffinity_PinnedEndpointRemoved(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	p := newLocalNodePortProxy(ctx, v1.IPv4Protocol, testingclock.NewFakeClock(time.Now()))
	defer p.Shutdown()

	b1, port1, ep1 := startPortReportingServer(t, "tcp4")
	defer b1.Close() //nolint:errcheck
	b2, port2, ep2 := startPortReportingServer(t, "tcp4")
	defer b2.Close() //nolint:errcheck

	nodePort := randomNodePort()

	svcName := makeServicePortName("default", "sticky-svc", "http")
	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName:     svcName,
		Protocol:            v1.ProtocolTCP,
		NodePort:            nodePort,
		Endpoints:           []string{ep1, ep2},
		SessionAffinityType: v1.ServiceAffinityClientIP,
		StickyMaxAgeSeconds: 10800,
	}})

	addr := fmt.Sprintf("127.0.0.1:%d", nodePort)
	pinned := readBackendID(t, "tcp4", addr)

	// Drop the pinned endpoint from the set; remaining traffic must flow to
	// the surviving endpoint rather than silently dropping.
	remaining, remainingPort := ep2, port2
	if pinned == port2 {
		remaining, remainingPort = ep1, port1
	}
	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName:     svcName,
		Protocol:            v1.ProtocolTCP,
		NodePort:            nodePort,
		Endpoints:           []string{remaining},
		SessionAffinityType: v1.ServiceAffinityClientIP,
		StickyMaxAgeSeconds: 10800,
	}})

	if got := readBackendID(t, "tcp4", addr); got != remainingPort {
		t.Fatalf("After pinned endpoint removal: expected %d, got %d", remainingPort, got)
	}
}

func TestSessionAffinity_Expires(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	fakeClock := testingclock.NewFakeClock(time.Now())
	p := newLocalNodePortProxy(ctx, v1.IPv4Protocol, fakeClock)
	defer p.Shutdown()

	var backends []net.Listener
	var endpoints []string
	for range 3 {
		b, _, ep := startPortReportingServer(t, "tcp4")
		backends = append(backends, b)
		endpoints = append(endpoints, ep)
	}
	defer func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}()

	nodePort := randomNodePort()

	const affinitySeconds = 10800
	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName:     makeServicePortName("default", "sticky-svc", "http"),
		Protocol:            v1.ProtocolTCP,
		NodePort:            nodePort,
		Endpoints:           endpoints,
		SessionAffinityType: v1.ServiceAffinityClientIP,
		StickyMaxAgeSeconds: affinitySeconds,
	}})

	// Verify that the pin releases once the window lapses
	window := affinitySeconds*time.Second + time.Second
	addr := fmt.Sprintf("127.0.0.1:%d", nodePort)
	first := readBackendID(t, "tcp4", addr)
	for i := 0; ; i++ {
		if i == 20 {
			t.Fatalf("Affinity pin to %d never expired after %d attempts", first, i)
		}
		fakeClock.Step(window)
		if readBackendID(t, "tcp4", addr) != first {
			break
		}
	}
}

func TestNoSessionAffinity_SpreadsAcrossEndpoints(t *testing.T) {
	const numEndpoints = 3
	const maxAttempts = 100

	_, ctx := ktesting.NewTestContext(t)
	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	var backends []net.Listener
	var endpoints []string
	for range numEndpoints {
		b, _, ep := startPortReportingServer(t, "tcp4")
		backends = append(backends, b)
		endpoints = append(endpoints, ep)
	}
	defer func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}()

	nodePort := randomNodePort()

	p.SyncNodePorts([]NodePortSpec{{
		ServicePortName: makeServicePortName("default", "roundrobin-svc", "http"),
		Protocol:        v1.ProtocolTCP,
		NodePort:        nodePort,
		Endpoints:       endpoints,
	}})

	// Verify every endpoint is reached
	addr := fmt.Sprintf("127.0.0.1:%d", nodePort)
	seen := map[int]bool{}
	for i := 0; len(seen) < numEndpoints; i++ {
		if i == maxAttempts {
			t.Fatalf("Expected connections to spread across all %d endpoints, only reached %v", numEndpoints, seen)
		}
		seen[readBackendID(t, "tcp4", addr)] = true
	}
}

func TestSyncNodePorts_ListenersGaugeReflectsActiveCount(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeProxyNFTablesLocalhostNodePorts, true)
	proxymetrics.RegisterMetrics(kubeproxyconfig.ProxyModeNFTables)
	gauge := proxymetrics.LocalhostNodePortListeners.WithLabelValues(string(v1.IPv4Protocol))

	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	backend := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend.Close() //nolint:errcheck
	ep := net.JoinHostPort("127.0.0.1", strconv.Itoa(backend.Addr().(*net.TCPAddr).Port))

	p1, p2 := randomNodePort(), randomNodePort()

	p.SyncNodePorts([]NodePortSpec{
		{ServicePortName: makeServicePortName("ns", "svc1", "p"), Protocol: v1.ProtocolTCP, NodePort: p1, Endpoints: []string{ep}},
		{ServicePortName: makeServicePortName("ns", "svc2", "p"), Protocol: v1.ProtocolTCP, NodePort: p2, Endpoints: []string{ep}},
	})

	got, err := testutil.GetGaugeMetricValue(gauge)
	if err != nil {
		t.Fatalf("Failed to read gauge: %v", err)
	}
	if got != 2 {
		t.Fatalf("After adding 2 listeners: gauge = %v, want 2", got)
	}

	p.SyncNodePorts(nil)
	got, err = testutil.GetGaugeMetricValue(gauge)
	if err != nil {
		t.Fatalf("Failed to read gauge: %v", err)
	}
	if got != 0 {
		t.Fatalf("After removing all listeners: gauge = %v, want 0", got)
	}
}

func TestSyncNodePorts_ListenerCreationFailuresCounter(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeProxyNFTablesLocalhostNodePorts, true)
	proxymetrics.RegisterMetrics(kubeproxyconfig.ProxyModeNFTables)
	counter := proxymetrics.LocalhostNodePortListenerCreationFailuresTotal.WithLabelValues(string(v1.IPv4Protocol))
	before, err := testutil.GetCounterMetricValue(counter)
	if err != nil {
		t.Fatalf("Failed to read counter: %v", err)
	}

	p := NewLocalNodePortProxy(ctx, v1.IPv4Protocol)
	defer p.Shutdown()

	backend := startTCPEchoServer(t, "tcp4", "127.0.0.1:0")
	defer backend.Close() //nolint:errcheck
	ep := net.JoinHostPort("127.0.0.1", strconv.Itoa(backend.Addr().(*net.TCPAddr).Port))

	// Occupy a port so the proxy's bind to it fails.
	taken, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to occupy port: %v", err)
	}
	defer taken.Close() //nolint:errcheck
	takenPort := taken.Addr().(*net.TCPAddr).Port

	p.SyncNodePorts([]NodePortSpec{
		{ServicePortName: makeServicePortName("ns", "svc", "p"), Protocol: v1.ProtocolTCP, NodePort: takenPort, Endpoints: []string{ep}},
	})

	got, err := testutil.GetCounterMetricValue(counter)
	if err != nil {
		t.Fatalf("Failed to read counter: %v", err)
	}
	if got != before+1 {
		t.Fatalf("After a failed bind: counter = %v, want %v", got, before+1)
	}
}
