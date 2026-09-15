/*
Copyright 2019 The Kubernetes Authors.

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

package egressselector

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/server/egressselector/metrics"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	testingclock "k8s.io/utils/clock/testing"
	clientmetrics "sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/client/metrics"
	ccmetrics "sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/common/metrics"
	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

type fakeEgressSelection struct {
	directDialerCalled bool
}

func TestEgressSelector(t *testing.T) {
	testcases := []struct {
		name     string
		input    *apiserver.EgressSelectorConfiguration
		services []struct {
			egressType     EgressType
			validateDialer func(dialer utilnet.DialFunc, s *fakeEgressSelection) (bool, error)
			lookupError    *string
			dialerError    *string
		}
		expectedError *string
	}{
		{
			name: "direct",
			input: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
					{
						Name: "etcd",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
				},
			},
			services: []struct {
				egressType     EgressType
				validateDialer func(dialer utilnet.DialFunc, s *fakeEgressSelection) (bool, error)
				lookupError    *string
				dialerError    *string
			}{
				{
					Cluster,
					validateDirectDialer,
					nil,
					nil,
				},
				{
					ControlPlane,
					validateDirectDialer,
					nil,
					nil,
				},
				{
					Etcd,
					validateDirectDialer,
					nil,
					nil,
				},
			},
			expectedError: nil,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			// Setup the various pieces such as the fake dialer prior to initializing the egress selector.
			// Go doesn't allow function pointer comparison, nor does its reflect package
			// So overriding the default dialer to detect if it is returned.
			fake := &fakeEgressSelection{}
			directDialer = fake.fakeDirectDialer
			cs, err := NewEgressSelector(tc.input)
			if err == nil && tc.expectedError != nil {
				t.Errorf("calling NewEgressSelector expected error: %s, did not get it", *tc.expectedError)
			}
			if err != nil && tc.expectedError == nil {
				t.Errorf("unexpected error calling NewEgressSelector got: %#v", err)
			}
			if err != nil && tc.expectedError != nil && err.Error() != *tc.expectedError {
				t.Errorf("calling NewEgressSelector expected error: %s, got %#v", *tc.expectedError, err)
			}

			for _, service := range tc.services {
				networkContext := NetworkContext{EgressSelectionName: service.egressType}
				dialer, lookupErr := cs.Lookup(networkContext)
				if lookupErr == nil && service.lookupError != nil {
					t.Errorf("calling Lookup expected error: %s, did not get it", *service.lookupError)
				}
				if lookupErr != nil && service.lookupError == nil {
					t.Errorf("unexpected error calling Lookup got: %#v", lookupErr)
				}
				if lookupErr != nil && service.lookupError != nil && lookupErr.Error() != *service.lookupError {
					t.Errorf("calling Lookup expected error: %s, got %#v", *service.lookupError, lookupErr)
				}
				fake.directDialerCalled = false
				ok, dialerErr := service.validateDialer(dialer, fake)
				if dialerErr == nil && service.dialerError != nil {
					t.Errorf("calling Lookup expected error: %s, did not get it", *service.dialerError)
				}
				if dialerErr != nil && service.dialerError == nil {
					t.Errorf("unexpected error calling Lookup got: %#v", dialerErr)
				}
				if dialerErr != nil && service.dialerError != nil && dialerErr.Error() != *service.dialerError {
					t.Errorf("calling Lookup expected error: %s, got %#v", *service.dialerError, dialerErr)
				}
				if !ok {
					t.Errorf("Could not validate dialer for service %q", service.egressType)
				}
			}
		})
	}
}

func (s *fakeEgressSelection) fakeDirectDialer(ctx context.Context, network, address string) (net.Conn, error) {
	s.directDialerCalled = true
	return nil, nil
}

func validateDirectDialer(dialer utilnet.DialFunc, s *fakeEgressSelection) (bool, error) {
	conn, err := dialer(context.Background(), "tcp", "127.0.0.1:8080")
	if err != nil {
		return false, err
	}
	if conn != nil {
		return false, nil
	}
	return s.directDialerCalled, nil
}

type fakeProxyServerConnector struct {
	connectorErr bool
	proxierErr   bool
}

func (f *fakeProxyServerConnector) connect(context.Context) (proxier, error) {
	if f.connectorErr {
		return nil, fmt.Errorf("fake error")
	}
	return &fakeProxier{err: f.proxierErr}, nil
}

type fakeProxier struct {
	err bool
}

func (f *fakeProxier) proxy(_ context.Context, _ string) (net.Conn, error) {
	if f.err {
		return nil, fmt.Errorf("fake error")
	}
	return nil, nil
}

type proxyServerConnectorFunc func(context.Context) (proxier, error)

func (f proxyServerConnectorFunc) connect(ctx context.Context) (proxier, error) {
	return f(ctx)
}

func TestMetrics(t *testing.T) {
	testcases := map[string]struct {
		connectorErr bool
		proxierErr   bool
		metrics      []string
		want         string
	}{
		"connect to proxy server start": {
			connectorErr: true,
			proxierErr:   true,
			metrics:      []string{"apiserver_egress_dialer_dial_start_total"},
			want: `
	# HELP apiserver_egress_dialer_dial_start_total [ALPHA] Dial starts, labeled by the protocol (http-connect or grpc) and transport (tcp or uds).
	# TYPE apiserver_egress_dialer_dial_start_total counter
	apiserver_egress_dialer_dial_start_total{protocol="fake_protocol",transport="fake_transport"} 1
`,
		},
		"connect to proxy server error": {
			connectorErr: true,
			proxierErr:   false,
			metrics:      []string{"apiserver_egress_dialer_dial_failure_count"},
			want: `
	# HELP apiserver_egress_dialer_dial_failure_count [ALPHA] Dial failure count, labeled by the protocol (http-connect or grpc), transport (tcp or uds), and stage (connect or proxy). The stage indicates at which stage the dial failed
	# TYPE apiserver_egress_dialer_dial_failure_count counter
	apiserver_egress_dialer_dial_failure_count{protocol="fake_protocol",stage="connect",transport="fake_transport"} 1
`,
		},
		"connect succeeded, proxy failed": {
			connectorErr: false,
			proxierErr:   true,
			metrics:      []string{"apiserver_egress_dialer_dial_failure_count"},
			want: `
	# HELP apiserver_egress_dialer_dial_failure_count [ALPHA] Dial failure count, labeled by the protocol (http-connect or grpc), transport (tcp or uds), and stage (connect or proxy). The stage indicates at which stage the dial failed
	# TYPE apiserver_egress_dialer_dial_failure_count counter
	apiserver_egress_dialer_dial_failure_count{protocol="fake_protocol",stage="proxy",transport="fake_transport"} 1
`,
		},
		"successful": {
			connectorErr: false,
			proxierErr:   false,
			metrics:      []string{"apiserver_egress_dialer_dial_duration_seconds"},
			want: `
            # HELP apiserver_egress_dialer_dial_duration_seconds [ALPHA] Dial latency histogram in seconds, labeled by the protocol (http-connect or grpc), transport (tcp or uds)
            # TYPE apiserver_egress_dialer_dial_duration_seconds histogram
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="0.005"} 1
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="0.025"} 1
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="0.1"} 1
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="0.5"} 1
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="2.5"} 1
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="12.5"} 1
            apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="fake_protocol",transport="fake_transport",le="+Inf"} 1
            apiserver_egress_dialer_dial_duration_seconds_sum{protocol="fake_protocol",transport="fake_transport"} 0
            apiserver_egress_dialer_dial_duration_seconds_count{protocol="fake_protocol",transport="fake_transport"} 1
`,
		},
	}
	for tn, tc := range testcases {

		t.Run(tn, func(t *testing.T) {
			metrics.Metrics.Reset()
			metrics.Metrics.SetClock(testingclock.NewFakeClock(time.Now()))
			d := dialerCreator{
				connector: &fakeProxyServerConnector{
					connectorErr: tc.connectorErr,
					proxierErr:   tc.proxierErr,
				},
				options: metricsOptions{
					transport: "fake_transport",
					protocol:  "fake_protocol",
				},
			}
			dialer := d.createDialer()
			dialer(context.TODO(), "", "")
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metrics...); err != nil {
				t.Errorf("Err in comparing metrics %v", err)
			}
		})
	}
}

func TestKonnectivityClientMetrics(t *testing.T) {
	testcases := []struct {
		name    string
		metrics []string
		trigger func()
		want    string
	}{
		{
			name:    "stream packets",
			metrics: []string{"konnectivity_network_proxy_client_stream_packets_total"},
			trigger: func() {
				clientmetrics.Metrics.ObservePacket(ccmetrics.SegmentFromClient, client.PacketType_DIAL_REQ)
			},
			want: `
# HELP konnectivity_network_proxy_client_stream_packets_total Count of packets processed, by segment and packet type (example: from_client, DIAL_REQ)
# TYPE konnectivity_network_proxy_client_stream_packets_total counter
konnectivity_network_proxy_client_stream_packets_total{packet_type="DIAL_REQ",segment="from_client"} 1
`,
		},
		{
			name:    "stream errors",
			metrics: []string{"konnectivity_network_proxy_client_stream_errors_total"},
			trigger: func() {
				clientmetrics.Metrics.ObserveStreamError(ccmetrics.SegmentToClient, errors.New("example"), client.PacketType_DIAL_RSP)
			},
			want: `
# HELP konnectivity_network_proxy_client_stream_errors_total Count of gRPC stream errors, by segment, grpc Code, packet type. (example: from_agent, Code.Unavailable, DIAL_RSP)
# TYPE konnectivity_network_proxy_client_stream_errors_total counter
konnectivity_network_proxy_client_stream_errors_total{code="Unknown",packet_type="DIAL_RSP",segment="to_client"} 1
`,
		},
		{
			name:    "dial failure",
			metrics: []string{"konnectivity_network_proxy_client_dial_failure_total"},
			trigger: func() {
				clientmetrics.Metrics.ObserveDialFailure(clientmetrics.DialFailureTimeout)
			},
			want: `
# HELP konnectivity_network_proxy_client_dial_failure_total Number of dial failures observed, by reason (example: remote endpoint error)
# TYPE konnectivity_network_proxy_client_dial_failure_total counter
konnectivity_network_proxy_client_dial_failure_total{reason="timeout"} 1
`,
		},
		{
			name:    "client connections",
			metrics: []string{"konnectivity_network_proxy_client_client_connections"},
			trigger: func() {
				clientmetrics.Metrics.GetClientConnectionsMetric().WithLabelValues("dialing").Inc()
			},
			want: `
# HELP konnectivity_network_proxy_client_client_connections Number of open client connections, by status (Example: dialing)
# TYPE konnectivity_network_proxy_client_client_connections gauge
konnectivity_network_proxy_client_client_connections{status="dialing"} 1
`,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			tc.trigger()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.want), tc.metrics...); err != nil {
				t.Errorf("GatherAndCompare error: %v", err)
			}
		})
	}
}

func TestGetTLSConfig(t *testing.T) {
	tempDir := t.TempDir()

	certPEM, keyPEM, err := certutil.GenerateSelfSignedCertKey("localhost", nil, nil)
	if err != nil {
		t.Fatalf("Failed to generate test certificates: %v", err)
	}

	certPath := filepath.Join(tempDir, "cert.crt")
	keyPath := filepath.Join(tempDir, "cert.key")
	if err := os.WriteFile(certPath, certPEM, 0600); err != nil {
		t.Fatalf("Failed to write cert file: %v", err)
	}
	if err := os.WriteFile(keyPath, keyPEM, 0600); err != nil {
		t.Fatalf("Failed to write key file: %v", err)
	}

	testcases := []struct {
		name               string
		tlsConfig          *apiserver.TLSConfig
		expectedServerName string
	}{
		{
			name: "with TLSServerName set",
			tlsConfig: &apiserver.TLSConfig{
				CABundle:      certPath,
				ClientCert:    certPath,
				ClientKey:     keyPath,
				TLSServerName: "custom-server.example.com",
			},
			expectedServerName: "custom-server.example.com",
		},
		{
			name: "without TLSServerName (empty)",
			tlsConfig: &apiserver.TLSConfig{
				CABundle:      certPath,
				ClientCert:    certPath,
				ClientKey:     keyPath,
				TLSServerName: "",
			},
			expectedServerName: "",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			tlsConfig, err := getTLSConfig(tc.tlsConfig)
			if err != nil {
				t.Fatalf("getTLSConfig returned unexpected error: %v", err)
			}

			if tlsConfig.ServerName != tc.expectedServerName {
				t.Errorf("expected ServerName %q, got %q", tc.expectedServerName, tlsConfig.ServerName)
			}
		})
	}
}

func TestHTTPConnectProxierReturnsOnContextDeadline(t *testing.T) {
	clientConn, proxyConn := net.Pipe()
	defer func() {
		_ = clientConn.Close()
		_ = proxyConn.Close()
	}()

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		_, err := (&httpConnectProxier{
			conn:         clientConn,
			proxyAddress: "proxy",
		}).proxy(ctx, "webhook.default.svc:443")
		errCh <- err
	}()

	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("expected CONNECT to fail when proxy does not return 200 OK")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected context deadline exceeded, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for HTTP CONNECT proxy to return")
	}
}

func TestHTTPConnectProxierReturnsOnContextCancel(t *testing.T) {
	clientConn, proxyConn := net.Pipe()
	defer func() {
		_ = clientConn.Close()
		_ = proxyConn.Close()
	}()

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	requestReadCh := make(chan error, 1)
	go func() {
		req, err := http.ReadRequest(bufio.NewReader(proxyConn))
		if err == nil {
			_ = req.Body.Close()
		}
		requestReadCh <- err
	}()

	errCh := make(chan error, 1)
	go func() {
		_, err := (&httpConnectProxier{
			conn:         clientConn,
			proxyAddress: "proxy",
		}).proxy(ctx, "webhook.default.svc:443")
		errCh <- err
	}()

	select {
	case err := <-requestReadCh:
		if err != nil {
			t.Fatalf("failed to read CONNECT request: %v", err)
		}

	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for CONNECT request")
	}

	select {
	case <-errCh:
		t.Fatal("should be blocked in building tunnel")
	default:
	}

	cancel()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for HTTP CONNECT proxy to return")
	}
}

func TestDialerCreatorContextDeadline(t *testing.T) {
	captureDeadline := func(t *testing.T, ctx context.Context) time.Time {
		t.Helper()

		var deadline time.Time
		dialer := (&dialerCreator{connector: proxyServerConnectorFunc(func(ctx context.Context) (proxier, error) {
			deadline, _ = ctx.Deadline()
			return &fakeProxier{}, nil
		})}).createDialer()

		if _, err := dialer(ctx, "tcp", "webhook.default.svc:443"); err != nil {
			t.Fatalf("unexpected dial error: %v", err)
		}
		if deadline.IsZero() {
			t.Fatal("expected dial context to have a deadline")
		}
		return deadline
	}

	t.Run("adds default deadline", func(t *testing.T) {
		_ = captureDeadline(t, context.Background())
	})

	t.Run("preserves existing deadline", func(t *testing.T) {
		want := time.Now().Add(time.Hour)
		ctx, cancel := context.WithDeadline(context.Background(), want)
		defer cancel()

		got := captureDeadline(t, ctx)
		if !got.Equal(want) {
			t.Fatalf("expected deadline %v, got %v", want, got)
		}
	})
}

type fakeGRPCTunnel struct {
	dial func(context.Context, string, string) (net.Conn, error)
}

func (f *fakeGRPCTunnel) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	return f.dial(ctx, network, address)
}

func (*fakeGRPCTunnel) Done() <-chan struct{} {
	return nil
}

func TestGRPCProxierCancelsTunnel(t *testing.T) {
	t.Run("dial failure", func(t *testing.T) {
		dialErr := errors.New("dial failed")
		tunnelCtx, cancel := context.WithCancelCause(context.Background())
		proxier := &grpcProxier{
			tunnel: &fakeGRPCTunnel{dial: func(context.Context, string, string) (net.Conn, error) {
				return nil, dialErr
			}},
			cancel: cancel,
		}

		if _, err := proxier.proxy(t.Context(), "webhook.default.svc:443"); !errors.Is(err, dialErr) {
			t.Fatalf("expected dial error %v, got %v", dialErr, err)
		}

		if cause := context.Cause(tunnelCtx); !errors.Is(cause, dialErr) {
			t.Fatalf("expected tunnel cancellation cause %v, got %v", dialErr, cause)
		}
	})

	t.Run("connection close", func(t *testing.T) {
		clientConn, serverConn := net.Pipe()
		defer func() {
			_ = clientConn.Close()
			_ = serverConn.Close()
		}()

		tunnelCtx, cancel := context.WithCancelCause(context.Background())
		proxier := &grpcProxier{
			tunnel: &fakeGRPCTunnel{dial: func(context.Context, string, string) (net.Conn, error) {
				return clientConn, nil
			}},
			cancel: cancel,
		}

		conn, err := proxier.proxy(t.Context(), "webhook.default.svc:443")
		if err != nil {
			t.Fatalf("unexpected dial error: %v", err)
		}
		if err := conn.Close(); err != nil {
			t.Fatalf("unexpected close error: %v", err)
		}
		if cause := context.Cause(tunnelCtx); !errors.Is(cause, context.Canceled) {
			t.Fatalf("expected tunnel to be canceled when connection closes, got %v", cause)
		}
	})
}

func TestUDSGRPCConnectorTunnelOutlivesConnectContext(t *testing.T) {
	udsName := filepath.Join(t.TempDir(), "proxy.sock")
	listener, err := net.Listen("unix", udsName)
	if err != nil {
		t.Fatalf("failed to listen on UDS: %v", err)
	}

	proxyServer := &testGRPCProxyServer{t: t}
	grpcServer := grpc.NewServer()
	client.RegisterProxyServiceServer(grpcServer, proxyServer)

	serveErrCh := make(chan error, 1)
	go func() {
		serveErrCh <- grpcServer.Serve(listener)
	}()

	t.Cleanup(func() {
		grpcServer.Stop()
		select {
		case err := <-serveErrCh:
			if err != nil && !errors.Is(err, grpc.ErrServerStopped) {
				t.Logf("gRPC proxy server failed: %v", err)
			}
		case <-time.After(30 * time.Second):
			t.Fatal("failed to wait for gRPC proxy server to exit")
		}
	})

	canceledCtx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := (&udsGRPCConnector{udsName: udsName}).connect(canceledCtx); err == nil {
		t.Fatal("expected canceled connect context to stop connection setup")
	}

	connectCtx, cancelConnect := context.WithCancel(t.Context())
	defer cancelConnect()
	proxier, err := (&udsGRPCConnector{udsName: udsName}).connect(connectCtx)
	if err != nil {
		t.Fatalf("failed to connect to gRPC proxy: %v", err)
	}

	// tunnel context should be detached from connecting one.
	cancelConnect()

	dialCtx, cancelDial := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancelDial()
	conn, err := proxier.proxy(dialCtx, "webhook.default.svc:443")
	if err != nil {
		t.Fatalf("tunnel did not survive connect context cancellation: %v", err)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("failed to close tunneled connection: %v", err)
	}
}

type testGRPCProxyServer struct {
	t *testing.T
	client.UnimplementedProxyServiceServer
}

func (s *testGRPCProxyServer) Proxy(stream client.ProxyService_ProxyServer) error {
	dialPacket, err := stream.Recv()
	if err != nil {
		return err
	}

	s.t.Logf("received packet: %s", dialPacket.String())

	if dialPacket.Type != client.PacketType_DIAL_REQ {
		return fmt.Errorf("expected DIAL_REQ, got %v", dialPacket.Type)
	}
	if err := stream.Send(&client.Packet{
		Type: client.PacketType_DIAL_RSP,
		Payload: &client.Packet_DialResponse{DialResponse: &client.DialResponse{
			ConnectID: 1,
			Random:    dialPacket.GetDialRequest().Random,
		}},
	}); err != nil {
		return err
	}

	closePacket, err := stream.Recv()
	if err != nil {
		return err
	}
	if closePacket.Type != client.PacketType_CLOSE_REQ {
		return fmt.Errorf("expected CLOSE_REQ, got %v", closePacket.Type)
	}
	return stream.Send(&client.Packet{
		Type: client.PacketType_CLOSE_RSP,
		Payload: &client.Packet_CloseResponse{CloseResponse: &client.CloseResponse{
			ConnectID: closePacket.GetCloseRequest().ConnectID,
		}},
	})
}
