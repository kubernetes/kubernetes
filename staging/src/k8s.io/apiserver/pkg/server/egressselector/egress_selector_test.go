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
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/server/egressselector/metrics"
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
