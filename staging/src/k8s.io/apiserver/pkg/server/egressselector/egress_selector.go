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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/apis/apiserver"
	egressmetrics "k8s.io/apiserver/pkg/server/egressselector/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
	client "sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/client"
)

var directDialer utilnet.DialFunc = http.DefaultTransport.(*http.Transport).DialContext

func init() {
	client.Metrics.RegisterMetrics(legacyregistry.Registerer())
}

// EgressSelector is the map of network context type to context dialer, for network egress.
type EgressSelector struct {
	egressToDialer map[EgressType]utilnet.DialFunc
}

// EgressType is an indicator of which egress selection should be used for sending traffic.
// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/1281-network-proxy/README.md#network-context
type EgressType int

const (
	// ControlPlane is the EgressType for traffic intended to go to the control plane.
	ControlPlane EgressType = iota
	// Etcd is the EgressType for traffic intended to go to Kubernetes persistence store.
	Etcd
	// Cluster is the EgressType for traffic intended to go to the system being managed by Kubernetes.
	Cluster
)

// NetworkContext is the struct used by Kubernetes API Server to indicate where it intends traffic to be sent.
type NetworkContext struct {
	// EgressSelectionName is the unique name of the
	// EgressSelectorConfiguration which determines
	// the network we route the traffic to.
	EgressSelectionName EgressType
}

// Lookup is the interface to get the dialer function for the network context.
type Lookup func(networkContext NetworkContext) (utilnet.DialFunc, error)

// String returns the canonical string representation of the egress type
func (s EgressType) String() string {
	switch s {
	case ControlPlane:
		return "controlplane"
	case Etcd:
		return "etcd"
	case Cluster:
		return "cluster"
	default:
		return "invalid"
	}
}

// AsNetworkContext is a helper function to make it easy to get the basic NetworkContext objects.
func (s EgressType) AsNetworkContext() NetworkContext {
	return NetworkContext{EgressSelectionName: s}
}

func lookupServiceName(name string) (EgressType, error) {
	switch strings.ToLower(name) {
	case "controlplane":
		return ControlPlane, nil
	case "etcd":
		return Etcd, nil
	case "cluster":
		return Cluster, nil
	}
	return -1, fmt.Errorf("unrecognized service name %s", name)
}

func tunnelHTTPConnect(proxyConn net.Conn, proxyAddress, addr string) (net.Conn, error) {
	fmt.Fprintf(proxyConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", addr, "127.0.0.1")
	br := bufio.NewReader(proxyConn)
	res, err := http.ReadResponse(br, nil)
	if err != nil {
		proxyConn.Close()
		return nil, fmt.Errorf("reading HTTP response from CONNECT to %s via proxy %s failed: %v",
			addr, proxyAddress, err)
	}
	if res.StatusCode != 200 {
		proxyConn.Close()
		return nil, fmt.Errorf("proxy error from %s while dialing %s, code %d: %v",
			proxyAddress, addr, res.StatusCode, res.Status)
	}

	// It's safe to discard the bufio.Reader here and return the
	// original TCP conn directly because we only use this for
	// TLS, and in TLS the client speaks first, so we know there's
	// no unbuffered data. But we can double-check.
	if br.Buffered() > 0 {
		proxyConn.Close()
		return nil, fmt.Errorf("unexpected %d bytes of buffered data from CONNECT proxy %q",
			br.Buffered(), proxyAddress)
	}
	return proxyConn, nil
}

type proxier interface {
	// proxy returns a connection to addr.
	proxy(ctx context.Context, addr string) (net.Conn, error)
}

var _ proxier = &httpConnectProxier{}

type httpConnectProxier struct {
	conn         net.Conn
	proxyAddress string
}

func (t *httpConnectProxier) proxy(ctx context.Context, addr string) (net.Conn, error) {
	return tunnelHTTPConnect(t.conn, t.proxyAddress, addr)
}

var _ proxier = &grpcProxier{}

type grpcProxier struct {
	tunnel client.Tunnel
}

func (g *grpcProxier) proxy(ctx context.Context, addr string) (net.Conn, error) {
	return g.tunnel.DialContext(ctx, "tcp", addr)
}

type proxyServerConnector interface {
	// connect establishes connection to the proxy server, and returns a
	// proxier based on the connection.
	//
	// The provided Context must be non-nil. The context is used for connecting to the proxy only.
	// If the context expires before the connection is complete, an error is returned.
	// Once successfully connected to the proxy, any expiration of the context will not affect the connection.
	connect(context.Context) (proxier, error)
}

type tcpHTTPConnectConnector struct {
	proxyAddress string
	tlsConfig    *tls.Config
}

func (t *tcpHTTPConnectConnector) connect(ctx context.Context) (proxier, error) {
	d := tls.Dialer{
		Config: t.tlsConfig,
	}
	conn, err := d.DialContext(ctx, "tcp", t.proxyAddress)
	if err != nil {
		return nil, err
	}
	return &httpConnectProxier{conn: conn, proxyAddress: t.proxyAddress}, nil
}

type udsHTTPConnectConnector struct {
	udsName string
}

func (u *udsHTTPConnectConnector) connect(ctx context.Context) (proxier, error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "unix", u.udsName)
	if err != nil {
		return nil, err
	}
	return &httpConnectProxier{conn: conn, proxyAddress: u.udsName}, nil
}

type udsGRPCConnector struct {
	udsName string
}

// connect establishes a connection to a proxy over gRPC.
// TODO At the moment, it does not use the provided context.
func (u *udsGRPCConnector) connect(_ context.Context) (proxier, error) {
	udsName := u.udsName
	dialOption := grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
		var d net.Dialer
		c, err := d.DialContext(ctx, "unix", udsName)
		if err != nil {
			klog.Errorf("failed to create connection to uds name %s, error: %v", udsName, err)
		}
		return c, err
	})

	// CreateSingleUseGrpcTunnel() unfortunately couples dial and connection contexts. Because of that,
	// we cannot use ctx just for dialing and control the connection lifetime separately.
	// See https://github.com/kubernetes-sigs/apiserver-network-proxy/issues/357.
	tunnelCtx := context.TODO()
	tunnel, err := client.CreateSingleUseGrpcTunnel(tunnelCtx, udsName, dialOption,
		grpc.WithBlock(),
		grpc.WithReturnConnectionError(),
		grpc.WithTimeout(30*time.Second), // matches http.DefaultTransport dial timeout
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &grpcProxier{tunnel: tunnel}, nil
}

type dialerCreator struct {
	connector proxyServerConnector
	direct    bool
	options   metricsOptions
}

type metricsOptions struct {
	transport string
	protocol  string
}

func (d *dialerCreator) createDialer() utilnet.DialFunc {
	if d.direct {
		return directDialer
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		ctx, span := tracing.Start(ctx, fmt.Sprintf("Proxy via %s protocol over %s", d.options.protocol, d.options.transport), attribute.String("address", addr))
		defer span.End(500 * time.Millisecond)
		start := egressmetrics.Metrics.Clock().Now()
		egressmetrics.Metrics.ObserveDialStart(d.options.protocol, d.options.transport)
		proxier, err := d.connector.connect(ctx)
		if err != nil {
			egressmetrics.Metrics.ObserveDialFailure(d.options.protocol, d.options.transport, egressmetrics.StageConnect)
			return nil, err
		}
		conn, err := proxier.proxy(ctx, addr)
		if err != nil {
			egressmetrics.Metrics.ObserveDialFailure(d.options.protocol, d.options.transport, egressmetrics.StageProxy)
			return nil, err
		}
		egressmetrics.Metrics.ObserveDialLatency(egressmetrics.Metrics.Clock().Now().Sub(start), d.options.protocol, d.options.transport)
		return conn, nil
	}
}

func getTLSConfig(t *apiserver.TLSConfig) (*tls.Config, error) {
	clientCert := t.ClientCert
	clientKey := t.ClientKey
	caCert := t.CABundle
	clientCerts, err := tls.LoadX509KeyPair(clientCert, clientKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read key pair %s & %s, got %v", clientCert, clientKey, err)
	}
	certPool := x509.NewCertPool()
	if caCert != "" {
		certBytes, err := os.ReadFile(caCert)
		if err != nil {
			return nil, fmt.Errorf("failed to read cert file %s, got %v", caCert, err)
		}
		ok := certPool.AppendCertsFromPEM(certBytes)
		if !ok {
			return nil, fmt.Errorf("failed to append CA cert to the cert pool")
		}
	} else {
		// Use host's root CA set instead of providing our own
		certPool = nil
	}
	return &tls.Config{
		Certificates: []tls.Certificate{clientCerts},
		RootCAs:      certPool,
	}, nil
}

func getProxyAddress(urlString string) (string, error) {
	proxyURL, err := url.Parse(urlString)
	if err != nil {
		return "", fmt.Errorf("invalid proxy server url %q: %v", urlString, err)
	}
	return proxyURL.Host, nil
}

func connectionToDialerCreator(c apiserver.Connection) (*dialerCreator, error) {
	switch c.ProxyProtocol {

	case apiserver.ProtocolHTTPConnect:
		if c.Transport.UDS != nil {
			return &dialerCreator{
				connector: &udsHTTPConnectConnector{
					udsName: c.Transport.UDS.UDSName,
				},
				options: metricsOptions{
					transport: egressmetrics.TransportUDS,
					protocol:  egressmetrics.ProtocolHTTPConnect,
				},
			}, nil
		} else if c.Transport.TCP != nil {
			tlsConfig, err := getTLSConfig(c.Transport.TCP.TLSConfig)
			if err != nil {
				return nil, err
			}
			proxyAddress, err := getProxyAddress(c.Transport.TCP.URL)
			if err != nil {
				return nil, err
			}
			return &dialerCreator{
				connector: &tcpHTTPConnectConnector{
					tlsConfig:    tlsConfig,
					proxyAddress: proxyAddress,
				},
				options: metricsOptions{
					transport: egressmetrics.TransportTCP,
					protocol:  egressmetrics.ProtocolHTTPConnect,
				},
			}, nil
		} else {
			return nil, fmt.Errorf("Either a TCP or UDS transport must be specified")
		}
	case apiserver.ProtocolGRPC:
		if c.Transport.UDS != nil {
			return &dialerCreator{
				connector: &udsGRPCConnector{
					udsName: c.Transport.UDS.UDSName,
				},
				options: metricsOptions{
					transport: egressmetrics.TransportUDS,
					protocol:  egressmetrics.ProtocolGRPC,
				},
			}, nil
		}
		return nil, fmt.Errorf("UDS transport must be specified for GRPC")
	case apiserver.ProtocolDirect:
		return &dialerCreator{direct: true}, nil
	default:
		return nil, fmt.Errorf("unrecognized service connection protocol %q", c.ProxyProtocol)
	}

}

// NewEgressSelector configures lookup mechanism for Lookup.
// It does so based on a EgressSelectorConfiguration which was read at startup.
func NewEgressSelector(config *apiserver.EgressSelectorConfiguration) (*EgressSelector, error) {
	if config == nil || config.EgressSelections == nil {
		// No Connection Services configured, leaving the serviceMap empty, will return default dialer.
		return nil, nil
	}
	cs := &EgressSelector{
		egressToDialer: make(map[EgressType]utilnet.DialFunc),
	}
	for _, service := range config.EgressSelections {
		name, err := lookupServiceName(service.Name)
		if err != nil {
			return nil, err
		}
		dialerCreator, err := connectionToDialerCreator(service.Connection)
		if err != nil {
			return nil, fmt.Errorf("failed to create dialer for egressSelection %q: %v", name, err)
		}
		cs.egressToDialer[name] = dialerCreator.createDialer()
	}
	return cs, nil
}

// NewEgressSelectorWithMap returns a EgressSelector with the supplied EgressType to DialFunc map.
func NewEgressSelectorWithMap(m map[EgressType]utilnet.DialFunc) *EgressSelector {
	if m == nil {
		m = make(map[EgressType]utilnet.DialFunc)
	}
	return &EgressSelector{
		egressToDialer: m,
	}
}

// Lookup gets the dialer function for the network context.
// This is configured for the Kubernetes API Server at startup.
func (cs *EgressSelector) Lookup(networkContext NetworkContext) (utilnet.DialFunc, error) {
	if cs.egressToDialer == nil {
		// The round trip wrapper will over-ride the dialContext method appropriately
		return nil, nil
	}

	return cs.egressToDialer[networkContext.EgressSelectionName], nil
}
