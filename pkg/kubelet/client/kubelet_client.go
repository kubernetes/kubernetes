/*
Copyright 2014 The Kubernetes Authors.

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

package client

import (
	"context"
	"crypto/tls"
	"encoding/base32"
	"fmt"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/server/egressselector"
	"k8s.io/client-go/transport"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

// KubeletClientConfig defines config parameters for the kubelet client
type KubeletClientConfig struct {
	// Port specifies the default port - used if no information about Kubelet port can be found in Node.NodeStatus.DaemonEndpoints.
	Port uint

	// ReadOnlyPort specifies the Port for ReadOnly communications.
	ReadOnlyPort uint

	// PreferredAddressTypes - used to select an address from Node.NodeStatus.Addresses
	PreferredAddressTypes []string

	// TLSClientConfig contains settings to enable transport layer security
	TLSClientConfig KubeletTLSConfig

	// HTTPTimeout is used by the client to timeout http requests to Kubelet.
	HTTPTimeout time.Duration

	// Lookup will give us a dialer if the egress selector is configured for it
	Lookup egressselector.Lookup
}

type KubeletTLSConfig struct {
	// Server requires TLS client certificate authentication
	CertFile string
	// Server requires TLS client certificate authentication
	KeyFile string
	// Trusted root certificates for server
	CAFile string
	// Check that the kubelet's serving certificate common name matches the name of the kubelet
	ValidateNodeName bool
}

// ConnectionInfo provides the information needed to connect to a kubelet
type ConnectionInfo struct {
	Scheme                         string
	Hostname                       string
	Port                           string
	Transport                      http.RoundTripper
	InsecureSkipTLSVerifyTransport http.RoundTripper
}

// ConnectionInfoGetter provides ConnectionInfo for the kubelet running on a named node
type ConnectionInfoGetter interface {
	GetConnectionInfo(ctx context.Context, nodeName types.NodeName) (*ConnectionInfo, error)
}

type reqHostRewrite struct {
	commonName    string
	connCacheKey  string
	tlsServerName string
	tcpDialAddr   string
}

type reqHostRewriteAndH1 struct {
	hostRewrite reqHostRewrite
	onlyH1      bool
}

type reqHostRewriteKeyType int

const reqHostRewriteKey reqHostRewriteKeyType = iota

func withReqHostRewrite(ctx context.Context, hostRewrite reqHostRewrite, onlyH1 bool) context.Context {
	return context.WithValue(ctx, reqHostRewriteKey, reqHostRewriteAndH1{hostRewrite: hostRewrite, onlyH1: onlyH1})
}

func reqHostRewriteFrom(ctx context.Context) (reqHostRewrite, bool, bool) {
	hostRewriteAndH1, ok := ctx.Value(reqHostRewriteKey).(reqHostRewriteAndH1)
	return hostRewriteAndH1.hostRewrite, hostRewriteAndH1.onlyH1, ok
}

// MakeTransport creates a secure RoundTripper for HTTP Transport.
func MakeTransport(config *KubeletClientConfig) (http.RoundTripper, error) {
	return makeTransport(config, false)
}

// MakeInsecureTransport creates an insecure RoundTripper for HTTP Transport.
func MakeInsecureTransport(config *KubeletClientConfig) (http.RoundTripper, error) {
	return makeTransport(config, true)
}

// makeTransport creates a RoundTripper for HTTP Transport.
func makeTransport(config *KubeletClientConfig, insecureSkipTLSVerify bool) (http.RoundTripper, error) {
	// do the insecureSkipTLSVerify on the pre-transport *before* we go get a potentially cached connection.
	// transportConfig always produces a new struct pointer.
	transportConfig := config.transportConfig()
	if insecureSkipTLSVerify {
		transportConfig.TLS.Insecure = true
		transportConfig.TLS.CAFile = "" // we are only using files so we can ignore CAData
	}

	if config.Lookup != nil {
		// Assuming EgressSelector if SSHTunnel is not turned on.
		// We will not get a dialer if egress selector is disabled.
		networkContext := egressselector.Cluster.AsNetworkContext()
		dialer, err := config.Lookup(networkContext)
		if err != nil {
			return nil, fmt.Errorf("failed to get context dialer for 'cluster': got %v", err)
		}
		if dialer != nil {
			transportConfig.DialHolder = &transport.DialHolder{Dial: dialer}
		}
	}
	return transport.New(transportConfig)
}

// transportConfig converts a client config to an appropriate transport config.
func (c *KubeletClientConfig) transportConfig() *transport.Config {
	cfg := &transport.Config{
		// always bust the client-go TLS cache instead of only when KubeletClientConfig.Lookup is set.
		// this allows us to safely mutate the underlying *http.Transport in mutateTransportToValidateNodeName.
		// note that we do not need the client-go TLS cache since we make and re-use a small number of transports.
		// TODO add unit test
		//  1. unique transport per call for same inputs
		//  2. unwrapping still works at high log levels when the transport is actually wrapped
		Proxy: http.ProxyFromEnvironment,
		TLS: transport.TLSConfig{
			CAFile:   c.TLSClientConfig.CAFile,
			CertFile: c.TLSClientConfig.CertFile,
			KeyFile:  c.TLSClientConfig.KeyFile,
			// transport.loadTLSFiles would set this to true because we are only using files
			// it is clearer to set it explicitly here so we remember that this is happening
			ReloadTLSFiles: true,
		},
	}
	if !cfg.HasCA() {
		cfg.TLS.Insecure = true
	}
	return cfg
}

// NodeGetter defines an interface for looking up a node by name
type NodeGetter interface {
	Get(ctx context.Context, name string, options metav1.GetOptions) (*v1.Node, error)
}

// NodeGetterFunc allows implementing NodeGetter with a function
type NodeGetterFunc func(ctx context.Context, name string, options metav1.GetOptions) (*v1.Node, error)

// Get fetches information via NodeGetterFunc.
func (f NodeGetterFunc) Get(ctx context.Context, name string, options metav1.GetOptions) (*v1.Node, error) {
	return f(ctx, name, options)
}

// NodeConnectionInfoGetter obtains connection info from the status of a Node API object
type NodeConnectionInfoGetter struct {
	// nodes is used to look up Node objects
	nodes NodeGetter
	// scheme is the scheme to use to connect to all kubelets
	scheme string
	// defaultPort is the port to use if no Kubelet endpoint port is recorded in the node status
	defaultPort int
	// transport is the transport to use to send a request to all kubelets
	transport http.RoundTripper
	// check that the kubelet's serving certificate common name matches the name of the kubelet
	validateNodeName bool
	// insecureSkipTLSVerifyTransport is the transport to use if the kube-apiserver wants to skip verifying the TLS certificate of the kubelet
	insecureSkipTLSVerifyTransport http.RoundTripper
	// preferredAddressTypes specifies the preferred order to use to find a node address
	preferredAddressTypes []v1.NodeAddressType
}

// NewNodeConnectionInfoGetter creates a new NodeConnectionInfoGetter.
func NewNodeConnectionInfoGetter(nodes NodeGetter, config KubeletClientConfig) (ConnectionInfoGetter, error) {
	transport, err := MakeTransport(&config)
	if err != nil {
		return nil, err
	}
	if config.TLSClientConfig.ValidateNodeName {
		if err := mutateTransportToValidateNodeName(transport); err != nil {
			return nil, err
		}
	}

	insecureSkipTLSVerifyTransport, err := MakeInsecureTransport(&config)
	if err != nil {
		return nil, err
	}

	types := []v1.NodeAddressType{}
	for _, t := range config.PreferredAddressTypes {
		types = append(types, v1.NodeAddressType(t))
	}

	return &NodeConnectionInfoGetter{
		nodes:                          nodes,
		scheme:                         "https",
		defaultPort:                    int(config.Port),
		transport:                      transport,
		validateNodeName:               config.TLSClientConfig.ValidateNodeName,
		insecureSkipTLSVerifyTransport: insecureSkipTLSVerifyTransport,

		preferredAddressTypes: types,
	}, nil
}

// GetConnectionInfo retrieves connection info from the status of a Node API object.
func (k *NodeConnectionInfoGetter) GetConnectionInfo(ctx context.Context, nodeName types.NodeName) (*ConnectionInfo, error) {
	node, err := k.nodes.Get(ctx, string(nodeName), metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	// Find a kubelet-reported address, using preferred address type
	hostname, err := nodeutil.GetPreferredNodeAddress(node, k.preferredAddressTypes)
	if err != nil {
		return nil, err
	}

	// Use the kubelet-reported port, if present
	port := int(node.Status.DaemonEndpoints.KubeletEndpoint.Port)
	if port <= 0 {
		port = k.defaultPort
	}
	portStr := strconv.Itoa(port)

	rt := k.transport
	if k.validateNodeName {
		rt = withValidateNodeName(rt, string(nodeName), hostname, portStr)
	}

	return &ConnectionInfo{
		Scheme:                         k.scheme,
		Hostname:                       hostname,
		Port:                           portStr,
		Transport:                      rt,
		InsecureSkipTLSVerifyTransport: k.insecureSkipTLSVerifyTransport,
	}, nil
}

func withValidateNodeName(delegate http.RoundTripper, nodeName, hostname, port string) http.RoundTripper {
	// TODO should we hash the hostname before encoding it?
	base32Hostname := strings.ToLower(base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString([]byte(hostname)))
	return &validateNodeNameRoundTripper{
		delegate: delegate,
		hostRewrite: reqHostRewrite{
			commonName: "system:node:" + nodeName,
			// nodeName cannot contain ; or : due to ValidateNodeName
			// hostname could be anything so we lowercase base32 encode it -> it also cannot contain ; or :
			// therefore we know this is an unambiguous cache key that passes httpguts.ValidHostHeader and net.SplitHostPort
			connCacheKey:  net.JoinHostPort(nodeName+";"+base32Hostname, port),
			tlsServerName: hostname,
			tcpDialAddr:   net.JoinHostPort(hostname, port),
		},
	}
}

var _ utilnet.RoundTripperWrapper = &validateNodeNameRoundTripper{}

type validateNodeNameRoundTripper struct {
	delegate    http.RoundTripper
	hostRewrite reqHostRewrite
}

func (r *validateNodeNameRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return r.delegate
}

func (r *validateNodeNameRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "https" {
		return nil, fmt.Errorf(`%q must use scheme "https", got %q`, r.hostRewrite.commonName, req.URL.Scheme)
	}
	// TODO these checks enforce that proxy URL is not set, does that make sense?
	if req.URL.Host != r.hostRewrite.tcpDialAddr {
		return nil, fmt.Errorf("%q must use host %q, got %q", r.hostRewrite.commonName, r.hostRewrite.tcpDialAddr, req.URL.Host)
	}
	if len(req.Host) > 0 && req.Host != r.hostRewrite.tcpDialAddr {
		return nil, fmt.Errorf("%q must use host %q, got %q", r.hostRewrite.commonName, r.hostRewrite.tcpDialAddr, req.Host)
	}

	onlyH1 := wsstream.IsWebSocketRequest(req) // matches *http.Request.requiresHTTP1
	ctx := withReqHostRewrite(req.Context(), r.hostRewrite, onlyH1)
	req = req.Clone(ctx) // so we can mutate the request URL
	// set the "Host" header / ":authority" pseudo-header field so that servers
	// can correctly perform host based routing based on the original value.
	req.Host = r.hostRewrite.tcpDialAddr
	// override targetAddr used by *http.Transport.connectMethodForRequest which directly
	// controls the cache key used by *http.Transport.getConn and *http.Transport.dialConn.
	// we undo this rewrite in *http.Transport.DialTLSContext by passing in the original
	// value to *http.Transport.DialContext.  DialTLSContext maintains the regular TLS
	// verification semantics by setting *tls.Config.ServerName to the same value that
	// *http.persistConn.addTLS would have set if we did not rewrite the host value here.
	// this also works for http2 requests as http.http2authorityAddr uses this same value
	// to determine the http2ClientConnPool.GetClientConn input which is used as the map
	// key for tracking and re-using http.http2ClientConn values.
	req.URL.Host = r.hostRewrite.connCacheKey

	return r.delegate.RoundTrip(req)
}

func mutateTransportToValidateNodeName(baseRT http.RoundTripper) error {
	rt, err := httpTransportFor(baseRT)
	if err != nil {
		return fmt.Errorf("failed to get *http.Transport for transport: %w", err)
	}
	if rt.DialTLSContext != nil || rt.DialContext == nil || rt.TLSHandshakeTimeout == 0 ||
		rt.TLSClientConfig == nil || len(rt.TLSClientConfig.ServerName) != 0 || rt.TLSClientConfig.VerifyConnection != nil {
		return fmt.Errorf("*http.Transport has invalid state: %#v", rt) // sanity check client-go internals that we rely on
	}
	// mutate the underlying *http.Transport in-place
	rt.DialTLSContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if network != "tcp" {
			return nil, fmt.Errorf("only tcp connections are supported")
		}
		hostRewrite, onlyH1, ok := reqHostRewriteFrom(ctx)
		if !ok {
			return nil, fmt.Errorf("failed to get dial rewrite from context")
		}
		if address != hostRewrite.connCacheKey {
			return nil, fmt.Errorf("rewrite of %q->%q failed, got unexpected address %q", hostRewrite.connCacheKey, hostRewrite.tcpDialAddr, address)
		}
		plainConn, err := rt.DialContext(ctx, "tcp", hostRewrite.tcpDialAddr) // make the TCP connection to the original host
		if err != nil {
			return nil, fmt.Errorf("failed to make TCP connection: %w", err)
		}
		tlsConfig := rt.TLSClientConfig.Clone()                           // shallow clone so we can mutate it per connection
		tlsConfig.ServerName = hostRewrite.tlsServerName                  // maintain the existing SAN check via *x509.Certificate.VerifyHostname
		tlsConfig.VerifyConnection = func(cs tls.ConnectionState) error { // add the extra kubelet common name check
			leaf := cs.PeerCertificates[0]
			if leaf.Subject.CommonName != hostRewrite.commonName {
				return fmt.Errorf("invalid node name; expected %q, got %q", hostRewrite.commonName, leaf.Subject.CommonName)
			}
			if !slices.Contains(leaf.Subject.Organization, user.NodesGroup) {
				return fmt.Errorf("invalid node groups; expected to include %q, got %q", user.NodesGroup, leaf.Subject.Organization)
			}
			return nil
		}
		if onlyH1 {
			tlsConfig.NextProtos = nil // copied from *http.persistConn.addTLS
		}
		tlsConn, err := addTLS(ctx, plainConn, tlsConfig, rt.TLSHandshakeTimeout) // ensure TLSHandshakeTimeout is honored
		if err != nil {
			return nil, err
		}
		return tlsConn, nil
	}
	return nil
}

func httpTransportFor(rt http.RoundTripper) (*http.Transport, error) {
	switch t := rt.(type) {
	case *http.Transport:
		return t, nil
	case utilnet.RoundTripperWrapper: // TODO see if we can add a unit test to enforce that all http.RoundTripper in k/k implement this
		return httpTransportFor(t.WrappedRoundTripper())
	default:
		return nil, fmt.Errorf("unknown transport type: %T", t)
	}
}

// tlsHandshakeTimeoutError and addTLS below are copied with modifications from *http.persistConn.addTLS.
// we lose the exact timings of the tracing TLSHandshakeStart and TLSHandshakeDone hooks
// but not calling them in addTLS seems better than calling them multiple times for the same TLS connection.
// we need to copy addTLS to preserve the non-blocking cancellation logic for *http.Transport.TLSHandshakeTimeout.

type tlsHandshakeTimeoutError struct{}

func (tlsHandshakeTimeoutError) Timeout() bool   { return true }
func (tlsHandshakeTimeoutError) Temporary() bool { return true }
func (tlsHandshakeTimeoutError) Error() string   { return "net/http: TLS handshake timeout" }

func addTLS(ctx context.Context, plainConn net.Conn, tlsConfig *tls.Config, tlsHandshakeTimeout time.Duration) (*tls.Conn, error) {
	tlsConn := tls.Client(plainConn, tlsConfig)
	errc := make(chan error, 2)
	// for canceling TLS handshake
	timer := time.AfterFunc(tlsHandshakeTimeout, func() {
		errc <- tlsHandshakeTimeoutError{}
	})
	go func() {
		err := tlsConn.HandshakeContext(ctx)
		timer.Stop()
		errc <- err
	}()
	if err := <-errc; err != nil {
		_ = plainConn.Close()
		if err == (tlsHandshakeTimeoutError{}) {
			// Now that we have closed the connection,
			// wait for the call to HandshakeContext to return.
			<-errc
		}
		return nil, err
	}
	return tlsConn, nil
}
