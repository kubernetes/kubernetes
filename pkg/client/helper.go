/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"reflect"
	gruntime "runtime"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

// Config holds the common attributes that can be passed to a Kubernetes client on
// initialization.
type Config struct {
	// Host must be a host string, a host:port pair, or a URL to the base of the API.
	Host string
	// Prefix is the sub path of the server. If not specified, the client will set
	// a default value.  Use "/" to indicate the server root should be used
	Prefix string
	// Version is the API version to talk to. Must be provided when initializing
	// a RESTClient directly. When initializing a Client, will be set with the default
	// code version.
	Version string
	// LegacyBehavior defines whether the RESTClient should follow conventions that
	// existed prior to v1beta3 in Kubernetes - namely, namespace (if specified)
	// not being part of the path, and resource names allowing mixed case. Set to
	// true when using Kubernetes v1beta1 or v1beta2.
	LegacyBehavior bool
	// Codec specifies the encoding and decoding behavior for runtime.Objects passed
	// to a RESTClient or Client. Required when initializing a RESTClient, optional
	// when initializing a Client.
	Codec runtime.Codec

	// Server requires Basic authentication
	Username string
	Password string

	// Server requires Bearer authentication. This client will not attempt to use
	// refresh tokens for an OAuth2 flow.
	// TODO: demonstrate an OAuth2 compatible client.
	BearerToken string

	// TLSClientConfig contains settings to enable transport layer security
	TLSClientConfig

	// Server should be accessed without verifying the TLS
	// certificate. For testing only.
	Insecure bool

	// UserAgent is an optional field that specifies the caller of this request.
	UserAgent string

	// Transport may be used for custom HTTP behavior. This attribute may not
	// be specified with the TLS client certificate options. Use WrapTransport
	// for most client level operations.
	Transport http.RoundTripper
	// WrapTransport will be invoked for custom HTTP behavior after the underlying
	// transport is initialized (either the transport created from TLSClientConfig,
	// Transport, or http.DefaultTransport). The config may layer other RoundTrippers
	// on top of the returned RoundTripper.
	WrapTransport func(rt http.RoundTripper) http.RoundTripper

	// QPS indicates the maximum QPS to the master from this client.  If zero, QPS is unlimited.
	QPS float32

	// Maximum burst for throttle
	Burst int
}

type KubeletConfig struct {
	// ToDo: Add support for different kubelet instances exposing different ports
	Port        uint
	EnableHttps bool

	// TLSClientConfig contains settings to enable transport layer security
	TLSClientConfig

	// HTTPTimeout is used by the client to timeout http requests to Kubelet.
	HTTPTimeout time.Duration
}

// TLSClientConfig contains settings to enable transport layer security
type TLSClientConfig struct {
	// Server requires TLS client certificate authentication
	CertFile string
	// Server requires TLS client certificate authentication
	KeyFile string
	// Trusted root certificates for server
	CAFile string

	// CertData holds PEM-encoded bytes (typically read from a client certificate file).
	// CertData takes precedence over CertFile
	CertData []byte
	// KeyData holds PEM-encoded bytes (typically read from a client certificate key file).
	// KeyData takes precedence over KeyFile
	KeyData []byte
	// CAData holds PEM-encoded bytes (typically read from a root certificates bundle).
	// CAData takes precedence over CAFile
	CAData []byte
}

// New creates a Kubernetes client for the given config. This client works with pods,
// replication controllers and services. It allows operations such as list, get, update
// and delete on these objects. An error is returned if the provided configuration
// is not valid.
func New(c *Config) (*Client, error) {
	config := *c
	if err := SetKubernetesDefaults(&config); err != nil {
		return nil, err
	}
	client, err := RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &Client{client}, nil
}

func MatchesServerVersion(c *Config) error {
	client, err := New(c)
	if err != nil {
		return err
	}

	clientVersion := version.Get()
	serverVersion, err := client.ServerVersion()
	if err != nil {
		return fmt.Errorf("couldn't read version from server: %v\n", err)
	}
	if s := *serverVersion; !reflect.DeepEqual(clientVersion, s) {
		return fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", s, clientVersion)
	}

	return nil
}

// NewOrDie creates a Kubernetes client and panics if the provided API version is not recognized.
func NewOrDie(c *Config) *Client {
	client, err := New(c)
	if err != nil {
		panic(err)
	}
	return client
}

// SetKubernetesDefaults sets default values on the provided client config for accessing the
// Kubernetes API or returns an error if any of the defaults are impossible or invalid.
func SetKubernetesDefaults(config *Config) error {
	if config.Prefix == "" {
		config.Prefix = "/api"
	}
	if len(config.UserAgent) == 0 {
		config.UserAgent = DefaultKubernetesUserAgent()
	}
	if len(config.Version) == 0 {
		config.Version = defaultVersionFor(config)
	}
	version := config.Version
	versionInterfaces, err := latest.InterfacesFor(version)
	if err != nil {
		return fmt.Errorf("API version '%s' is not recognized (valid values: %s)", version, strings.Join(latest.Versions, ", "))
	}
	if config.Codec == nil {
		config.Codec = versionInterfaces.Codec
	}
	config.LegacyBehavior = (version == "v1beta1" || version == "v1beta2")
	if config.QPS == 0.0 {
		config.QPS = 5.0
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}

// RESTClientFor returns a RESTClient that satisfies the requested attributes on a client Config
// object. Note that a RESTClient may require fields that are optional when initializing a Client.
// A RESTClient created by this method is generic - it expects to operate on an API that follows
// the Kubernetes conventions, but may not be the Kubernetes API.
func RESTClientFor(config *Config) (*RESTClient, error) {
	if len(config.Version) == 0 {
		return nil, fmt.Errorf("version is required when initializing a RESTClient")
	}
	if config.Codec == nil {
		return nil, fmt.Errorf("Codec is required when initializing a RESTClient")
	}

	baseURL, err := defaultServerUrlFor(config)
	if err != nil {
		return nil, err
	}

	client := NewRESTClient(baseURL, config.Version, config.Codec, config.LegacyBehavior, config.QPS, config.Burst)

	transport, err := TransportFor(config)
	if err != nil {
		return nil, err
	}

	if transport != http.DefaultTransport {
		client.Client = &http.Client{Transport: transport}
	}
	return client, nil
}

// TransportFor returns an http.RoundTripper that will provide the authentication
// or transport level security defined by the provided Config. Will return the
// default http.DefaultTransport if no special case behavior is needed.
func TransportFor(config *Config) (http.RoundTripper, error) {
	hasCA := len(config.CAFile) > 0 || len(config.CAData) > 0
	hasCert := len(config.CertFile) > 0 || len(config.CertData) > 0

	// Set transport level security
	if config.Transport != nil && (hasCA || hasCert || config.Insecure) {
		return nil, fmt.Errorf("using a custom transport with TLS certificate options or the insecure flag is not allowed")
	}

	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		return nil, err
	}

	var transport http.RoundTripper
	if config.Transport != nil {
		transport = config.Transport
	} else {
		if tlsConfig != nil {
			transport = &http.Transport{
				TLSClientConfig: tlsConfig,
				Proxy:           http.ProxyFromEnvironment,
				Dial: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).Dial,
				TLSHandshakeTimeout: 10 * time.Second,
			}
		} else {
			transport = http.DefaultTransport
		}
	}
	if config.WrapTransport != nil {
		transport = config.WrapTransport(transport)
	}

	transport, err = HTTPWrappersForConfig(config, transport)
	if err != nil {
		return nil, err
	}

	// TODO: use the config context to wrap a transport

	return transport, nil
}

// HTTPWrappersForConfig wraps a round tripper with any relevant layered behavior from the
// config. Exposed to allow more clients that need HTTP-like behavior but then must hijack
// the underlying connection (like WebSocket or HTTP2 clients). Pure HTTP clients should use
// the higher level TransportFor or RESTClientFor methods.
func HTTPWrappersForConfig(config *Config, rt http.RoundTripper) (http.RoundTripper, error) {
	// Set authentication wrappers
	hasBasicAuth := config.Username != "" || config.Password != ""
	if hasBasicAuth && config.BearerToken != "" {
		return nil, fmt.Errorf("username/password or bearer token may be set, but not both")
	}
	switch {
	case config.BearerToken != "":
		rt = NewBearerAuthRoundTripper(config.BearerToken, rt)
	case hasBasicAuth:
		rt = NewBasicAuthRoundTripper(config.Username, config.Password, rt)
	}
	if len(config.UserAgent) > 0 {
		rt = NewUserAgentRoundTripper(config.UserAgent, rt)
	}
	return rt, nil
}

// DefaultServerURL converts a host, host:port, or URL string to the default base server API path
// to use with a Client at a given API version following the standard conventions for a
// Kubernetes API.
func DefaultServerURL(host, prefix, version string, defaultTLS bool) (*url.URL, error) {
	if host == "" {
		return nil, fmt.Errorf("host must be a URL or a host:port pair")
	}
	if version == "" {
		return nil, fmt.Errorf("version must be set")
	}
	base := host
	hostURL, err := url.Parse(base)
	if err != nil {
		return nil, err
	}
	if hostURL.Scheme == "" {
		scheme := "http://"
		if defaultTLS {
			scheme = "https://"
		}
		hostURL, err = url.Parse(scheme + base)
		if err != nil {
			return nil, err
		}
		if hostURL.Path != "" && hostURL.Path != "/" {
			return nil, fmt.Errorf("host must be a URL or a host:port pair: %s", base)
		}
	}

	// If the user specified a URL without a path component (http://server.com), automatically
	// append the default prefix
	if hostURL.Path == "" {
		if prefix == "" {
			prefix = "/"
		}
		hostURL.Path = prefix
	}

	// Add the version to the end of the path
	hostURL.Path = path.Join(hostURL.Path, version)

	return hostURL, nil
}

// IsConfigTransportTLS returns true iff the provided config will result in a protected
// connection to the server when it is passed to client.New() or client.RESTClientFor().
// Use to determine when to send credentials over the wire.
//
// Note: the Insecure flag is ignored when testing for this value, so MITM attacks are
// still possible.
func IsConfigTransportTLS(config Config) bool {
	// determination of TLS transport does not logically require a version to be specified
	// modify the copy of the config we got to satisfy preconditions for defaultServerUrlFor
	config.Version = defaultVersionFor(&config)

	baseURL, err := defaultServerUrlFor(&config)
	if err != nil {
		return false
	}
	return baseURL.Scheme == "https"
}

// defaultServerUrlFor is shared between IsConfigTransportTLS and RESTClientFor. It
// requires Host and Version to be set prior to being called.
func defaultServerUrlFor(config *Config) (*url.URL, error) {
	// TODO: move the default to secure when the apiserver supports TLS by default
	// config.Insecure is taken to mean "I want HTTPS but don't bother checking the certs against a CA."
	hasCA := len(config.CAFile) != 0 || len(config.CAData) != 0
	hasCert := len(config.CertFile) != 0 || len(config.CertData) != 0
	defaultTLS := hasCA || hasCert || config.Insecure
	host := config.Host
	if host == "" {
		host = "localhost"
	}
	return DefaultServerURL(host, config.Prefix, config.Version, defaultTLS)
}

// defaultVersionFor is shared between defaultServerUrlFor and RESTClientFor
func defaultVersionFor(config *Config) string {
	version := config.Version
	if version == "" {
		// Clients default to the preferred code API version
		// TODO: implement version negotiation (highest version supported by server)
		version = latest.Version
	}
	return version
}

// DefaultKubernetesUserAgent returns the default user agent that clients can use.
func DefaultKubernetesUserAgent() string {
	commit := version.Get().GitCommit
	if len(commit) > 7 {
		commit = commit[:7]
	}
	if len(commit) == 0 {
		commit = "unknown"
	}
	version := version.Get().GitVersion
	seg := strings.SplitN(version, "-", 2)
	version = seg[0]
	return fmt.Sprintf("%s/%s (%s/%s) kubernetes/%s", path.Base(os.Args[0]), version, gruntime.GOOS, gruntime.GOARCH, commit)
}
