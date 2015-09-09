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

package unversioned

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"reflect"
	gruntime "runtime"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"
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

	// Dial is a custom dialer used for the client
	Dial func(net, addr string) (net.Conn, error)
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
// replication controllers, daemons, and services. It allows operations such as list, get, update
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
	experimentalConfig := *c
	experimentalClient, err := NewExperimental(&experimentalConfig)
	if err != nil {
		return nil, err
	}
	return &Client{RESTClient: client, ExperimentalClient: experimentalClient}, nil
}

// MatchesServerVersion queries the server to compares the build version
// (git hash) of the client with the server's build version. It returns an error
// if it failed to contact the server or if the versions are not an exact match.
func MatchesServerVersion(client *Client, c *Config) error {
	var err error
	if client == nil {
		client, err = New(c)
		if err != nil {
			return err
		}
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

// NegotiateVersion queries the server's supported api versions to find
// a version that both client and server support.
// - If no version is provided, try registered client versions in order of
//   preference.
// - If version is provided, but not default config (explicitly requested via
//   commandline flag), and is unsupported by the server, print a warning to
//   stderr and try client's registered versions in order of preference.
// - If version is config default, and the server does not support it,
//   return an error.
func NegotiateVersion(client *Client, c *Config, version string, clientRegisteredVersions []string) (string, error) {
	var err error
	if client == nil {
		client, err = New(c)
		if err != nil {
			return "", err
		}
	}
	clientVersions := sets.String{}
	for _, v := range clientRegisteredVersions {
		clientVersions.Insert(v)
	}
	apiVersions, err := client.ServerAPIVersions()
	if err != nil {
		return "", fmt.Errorf("couldn't read version from server: %v", err)
	}
	serverVersions := sets.String{}
	for _, v := range apiVersions.Versions {
		serverVersions.Insert(v)
	}
	// If no version requested, use config version (may also be empty).
	if len(version) == 0 {
		version = c.Version
	}
	// If version explicitly requested verify that both client and server support it.
	// If server does not support warn, but try to negotiate a lower version.
	if len(version) != 0 {
		if !clientVersions.Has(version) {
			return "", fmt.Errorf("Client does not support API version '%s'. Client supported API versions: %v", version, clientVersions)

		}
		if serverVersions.Has(version) {
			return version, nil
		}
		// If we are using an explicit config version the server does not support, fail.
		if version == c.Version {
			return "", fmt.Errorf("Server does not support API version '%s'.", version)
		}
	}

	for _, clientVersion := range clientRegisteredVersions {
		if serverVersions.Has(clientVersion) {
			// Version was not explicitly requested in command config (--api-version).
			// Ok to fall back to a supported version with a warning.
			if len(version) != 0 {
				glog.Warningf("Server does not support API version '%s'. Falling back to '%s'.", version, clientVersion)
			}
			return clientVersion, nil
		}
	}
	return "", fmt.Errorf("Failed to negotiate an api version. Server supports: %v. Client supports: %v.",
		serverVersions, clientRegisteredVersions)
}

// NewOrDie creates a Kubernetes client and panics if the provided API version is not recognized.
func NewOrDie(c *Config) *Client {
	client, err := New(c)
	if err != nil {
		panic(err)
	}
	return client
}

// InClusterConfig returns a config object which uses the service account
// kubernetes gives to pods. It's intended for clients that expect to be
// running inside a pod running on kuberenetes. It will return an error if
// called from a process not running in a kubernetes environment.
func InClusterConfig() (*Config, error) {
	token, err := ioutil.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/" + api.ServiceAccountTokenKey)
	if err != nil {
		return nil, err
	}
	tlsClientConfig := TLSClientConfig{}
	rootCAFile := "/var/run/secrets/kubernetes.io/serviceaccount/" + api.ServiceAccountRootCAKey
	if _, err := util.CertPoolFromFile(rootCAFile); err != nil {
		glog.Errorf("expected to load root CA config from %s, but got err: %v", rootCAFile, err)
	} else {
		tlsClientConfig.CAFile = rootCAFile
	}

	return &Config{
		// TODO: switch to using cluster DNS.
		Host:            "https://" + net.JoinHostPort(os.Getenv("KUBERNETES_SERVICE_HOST"), os.Getenv("KUBERNETES_SERVICE_PORT")),
		BearerToken:     string(token),
		TLSClientConfig: tlsClientConfig,
	}, nil
}

// NewInCluster is a shortcut for calling InClusterConfig() and then New().
func NewInCluster() (*Client, error) {
	cc, err := InClusterConfig()
	if err != nil {
		return nil, err
	}
	return New(cc)
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

	client := NewRESTClient(baseURL, config.Version, config.Codec, config.QPS, config.Burst)

	transport, err := TransportFor(config)
	if err != nil {
		return nil, err
	}

	if transport != http.DefaultTransport {
		client.Client = &http.Client{Transport: transport}
	}
	return client, nil
}

var (
	// tlsTransports stores reusable round trippers with custom TLSClientConfig options
	tlsTransports = map[string]*http.Transport{}

	// tlsTransportLock protects retrieval and storage of round trippers into the tlsTransports map
	tlsTransportLock sync.Mutex
)

// tlsTransportFor returns a http.RoundTripper for the given config, or an error
// The same RoundTripper will be returned for configs with identical TLS options
// If the config has no custom TLS options, http.DefaultTransport is returned
func tlsTransportFor(config *Config) (http.RoundTripper, error) {
	// Get a unique key for the TLS options in the config
	key, err := tlsConfigKey(config)
	if err != nil {
		return nil, err
	}

	// Ensure we only create a single transport for the given TLS options
	tlsTransportLock.Lock()
	defer tlsTransportLock.Unlock()

	// See if we already have a custom transport for this config
	if cachedTransport, ok := tlsTransports[key]; ok {
		return cachedTransport, nil
	}

	// Get the TLS options for this client config
	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		return nil, err
	}
	// The options didn't require a custom TLS config
	if tlsConfig == nil {
		return http.DefaultTransport, nil
	}

	// Cache a single transport for these options
	tlsTransports[key] = &http.Transport{
		TLSClientConfig: tlsConfig,
		Proxy:           http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
	}
	return tlsTransports[key], nil
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

	var (
		transport http.RoundTripper
		err       error
	)

	if config.Transport != nil {
		transport = config.Transport
	} else {
		transport, err = tlsTransportFor(config)
		if err != nil {
			return nil, err
		}
	}

	// Call wrap prior to adding debugging wrappers
	if config.WrapTransport != nil {
		transport = config.WrapTransport(transport)
	}

	switch {
	case bool(glog.V(9)):
		transport = NewDebuggingRoundTripper(transport, CurlCommand, URLTiming, ResponseHeaders)
	case bool(glog.V(8)):
		transport = NewDebuggingRoundTripper(transport, JustURL, RequestHeaders, ResponseStatus, ResponseHeaders)
	case bool(glog.V(7)):
		transport = NewDebuggingRoundTripper(transport, JustURL, RequestHeaders, ResponseStatus)
	case bool(glog.V(6)):
		transport = NewDebuggingRoundTripper(transport, URLTiming)
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
			return nil, fmt.Errorf("host must be a URL or a host:port pair: %q", base)
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
