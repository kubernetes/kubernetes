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
	"encoding/json"
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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"
)

const (
	legacyAPIPath  = "/api"
	defaultAPIPath = "/apis"
)

// Config holds the common attributes that can be passed to a Kubernetes client on
// initialization.
type Config struct {
	// Host must be a host string, a host:port pair, or a URL to the base of the apiserver.
	// If a URL is given then the (optional) Path of that URL represents a prefix that must
	// be appended to all request URIs used to access the apiserver. This allows a frontend
	// proxy to easily relocate all of the apiserver endpoints.
	Host string
	// APIPath is a sub-path that points to an API root.
	APIPath string
	// GroupVersion is the API version to talk to. Must be provided when initializing
	// a RESTClient directly. When initializing a Client, will be set with the default
	// code version.
	GroupVersion *unversioned.GroupVersion
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

	discoveryConfig := *c
	discoveryClient, err := NewDiscoveryClient(&discoveryConfig)
	if err != nil {
		return nil, err
	}

	if _, err := registered.Group(extensions.GroupName); err != nil {
		return &Client{RESTClient: client, ExtensionsClient: nil, DiscoveryClient: discoveryClient}, nil
	}
	experimentalConfig := *c
	experimentalClient, err := NewExtensions(&experimentalConfig)
	if err != nil {
		return nil, err
	}

	return &Client{RESTClient: client, ExtensionsClient: experimentalClient, DiscoveryClient: discoveryClient}, nil
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
	serverVersion, err := client.Discovery().ServerVersion()
	if err != nil {
		return fmt.Errorf("couldn't read version from server: %v\n", err)
	}
	if s := *serverVersion; !reflect.DeepEqual(clientVersion, s) {
		return fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", s, clientVersion)
	}

	return nil
}

func ExtractGroupVersions(l *unversioned.APIGroupList) []string {
	var groupVersions []string
	for _, g := range l.Groups {
		for _, gv := range g.Versions {
			groupVersions = append(groupVersions, gv.GroupVersion)
		}
	}
	return groupVersions
}

// ServerAPIVersions returns the GroupVersions supported by the API server.
// It creates a RESTClient based on the passed in config, but it doesn't rely
// on the Version and Codec of the config, because it uses AbsPath and
// takes the raw response.
func ServerAPIVersions(c *Config) (groupVersions []string, err error) {
	transport, err := TransportFor(c)
	if err != nil {
		return nil, err
	}
	client := http.Client{Transport: transport}

	configCopy := *c
	configCopy.GroupVersion = nil
	configCopy.APIPath = ""
	baseURL, _, err := defaultServerUrlFor(&configCopy)
	if err != nil {
		return nil, err
	}
	// Get the groupVersions exposed at /api
	originalPath := baseURL.Path
	baseURL.Path = path.Join(originalPath, legacyAPIPath)
	resp, err := client.Get(baseURL.String())
	if err != nil {
		return nil, err
	}
	var v unversioned.APIVersions
	defer resp.Body.Close()
	err = json.NewDecoder(resp.Body).Decode(&v)
	if err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}

	groupVersions = append(groupVersions, v.Versions...)
	// Get the groupVersions exposed at /apis
	baseURL.Path = path.Join(originalPath, defaultAPIPath)
	resp2, err := client.Get(baseURL.String())
	if err != nil {
		return nil, err
	}
	var apiGroupList unversioned.APIGroupList
	defer resp2.Body.Close()
	err = json.NewDecoder(resp2.Body).Decode(&apiGroupList)
	if err != nil {
		return nil, fmt.Errorf("unexpected error: %v", err)
	}
	groupVersions = append(groupVersions, ExtractGroupVersions(&apiGroupList)...)

	return groupVersions, nil
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
func NegotiateVersion(client *Client, c *Config, requestedGV *unversioned.GroupVersion, clientRegisteredGVs []unversioned.GroupVersion) (*unversioned.GroupVersion, error) {
	var err error
	if client == nil {
		client, err = New(c)
		if err != nil {
			return nil, err
		}
	}
	clientVersions := sets.String{}
	for _, gv := range clientRegisteredGVs {
		clientVersions.Insert(gv.String())
	}
	groups, err := client.ServerGroups()
	if err != nil {
		// This is almost always a connection error, and higher level code should treat this as a generic error,
		// not a negotiation specific error.
		return nil, err
	}
	versions := ExtractGroupVersions(groups)
	serverVersions := sets.String{}
	for _, v := range versions {
		serverVersions.Insert(v)
	}

	// If no version requested, use config version (may also be empty).
	// make a copy of the original so we don't risk mutating input here or in the returned value
	var preferredGV *unversioned.GroupVersion
	switch {
	case requestedGV != nil:
		t := *requestedGV
		preferredGV = &t
	case c.GroupVersion != nil:
		t := *c.GroupVersion
		preferredGV = &t
	}

	// If version explicitly requested verify that both client and server support it.
	// If server does not support warn, but try to negotiate a lower version.
	if preferredGV != nil {
		if !clientVersions.Has(preferredGV.String()) {
			return nil, fmt.Errorf("client does not support API version %q; client supported API versions: %v", preferredGV, clientVersions)

		}
		if serverVersions.Has(preferredGV.String()) {
			return preferredGV, nil
		}
		// If we are using an explicit config version the server does not support, fail.
		if (c.GroupVersion != nil) && (*preferredGV == *c.GroupVersion) {
			return nil, fmt.Errorf("server does not support API version %q", preferredGV)
		}
	}

	for _, clientGV := range clientRegisteredGVs {
		if serverVersions.Has(clientGV.String()) {
			// Version was not explicitly requested in command config (--api-version).
			// Ok to fall back to a supported version with a warning.
			// TODO: caesarxuchao: enable the warning message when we have
			// proper fix. Please refer to issue #14895.
			// if len(version) != 0 {
			// 	glog.Warningf("Server does not support API version '%s'. Falling back to '%s'.", version, clientVersion)
			// }
			t := clientGV
			return &t, nil
		}
	}
	return nil, fmt.Errorf("failed to negotiate an api version; server supports: %v, client supports: %v",
		serverVersions, clientVersions)
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
	host, port := os.Getenv("KUBERNETES_SERVICE_HOST"), os.Getenv("KUBERNETES_SERVICE_PORT")
	if len(host) == 0 || len(port) == 0 {
		return nil, fmt.Errorf("unable to load in-cluster configuration, KUBERNETES_SERVICE_HOST and KUBERNETES_SERVICE_PORT must be defined")
	}

	token, err := ioutil.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/" + api.ServiceAccountTokenKey)
	if err != nil {
		return nil, err
	}
	tlsClientConfig := TLSClientConfig{}
	rootCAFile := "/var/run/secrets/kubernetes.io/serviceaccount/" + api.ServiceAccountRootCAKey
	if _, err := util.CertPoolFromFile(rootCAFile); err != nil {
		glog.Errorf("Expected to load root CA config from %s, but got err: %v", rootCAFile, err)
	} else {
		tlsClientConfig.CAFile = rootCAFile
	}

	return &Config{
		// TODO: switch to using cluster DNS.
		Host:            "https://" + net.JoinHostPort(host, port),
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
// TODO: this method needs to be split into one that sets defaults per group, expected to be fix in PR "Refactoring clientcache.go and helper.go #14592"
func SetKubernetesDefaults(config *Config) error {
	if config.APIPath == "" {
		config.APIPath = legacyAPIPath
	}
	if len(config.UserAgent) == 0 {
		config.UserAgent = DefaultKubernetesUserAgent()
	}
	g, err := registered.Group(api.GroupName)
	if err != nil {
		return err
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	if config.Codec == nil {
		config.Codec = api.Codecs.LegacyCodec(*config.GroupVersion)
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
	if config.GroupVersion == nil {
		return nil, fmt.Errorf("GroupVersion is required when initializing a RESTClient")
	}
	if config.Codec == nil {
		return nil, fmt.Errorf("Codec is required when initializing a RESTClient")
	}

	baseURL, versionedAPIPath, err := defaultServerUrlFor(config)
	if err != nil {
		return nil, err
	}

	client := NewRESTClient(baseURL, versionedAPIPath, *config.GroupVersion, config.Codec, config.QPS, config.Burst)

	transport, err := TransportFor(config)
	if err != nil {
		return nil, err
	}

	if transport != http.DefaultTransport {
		client.Client = &http.Client{Transport: transport}
	}
	return client, nil
}

// UnversionedRESTClientFor is the same as RESTClientFor, except that it allows
// the config.Version to be empty.
func UnversionedRESTClientFor(config *Config) (*RESTClient, error) {
	if config.Codec == nil {
		return nil, fmt.Errorf("Codec is required when initializing a RESTClient")
	}

	baseURL, versionedAPIPath, err := defaultServerUrlFor(config)
	if err != nil {
		return nil, err
	}

	client := NewRESTClient(baseURL, versionedAPIPath, unversioned.SchemeGroupVersion, config.Codec, config.QPS, config.Burst)

	transport, err := TransportFor(config)
	if err != nil {
		return nil, err
	}

	if transport != http.DefaultTransport {
		client.Client = &http.Client{Transport: transport}
	}
	return client, nil
}

// DefaultServerURL converts a host, host:port, or URL string to the default base server API path
// to use with a Client at a given API version following the standard conventions for a
// Kubernetes API.
func DefaultServerURL(host, apiPath string, groupVersion unversioned.GroupVersion, defaultTLS bool) (*url.URL, string, error) {
	if host == "" {
		return nil, "", fmt.Errorf("host must be a URL or a host:port pair")
	}
	base := host
	hostURL, err := url.Parse(base)
	if err != nil {
		return nil, "", err
	}
	if hostURL.Scheme == "" {
		scheme := "http://"
		if defaultTLS {
			scheme = "https://"
		}
		hostURL, err = url.Parse(scheme + base)
		if err != nil {
			return nil, "", err
		}
		if hostURL.Path != "" && hostURL.Path != "/" {
			return nil, "", fmt.Errorf("host must be a URL or a host:port pair: %q", base)
		}
	}

	// hostURL.Path is optional; a non-empty Path is treated as a prefix that is to be applied to
	// all URIs used to access the host. this is useful when there's a proxy in front of the
	// apiserver that has relocated the apiserver endpoints, forwarding all requests from, for
	// example, /a/b/c to the apiserver. in this case the Path should be /a/b/c.
	//
	// if running without a frontend proxy (that changes the location of the apiserver), then
	// hostURL.Path should be blank.
	//
	// versionedAPIPath, a path relative to baseURL.Path, points to a versioned API base
	versionedAPIPath := path.Join("/", apiPath)

	// Add the version to the end of the path
	if len(groupVersion.Group) > 0 {
		versionedAPIPath = path.Join(versionedAPIPath, groupVersion.Group, groupVersion.Version)

	} else {
		versionedAPIPath = path.Join(versionedAPIPath, groupVersion.Version)

	}

	return hostURL, versionedAPIPath, nil
}

// IsConfigTransportTLS returns true if and only if the provided config will result in a protected
// connection to the server when it is passed to client.New() or client.RESTClientFor().
// Use to determine when to send credentials over the wire.
//
// Note: the Insecure flag is ignored when testing for this value, so MITM attacks are
// still possible.
func IsConfigTransportTLS(config Config) bool {
	// determination of TLS transport does not logically require a version to be specified
	// modify the copy of the config we got to satisfy preconditions for defaultServerUrlFor
	config.GroupVersion = defaultVersionFor(&config)

	baseURL, _, err := defaultServerUrlFor(&config)
	if err != nil {
		return false
	}
	return baseURL.Scheme == "https"
}

// defaultServerUrlFor is shared between IsConfigTransportTLS and RESTClientFor. It
// requires Host and Version to be set prior to being called.
func defaultServerUrlFor(config *Config) (*url.URL, string, error) {
	// TODO: move the default to secure when the apiserver supports TLS by default
	// config.Insecure is taken to mean "I want HTTPS but don't bother checking the certs against a CA."
	hasCA := len(config.CAFile) != 0 || len(config.CAData) != 0
	hasCert := len(config.CertFile) != 0 || len(config.CertData) != 0
	defaultTLS := hasCA || hasCert || config.Insecure
	host := config.Host
	if host == "" {
		host = "localhost"
	}

	if config.GroupVersion != nil {
		return DefaultServerURL(host, config.APIPath, *config.GroupVersion, defaultTLS)
	}
	return DefaultServerURL(host, config.APIPath, unversioned.GroupVersion{}, defaultTLS)
}

// defaultVersionFor is shared between IsConfigTransportTLS and RESTClientFor
func defaultVersionFor(config *Config) *unversioned.GroupVersion {
	if config.GroupVersion == nil {
		// Clients default to the preferred code API version
		// TODO: implement version negotiation (highest version supported by server)
		// TODO this drops out when groupmeta is refactored
		copyGroupVersion := registered.GroupOrDie(api.GroupName).GroupVersion
		return &copyGroupVersion
	}

	return config.GroupVersion
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

// LoadTLSFiles copies the data from the CertFile, KeyFile, and CAFile fields into the CertData,
// KeyData, and CAFile fields, or returns an error. If no error is returned, all three fields are
// either populated or were empty to start.
func LoadTLSFiles(c *Config) error {
	var err error
	c.CAData, err = dataFromSliceOrFile(c.CAData, c.CAFile)
	if err != nil {
		return err
	}

	c.CertData, err = dataFromSliceOrFile(c.CertData, c.CertFile)
	if err != nil {
		return err
	}

	c.KeyData, err = dataFromSliceOrFile(c.KeyData, c.KeyFile)
	if err != nil {
		return err
	}
	return nil
}

// dataFromSliceOrFile returns data from the slice (if non-empty), or from the file,
// or an error if an error occurred reading the file
func dataFromSliceOrFile(data []byte, file string) ([]byte, error) {
	if len(data) > 0 {
		return data, nil
	}
	if len(file) > 0 {
		fileData, err := ioutil.ReadFile(file)
		if err != nil {
			return []byte{}, err
		}
		return fileData, nil
	}
	return nil, nil
}
