/*
Copyright 2014 Google Inc. All rights reserved.

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
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
)

// Config holds the common attributes that can be passed to a Kubernetes client on
// initialization.
type Config struct {
	// Host must be a host string, a host:port pair, or a URL to the base of the API.
	Host string
	// Prefix is the sub path of the server. If not specified, the client will set
	// a default value.  Use "/" to indicate the server root should be used
	Prefix string
	// Version is the API version to talk to. If not specified, the client will use
	// the preferred version.
	Version string

	// Server requires Basic authentication
	Username string
	Password string

	// Server requires Bearer authentication. This client will not attempt to use
	// refresh tokens for an OAuth2 flow.
	// TODO: demonstrate an OAuth2 compatible client.
	BearerToken string

	// Server requires TLS client certificate authentication
	CertFile string
	KeyFile  string
	CAFile   string

	// Server should be accessed without verifying the TLS
	// certificate. For testing only.
	Insecure bool

	// Transport may be used for custom HTTP behavior. This attribute may not
	// be specified with the TLS client certificate options.
	Transport http.RoundTripper
}

type KubeletConfig struct {
	// ToDo: Add support for different kubelet instances exposing different ports
	Port        uint
	EnableHttps bool

	// TLS Configuration, only applies if EnableHttps is true.
	CertFile string
	// TLS Configuration, only applies if EnableHttps is true.
	KeyFile string
	// TLS Configuration, only applies if EnableHttps is true.
	CAFile string
}

// New creates a Kubernetes client for the given config. This client works with pods,
// replication controllers and services. It allows operations such as list, get, update
// and delete on these objects. An error is returned if the provided configuration
// is not valid.
func New(c *Config) (*Client, error) {
	config := *c
	if config.Prefix == "" {
		config.Prefix = "/api"
	}
	client, err := RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &Client{client}, nil
}

// NewOrDie creates a Kubernetes client and panics if the provided API version is not recognized.
func NewOrDie(c *Config) *Client {
	client, err := New(c)
	if err != nil {
		panic(err)
	}
	return client
}

// RESTClientFor returns a RESTClient that satisfies the requested attributes on a client Config
// object.
func RESTClientFor(config *Config) (*RESTClient, error) {
	version := defaultVersionFor(config)

	// Set version
	versionInterfaces, err := latest.InterfacesFor(version)
	if err != nil {
		return nil, fmt.Errorf("API version '%s' is not recognized (valid values: %s)", version, strings.Join(latest.Versions, ", "))
	}

	baseURL, err := defaultServerUrlFor(config)
	if err != nil {
		return nil, err
	}

	client := NewRESTClient(baseURL, versionInterfaces.Codec)

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
	// Set transport level security
	if config.Transport != nil && (config.CertFile != "" || config.Insecure) {
		return nil, fmt.Errorf("using a custom transport with TLS certificate options or the insecure flag is not allowed")
	}
	var transport http.RoundTripper
	switch {
	case config.Transport != nil:
		transport = config.Transport
	case config.CertFile != "":
		t, err := NewClientCertTLSTransport(config.CertFile, config.KeyFile, config.CAFile)
		if err != nil {
			return nil, err
		}
		transport = t
	case config.Insecure:
		transport = NewUnsafeTLSTransport()
	default:
		transport = http.DefaultTransport
	}

	// Set authentication wrappers
	hasBasicAuth := config.Username != "" || config.Password != ""
	if hasBasicAuth && config.BearerToken != "" {
		return nil, fmt.Errorf("username/password or bearer token may be set, but not both")
	}
	switch {
	case config.BearerToken != "":
		transport = NewBearerAuthRoundTripper(config.BearerToken, transport)
	case hasBasicAuth:
		transport = NewBasicAuthRoundTripper(config.Username, config.Password, transport)
	}

	// TODO: use the config context to wrap a transport

	return transport, nil
}

// DefaultServerURL converts a host, host:port, or URL string to the default base server API path
// to use with a Client at a given API version following the standard conventions for a
// Kubernetes API.
func DefaultServerURL(host, prefix, version string, defaultSecure bool) (*url.URL, error) {
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
		if defaultSecure {
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

// IsConfigTransportSecure returns true iff the provided config will result in a protected
// connection to the server when it is passed to client.New() or client.RESTClientFor().
// Use to determine when to send credentials over the wire.
//
// Note: the Insecure flag is ignored when testing for this value, so MITM attacks are
// still possible.
func IsConfigTransportSecure(config *Config) bool {
	baseURL, err := defaultServerUrlFor(config)
	if err != nil {
		return false
	}
	return baseURL.Scheme == "https"
}

// defaultServerUrlFor is shared between IsConfigSecure and RESTClientFor
func defaultServerUrlFor(config *Config) (*url.URL, error) {
	version := defaultVersionFor(config)
	// TODO: move the default to secure when the apiserver supports TLS by default
	defaultSecure := config.CertFile != ""
	host := config.Host
	if host == "" {
		host = "localhost"
	}
	return DefaultServerURL(host, config.Prefix, version, defaultSecure)
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
