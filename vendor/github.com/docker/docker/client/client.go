/*
Package client is a Go client for the Docker Engine API.

For more information about the Engine API, see the documentation:
https://docs.docker.com/engine/reference/api/

Usage

You use the library by creating a client object and calling methods on it. The
client can be created either from environment variables with NewEnvClient, or
configured manually with NewClient.

For example, to list running containers (the equivalent of "docker ps"):

	package main

	import (
		"context"
		"fmt"

		"github.com/docker/docker/api/types"
		"github.com/docker/docker/client"
	)

	func main() {
		cli, err := client.NewEnvClient()
		if err != nil {
			panic(err)
		}

		containers, err := cli.ContainerList(context.Background(), types.ContainerListOptions{})
		if err != nil {
			panic(err)
		}

		for _, container := range containers {
			fmt.Printf("%s %s\n", container.ID[:10], container.Image)
		}
	}

*/
package client

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
	"golang.org/x/net/context"
)

// ErrRedirect is the error returned by checkRedirect when the request is non-GET.
var ErrRedirect = errors.New("unexpected redirect in response")

// Client is the API client that performs all operations
// against a docker server.
type Client struct {
	// scheme sets the scheme for the client
	scheme string
	// host holds the server address to connect to
	host string
	// proto holds the client protocol i.e. unix.
	proto string
	// addr holds the client address.
	addr string
	// basePath holds the path to prepend to the requests.
	basePath string
	// client used to send and receive http requests.
	client *http.Client
	// version of the server to talk to.
	version string
	// custom http headers configured by users.
	customHTTPHeaders map[string]string
	// manualOverride is set to true when the version was set by users.
	manualOverride bool
}

// CheckRedirect specifies the policy for dealing with redirect responses:
// If the request is non-GET return `ErrRedirect`. Otherwise use the last response.
//
// Go 1.8 changes behavior for HTTP redirects (specifically 301, 307, and 308) in the client .
// The Docker client (and by extension docker API client) can be made to to send a request
// like POST /containers//start where what would normally be in the name section of the URL is empty.
// This triggers an HTTP 301 from the daemon.
// In go 1.8 this 301 will be converted to a GET request, and ends up getting a 404 from the daemon.
// This behavior change manifests in the client in that before the 301 was not followed and
// the client did not generate an error, but now results in a message like Error response from daemon: page not found.
func CheckRedirect(req *http.Request, via []*http.Request) error {
	if via[0].Method == http.MethodGet {
		return http.ErrUseLastResponse
	}
	return ErrRedirect
}

// NewEnvClient initializes a new API client based on environment variables.
// Use DOCKER_HOST to set the url to the docker server.
// Use DOCKER_API_VERSION to set the version of the API to reach, leave empty for latest.
// Use DOCKER_CERT_PATH to load the TLS certificates from.
// Use DOCKER_TLS_VERIFY to enable or disable TLS verification, off by default.
func NewEnvClient() (*Client, error) {
	var client *http.Client
	if dockerCertPath := os.Getenv("DOCKER_CERT_PATH"); dockerCertPath != "" {
		options := tlsconfig.Options{
			CAFile:             filepath.Join(dockerCertPath, "ca.pem"),
			CertFile:           filepath.Join(dockerCertPath, "cert.pem"),
			KeyFile:            filepath.Join(dockerCertPath, "key.pem"),
			InsecureSkipVerify: os.Getenv("DOCKER_TLS_VERIFY") == "",
		}
		tlsc, err := tlsconfig.Client(options)
		if err != nil {
			return nil, err
		}

		client = &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: tlsc,
			},
			CheckRedirect: CheckRedirect,
		}
	}

	host := os.Getenv("DOCKER_HOST")
	if host == "" {
		host = DefaultDockerHost
	}
	version := os.Getenv("DOCKER_API_VERSION")
	if version == "" {
		version = api.DefaultVersion
	}

	cli, err := NewClient(host, version, client, nil)
	if err != nil {
		return cli, err
	}
	if os.Getenv("DOCKER_API_VERSION") != "" {
		cli.manualOverride = true
	}
	return cli, nil
}

// NewClient initializes a new API client for the given host and API version.
// It uses the given http client as transport.
// It also initializes the custom http headers to add to each request.
//
// It won't send any version information if the version number is empty. It is
// highly recommended that you set a version or your client may break if the
// server is upgraded.
func NewClient(host string, version string, client *http.Client, httpHeaders map[string]string) (*Client, error) {
	hostURL, err := ParseHostURL(host)
	if err != nil {
		return nil, err
	}

	if client != nil {
		if _, ok := client.Transport.(http.RoundTripper); !ok {
			return nil, fmt.Errorf("unable to verify TLS configuration, invalid transport %v", client.Transport)
		}
	} else {
		transport := new(http.Transport)
		sockets.ConfigureTransport(transport, hostURL.Scheme, hostURL.Host)
		client = &http.Client{
			Transport:     transport,
			CheckRedirect: CheckRedirect,
		}
	}

	scheme := "http"
	tlsConfig := resolveTLSConfig(client.Transport)
	if tlsConfig != nil {
		// TODO(stevvooe): This isn't really the right way to write clients in Go.
		// `NewClient` should probably only take an `*http.Client` and work from there.
		// Unfortunately, the model of having a host-ish/url-thingy as the connection
		// string has us confusing protocol and transport layers. We continue doing
		// this to avoid breaking existing clients but this should be addressed.
		scheme = "https"
	}

	// TODO: store URL instead of proto/addr/basePath
	return &Client{
		scheme:            scheme,
		host:              host,
		proto:             hostURL.Scheme,
		addr:              hostURL.Host,
		basePath:          hostURL.Path,
		client:            client,
		version:           version,
		customHTTPHeaders: httpHeaders,
	}, nil
}

// Close the transport used by the client
func (cli *Client) Close() error {
	if t, ok := cli.client.Transport.(*http.Transport); ok {
		t.CloseIdleConnections()
	}
	return nil
}

// getAPIPath returns the versioned request path to call the api.
// It appends the query parameters to the path if they are not empty.
func (cli *Client) getAPIPath(p string, query url.Values) string {
	var apiPath string
	if cli.version != "" {
		v := strings.TrimPrefix(cli.version, "v")
		apiPath = path.Join(cli.basePath, "/v"+v, p)
	} else {
		apiPath = path.Join(cli.basePath, p)
	}
	return (&url.URL{Path: apiPath, RawQuery: query.Encode()}).String()
}

// ClientVersion returns the API version used by this client.
func (cli *Client) ClientVersion() string {
	return cli.version
}

// NegotiateAPIVersion queries the API and updates the version to match the
// API version. Any errors are silently ignored.
func (cli *Client) NegotiateAPIVersion(ctx context.Context) {
	ping, _ := cli.Ping(ctx)
	cli.NegotiateAPIVersionPing(ping)
}

// NegotiateAPIVersionPing updates the client version to match the Ping.APIVersion
// if the ping version is less than the default version.
func (cli *Client) NegotiateAPIVersionPing(p types.Ping) {
	if cli.manualOverride {
		return
	}

	// try the latest version before versioning headers existed
	if p.APIVersion == "" {
		p.APIVersion = "1.24"
	}

	// if the client is not initialized with a version, start with the latest supported version
	if cli.version == "" {
		cli.version = api.DefaultVersion
	}

	// if server version is lower than the client version, downgrade
	if versions.LessThan(p.APIVersion, cli.version) {
		cli.version = p.APIVersion
	}
}

// DaemonHost returns the host address used by the client
func (cli *Client) DaemonHost() string {
	return cli.host
}

// ParseHost parses a url string, validates the strings is a host url, and returns
// the parsed host as: protocol, address, and base path
// Deprecated: use ParseHostURL
func ParseHost(host string) (string, string, string, error) {
	hostURL, err := ParseHostURL(host)
	if err != nil {
		return "", "", "", err
	}
	return hostURL.Scheme, hostURL.Host, hostURL.Path, nil
}

// ParseHostURL parses a url string, validates the string is a host url, and
// returns the parsed URL
func ParseHostURL(host string) (*url.URL, error) {
	protoAddrParts := strings.SplitN(host, "://", 2)
	if len(protoAddrParts) == 1 {
		return nil, fmt.Errorf("unable to parse docker host `%s`", host)
	}

	var basePath string
	proto, addr := protoAddrParts[0], protoAddrParts[1]
	if proto == "tcp" {
		parsed, err := url.Parse("tcp://" + addr)
		if err != nil {
			return nil, err
		}
		addr = parsed.Host
		basePath = parsed.Path
	}
	return &url.URL{
		Scheme: proto,
		Host:   addr,
		Path:   basePath,
	}, nil
}

// CustomHTTPHeaders returns the custom http headers stored by the client.
func (cli *Client) CustomHTTPHeaders() map[string]string {
	m := make(map[string]string)
	for k, v := range cli.customHTTPHeaders {
		m[k] = v
	}
	return m
}

// SetCustomHTTPHeaders that will be set on every HTTP request made by the client.
func (cli *Client) SetCustomHTTPHeaders(headers map[string]string) {
	cli.customHTTPHeaders = headers
}
