/*
Package client is a Go client for the Docker Engine API.

The "docker" command uses this package to communicate with the daemon. It can also
be used by your own Go applications to do anything the command-line interface does
– running containers, pulling images, managing swarms, etc.

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
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
)

// DefaultVersion is the version of the current stable API
const DefaultVersion string = "1.25"

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

// NewEnvClient initializes a new API client based on environment variables.
// Use DOCKER_HOST to set the url to the docker server.
// Use DOCKER_API_VERSION to set the version of the API to reach, leave empty for latest.
// Use DOCKER_CERT_PATH to load the tls certificates from.
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
		}
	}

	host := os.Getenv("DOCKER_HOST")
	if host == "" {
		host = DefaultDockerHost
	}
	version := os.Getenv("DOCKER_API_VERSION")
	if version == "" {
		version = DefaultVersion
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
	proto, addr, basePath, err := ParseHost(host)
	if err != nil {
		return nil, err
	}

	if client != nil {
		if _, ok := client.Transport.(*http.Transport); !ok {
			return nil, fmt.Errorf("unable to verify TLS configuration, invalid transport %v", client.Transport)
		}
	} else {
		transport := new(http.Transport)
		sockets.ConfigureTransport(transport, proto, addr)
		client = &http.Client{
			Transport: transport,
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

	return &Client{
		scheme:            scheme,
		host:              host,
		proto:             proto,
		addr:              addr,
		basePath:          basePath,
		client:            client,
		version:           version,
		customHTTPHeaders: httpHeaders,
	}, nil
}

// Close ensures that transport.Client is closed
// especially needed while using NewClient with *http.Client = nil
// for example
// client.NewClient("unix:///var/run/docker.sock", nil, "v1.18", map[string]string{"User-Agent": "engine-api-cli-1.0"})
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
		apiPath = fmt.Sprintf("%s/v%s%s", cli.basePath, v, p)
	} else {
		apiPath = fmt.Sprintf("%s%s", cli.basePath, p)
	}

	u := &url.URL{
		Path: apiPath,
	}
	if len(query) > 0 {
		u.RawQuery = query.Encode()
	}
	return u.String()
}

// ClientVersion returns the version string associated with this
// instance of the Client. Note that this value can be changed
// via the DOCKER_API_VERSION env var.
func (cli *Client) ClientVersion() string {
	return cli.version
}

// UpdateClientVersion updates the version string associated with this
// instance of the Client.
func (cli *Client) UpdateClientVersion(v string) {
	if !cli.manualOverride {
		cli.version = v
	}

}

// ParseHost verifies that the given host strings is valid.
func ParseHost(host string) (string, string, string, error) {
	protoAddrParts := strings.SplitN(host, "://", 2)
	if len(protoAddrParts) == 1 {
		return "", "", "", fmt.Errorf("unable to parse docker host `%s`", host)
	}

	var basePath string
	proto, addr := protoAddrParts[0], protoAddrParts[1]
	if proto == "tcp" {
		parsed, err := url.Parse("tcp://" + addr)
		if err != nil {
			return "", "", "", err
		}
		addr = parsed.Host
		basePath = parsed.Path
	}
	return proto, addr, basePath, nil
}
