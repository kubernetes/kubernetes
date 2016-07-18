package client

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/engine-api/client/transport"
	"github.com/docker/go-connections/tlsconfig"
)

// DefaultVersion is the version of the current stable API
const DefaultVersion string = "1.23"

// Client is the API client that performs all operations
// against a docker server.
type Client struct {
	// proto holds the client protocol i.e. unix.
	proto string
	// addr holds the client address.
	addr string
	// basePath holds the path to prepend to the requests.
	basePath string
	// transport is the interface to send request with, it implements transport.Client.
	transport transport.Client
	// version of the server to talk to.
	version string
	// custom http headers configured by users.
	customHTTPHeaders map[string]string
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

	return NewClient(host, version, client, nil)
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

	transport, err := transport.NewTransportWithHTTP(proto, addr, client)
	if err != nil {
		return nil, err
	}

	return &Client{
		proto:             proto,
		addr:              addr,
		basePath:          basePath,
		transport:         transport,
		version:           version,
		customHTTPHeaders: httpHeaders,
	}, nil
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
	cli.version = v
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
