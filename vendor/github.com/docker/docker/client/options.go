package client

import (
	"context"
	"net"
	"net/http"
	"os"
	"path/filepath"

	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/pkg/errors"
)

// FromEnv configures the client with values from environment variables.
//
// Supported environment variables:
// DOCKER_HOST to set the url to the docker server.
// DOCKER_API_VERSION to set the version of the API to reach, leave empty for latest.
// DOCKER_CERT_PATH to load the TLS certificates from.
// DOCKER_TLS_VERIFY to enable or disable TLS verification, off by default.
func FromEnv(c *Client) error {
	if dockerCertPath := os.Getenv("DOCKER_CERT_PATH"); dockerCertPath != "" {
		options := tlsconfig.Options{
			CAFile:             filepath.Join(dockerCertPath, "ca.pem"),
			CertFile:           filepath.Join(dockerCertPath, "cert.pem"),
			KeyFile:            filepath.Join(dockerCertPath, "key.pem"),
			InsecureSkipVerify: os.Getenv("DOCKER_TLS_VERIFY") == "",
		}
		tlsc, err := tlsconfig.Client(options)
		if err != nil {
			return err
		}

		c.client = &http.Client{
			Transport:     &http.Transport{TLSClientConfig: tlsc},
			CheckRedirect: CheckRedirect,
		}
	}

	if host := os.Getenv("DOCKER_HOST"); host != "" {
		if err := WithHost(host)(c); err != nil {
			return err
		}
	}

	if version := os.Getenv("DOCKER_API_VERSION"); version != "" {
		if err := WithVersion(version)(c); err != nil {
			return err
		}
	}
	return nil
}

// WithDialer applies the dialer.DialContext to the client transport. This can be
// used to set the Timeout and KeepAlive settings of the client.
// Deprecated: use WithDialContext
func WithDialer(dialer *net.Dialer) func(*Client) error {
	return WithDialContext(dialer.DialContext)
}

// WithDialContext applies the dialer to the client transport. This can be
// used to set the Timeout and KeepAlive settings of the client.
func WithDialContext(dialContext func(ctx context.Context, network, addr string) (net.Conn, error)) func(*Client) error {
	return func(c *Client) error {
		if transport, ok := c.client.Transport.(*http.Transport); ok {
			transport.DialContext = dialContext
			return nil
		}
		return errors.Errorf("cannot apply dialer to transport: %T", c.client.Transport)
	}
}

// WithHost overrides the client host with the specified one.
func WithHost(host string) func(*Client) error {
	return func(c *Client) error {
		hostURL, err := ParseHostURL(host)
		if err != nil {
			return err
		}
		c.host = host
		c.proto = hostURL.Scheme
		c.addr = hostURL.Host
		c.basePath = hostURL.Path
		if transport, ok := c.client.Transport.(*http.Transport); ok {
			return sockets.ConfigureTransport(transport, c.proto, c.addr)
		}
		return errors.Errorf("cannot apply host to transport: %T", c.client.Transport)
	}
}

// WithHTTPClient overrides the client http client with the specified one
func WithHTTPClient(client *http.Client) func(*Client) error {
	return func(c *Client) error {
		if client != nil {
			c.client = client
		}
		return nil
	}
}

// WithHTTPHeaders overrides the client default http headers
func WithHTTPHeaders(headers map[string]string) func(*Client) error {
	return func(c *Client) error {
		c.customHTTPHeaders = headers
		return nil
	}
}

// WithScheme overrides the client scheme with the specified one
func WithScheme(scheme string) func(*Client) error {
	return func(c *Client) error {
		c.scheme = scheme
		return nil
	}
}

// WithTLSClientConfig applies a tls config to the client transport.
func WithTLSClientConfig(cacertPath, certPath, keyPath string) func(*Client) error {
	return func(c *Client) error {
		opts := tlsconfig.Options{
			CAFile:             cacertPath,
			CertFile:           certPath,
			KeyFile:            keyPath,
			ExclusiveRootPools: true,
		}
		config, err := tlsconfig.Client(opts)
		if err != nil {
			return errors.Wrap(err, "failed to create tls config")
		}
		if transport, ok := c.client.Transport.(*http.Transport); ok {
			transport.TLSClientConfig = config
			return nil
		}
		return errors.Errorf("cannot apply tls config to transport: %T", c.client.Transport)
	}
}

// WithVersion overrides the client version with the specified one
func WithVersion(version string) func(*Client) error {
	return func(c *Client) error {
		c.version = version
		c.manualOverride = true
		return nil
	}
}
