package request

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api"
	dclient "github.com/docker/docker/client"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/testutil"
	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/pkg/errors"
)

// Method creates a modifier that sets the specified string as the request method
func Method(method string) func(*http.Request) error {
	return func(req *http.Request) error {
		req.Method = method
		return nil
	}
}

// RawString sets the specified string as body for the request
func RawString(content string) func(*http.Request) error {
	return RawContent(ioutil.NopCloser(strings.NewReader(content)))
}

// RawContent sets the specified reader as body for the request
func RawContent(reader io.ReadCloser) func(*http.Request) error {
	return func(req *http.Request) error {
		req.Body = reader
		return nil
	}
}

// ContentType sets the specified Content-Type request header
func ContentType(contentType string) func(*http.Request) error {
	return func(req *http.Request) error {
		req.Header.Set("Content-Type", contentType)
		return nil
	}
}

// JSON sets the Content-Type request header to json
func JSON(req *http.Request) error {
	return ContentType("application/json")(req)
}

// JSONBody creates a modifier that encodes the specified data to a JSON string and set it as request body. It also sets
// the Content-Type header of the request.
func JSONBody(data interface{}) func(*http.Request) error {
	return func(req *http.Request) error {
		jsonData := bytes.NewBuffer(nil)
		if err := json.NewEncoder(jsonData).Encode(data); err != nil {
			return err
		}
		req.Body = ioutil.NopCloser(jsonData)
		req.Header.Set("Content-Type", "application/json")
		return nil
	}
}

// Post creates and execute a POST request on the specified host and endpoint, with the specified request modifiers
func Post(endpoint string, modifiers ...func(*http.Request) error) (*http.Response, io.ReadCloser, error) {
	return Do(endpoint, append(modifiers, Method(http.MethodPost))...)
}

// Delete creates and execute a DELETE request on the specified host and endpoint, with the specified request modifiers
func Delete(endpoint string, modifiers ...func(*http.Request) error) (*http.Response, io.ReadCloser, error) {
	return Do(endpoint, append(modifiers, Method(http.MethodDelete))...)
}

// Get creates and execute a GET request on the specified host and endpoint, with the specified request modifiers
func Get(endpoint string, modifiers ...func(*http.Request) error) (*http.Response, io.ReadCloser, error) {
	return Do(endpoint, modifiers...)
}

// Do creates and execute a request on the specified endpoint, with the specified request modifiers
func Do(endpoint string, modifiers ...func(*http.Request) error) (*http.Response, io.ReadCloser, error) {
	return DoOnHost(DaemonHost(), endpoint, modifiers...)
}

// DoOnHost creates and execute a request on the specified host and endpoint, with the specified request modifiers
func DoOnHost(host, endpoint string, modifiers ...func(*http.Request) error) (*http.Response, io.ReadCloser, error) {
	req, err := New(host, endpoint, modifiers...)
	if err != nil {
		return nil, nil, err
	}
	client, err := NewHTTPClient(host)
	if err != nil {
		return nil, nil, err
	}
	resp, err := client.Do(req)
	var body io.ReadCloser
	if resp != nil {
		body = ioutils.NewReadCloserWrapper(resp.Body, func() error {
			defer resp.Body.Close()
			return nil
		})
	}
	return resp, body, err
}

// New creates a new http Request to the specified host and endpoint, with the specified request modifiers
func New(host, endpoint string, modifiers ...func(*http.Request) error) (*http.Request, error) {
	_, addr, _, err := dclient.ParseHost(host)
	if err != nil {
		return nil, err
	}
	if err != nil {
		return nil, errors.Wrapf(err, "could not parse url %q", host)
	}
	req, err := http.NewRequest("GET", endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("could not create new request: %v", err)
	}

	req.URL.Scheme = "http"
	req.URL.Host = addr

	for _, config := range modifiers {
		if err := config(req); err != nil {
			return nil, err
		}
	}
	return req, nil
}

// NewHTTPClient creates an http client for the specific host
func NewHTTPClient(host string) (*http.Client, error) {
	// FIXME(vdemeester) 10*time.Second timeout of SockRequest… ?
	proto, addr, _, err := dclient.ParseHost(host)
	if err != nil {
		return nil, err
	}
	transport := new(http.Transport)
	if proto == "tcp" && os.Getenv("DOCKER_TLS_VERIFY") != "" {
		// Setup the socket TLS configuration.
		tlsConfig, err := getTLSConfig()
		if err != nil {
			return nil, err
		}
		transport = &http.Transport{TLSClientConfig: tlsConfig}
	}
	transport.DisableKeepAlives = true
	err = sockets.ConfigureTransport(transport, proto, addr)
	return &http.Client{
		Transport: transport,
	}, err
}

// NewClient returns a new Docker API client
func NewClient() (dclient.APIClient, error) {
	host := DaemonHost()
	httpClient, err := NewHTTPClient(host)
	if err != nil {
		return nil, err
	}
	return dclient.NewClient(host, api.DefaultVersion, httpClient, nil)
}

// FIXME(vdemeester) httputil.ClientConn is deprecated, use http.Client instead (closer to actual client)
// Deprecated: Use New instead of NewRequestClient
// Deprecated: use request.Do (or Get, Delete, Post) instead
func newRequestClient(method, endpoint string, data io.Reader, ct, daemon string, modifiers ...func(*http.Request)) (*http.Request, *httputil.ClientConn, error) {
	c, err := SockConn(time.Duration(10*time.Second), daemon)
	if err != nil {
		return nil, nil, fmt.Errorf("could not dial docker daemon: %v", err)
	}

	client := httputil.NewClientConn(c, nil)

	req, err := http.NewRequest(method, endpoint, data)
	if err != nil {
		client.Close()
		return nil, nil, fmt.Errorf("could not create new request: %v", err)
	}

	for _, opt := range modifiers {
		opt(req)
	}

	if ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	return req, client, nil
}

// SockRequest create a request against the specified host (with method, endpoint and other request modifier) and
// returns the status code, and the content as an byte slice
// Deprecated: use request.Do instead
func SockRequest(method, endpoint string, data interface{}, daemon string, modifiers ...func(*http.Request)) (int, []byte, error) {
	jsonData := bytes.NewBuffer(nil)
	if err := json.NewEncoder(jsonData).Encode(data); err != nil {
		return -1, nil, err
	}

	res, body, err := SockRequestRaw(method, endpoint, jsonData, "application/json", daemon, modifiers...)
	if err != nil {
		return -1, nil, err
	}
	b, err := testutil.ReadBody(body)
	return res.StatusCode, b, err
}

// SockRequestRaw create a request against the specified host (with method, endpoint and other request modifier) and
// returns the http response, the output as a io.ReadCloser
// Deprecated: use request.Do (or Get, Delete, Post) instead
func SockRequestRaw(method, endpoint string, data io.Reader, ct, daemon string, modifiers ...func(*http.Request)) (*http.Response, io.ReadCloser, error) {
	req, client, err := newRequestClient(method, endpoint, data, ct, daemon, modifiers...)
	if err != nil {
		return nil, nil, err
	}

	resp, err := client.Do(req)
	if err != nil {
		client.Close()
		return resp, nil, err
	}
	body := ioutils.NewReadCloserWrapper(resp.Body, func() error {
		defer resp.Body.Close()
		return client.Close()
	})

	return resp, body, err
}

// SockRequestHijack creates a connection to specified host (with method, contenttype, …) and returns a hijacked connection
// and the output as a `bufio.Reader`
func SockRequestHijack(method, endpoint string, data io.Reader, ct string, daemon string, modifiers ...func(*http.Request)) (net.Conn, *bufio.Reader, error) {
	req, client, err := newRequestClient(method, endpoint, data, ct, daemon, modifiers...)
	if err != nil {
		return nil, nil, err
	}

	client.Do(req)
	conn, br := client.Hijack()
	return conn, br, nil
}

// SockConn opens a connection on the specified socket
func SockConn(timeout time.Duration, daemon string) (net.Conn, error) {
	daemonURL, err := url.Parse(daemon)
	if err != nil {
		return nil, errors.Wrapf(err, "could not parse url %q", daemon)
	}

	var c net.Conn
	switch daemonURL.Scheme {
	case "npipe":
		return npipeDial(daemonURL.Path, timeout)
	case "unix":
		return net.DialTimeout(daemonURL.Scheme, daemonURL.Path, timeout)
	case "tcp":
		if os.Getenv("DOCKER_TLS_VERIFY") != "" {
			// Setup the socket TLS configuration.
			tlsConfig, err := getTLSConfig()
			if err != nil {
				return nil, err
			}
			dialer := &net.Dialer{Timeout: timeout}
			return tls.DialWithDialer(dialer, daemonURL.Scheme, daemonURL.Host, tlsConfig)
		}
		return net.DialTimeout(daemonURL.Scheme, daemonURL.Host, timeout)
	default:
		return c, errors.Errorf("unknown scheme %v (%s)", daemonURL.Scheme, daemon)
	}
}

func getTLSConfig() (*tls.Config, error) {
	dockerCertPath := os.Getenv("DOCKER_CERT_PATH")

	if dockerCertPath == "" {
		return nil, errors.New("DOCKER_TLS_VERIFY specified, but no DOCKER_CERT_PATH environment variable")
	}

	option := &tlsconfig.Options{
		CAFile:   filepath.Join(dockerCertPath, "ca.pem"),
		CertFile: filepath.Join(dockerCertPath, "cert.pem"),
		KeyFile:  filepath.Join(dockerCertPath, "key.pem"),
	}
	tlsConfig, err := tlsconfig.Client(*option)
	if err != nil {
		return nil, err
	}

	return tlsConfig, nil
}

// DaemonHost return the daemon host string for this test execution
func DaemonHost() string {
	daemonURLStr := "unix://" + opts.DefaultUnixSocket
	if daemonHostVar := os.Getenv("DOCKER_HOST"); daemonHostVar != "" {
		daemonURLStr = daemonHostVar
	}
	return daemonURLStr
}
