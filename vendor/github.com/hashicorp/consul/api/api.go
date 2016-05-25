package api

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/hashicorp/go-cleanhttp"
)

// QueryOptions are used to parameterize a query
type QueryOptions struct {
	// Providing a datacenter overwrites the DC provided
	// by the Config
	Datacenter string

	// AllowStale allows any Consul server (non-leader) to service
	// a read. This allows for lower latency and higher throughput
	AllowStale bool

	// RequireConsistent forces the read to be fully consistent.
	// This is more expensive but prevents ever performing a stale
	// read.
	RequireConsistent bool

	// WaitIndex is used to enable a blocking query. Waits
	// until the timeout or the next index is reached
	WaitIndex uint64

	// WaitTime is used to bound the duration of a wait.
	// Defaults to that of the Config, but can be overridden.
	WaitTime time.Duration

	// Token is used to provide a per-request ACL token
	// which overrides the agent's default token.
	Token string

	// Near is used to provide a node name that will sort the results
	// in ascending order based on the estimated round trip time from
	// that node. Setting this to "_agent" will use the agent's node
	// for the sort.
	Near string
}

// WriteOptions are used to parameterize a write
type WriteOptions struct {
	// Providing a datacenter overwrites the DC provided
	// by the Config
	Datacenter string

	// Token is used to provide a per-request ACL token
	// which overrides the agent's default token.
	Token string
}

// QueryMeta is used to return meta data about a query
type QueryMeta struct {
	// LastIndex. This can be used as a WaitIndex to perform
	// a blocking query
	LastIndex uint64

	// Time of last contact from the leader for the
	// server servicing the request
	LastContact time.Duration

	// Is there a known leader
	KnownLeader bool

	// How long did the request take
	RequestTime time.Duration
}

// WriteMeta is used to return meta data about a write
type WriteMeta struct {
	// How long did the request take
	RequestTime time.Duration
}

// HttpBasicAuth is used to authenticate http client with HTTP Basic Authentication
type HttpBasicAuth struct {
	// Username to use for HTTP Basic Authentication
	Username string

	// Password to use for HTTP Basic Authentication
	Password string
}

// Config is used to configure the creation of a client
type Config struct {
	// Address is the address of the Consul server
	Address string

	// Scheme is the URI scheme for the Consul server
	Scheme string

	// Datacenter to use. If not provided, the default agent datacenter is used.
	Datacenter string

	// HttpClient is the client to use. Default will be
	// used if not provided.
	HttpClient *http.Client

	// HttpAuth is the auth info to use for http access.
	HttpAuth *HttpBasicAuth

	// WaitTime limits how long a Watch will block. If not provided,
	// the agent default values will be used.
	WaitTime time.Duration

	// Token is used to provide a per-request ACL token
	// which overrides the agent's default token.
	Token string
}

// TLSConfig is used to generate a TLSClientConfig that's useful for talking to
// Consul using TLS.
type TLSConfig struct {
	// Address is the optional address of the Consul server. The port, if any
	// will be removed from here and this will be set to the ServerName of the
	// resulting config.
	Address string

	// CAFile is the optional path to the CA certificate used for Consul
	// communication, defaults to the system bundle if not specified.
	CAFile string

	// CertFile is the optional path to the certificate for Consul
	// communication. If this is set then you need to also set KeyFile.
	CertFile string

	// KeyFile is the optional path to the private key for Consul communication.
	// If this is set then you need to also set CertFile.
	KeyFile string

	// InsecureSkipVerify if set to true will disable TLS host verification.
	InsecureSkipVerify bool
}

// DefaultConfig returns a default configuration for the client. By default this
// will pool and reuse idle connections to Consul. If you have a long-lived
// client object, this is the desired behavior and should make the most efficient
// use of the connections to Consul. If you don't reuse a client object , which
// is not recommended, then you may notice idle connections building up over
// time. To avoid this, use the DefaultNonPooledConfig() instead.
func DefaultConfig() *Config {
	return defaultConfig(cleanhttp.DefaultPooledTransport)
}

// DefaultNonPooledConfig returns a default configuration for the client which
// does not pool connections. This isn't a recommended configuration because it
// will reconnect to Consul on every request, but this is useful to avoid the
// accumulation of idle connections if you make many client objects during the
// lifetime of your application.
func DefaultNonPooledConfig() *Config {
	return defaultConfig(cleanhttp.DefaultTransport)
}

// defaultConfig returns the default configuration for the client, using the
// given function to make the transport.
func defaultConfig(transportFn func() *http.Transport) *Config {
	config := &Config{
		Address: "127.0.0.1:8500",
		Scheme:  "http",
		HttpClient: &http.Client{
			Transport: transportFn(),
		},
	}

	if addr := os.Getenv("CONSUL_HTTP_ADDR"); addr != "" {
		config.Address = addr
	}

	if token := os.Getenv("CONSUL_HTTP_TOKEN"); token != "" {
		config.Token = token
	}

	if auth := os.Getenv("CONSUL_HTTP_AUTH"); auth != "" {
		var username, password string
		if strings.Contains(auth, ":") {
			split := strings.SplitN(auth, ":", 2)
			username = split[0]
			password = split[1]
		} else {
			username = auth
		}

		config.HttpAuth = &HttpBasicAuth{
			Username: username,
			Password: password,
		}
	}

	if ssl := os.Getenv("CONSUL_HTTP_SSL"); ssl != "" {
		enabled, err := strconv.ParseBool(ssl)
		if err != nil {
			log.Printf("[WARN] client: could not parse CONSUL_HTTP_SSL: %s", err)
		}

		if enabled {
			config.Scheme = "https"
		}
	}

	if verify := os.Getenv("CONSUL_HTTP_SSL_VERIFY"); verify != "" {
		doVerify, err := strconv.ParseBool(verify)
		if err != nil {
			log.Printf("[WARN] client: could not parse CONSUL_HTTP_SSL_VERIFY: %s", err)
		}

		if !doVerify {
			tlsClientConfig, err := SetupTLSConfig(&TLSConfig{
				InsecureSkipVerify: true,
			})

			// We don't expect this to fail given that we aren't
			// parsing any of the input, but we panic just in case
			// since this doesn't have an error return.
			if err != nil {
				panic(err)
			}

			transport := transportFn()
			transport.TLSClientConfig = tlsClientConfig
			config.HttpClient.Transport = transport
		}
	}

	return config
}

// TLSConfig is used to generate a TLSClientConfig that's useful for talking to
// Consul using TLS.
func SetupTLSConfig(tlsConfig *TLSConfig) (*tls.Config, error) {
	tlsClientConfig := &tls.Config{
		InsecureSkipVerify: tlsConfig.InsecureSkipVerify,
	}

	if tlsConfig.Address != "" {
		server := tlsConfig.Address
		hasPort := strings.LastIndex(server, ":") > strings.LastIndex(server, "]")
		if hasPort {
			var err error
			server, _, err = net.SplitHostPort(server)
			if err != nil {
				return nil, err
			}
		}
		tlsClientConfig.ServerName = server
	}

	if tlsConfig.CertFile != "" && tlsConfig.KeyFile != "" {
		tlsCert, err := tls.LoadX509KeyPair(tlsConfig.CertFile, tlsConfig.KeyFile)
		if err != nil {
			return nil, err
		}
		tlsClientConfig.Certificates = []tls.Certificate{tlsCert}
	}

	if tlsConfig.CAFile != "" {
		data, err := ioutil.ReadFile(tlsConfig.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA file: %v", err)
		}

		caPool := x509.NewCertPool()
		if !caPool.AppendCertsFromPEM(data) {
			return nil, fmt.Errorf("failed to parse CA certificate")
		}
		tlsClientConfig.RootCAs = caPool
	}

	return tlsClientConfig, nil
}

// Client provides a client to the Consul API
type Client struct {
	config Config
}

// NewClient returns a new client
func NewClient(config *Config) (*Client, error) {
	// bootstrap the config
	defConfig := DefaultConfig()

	if len(config.Address) == 0 {
		config.Address = defConfig.Address
	}

	if len(config.Scheme) == 0 {
		config.Scheme = defConfig.Scheme
	}

	if config.HttpClient == nil {
		config.HttpClient = defConfig.HttpClient
	}

	if parts := strings.SplitN(config.Address, "unix://", 2); len(parts) == 2 {
		trans := cleanhttp.DefaultTransport()
		trans.Dial = func(_, _ string) (net.Conn, error) {
			return net.Dial("unix", parts[1])
		}
		config.HttpClient = &http.Client{
			Transport: trans,
		}
		config.Address = parts[1]
	}

	client := &Client{
		config: *config,
	}
	return client, nil
}

// request is used to help build up a request
type request struct {
	config *Config
	method string
	url    *url.URL
	params url.Values
	body   io.Reader
	obj    interface{}
}

// setQueryOptions is used to annotate the request with
// additional query options
func (r *request) setQueryOptions(q *QueryOptions) {
	if q == nil {
		return
	}
	if q.Datacenter != "" {
		r.params.Set("dc", q.Datacenter)
	}
	if q.AllowStale {
		r.params.Set("stale", "")
	}
	if q.RequireConsistent {
		r.params.Set("consistent", "")
	}
	if q.WaitIndex != 0 {
		r.params.Set("index", strconv.FormatUint(q.WaitIndex, 10))
	}
	if q.WaitTime != 0 {
		r.params.Set("wait", durToMsec(q.WaitTime))
	}
	if q.Token != "" {
		r.params.Set("token", q.Token)
	}
	if q.Near != "" {
		r.params.Set("near", q.Near)
	}
}

// durToMsec converts a duration to a millisecond specified string. If the
// user selected a positive value that rounds to 0 ms, then we will use 1 ms
// so they get a short delay, otherwise Consul will translate the 0 ms into
// a huge default delay.
func durToMsec(dur time.Duration) string {
	ms := dur / time.Millisecond
	if dur > 0 && ms == 0 {
		ms = 1
	}
	return fmt.Sprintf("%dms", ms)
}

// serverError is a string we look for to detect 500 errors.
const serverError = "Unexpected response code: 500"

// IsServerError returns true for 500 errors from the Consul servers, these are
// usually retryable at a later time.
func IsServerError(err error) bool {
	if err == nil {
		return false
	}

	// TODO (slackpad) - Make a real error type here instead of using
	// a string check.
	return strings.Contains(err.Error(), serverError)
}

// setWriteOptions is used to annotate the request with
// additional write options
func (r *request) setWriteOptions(q *WriteOptions) {
	if q == nil {
		return
	}
	if q.Datacenter != "" {
		r.params.Set("dc", q.Datacenter)
	}
	if q.Token != "" {
		r.params.Set("token", q.Token)
	}
}

// toHTTP converts the request to an HTTP request
func (r *request) toHTTP() (*http.Request, error) {
	// Encode the query parameters
	r.url.RawQuery = r.params.Encode()

	// Check if we should encode the body
	if r.body == nil && r.obj != nil {
		if b, err := encodeBody(r.obj); err != nil {
			return nil, err
		} else {
			r.body = b
		}
	}

	// Create the HTTP request
	req, err := http.NewRequest(r.method, r.url.RequestURI(), r.body)
	if err != nil {
		return nil, err
	}

	req.URL.Host = r.url.Host
	req.URL.Scheme = r.url.Scheme
	req.Host = r.url.Host

	// Setup auth
	if r.config.HttpAuth != nil {
		req.SetBasicAuth(r.config.HttpAuth.Username, r.config.HttpAuth.Password)
	}

	return req, nil
}

// newRequest is used to create a new request
func (c *Client) newRequest(method, path string) *request {
	r := &request{
		config: &c.config,
		method: method,
		url: &url.URL{
			Scheme: c.config.Scheme,
			Host:   c.config.Address,
			Path:   path,
		},
		params: make(map[string][]string),
	}
	if c.config.Datacenter != "" {
		r.params.Set("dc", c.config.Datacenter)
	}
	if c.config.WaitTime != 0 {
		r.params.Set("wait", durToMsec(r.config.WaitTime))
	}
	if c.config.Token != "" {
		r.params.Set("token", r.config.Token)
	}
	return r
}

// doRequest runs a request with our client
func (c *Client) doRequest(r *request) (time.Duration, *http.Response, error) {
	req, err := r.toHTTP()
	if err != nil {
		return 0, nil, err
	}
	start := time.Now()
	resp, err := c.config.HttpClient.Do(req)
	diff := time.Now().Sub(start)
	return diff, resp, err
}

// Query is used to do a GET request against an endpoint
// and deserialize the response into an interface using
// standard Consul conventions.
func (c *Client) query(endpoint string, out interface{}, q *QueryOptions) (*QueryMeta, error) {
	r := c.newRequest("GET", endpoint)
	r.setQueryOptions(q)
	rtt, resp, err := requireOK(c.doRequest(r))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	if err := decodeBody(resp, out); err != nil {
		return nil, err
	}
	return qm, nil
}

// write is used to do a PUT request against an endpoint
// and serialize/deserialized using the standard Consul conventions.
func (c *Client) write(endpoint string, in, out interface{}, q *WriteOptions) (*WriteMeta, error) {
	r := c.newRequest("PUT", endpoint)
	r.setWriteOptions(q)
	r.obj = in
	rtt, resp, err := requireOK(c.doRequest(r))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	wm := &WriteMeta{RequestTime: rtt}
	if out != nil {
		if err := decodeBody(resp, &out); err != nil {
			return nil, err
		}
	}
	return wm, nil
}

// parseQueryMeta is used to help parse query meta-data
func parseQueryMeta(resp *http.Response, q *QueryMeta) error {
	header := resp.Header

	// Parse the X-Consul-Index
	index, err := strconv.ParseUint(header.Get("X-Consul-Index"), 10, 64)
	if err != nil {
		return fmt.Errorf("Failed to parse X-Consul-Index: %v", err)
	}
	q.LastIndex = index

	// Parse the X-Consul-LastContact
	last, err := strconv.ParseUint(header.Get("X-Consul-LastContact"), 10, 64)
	if err != nil {
		return fmt.Errorf("Failed to parse X-Consul-LastContact: %v", err)
	}
	q.LastContact = time.Duration(last) * time.Millisecond

	// Parse the X-Consul-KnownLeader
	switch header.Get("X-Consul-KnownLeader") {
	case "true":
		q.KnownLeader = true
	default:
		q.KnownLeader = false
	}
	return nil
}

// decodeBody is used to JSON decode a body
func decodeBody(resp *http.Response, out interface{}) error {
	dec := json.NewDecoder(resp.Body)
	return dec.Decode(out)
}

// encodeBody is used to encode a request body
func encodeBody(obj interface{}) (io.Reader, error) {
	buf := bytes.NewBuffer(nil)
	enc := json.NewEncoder(buf)
	if err := enc.Encode(obj); err != nil {
		return nil, err
	}
	return buf, nil
}

// requireOK is used to wrap doRequest and check for a 200
func requireOK(d time.Duration, resp *http.Response, e error) (time.Duration, *http.Response, error) {
	if e != nil {
		if resp != nil {
			resp.Body.Close()
		}
		return d, nil, e
	}
	if resp.StatusCode != 200 {
		var buf bytes.Buffer
		io.Copy(&buf, resp.Body)
		resp.Body.Close()
		return d, nil, fmt.Errorf("Unexpected response code: %d (%s)", resp.StatusCode, buf.Bytes())
	}
	return d, resp, nil
}
