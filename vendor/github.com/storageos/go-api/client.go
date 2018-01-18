package storageos

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"
)

const (
	userAgent         = "go-storageosclient"
	unixProtocol      = "unix"
	namedPipeProtocol = "npipe"
	DefaultVersionStr = "1"
	DefaultVersion    = 1
	defaultNamespace  = "default"
)

var (
	// ErrInvalidEndpoint is returned when the endpoint is not a valid HTTP URL.
	ErrInvalidEndpoint = errors.New("invalid endpoint")

	// ErrConnectionRefused is returned when the client cannot connect to the given endpoint.
	ErrConnectionRefused = errors.New("cannot connect to StorageOS API endpoint")

	// ErrInactivityTimeout is returned when a streamable call has been inactive for some time.
	ErrInactivityTimeout = errors.New("inactivity time exceeded timeout")

	// ErrInvalidVersion is returned when a versioned client was requested but no version specified.
	ErrInvalidVersion = errors.New("invalid version")

	// DefaultHost is the default API host
	DefaultHost = "tcp://localhost:5705"
)

// APIVersion is an internal representation of a version of the Remote API.
type APIVersion int

// NewAPIVersion returns an instance of APIVersion for the given string.
//
// The given string must be in the form <major>
func NewAPIVersion(input string) (APIVersion, error) {
	if input == "" {
		return DefaultVersion, ErrInvalidVersion
	}
	version, err := strconv.Atoi(input)
	if err != nil {
		return 0, fmt.Errorf("Unable to parse version %q", input)
	}
	return APIVersion(version), nil
}

func (version APIVersion) String() string {
	return fmt.Sprintf("v%d", version)
}

// Client is the basic type of this package. It provides methods for
// interaction with the API.
type Client struct {
	SkipServerVersionCheck bool
	HTTPClient             *http.Client
	TLSConfig              *tls.Config
	Dialer                 Dialer
	endpoint               string
	endpointURL            *url.URL
	username               string
	secret                 string
	requestedAPIVersion    APIVersion
	serverAPIVersion       APIVersion
	expectedAPIVersion     APIVersion
	nativeHTTPClient       *http.Client
}

// ClientVersion returns the API version of the client
func (c *Client) ClientVersion() string {
	return DefaultVersionStr
}

// Dialer is an interface that allows network connections to be dialed
// (net.Dialer fulfills this interface) and named pipes (a shim using
// winio.DialPipe)
type Dialer interface {
	Dial(network, address string) (net.Conn, error)
}

// NewClient returns a Client instance ready for communication with the given
// server endpoint. It will use the latest remote API version available in the
// server.
func NewClient(endpoint string) (*Client, error) {
	client, err := NewVersionedClient(endpoint, "")
	if err != nil {
		return nil, err
	}
	client.SkipServerVersionCheck = true
	return client, nil
}

// NewTLSClient returns a Client instance ready for TLS communications with the given
// server endpoint, key and certificates . It will use the latest remote API version
// available in the server.
func NewTLSClient(endpoint string, cert, key, ca string) (*Client, error) {
	client, err := NewVersionedTLSClient(endpoint, cert, key, ca, "")
	if err != nil {
		return nil, err
	}
	client.SkipServerVersionCheck = true
	return client, nil
}

// NewVersionedClient returns a Client instance ready for communication with
// the given server endpoint, using a specific remote API version.
func NewVersionedClient(endpoint string, apiVersionString string) (*Client, error) {
	u, err := parseEndpoint(endpoint, false)
	if err != nil {
		return nil, err
	}

	c := &Client{
		HTTPClient:  defaultClient(),
		Dialer:      &net.Dialer{},
		endpoint:    endpoint,
		endpointURL: u,
	}

	if apiVersionString != "" {
		version, err := strconv.Atoi(apiVersionString)
		if err != nil {
			return nil, err
		}
		c.requestedAPIVersion = APIVersion(version)
	}

	c.initializeNativeClient()
	return c, nil
}

// NewVersionedTLSClient returns a Client instance ready for TLS communications with the givens
// server endpoint, key and certificates, using a specific remote API version.
func NewVersionedTLSClient(endpoint string, cert, key, ca, apiVersionString string) (*Client, error) {
	var certPEMBlock []byte
	var keyPEMBlock []byte
	var caPEMCert []byte
	if _, err := os.Stat(cert); !os.IsNotExist(err) {
		certPEMBlock, err = ioutil.ReadFile(cert)
		if err != nil {
			return nil, err
		}
	}
	if _, err := os.Stat(key); !os.IsNotExist(err) {
		keyPEMBlock, err = ioutil.ReadFile(key)
		if err != nil {
			return nil, err
		}
	}
	if _, err := os.Stat(ca); !os.IsNotExist(err) {
		caPEMCert, err = ioutil.ReadFile(ca)
		if err != nil {
			return nil, err
		}
	}
	return NewVersionedTLSClientFromBytes(endpoint, certPEMBlock, keyPEMBlock, caPEMCert, apiVersionString)
}

// NewVersionedTLSClientFromBytes returns a Client instance ready for TLS communications with the givens
// server endpoint, key and certificates (passed inline to the function as opposed to being
// read from a local file), using a specific remote API version.
func NewVersionedTLSClientFromBytes(endpoint string, certPEMBlock, keyPEMBlock, caPEMCert []byte, apiVersionString string) (*Client, error) {
	u, err := parseEndpoint(endpoint, true)
	if err != nil {
		return nil, err
	}

	tlsConfig := &tls.Config{}
	if certPEMBlock != nil && keyPEMBlock != nil {
		tlsCert, err := tls.X509KeyPair(certPEMBlock, keyPEMBlock)
		if err != nil {
			return nil, err
		}
		tlsConfig.Certificates = []tls.Certificate{tlsCert}
	}
	if caPEMCert == nil {
		tlsConfig.InsecureSkipVerify = true
	} else {
		caPool := x509.NewCertPool()
		if !caPool.AppendCertsFromPEM(caPEMCert) {
			return nil, errors.New("Could not add RootCA pem")
		}
		tlsConfig.RootCAs = caPool
	}
	tr := defaultTransport()
	tr.TLSClientConfig = tlsConfig
	if err != nil {
		return nil, err
	}
	c := &Client{
		HTTPClient:  &http.Client{Transport: tr},
		TLSConfig:   tlsConfig,
		Dialer:      &net.Dialer{},
		endpoint:    endpoint,
		endpointURL: u,
	}

	if apiVersionString != "" {
		version, err := strconv.Atoi(apiVersionString)
		if err != nil {
			return nil, err
		}
		c.requestedAPIVersion = APIVersion(version)
	}

	c.initializeNativeClient()
	return c, nil
}

// SetAuth sets the API username and secret to be used for all API requests.
// It should not be called concurrently with any other Client methods.
func (c *Client) SetAuth(username string, secret string) {
	if username != "" {
		c.username = username
	}
	if secret != "" {
		c.secret = secret
	}
}

// SetTimeout takes a timeout and applies it to both the HTTPClient and
// nativeHTTPClient. It should not be called concurrently with any other Client
// methods.
func (c *Client) SetTimeout(t time.Duration) {
	if c.HTTPClient != nil {
		c.HTTPClient.Timeout = t
	}
	if c.nativeHTTPClient != nil {
		c.nativeHTTPClient.Timeout = t
	}
}

func (c *Client) checkAPIVersion() error {
	serverAPIVersionString, err := c.getServerAPIVersionString()
	if err != nil {
		return err
	}
	c.serverAPIVersion, err = NewAPIVersion(serverAPIVersionString)
	if err != nil {
		return err
	}
	if c.requestedAPIVersion == 0 {
		c.expectedAPIVersion = c.serverAPIVersion
	} else {
		c.expectedAPIVersion = c.requestedAPIVersion
	}
	return nil
}

// Endpoint returns the current endpoint. It's useful for getting the endpoint
// when using functions that get this data from the environment (like
// NewClientFromEnv.
func (c *Client) Endpoint() string {
	return c.endpoint
}

// Ping pings the API server
//
// See https://goo.gl/wYfgY1 for more details.
func (c *Client) Ping() error {
	urlpath := "/_ping"
	resp, err := c.do("GET", urlpath, doOptions{})
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return newError(resp)
	}
	resp.Body.Close()
	return nil
}

func (c *Client) getServerAPIVersionString() (version string, err error) {
	v, err := c.ServerVersion(context.Background())
	if err != nil {
		return "", err
	}
	return v.APIVersion, nil
}

type doOptions struct {
	data          interface{}
	fieldSelector string
	labelSelector string
	namespace     string
	forceJSON     bool
	force         bool
	values        url.Values
	headers       map[string]string
	unversioned   bool
	context       context.Context
}

func (c *Client) do(method, urlpath string, doOptions doOptions) (*http.Response, error) {
	var params io.Reader
	if doOptions.data != nil || doOptions.forceJSON {
		buf, err := json.Marshal(doOptions.data)
		if err != nil {
			return nil, err
		}
		params = bytes.NewBuffer(buf)
	}

	// Prefix the path with the namespace if given.  The caller should only set
	// the namespace if this is desired.
	if doOptions.namespace != "" {
		urlpath = "/" + NamespaceAPIPrefix + "/" + doOptions.namespace + "/" + urlpath
	}

	if !c.SkipServerVersionCheck && !doOptions.unversioned {
		err := c.checkAPIVersion()
		if err != nil {
			return nil, err
		}
	}

	query := url.Values{}
	if doOptions.values != nil {
		query = doOptions.values
	}
	if doOptions.force {
		query.Add("force", "1")
	}

	httpClient := c.HTTPClient
	protocol := c.endpointURL.Scheme
	var u string
	switch protocol {
	case unixProtocol, namedPipeProtocol:
		httpClient = c.nativeHTTPClient
		u = c.getFakeNativeURL(urlpath, doOptions.unversioned)
	default:
		u = c.getAPIPath(urlpath, query, doOptions.unversioned)
	}

	req, err := http.NewRequest(method, u, params)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if doOptions.data != nil {
		req.Header.Set("Content-Type", "application/json")
	} else if method == "POST" {
		req.Header.Set("Content-Type", "plain/text")
	}
	if c.username != "" && c.secret != "" {
		req.SetBasicAuth(c.username, c.secret)
	}

	for k, v := range doOptions.headers {
		req.Header.Set(k, v)
	}

	ctx := doOptions.context
	if ctx == nil {
		ctx = context.Background()
	}

	resp, err := httpClient.Do(req.WithContext(ctx))
	if err != nil {
		if strings.Contains(err.Error(), "connection refused") {
			return nil, ErrConnectionRefused
		}
		return nil, chooseError(ctx, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return nil, newError(resp)
	}
	return resp, nil
}

// if error in context, return that instead of generic http error
func chooseError(ctx context.Context, err error) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return err
	}
}

func (c *Client) getURL(path string, unversioned bool) string {

	urlStr := strings.TrimRight(c.endpointURL.String(), "/")
	path = strings.TrimLeft(path, "/")
	if c.endpointURL.Scheme == unixProtocol || c.endpointURL.Scheme == namedPipeProtocol {
		urlStr = ""
	}
	if unversioned {
		return fmt.Sprintf("%s/%s", urlStr, path)
	}
	return fmt.Sprintf("%s/%s/%s", urlStr, c.requestedAPIVersion, path)

}

func (c *Client) getAPIPath(path string, query url.Values, unversioned bool) string {
	var apiPath string
	urlStr := strings.TrimRight(c.endpointURL.String(), "/")
	path = strings.TrimLeft(path, "/")
	if c.endpointURL.Scheme == unixProtocol || c.endpointURL.Scheme == namedPipeProtocol {
		urlStr = ""
	}
	if unversioned {
		apiPath = fmt.Sprintf("%s/%s", urlStr, path)
	} else {
		apiPath = fmt.Sprintf("%s/%s/%s", urlStr, c.requestedAPIVersion, path)
	}

	if len(query) > 0 {
		apiPath = apiPath + "?" + query.Encode()
	}

	return apiPath
}

// getFakeNativeURL returns the URL needed to make an HTTP request over a UNIX
// domain socket to the given path.
func (c *Client) getFakeNativeURL(path string, unversioned bool) string {
	u := *c.endpointURL // Copy.

	// Override URL so that net/http will not complain.
	u.Scheme = "http"
	u.Host = "unix.sock" // Doesn't matter what this is - it's not used.
	u.Path = ""
	urlStr := strings.TrimRight(u.String(), "/")
	path = strings.TrimLeft(path, "/")
	if unversioned {
		return fmt.Sprintf("%s/%s", urlStr, path)
	}
	return fmt.Sprintf("%s/%s/%s", urlStr, c.requestedAPIVersion, path)
}

type jsonMessage struct {
	Status   string `json:"status,omitempty"`
	Progress string `json:"progress,omitempty"`
	Error    string `json:"error,omitempty"`
	Stream   string `json:"stream,omitempty"`
}

func queryString(opts interface{}) string {
	if opts == nil {
		return ""
	}
	value := reflect.ValueOf(opts)
	if value.Kind() == reflect.Ptr {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return ""
	}
	items := url.Values(map[string][]string{})
	for i := 0; i < value.NumField(); i++ {
		field := value.Type().Field(i)
		if field.PkgPath != "" {
			continue
		}
		key := field.Tag.Get("qs")
		if key == "" {
			key = strings.ToLower(field.Name)
		} else if key == "-" {
			continue
		}
		addQueryStringValue(items, key, value.Field(i))
	}
	return items.Encode()
}

func addQueryStringValue(items url.Values, key string, v reflect.Value) {
	switch v.Kind() {
	case reflect.Bool:
		if v.Bool() {
			items.Add(key, "1")
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if v.Int() > 0 {
			items.Add(key, strconv.FormatInt(v.Int(), 10))
		}
	case reflect.Float32, reflect.Float64:
		if v.Float() > 0 {
			items.Add(key, strconv.FormatFloat(v.Float(), 'f', -1, 64))
		}
	case reflect.String:
		if v.String() != "" {
			items.Add(key, v.String())
		}
	case reflect.Ptr:
		if !v.IsNil() {
			if b, err := json.Marshal(v.Interface()); err == nil {
				items.Add(key, string(b))
			}
		}
	case reflect.Map:
		if len(v.MapKeys()) > 0 {
			if b, err := json.Marshal(v.Interface()); err == nil {
				items.Add(key, string(b))
			}
		}
	case reflect.Array, reflect.Slice:
		vLen := v.Len()
		if vLen > 0 {
			for i := 0; i < vLen; i++ {
				addQueryStringValue(items, key, v.Index(i))
			}
		}
	}
}

// Error represents failures in the API. It represents a failure from the API.
type Error struct {
	Status  int
	Message string
}

func newError(resp *http.Response) *Error {
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return &Error{Status: resp.StatusCode, Message: fmt.Sprintf("cannot read body, err: %v", err)}
	}
	return &Error{Status: resp.StatusCode, Message: string(data)}
}

func (e *Error) Error() string {
	return fmt.Sprintf("API error (%d): %s", e.Status, e.Message)
}

func parseEndpoint(endpoint string, tls bool) (*url.URL, error) {
	if endpoint != "" && !strings.Contains(endpoint, "://") {
		endpoint = "tcp://" + endpoint
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, ErrInvalidEndpoint
	}
	if tls && u.Scheme != "unix" {
		u.Scheme = "https"
	}
	switch u.Scheme {
	case unixProtocol, namedPipeProtocol:
		return u, nil
	case "http", "https", "tcp":
		_, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			if e, ok := err.(*net.AddrError); ok {
				if e.Err == "missing port in address" {
					return u, nil
				}
			}
			return nil, ErrInvalidEndpoint
		}
		number, err := strconv.ParseInt(port, 10, 64)
		if err == nil && number > 0 && number < 65536 {
			if u.Scheme == "tcp" {
				if tls {
					u.Scheme = "https"
				} else {
					u.Scheme = "http"
				}
			}
			return u, nil
		}
		return nil, ErrInvalidEndpoint
	default:
		return nil, ErrInvalidEndpoint
	}
}

// defaultTransport returns a new http.Transport with the same default values
// as http.DefaultTransport, but with idle connections and keepalives disabled.
func defaultTransport() *http.Transport {
	transport := defaultPooledTransport()
	transport.DisableKeepAlives = true
	transport.MaxIdleConnsPerHost = -1
	return transport
}

// defaultPooledTransport returns a new http.Transport with similar default
// values to http.DefaultTransport. Do not use this for transient transports as
// it can leak file descriptors over time. Only use this for transports that
// will be re-used for the same host(s).
func defaultPooledTransport() *http.Transport {
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		DisableKeepAlives:   false,
		MaxIdleConnsPerHost: 1,
	}
	return transport
}

// defaultClient returns a new http.Client with similar default values to
// http.Client, but with a non-shared Transport, idle connections disabled, and
// keepalives disabled.
func defaultClient() *http.Client {
	return &http.Client{
		Transport: defaultTransport(),
	}
}
