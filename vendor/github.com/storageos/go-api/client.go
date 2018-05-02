package storageos

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/storageos/go-api/netutil"
	"github.com/storageos/go-api/serror"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

const (
	userAgent         = "go-storageosclient"
	DefaultVersionStr = "1"
	DefaultVersion    = 1
)

var (
	// ErrConnectionRefused is returned when the client cannot connect to the given endpoint.
	ErrConnectionRefused = errors.New("cannot connect to StorageOS API endpoint")

	// ErrInactivityTimeout is returned when a streamable call has been inactive for some time.
	ErrInactivityTimeout = errors.New("inactivity time exceeded timeout")

	// ErrInvalidVersion is returned when a versioned client was requested but no version specified.
	ErrInvalidVersion = errors.New("invalid version")

	// DefaultPort is the default API port
	DefaultPort = "5705"

	// DataplaneHealthPort is the the port used by the dataplane health-check service
	DataplaneHealthPort = "5704"

	// DefaultHost is the default API host
	DefaultHost = "tcp://localhost:" + DefaultPort
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
	username               string
	secret                 string
	requestedAPIVersion    APIVersion
	serverAPIVersion       APIVersion
	expectedAPIVersion     APIVersion
	nativeHTTPClient       *http.Client
	useTLS                 bool
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
func NewClient(nodes string) (*Client, error) {
	client, err := NewVersionedClient(nodes, "")
	if err != nil {
		return nil, err
	}
	client.SkipServerVersionCheck = true
	return client, nil
}

// NewVersionedClient returns a Client instance ready for communication with
// the given server endpoint, using a specific remote API version.
func NewVersionedClient(nodestring string, apiVersionString string) (*Client, error) {
	nodes := strings.Split(nodestring, ",")

	d, err := netutil.NewMultiDialer(nodes, nil)
	if err != nil {
		return nil, err
	}

	var useTLS bool
	if len(nodes) > 0 {
		if u, err := url.Parse(nodes[0]); err != nil && u.Scheme == "https" {
			useTLS = true
		}
	}

	c := &Client{
		HTTPClient: defaultClient(d),
		useTLS:     useTLS,
	}

	if apiVersionString != "" {
		version, err := strconv.Atoi(apiVersionString)
		if err != nil {
			return nil, err
		}
		c.requestedAPIVersion = APIVersion(version)
	}

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
	u := c.getAPIPath(urlpath, query, doOptions.unversioned)

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
		// If it is a custom error, return it. It probably knows more than us
		if serror.IsStorageOSError(err) {
			return nil, err
		}

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

func (c *Client) getAPIPath(path string, query url.Values, unversioned bool) string {
	// The custom dialer contacts the hosts for us, making this hosname irrelevant
	var urlStr string
	if c.useTLS {
		urlStr = "https://storageos-cluster"
	} else {
		urlStr = "http://storageos-cluster"
	}

	var apiPath string

	path = strings.TrimLeft(path, "/")
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
	type jsonError struct {
		Message string `json:"message"`
	}

	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return &Error{Status: resp.StatusCode, Message: fmt.Sprintf("cannot read body, err: %v", err)}
	}

	// attempt to unmarshal the error if in json format
	jerr := &jsonError{}
	err = json.Unmarshal(data, jerr)
	if err != nil {
		return &Error{Status: resp.StatusCode, Message: string(data)} // Failed, just return string
	}

	return &Error{Status: resp.StatusCode, Message: jerr.Message}
}

func (e *Error) Error() string {
	var niceStatus string

	switch e.Status {
	case 400, 500:
		niceStatus = "Server failed to process your request. Was the data correct?"
	case 401:
		niceStatus = "Unauthenticated access of secure endpoint, please retry after authentication"
	case 403:
		niceStatus = "Forbidden request. Your user cannot perform this action"
	case 404:
		niceStatus = "Requested object not found. Does this item exist?"
	}

	if niceStatus != "" {
		return fmt.Sprintf("API error (%s): %s", niceStatus, e.Message)
	}
	return fmt.Sprintf("API error (%s): %s", http.StatusText(e.Status), e.Message)
}

// defaultTransport returns a new http.Transport with the same default values
// as http.DefaultTransport, but with idle connections and keepalives disabled.
func defaultTransport(d Dialer) *http.Transport {
	transport := defaultPooledTransport(d)
	transport.DisableKeepAlives = true
	transport.MaxIdleConnsPerHost = -1
	return transport
}

// defaultPooledTransport returns a new http.Transport with similar default
// values to http.DefaultTransport. Do not use this for transient transports as
// it can leak file descriptors over time. Only use this for transports that
// will be re-used for the same host(s).
func defaultPooledTransport(d Dialer) *http.Transport {
	transport := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		Dial:                d.Dial,
		TLSHandshakeTimeout: 5 * time.Second,
		DisableKeepAlives:   false,
		MaxIdleConnsPerHost: 1,
	}
	return transport
}

// defaultClient returns a new http.Client with similar default values to
// http.Client, but with a non-shared Transport, idle connections disabled, and
// keepalives disabled.
// If a custom dialer is not provided, one with sane defaults will be created.
func defaultClient(d Dialer) *http.Client {
	if d == nil {
		d = &net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 5 * time.Second,
		}
	}

	return &http.Client{
		Transport: defaultTransport(d),
	}
}
