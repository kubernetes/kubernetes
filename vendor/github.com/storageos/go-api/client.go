package storageos

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/storageos/go-api/netutil"
	"github.com/storageos/go-api/serror"
)

const (
	// DefaultUserAgent is the default User-Agent header to include in HTTP requests.
	DefaultUserAgent = "go-storageosclient"
	// DefaultVersionStr is the string value of the default API version.
	DefaultVersionStr = "1"
	// DefaultVersion is the default API version.
	DefaultVersion = 1
)

var (
	// ErrConnectionRefused is returned when the client cannot connect to the given endpoint.
	ErrConnectionRefused = errors.New("cannot connect to StorageOS API endpoint")

	// ErrInactivityTimeout is returned when a streamable call has been inactive for some time.
	ErrInactivityTimeout = errors.New("inactivity time exceeded timeout")

	// ErrInvalidVersion is returned when a versioned client was requested but no version specified.
	ErrInvalidVersion = errors.New("invalid version")

	// ErrProxyNotSupported is returned when a client is unable to set a proxy for http requests.
	ErrProxyNotSupported = errors.New("client does not support http proxy")

	// ErrDialerNotSupported is returned when a client is unable to set a DialContext for http requests.
	ErrDialerNotSupported = errors.New("client does not support setting DialContext")

	// DefaultPort is the default API port.
	DefaultPort = "5705"

	// DataplaneHealthPort is the the port used by the dataplane health-check service.
	DataplaneHealthPort = "5704"

	// DefaultHost is the default API host.
	DefaultHost = "http://localhost:" + DefaultPort
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
	httpClient *http.Client

	addresses []string
	username  string
	secret    string
	userAgent string

	configLock  *sync.RWMutex // Lock for config changes
	addressLock *sync.Mutex   // Lock used to copy/update the address slice

	requestedAPIVersion APIVersion
	serverAPIVersion    APIVersion
	expectedAPIVersion  APIVersion

	SkipServerVersionCheck bool
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

type dialContext = func(ctx context.Context, network, address string) (net.Conn, error)

// NewClient returns a Client instance ready for communication with the given
// server endpoint. It will use the latest remote API version available in the
// server.
func NewClient(nodes string) (*Client, error) {
	client, err := NewVersionedClient(nodes, "")
	if err != nil {
		return nil, err
	}
	client.SkipServerVersionCheck = true
	client.userAgent = DefaultUserAgent
	return client, nil
}

// NewVersionedClient returns a Client instance ready for communication with
// the given server endpoint, using a specific remote API version.
func NewVersionedClient(nodestring string, apiVersionString string) (*Client, error) {
	nodes := strings.Split(nodestring, ",")
	addresses, err := netutil.AddressesFromNodes(nodes)
	if err != nil {
		return nil, err
	}

	if len(addresses) > 1 {
		// Shuffle returned addresses in attempt to spread the load
		rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
		rnd.Shuffle(len(addresses), func(i, j int) {
			addresses[i], addresses[j] = addresses[j], addresses[i]
		})
	}

	client := &Client{
		httpClient:  defaultClient(),
		addresses:   addresses,
		configLock:  &sync.RWMutex{},
		addressLock: &sync.Mutex{},
	}

	if apiVersionString != "" {
		version, err := strconv.Atoi(apiVersionString)
		if err != nil {
			return nil, err
		}
		client.requestedAPIVersion = APIVersion(version)
	}

	return client, nil
}

// SetUserAgent sets the client useragent.
func (c *Client) SetUserAgent(useragent string) {
	c.configLock.Lock()
	defer c.configLock.Unlock()
	c.userAgent = useragent
}

// SetAuth sets the API username and secret to be used for all API requests.
// It should not be called concurrently with any other Client methods.
func (c *Client) SetAuth(username string, secret string) {
	c.configLock.Lock()
	defer c.configLock.Unlock()
	if username != "" {
		c.username = username
	}
	if secret != "" {
		c.secret = secret
	}
}

// SetProxy will set the proxy URL for both the HTTPClient.
// If the transport method does not support usage
// of proxies, an error will be returned.
func (c *Client) SetProxy(proxy *url.URL) error {
	c.configLock.Lock()
	defer c.configLock.Unlock()

	if client := c.httpClient; client != nil {
		transport, supported := client.Transport.(*http.Transport)
		if !supported {
			return ErrProxyNotSupported
		}
		transport.Proxy = http.ProxyURL(proxy)
	}
	return nil
}

// SetTimeout takes a timeout and applies it to both the HTTPClient and
// nativeHTTPClient. It should not be called concurrently with any other Client
// methods.
func (c *Client) SetTimeout(t time.Duration) {
	c.configLock.Lock()
	defer c.configLock.Unlock()
	if c.httpClient != nil {
		c.httpClient.Timeout = t
	}
}

// GetDialContext returns the current DialContext function, or nil if there is none.
func (c *Client) GetDialContext() dialContext {
	c.configLock.RLock()
	defer c.configLock.RUnlock()

	if c.httpClient == nil {
		return nil
	}
	transport, supported := c.httpClient.Transport.(*http.Transport)
	if !supported {
		return nil
	}
	return transport.DialContext
}

// SetDialContext uses the given dial function to establish TCP connections in the HTTPClient.
func (c *Client) SetDialContext(dial dialContext) error {
	c.configLock.Lock()
	defer c.configLock.Unlock()

	if client := c.httpClient; client != nil {
		transport, supported := client.Transport.(*http.Transport)
		if !supported {
			return ErrDialerNotSupported
		}
		transport.DialContext = dial
	}
	return nil
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
	c.configLock.Lock()
	defer c.configLock.Unlock()
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
	return resp.Body.Close()
}

func (c *Client) getServerAPIVersionString() (version string, err error) {
	v, err := c.ServerVersion(context.Background())
	if err != nil {
		return "", err
	}
	return v.APIVersion, nil
}

type doOptions struct {
	context context.Context
	data    interface{}

	values  url.Values
	headers map[string]string

	fieldSelector string
	labelSelector string
	namespace     string

	forceJSON   bool
	force       bool
	unversioned bool

	retryOn []int // http.status codes
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

	// Obtain a reader lock to prevent the http client from being
	// modified underneath us during a do().
	c.configLock.RLock()
	defer c.configLock.RUnlock() // This defer matches both the initial and the above lock

	httpClient := c.httpClient
	endpoint := c.getAPIPath(urlpath, query, doOptions.unversioned)

	// The doOptions Context is shared for every attempted request in the do.
	ctx := doOptions.context
	if ctx == nil {
		ctx = context.Background()
	}

	var failedAddresses = map[string]struct{}{}

	c.addressLock.Lock()
	var addresses = make([]string, len(c.addresses))
	copy(addresses, c.addresses)
	c.addressLock.Unlock()

	for _, address := range addresses {
		target := address + endpoint

		req, err := http.NewRequest(method, target, params)
		if err != nil {
			// Probably should not try and continue if we're unable
			// to create the request.
			return nil, err
		}
		req.Header.Set("User-Agent", c.userAgent)
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

		resp, err := httpClient.Do(req.WithContext(ctx))
		if err != nil {

			// If it is a custom error, return it. It probably knows more than us
			if serror.IsStorageOSError(err) {
				switch serror.ErrorKind(err) {
				case serror.APIUncontactable:
					// If API isn't contactable we should try the next address
					failedAddresses[address] = struct{}{}
					continue
				case serror.InvalidHostConfig:
					// If invalid host or unknown error, we should report back
					fallthrough
				case serror.UnknownError:
					return nil, err
				}
			}

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
				if _, ok := err.(net.Error); ok {
					// Be optimistic and try the next endpoint
					failedAddresses[address] = struct{}{}
					continue
				}
				return nil, err
			}
		}

		var shouldretry bool
		if doOptions.retryOn != nil {
			for _, code := range doOptions.retryOn {
				if resp.StatusCode == code {
					failedAddresses[address] = struct{}{}
					shouldretry = true
				}

			}
		}

		// If we get to the point of response, we should move any failed
		// addresses to the back.
		failed := len(failedAddresses)
		if failed > 0 {
			// Copy addresses we think are okay into the head of the list
			newOrder := make([]string, 0, len(addresses)-failed)

			for _, addr := range addresses {
				if _, exists := failedAddresses[addr]; !exists {
					newOrder = append(newOrder, addr)
				}
			}
			for addr := range failedAddresses {
				newOrder = append(newOrder, addr)
			}

			c.addressLock.Lock()
			// Bring in the new order
			c.addresses = newOrder
			c.addressLock.Unlock()
		}

		if shouldretry {
			continue
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 400 {
			return nil, newError(resp) // These status codes are likely to be fatal
		}
		return resp, nil
	}

	return nil, netutil.ErrAllFailed(addresses)
}

func (c *Client) getAPIPath(path string, query url.Values, unversioned bool) string {
	var apiPath = strings.TrimLeft(path, "/")

	if !unversioned {
		apiPath = fmt.Sprintf("/%s/%s", c.requestedAPIVersion, apiPath)
	} else {
		apiPath = fmt.Sprintf("/%s", apiPath)
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

// defaultPooledTransport returns a new http.Transport with similar default
// values to http.DefaultTransport. Do not use this for transient transports as
// it can leak file descriptors over time. Only use this for transports that
// will be re-used for the same host(s).
func defaultPooledTransport(dialer Dialer) *http.Transport {
	transport := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		Dial:                dialer.Dial,
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
func defaultClient() *http.Client {
	dialer := &net.Dialer{
		Timeout:   5 * time.Second,
		KeepAlive: 5 * time.Second,
	}

	return &http.Client{
		Transport: defaultPooledTransport(dialer),
	}
}
