package api

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http2"

	"github.com/hashicorp/go-cleanhttp"
	"github.com/hashicorp/go-rootcerts"
	"github.com/hashicorp/vault/helper/parseutil"
	"github.com/sethgrid/pester"
)

const EnvVaultAddress = "VAULT_ADDR"
const EnvVaultCACert = "VAULT_CACERT"
const EnvVaultCAPath = "VAULT_CAPATH"
const EnvVaultClientCert = "VAULT_CLIENT_CERT"
const EnvVaultClientKey = "VAULT_CLIENT_KEY"
const EnvVaultClientTimeout = "VAULT_CLIENT_TIMEOUT"
const EnvVaultInsecure = "VAULT_SKIP_VERIFY"
const EnvVaultTLSServerName = "VAULT_TLS_SERVER_NAME"
const EnvVaultWrapTTL = "VAULT_WRAP_TTL"
const EnvVaultMaxRetries = "VAULT_MAX_RETRIES"
const EnvVaultToken = "VAULT_TOKEN"

// WrappingLookupFunc is a function that, given an HTTP verb and a path,
// returns an optional string duration to be used for response wrapping (e.g.
// "15s", or simply "15"). The path will not begin with "/v1/" or "v1/" or "/",
// however, end-of-path forward slashes are not trimmed, so must match your
// called path precisely.
type WrappingLookupFunc func(operation, path string) string

// Config is used to configure the creation of the client.
type Config struct {
	// Address is the address of the Vault server. This should be a complete
	// URL such as "http://vault.example.com". If you need a custom SSL
	// cert or want to enable insecure mode, you need to specify a custom
	// HttpClient.
	Address string

	// HttpClient is the HTTP client to use, which will currently always have the
	// same values as http.DefaultClient. This is used to control redirect behavior.
	HttpClient *http.Client

	redirectSetup sync.Once

	// MaxRetries controls the maximum number of times to retry when a 5xx error
	// occurs. Set to 0 or less to disable retrying. Defaults to 0.
	MaxRetries int

	// Timeout is for setting custom timeout parameter in the HttpClient
	Timeout time.Duration
}

// TLSConfig contains the parameters needed to configure TLS on the HTTP client
// used to communicate with Vault.
type TLSConfig struct {
	// CACert is the path to a PEM-encoded CA cert file to use to verify the
	// Vault server SSL certificate.
	CACert string

	// CAPath is the path to a directory of PEM-encoded CA cert files to verify
	// the Vault server SSL certificate.
	CAPath string

	// ClientCert is the path to the certificate for Vault communication
	ClientCert string

	// ClientKey is the path to the private key for Vault communication
	ClientKey string

	// TLSServerName, if set, is used to set the SNI host when connecting via
	// TLS.
	TLSServerName string

	// Insecure enables or disables SSL verification
	Insecure bool
}

// DefaultConfig returns a default configuration for the client. It is
// safe to modify the return value of this function.
//
// The default Address is https://127.0.0.1:8200, but this can be overridden by
// setting the `VAULT_ADDR` environment variable.
func DefaultConfig() *Config {
	config := &Config{
		Address:    "https://127.0.0.1:8200",
		HttpClient: cleanhttp.DefaultClient(),
	}
	config.HttpClient.Timeout = time.Second * 60
	transport := config.HttpClient.Transport.(*http.Transport)
	transport.TLSHandshakeTimeout = 10 * time.Second
	transport.TLSClientConfig = &tls.Config{
		MinVersion: tls.VersionTLS12,
	}

	if v := os.Getenv(EnvVaultAddress); v != "" {
		config.Address = v
	}

	return config
}

// ConfigureTLS takes a set of TLS configurations and applies those to the the HTTP client.
func (c *Config) ConfigureTLS(t *TLSConfig) error {
	if c.HttpClient == nil {
		c.HttpClient = DefaultConfig().HttpClient
	}

	var clientCert tls.Certificate
	foundClientCert := false
	if t.CACert != "" || t.CAPath != "" || t.ClientCert != "" || t.ClientKey != "" || t.Insecure {
		if t.ClientCert != "" && t.ClientKey != "" {
			var err error
			clientCert, err = tls.LoadX509KeyPair(t.ClientCert, t.ClientKey)
			if err != nil {
				return err
			}
			foundClientCert = true
		} else if t.ClientCert != "" || t.ClientKey != "" {
			return fmt.Errorf("Both client cert and client key must be provided")
		}
	}

	clientTLSConfig := c.HttpClient.Transport.(*http.Transport).TLSClientConfig
	rootConfig := &rootcerts.Config{
		CAFile: t.CACert,
		CAPath: t.CAPath,
	}
	if err := rootcerts.ConfigureTLS(clientTLSConfig, rootConfig); err != nil {
		return err
	}

	clientTLSConfig.InsecureSkipVerify = t.Insecure

	if foundClientCert {
		clientTLSConfig.Certificates = []tls.Certificate{clientCert}
	}
	if t.TLSServerName != "" {
		clientTLSConfig.ServerName = t.TLSServerName
	}

	return nil
}

// ReadEnvironment reads configuration information from the
// environment. If there is an error, no configuration value
// is updated.
func (c *Config) ReadEnvironment() error {
	var envAddress string
	var envCACert string
	var envCAPath string
	var envClientCert string
	var envClientKey string
	var envClientTimeout time.Duration
	var envInsecure bool
	var envTLSServerName string
	var envMaxRetries *uint64

	// Parse the environment variables
	if v := os.Getenv(EnvVaultAddress); v != "" {
		envAddress = v
	}
	if v := os.Getenv(EnvVaultMaxRetries); v != "" {
		maxRetries, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return err
		}
		envMaxRetries = &maxRetries
	}
	if v := os.Getenv(EnvVaultCACert); v != "" {
		envCACert = v
	}
	if v := os.Getenv(EnvVaultCAPath); v != "" {
		envCAPath = v
	}
	if v := os.Getenv(EnvVaultClientCert); v != "" {
		envClientCert = v
	}
	if v := os.Getenv(EnvVaultClientKey); v != "" {
		envClientKey = v
	}
	if t := os.Getenv(EnvVaultClientTimeout); t != "" {
		clientTimeout, err := parseutil.ParseDurationSecond(t)
		if err != nil {
			return fmt.Errorf("Could not parse %s", EnvVaultClientTimeout)
		}
		envClientTimeout = clientTimeout
	}
	if v := os.Getenv(EnvVaultInsecure); v != "" {
		var err error
		envInsecure, err = strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("Could not parse VAULT_SKIP_VERIFY")
		}
	}
	if v := os.Getenv(EnvVaultTLSServerName); v != "" {
		envTLSServerName = v
	}

	// Configure the HTTP clients TLS configuration.
	t := &TLSConfig{
		CACert:        envCACert,
		CAPath:        envCAPath,
		ClientCert:    envClientCert,
		ClientKey:     envClientKey,
		TLSServerName: envTLSServerName,
		Insecure:      envInsecure,
	}
	if err := c.ConfigureTLS(t); err != nil {
		return err
	}

	if envAddress != "" {
		c.Address = envAddress
	}

	if envMaxRetries != nil {
		c.MaxRetries = int(*envMaxRetries) + 1
	}

	if envClientTimeout != 0 {
		c.Timeout = envClientTimeout
	}

	return nil
}

// Client is the client to the Vault API. Create a client with
// NewClient.
type Client struct {
	addr               *url.URL
	config             *Config
	token              string
	wrappingLookupFunc WrappingLookupFunc
}

// NewClient returns a new client for the given configuration.
//
// If the environment variable `VAULT_TOKEN` is present, the token will be
// automatically added to the client. Otherwise, you must manually call
// `SetToken()`.
func NewClient(c *Config) (*Client, error) {
	if c == nil {
		c = DefaultConfig()
		if err := c.ReadEnvironment(); err != nil {
			return nil, fmt.Errorf("error reading environment: %v", err)
		}
	}

	u, err := url.Parse(c.Address)
	if err != nil {
		return nil, err
	}

	if c.HttpClient == nil {
		c.HttpClient = DefaultConfig().HttpClient
	}

	tp := c.HttpClient.Transport.(*http.Transport)
	if err := http2.ConfigureTransport(tp); err != nil {
		return nil, err
	}

	redirFunc := func() {
		// Ensure redirects are not automatically followed
		// Note that this is sane for the API client as it has its own
		// redirect handling logic (and thus also for command/meta),
		// but in e.g. http_test actual redirect handling is necessary
		c.HttpClient.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			// Returning this value causes the Go net library to not close the
			// response body and to nil out the error. Otherwise pester tries
			// three times on every redirect because it sees an error from this
			// function (to prevent redirects) passing through to it.
			return http.ErrUseLastResponse
		}
	}

	c.redirectSetup.Do(redirFunc)

	client := &Client{
		addr:   u,
		config: c,
	}

	if token := os.Getenv(EnvVaultToken); token != "" {
		client.SetToken(token)
	}

	return client, nil
}

// Sets the address of Vault in the client. The format of address should be
// "<Scheme>://<Host>:<Port>". Setting this on a client will override the
// value of VAULT_ADDR environment variable.
func (c *Client) SetAddress(addr string) error {
	var err error
	if c.addr, err = url.Parse(addr); err != nil {
		return fmt.Errorf("failed to set address: %v", err)
	}

	return nil
}

// Address returns the Vault URL the client is configured to connect to
func (c *Client) Address() string {
	return c.addr.String()
}

// SetMaxRetries sets the number of retries that will be used in the case of certain errors
func (c *Client) SetMaxRetries(retries int) {
	c.config.MaxRetries = retries
}

// SetClientTimeout sets the client request timeout
func (c *Client) SetClientTimeout(timeout time.Duration) {
	c.config.Timeout = timeout
}

// SetWrappingLookupFunc sets a lookup function that returns desired wrap TTLs
// for a given operation and path
func (c *Client) SetWrappingLookupFunc(lookupFunc WrappingLookupFunc) {
	c.wrappingLookupFunc = lookupFunc
}

// Token returns the access token being used by this client. It will
// return the empty string if there is no token set.
func (c *Client) Token() string {
	return c.token
}

// SetToken sets the token directly. This won't perform any auth
// verification, it simply sets the token properly for future requests.
func (c *Client) SetToken(v string) {
	c.token = v
}

// ClearToken deletes the token if it is set or does nothing otherwise.
func (c *Client) ClearToken() {
	c.token = ""
}

// Clone creates a copy of this client.
func (c *Client) Clone() (*Client, error) {
	return NewClient(c.config)
}

// NewRequest creates a new raw request object to query the Vault server
// configured for this client. This is an advanced method and generally
// doesn't need to be called externally.
func (c *Client) NewRequest(method, requestPath string) *Request {
	// if SRV records exist (see https://tools.ietf.org/html/draft-andrews-http-srv-02), lookup the SRV
	// record and take the highest match; this is not designed for high-availability, just discovery
	var host string = c.addr.Host
	if c.addr.Port() == "" {
		// Internet Draft specifies that the SRV record is ignored if a port is given
		_, addrs, err := net.LookupSRV("http", "tcp", c.addr.Hostname())
		if err == nil && len(addrs) > 0 {
			host = fmt.Sprintf("%s:%d", addrs[0].Target, addrs[0].Port)
		}
	}

	req := &Request{
		Method: method,
		URL: &url.URL{
			User:   c.addr.User,
			Scheme: c.addr.Scheme,
			Host:   host,
			Path:   path.Join(c.addr.Path, requestPath),
		},
		ClientToken: c.token,
		Params:      make(map[string][]string),
	}

	var lookupPath string
	switch {
	case strings.HasPrefix(requestPath, "/v1/"):
		lookupPath = strings.TrimPrefix(requestPath, "/v1/")
	case strings.HasPrefix(requestPath, "v1/"):
		lookupPath = strings.TrimPrefix(requestPath, "v1/")
	default:
		lookupPath = requestPath
	}
	if c.wrappingLookupFunc != nil {
		req.WrapTTL = c.wrappingLookupFunc(method, lookupPath)
	} else {
		req.WrapTTL = DefaultWrappingLookupFunc(method, lookupPath)
	}
	if c.config.Timeout != 0 {
		c.config.HttpClient.Timeout = c.config.Timeout
	}

	return req
}

// RawRequest performs the raw request given. This request may be against
// a Vault server not configured with this client. This is an advanced operation
// that generally won't need to be called externally.
func (c *Client) RawRequest(r *Request) (*Response, error) {
	redirectCount := 0
START:
	req, err := r.ToHTTP()
	if err != nil {
		return nil, err
	}

	client := pester.NewExtendedClient(c.config.HttpClient)
	client.Backoff = pester.LinearJitterBackoff
	client.MaxRetries = c.config.MaxRetries

	var result *Response
	resp, err := client.Do(req)
	if resp != nil {
		result = &Response{Response: resp}
	}
	if err != nil {
		if strings.Contains(err.Error(), "tls: oversized") {
			err = fmt.Errorf(
				"%s\n\n"+
					"This error usually means that the server is running with TLS disabled\n"+
					"but the client is configured to use TLS. Please either enable TLS\n"+
					"on the server or run the client with -address set to an address\n"+
					"that uses the http protocol:\n\n"+
					"    vault <command> -address http://<address>\n\n"+
					"You can also set the VAULT_ADDR environment variable:\n\n\n"+
					"    VAULT_ADDR=http://<address> vault <command>\n\n"+
					"where <address> is replaced by the actual address to the server.",
				err)
		}
		return result, err
	}

	// Check for a redirect, only allowing for a single redirect
	if (resp.StatusCode == 301 || resp.StatusCode == 302 || resp.StatusCode == 307) && redirectCount == 0 {
		// Parse the updated location
		respLoc, err := resp.Location()
		if err != nil {
			return result, err
		}

		// Ensure a protocol downgrade doesn't happen
		if req.URL.Scheme == "https" && respLoc.Scheme != "https" {
			return result, fmt.Errorf("redirect would cause protocol downgrade")
		}

		// Update the request
		r.URL = respLoc

		// Reset the request body if any
		if err := r.ResetJSONBody(); err != nil {
			return result, err
		}

		// Retry the request
		redirectCount++
		goto START
	}

	if err := result.Error(); err != nil {
		return result, err
	}

	return result, nil
}
