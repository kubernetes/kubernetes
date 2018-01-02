package plugins

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"time"

	"github.com/docker/docker/pkg/plugins/transport"
	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/sirupsen/logrus"
)

const (
	defaultTimeOut = 30
)

func newTransport(addr string, tlsConfig *tlsconfig.Options) (transport.Transport, error) {
	tr := &http.Transport{}

	if tlsConfig != nil {
		c, err := tlsconfig.Client(*tlsConfig)
		if err != nil {
			return nil, err
		}
		tr.TLSClientConfig = c
	}

	u, err := url.Parse(addr)
	if err != nil {
		return nil, err
	}
	socket := u.Host
	if socket == "" {
		// valid local socket addresses have the host empty.
		socket = u.Path
	}
	if err := sockets.ConfigureTransport(tr, u.Scheme, socket); err != nil {
		return nil, err
	}
	scheme := httpScheme(u)

	return transport.NewHTTPTransport(tr, scheme, socket), nil
}

// NewClient creates a new plugin client (http).
func NewClient(addr string, tlsConfig *tlsconfig.Options) (*Client, error) {
	clientTransport, err := newTransport(addr, tlsConfig)
	if err != nil {
		return nil, err
	}
	return newClientWithTransport(clientTransport, 0), nil
}

// NewClientWithTimeout creates a new plugin client (http).
func NewClientWithTimeout(addr string, tlsConfig *tlsconfig.Options, timeout time.Duration) (*Client, error) {
	clientTransport, err := newTransport(addr, tlsConfig)
	if err != nil {
		return nil, err
	}
	return newClientWithTransport(clientTransport, timeout), nil
}

// newClientWithTransport creates a new plugin client with a given transport.
func newClientWithTransport(tr transport.Transport, timeout time.Duration) *Client {
	return &Client{
		http: &http.Client{
			Transport: tr,
			Timeout:   timeout,
		},
		requestFactory: tr,
	}
}

// Client represents a plugin client.
type Client struct {
	http           *http.Client // http client to use
	requestFactory transport.RequestFactory
}

// Call calls the specified method with the specified arguments for the plugin.
// It will retry for 30 seconds if a failure occurs when calling.
func (c *Client) Call(serviceMethod string, args interface{}, ret interface{}) error {
	var buf bytes.Buffer
	if args != nil {
		if err := json.NewEncoder(&buf).Encode(args); err != nil {
			return err
		}
	}
	body, err := c.callWithRetry(serviceMethod, &buf, true)
	if err != nil {
		return err
	}
	defer body.Close()
	if ret != nil {
		if err := json.NewDecoder(body).Decode(&ret); err != nil {
			logrus.Errorf("%s: error reading plugin resp: %v", serviceMethod, err)
			return err
		}
	}
	return nil
}

// Stream calls the specified method with the specified arguments for the plugin and returns the response body
func (c *Client) Stream(serviceMethod string, args interface{}) (io.ReadCloser, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(args); err != nil {
		return nil, err
	}
	return c.callWithRetry(serviceMethod, &buf, true)
}

// SendFile calls the specified method, and passes through the IO stream
func (c *Client) SendFile(serviceMethod string, data io.Reader, ret interface{}) error {
	body, err := c.callWithRetry(serviceMethod, data, true)
	if err != nil {
		return err
	}
	defer body.Close()
	if err := json.NewDecoder(body).Decode(&ret); err != nil {
		logrus.Errorf("%s: error reading plugin resp: %v", serviceMethod, err)
		return err
	}
	return nil
}

func (c *Client) callWithRetry(serviceMethod string, data io.Reader, retry bool) (io.ReadCloser, error) {
	var retries int
	start := time.Now()

	for {
		req, err := c.requestFactory.NewRequest(serviceMethod, data)
		if err != nil {
			return nil, err
		}

		resp, err := c.http.Do(req)
		if err != nil {
			if !retry {
				return nil, err
			}

			timeOff := backoff(retries)
			if abort(start, timeOff) {
				return nil, err
			}
			retries++
			logrus.Warnf("Unable to connect to plugin: %s%s: %v, retrying in %v", req.URL.Host, req.URL.Path, err, timeOff)
			time.Sleep(timeOff)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			b, err := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, &statusError{resp.StatusCode, serviceMethod, err.Error()}
			}

			// Plugins' Response(s) should have an Err field indicating what went
			// wrong. Try to unmarshal into ResponseErr. Otherwise fallback to just
			// return the string(body)
			type responseErr struct {
				Err string
			}
			remoteErr := responseErr{}
			if err := json.Unmarshal(b, &remoteErr); err == nil {
				if remoteErr.Err != "" {
					return nil, &statusError{resp.StatusCode, serviceMethod, remoteErr.Err}
				}
			}
			// old way...
			return nil, &statusError{resp.StatusCode, serviceMethod, string(b)}
		}
		return resp.Body, nil
	}
}

func backoff(retries int) time.Duration {
	b, max := 1, defaultTimeOut
	for b < max && retries > 0 {
		b *= 2
		retries--
	}
	if b > max {
		b = max
	}
	return time.Duration(b) * time.Second
}

func abort(start time.Time, timeOff time.Duration) bool {
	return timeOff+time.Since(start) >= time.Duration(defaultTimeOut)*time.Second
}

func httpScheme(u *url.URL) string {
	scheme := u.Scheme
	if scheme != "https" {
		scheme = "http"
	}
	return scheme
}
