/*
Provides an HTTP Transport that implements the `RoundTripper` interface and
can be used as a built in replacement for the standard library's, providing:

	* connection timeouts
	* request timeouts

This is a thin wrapper around `http.Transport` that sets dial timeouts and uses
Go's internal timer scheduler to call the Go 1.1+ `CancelRequest()` API.
*/
package httpclient

import (
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"
)

// returns the current version of the package
func Version() string {
	return "0.4.1"
}

// Transport implements the RoundTripper interface and can be used as a replacement
// for Go's built in http.Transport implementing end-to-end request timeouts.
//
// 	transport := &httpclient.Transport{
// 	    ConnectTimeout: 1*time.Second,
// 	    ResponseHeaderTimeout: 5*time.Second,
// 	    RequestTimeout: 10*time.Second,
// 	}
// 	defer transport.Close()
//
// 	client := &http.Client{Transport: transport}
// 	req, _ := http.NewRequest("GET", "http://127.0.0.1/test", nil)
// 	resp, err := client.Do(req)
// 	if err != nil {
// 	    return err
// 	}
// 	defer resp.Body.Close()
//
type Transport struct {
	// Proxy specifies a function to return a proxy for a given
	// *http.Request. If the function returns a non-nil error, the
	// request is aborted with the provided error.
	// If Proxy is nil or returns a nil *url.URL, no proxy is used.
	Proxy func(*http.Request) (*url.URL, error)

	// Dial specifies the dial function for creating TCP
	// connections. This will override the Transport's ConnectTimeout and
	// ReadWriteTimeout settings.
	// If Dial is nil, a dialer is generated on demand matching the Transport's
	// options.
	Dial func(network, addr string) (net.Conn, error)

	// TLSClientConfig specifies the TLS configuration to use with
	// tls.Client. If nil, the default configuration is used.
	TLSClientConfig *tls.Config

	// DisableKeepAlives, if true, prevents re-use of TCP connections
	// between different HTTP requests.
	DisableKeepAlives bool

	// DisableCompression, if true, prevents the Transport from
	// requesting compression with an "Accept-Encoding: gzip"
	// request header when the Request contains no existing
	// Accept-Encoding value. If the Transport requests gzip on
	// its own and gets a gzipped response, it's transparently
	// decoded in the Response.Body. However, if the user
	// explicitly requested gzip it is not automatically
	// uncompressed.
	DisableCompression bool

	// MaxIdleConnsPerHost, if non-zero, controls the maximum idle
	// (keep-alive) to keep per-host.  If zero,
	// http.DefaultMaxIdleConnsPerHost is used.
	MaxIdleConnsPerHost int

	// ConnectTimeout, if non-zero, is the maximum amount of time a dial will wait for
	// a connect to complete.
	ConnectTimeout time.Duration

	// ResponseHeaderTimeout, if non-zero, specifies the amount of
	// time to wait for a server's response headers after fully
	// writing the request (including its body, if any). This
	// time does not include the time to read the response body.
	ResponseHeaderTimeout time.Duration

	// RequestTimeout, if non-zero, specifies the amount of time for the entire
	// request to complete (including all of the above timeouts + entire response body).
	// This should never be less than the sum total of the above two timeouts.
	RequestTimeout time.Duration

	// ReadWriteTimeout, if non-zero, will set a deadline for every Read and
	// Write operation on the request connection.
	ReadWriteTimeout time.Duration

	// TCPWriteBufferSize, the size of the operating system's write
	// buffer associated with the connection.
	TCPWriteBufferSize int

	// TCPReadBuffserSize, the size of the operating system's read
	// buffer associated with the connection.
	TCPReadBufferSize int

	starter   sync.Once
	transport *http.Transport
}

// Close cleans up the Transport, currently a no-op
func (t *Transport) Close() error {
	return nil
}

func (t *Transport) lazyStart() {
	if t.Dial == nil {
		t.Dial = func(netw, addr string) (net.Conn, error) {
			c, err := net.DialTimeout(netw, addr, t.ConnectTimeout)
			if err != nil {
				return nil, err
			}

			if t.TCPReadBufferSize != 0 || t.TCPWriteBufferSize != 0 {
				if tcpCon, ok := c.(*net.TCPConn); ok {
					if t.TCPWriteBufferSize != 0 {
						if err = tcpCon.SetWriteBuffer(t.TCPWriteBufferSize); err != nil {
							return nil, err
						}
					}
					if t.TCPReadBufferSize != 0 {
						if err = tcpCon.SetReadBuffer(t.TCPReadBufferSize); err != nil {
							return nil, err
						}
					}
				} else {
					err = errors.New("Not Tcp Connection")
					return nil, err
				}
			}

			if t.ReadWriteTimeout > 0 {
				timeoutConn := &rwTimeoutConn{
					TCPConn:   c.(*net.TCPConn),
					rwTimeout: t.ReadWriteTimeout,
				}
				return timeoutConn, nil
			}
			return c, nil
		}
	}

	t.transport = &http.Transport{
		Dial:                  t.Dial,
		Proxy:                 t.Proxy,
		TLSClientConfig:       t.TLSClientConfig,
		DisableKeepAlives:     t.DisableKeepAlives,
		DisableCompression:    t.DisableCompression,
		MaxIdleConnsPerHost:   t.MaxIdleConnsPerHost,
		ResponseHeaderTimeout: t.ResponseHeaderTimeout,
	}
}

func (t *Transport) CancelRequest(req *http.Request) {
	t.starter.Do(t.lazyStart)

	t.transport.CancelRequest(req)
}

func (t *Transport) CloseIdleConnections() {
	t.starter.Do(t.lazyStart)

	t.transport.CloseIdleConnections()
}

func (t *Transport) RegisterProtocol(scheme string, rt http.RoundTripper) {
	t.starter.Do(t.lazyStart)

	t.transport.RegisterProtocol(scheme, rt)
}

func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	t.starter.Do(t.lazyStart)

	if t.RequestTimeout > 0 {
		timer := time.AfterFunc(t.RequestTimeout, func() {
			t.transport.CancelRequest(req)
		})

		resp, err = t.transport.RoundTrip(req)
		if err != nil {
			timer.Stop()
		} else {
			resp.Body = &bodyCloseInterceptor{ReadCloser: resp.Body, timer: timer}
		}
	} else {
		resp, err = t.transport.RoundTrip(req)
	}

	return
}

type bodyCloseInterceptor struct {
	io.ReadCloser
	timer *time.Timer
}

func (bci *bodyCloseInterceptor) Close() error {
	bci.timer.Stop()
	return bci.ReadCloser.Close()
}

// A net.Conn that sets a deadline for every Read or Write operation
type rwTimeoutConn struct {
	*net.TCPConn
	rwTimeout time.Duration
}

func (c *rwTimeoutConn) Read(b []byte) (int, error) {
	err := c.TCPConn.SetDeadline(time.Now().Add(c.rwTimeout))
	if err != nil {
		return 0, err
	}
	return c.TCPConn.Read(b)
}

func (c *rwTimeoutConn) Write(b []byte) (int, error) {
	err := c.TCPConn.SetDeadline(time.Now().Add(c.rwTimeout))
	if err != nil {
		return 0, err
	}
	return c.TCPConn.Write(b)
}
