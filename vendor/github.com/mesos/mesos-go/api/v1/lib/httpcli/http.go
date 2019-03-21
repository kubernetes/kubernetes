package httpcli

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/mesos/mesos-go/api/v1/lib"
	"github.com/mesos/mesos-go/api/v1/lib/client"
	logger "github.com/mesos/mesos-go/api/v1/lib/debug"
	"github.com/mesos/mesos-go/api/v1/lib/encoding"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/codecs"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/framing"
	"github.com/mesos/mesos-go/api/v1/lib/httpcli/apierrors"
	"github.com/mesos/mesos-go/api/v1/lib/recordio"
)

func noRedirect(req *http.Request, via []*http.Request) error { return http.ErrUseLastResponse }

// ProtocolError is returned when we receive a response from Mesos that is outside of the HTTP API specification.
// Receipt of the following will yield protocol errors:
//   - any unexpected non-error HTTP response codes (e.g. 199)
//   - any unexpected Content-Type
type ProtocolError string

// Error implements error interface
func (pe ProtocolError) Error() string { return string(pe) }

const (
	debug             = logger.Logger(false)
	mediaTypeRecordIO = encoding.MediaType("application/recordio")
)

// DoFunc sends an HTTP request and returns an HTTP response.
//
// An error is returned if caused by client policy (such as
// http.Client.CheckRedirect), or if there was an HTTP protocol error. A
// non-2xx response doesn't cause an error.
//
// When err is nil, resp always contains a non-nil resp.Body.
//
// Callers should close resp.Body when done reading from it. If resp.Body is
// not closed, an underlying RoundTripper (typically Transport) may not be able
// to re-use a persistent TCP connection to the server for a subsequent
// "keep-alive" request.
//
// The request Body, if non-nil, will be closed by an underlying Transport,
// even on errors.
type DoFunc func(*http.Request) (*http.Response, error)

// Response captures the output of a Mesos HTTP API operation. Callers are responsible for invoking
// Close when they're finished processing the response otherwise there may be connection leaks.
type Response struct {
	io.Closer
	encoding.Decoder
	Header http.Header
}

// ErrorMapperFunc generates an error for the given response.
type ErrorMapperFunc func(*http.Response) error

// ResponseHandler is invoked to process an HTTP response. Callers SHALL invoke Close for
// a non-nil Response, even when errors are returned.
type ResponseHandler func(*http.Response, client.ResponseClass, error) (mesos.Response, error)

// A Client is a Mesos HTTP APIs client.
type Client struct {
	url              string
	do               DoFunc
	header           http.Header
	codec            encoding.Codec
	errorMapper      ErrorMapperFunc
	requestOpts      []RequestOpt
	buildRequestFunc func(client.Request, client.ResponseClass, ...RequestOpt) (*http.Request, error)
	handleResponse   ResponseHandler
}

var (
	DefaultCodec   = codecs.ByMediaType[codecs.MediaTypeProtobuf]
	DefaultHeaders = http.Header{}

	// DefaultConfigOpt represents the default client config options.
	DefaultConfigOpt = []ConfigOpt{
		Transport(func(t *http.Transport) {
			// all calls should be ack'd by the server within this interval.
			t.ResponseHeaderTimeout = 15 * time.Second
			t.MaxIdleConnsPerHost = 2 // don't depend on go's default
		}),
	}

	DefaultErrorMapper = ErrorMapperFunc(apierrors.FromResponse)
)

// New returns a new Client with the given Opts applied.
// Callers are expected to configure the URL, Do, and Codec options prior to
// invoking Do.
func New(opts ...Opt) *Client {
	c := &Client{
		codec:       DefaultCodec,
		do:          With(DefaultConfigOpt...),
		header:      cloneHeaders(DefaultHeaders),
		errorMapper: DefaultErrorMapper,
	}
	c.buildRequestFunc = c.buildRequest
	c.handleResponse = c.HandleResponse
	c.With(opts...)
	return c
}

func cloneHeaders(hs http.Header) http.Header {
	result := make(http.Header)
	for k, v := range hs {
		cloned := make([]string, len(v))
		copy(cloned, v)
		result[k] = cloned
	}
	return result
}

// Endpoint returns the current Mesos API endpoint URL that the caller is set to invoke
func (c *Client) Endpoint() string {
	return c.url
}

// RequestOpt defines a functional option for an http.Request.
type RequestOpt func(*http.Request)

// RequestOpts is a convenience type
type RequestOpts []RequestOpt

// Apply this set of request options to the given HTTP request.
func (opts RequestOpts) Apply(req *http.Request) {
	// apply per-request options
	for _, o := range opts {
		if o != nil {
			o(req)
		}
	}
}

// With applies the given Opts to a Client and returns itself.
func (c *Client) With(opts ...Opt) Opt {
	return Opts(opts).Merged().Apply(c)
}

// WithTemporary configures the Client with the temporary option and returns the results of
// invoking f(). Changes made to the Client by the temporary option are reverted before this
// func returns.
func (c *Client) WithTemporary(opt Opt, f func() error) error {
	if opt != nil {
		undo := c.With(opt)
		defer c.With(undo)
	}
	return f()
}

// Mesos returns a mesos.Client variant backed by this implementation.
// Deprecated.
func (c *Client) Mesos(opts ...RequestOpt) mesos.Client {
	return mesos.ClientFunc(func(m encoding.Marshaler) (mesos.Response, error) {
		return c.Do(m, opts...)
	})
}

func prepareForResponse(rc client.ResponseClass, codec encoding.Codec) (RequestOpts, error) {
	// We need to tell Mesos both the content-type and message-content-type that we're expecting, otherwise
	// the server may give us validation problems, or else send back a vague content-type (w/o a
	// message-content-type). In order to communicate these things we need to understand the desired response
	// type from the perspective of the caller --> client.ResponseClass.
	var accept RequestOpts
	switch rc {
	case client.ResponseClassSingleton, client.ResponseClassAuto, client.ResponseClassNoData:
		accept = append(accept, Header("Accept", codec.Type.ContentType()))
	case client.ResponseClassStreaming:
		accept = append(accept, Header("Accept", mediaTypeRecordIO.ContentType()))
		accept = append(accept, Header("Message-Accept", codec.Type.ContentType()))
	default:
		return nil, ProtocolError(fmt.Sprintf("illegal response class requested: %v", rc))
	}
	return accept, nil
}

// buildRequest is a factory func that generates and returns an http.Request for the
// given marshaler and request options.
func (c *Client) buildRequest(cr client.Request, rc client.ResponseClass, opt ...RequestOpt) (*http.Request, error) {
	if crs, ok := cr.(client.RequestStreaming); ok {
		return c.buildRequestStream(crs.Marshaler, rc, opt...)
	}
	accept, err := prepareForResponse(rc, c.codec)
	if err != nil {
		return nil, err
	}

	//TODO(jdef): use a pool to allocate these (and reduce garbage)?
	// .. or else, use a pipe (like streaming does) to avoid the intermediate buffer?
	var body bytes.Buffer
	if err := c.codec.NewEncoder(encoding.SinkWriter(&body)).Encode(cr.Marshaler()); err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", c.url, &body)
	if err != nil {
		return nil, err
	}

	helper := HTTPRequestHelper{req}
	return helper.
		withOptions(c.requestOpts, opt).
		withHeaders(c.header).
		withHeader("Content-Type", c.codec.Type.ContentType()).
		withHeader("Accept", c.codec.Type.ContentType()).
		withOptions(accept).
		Request, nil
}

func (c *Client) buildRequestStream(f func() encoding.Marshaler, rc client.ResponseClass, opt ...RequestOpt) (*http.Request, error) {
	accept, err := prepareForResponse(rc, c.codec)
	if err != nil {
		return nil, err
	}

	var (
		pr, pw = io.Pipe()
		enc    = c.codec.NewEncoder(func() framing.Writer { return recordio.NewWriter(pw) })
	)
	req, err := http.NewRequest("POST", c.url, pr)
	if err != nil {
		pw.Close() // ignore error
		return nil, err
	}

	go func() {
		var closeOnce sync.Once
		defer closeOnce.Do(func() {
			pw.Close()
		})
		for {
			m := f()
			if m == nil {
				// no more messages to send; end of the stream
				break
			}
			err := enc.Encode(m)
			if err != nil {
				closeOnce.Do(func() {
					pw.CloseWithError(err)
				})
				break
			}
		}
	}()

	helper := HTTPRequestHelper{req}
	return helper.
		withOptions(c.requestOpts, opt).
		withHeaders(c.header).
		withHeader("Content-Type", mediaTypeRecordIO.ContentType()).
		withHeader("Message-Content-Type", c.codec.Type.ContentType()).
		withOptions(accept).
		Request, nil
}

func validateSuccessfulResponse(codec encoding.Codec, res *http.Response, rc client.ResponseClass) error {
	switch res.StatusCode {
	case http.StatusOK:
		ct := res.Header.Get("Content-Type")
		switch rc {
		case client.ResponseClassNoData:
			if ct != "" {
				return ProtocolError(fmt.Sprintf("unexpected content type: %q", ct))
			}
		case client.ResponseClassSingleton, client.ResponseClassAuto:
			if ct != codec.Type.ContentType() {
				return ProtocolError(fmt.Sprintf("unexpected content type: %q", ct))
			}
		case client.ResponseClassStreaming:
			if ct != mediaTypeRecordIO.ContentType() {
				return ProtocolError(fmt.Sprintf("unexpected content type: %q", ct))
			}
			ct = res.Header.Get("Message-Content-Type")
			if ct != codec.Type.ContentType() {
				return ProtocolError(fmt.Sprintf("unexpected message content type: %q", ct))
			}
		default:
			return ProtocolError(fmt.Sprintf("unsupported response-class: %q", rc))
		}

	case http.StatusAccepted:
		// nothing to validate, we're not expecting any response entity in this case.
		// TODO(jdef) perhaps check Content-Length == 0 here?
	}
	return nil
}

func newSourceFactory(rc client.ResponseClass) encoding.SourceFactoryFunc {
	switch rc {
	case client.ResponseClassNoData:
		return nil
	case client.ResponseClassSingleton:
		return encoding.SourceReader
	case client.ResponseClassStreaming, client.ResponseClassAuto:
		return recordIOSourceFactory
	default:
		panic(fmt.Sprintf("unsupported response-class: %q", rc))
	}
}

func recordIOSourceFactory(r io.Reader) encoding.Source {
	return func() framing.Reader { return recordio.NewReader(r) }
}

// HandleResponse parses an HTTP response from a Mesos service endpoint, transforming the
// raw HTTP response into a mesos.Response.
func (c *Client) HandleResponse(res *http.Response, rc client.ResponseClass, err error) (mesos.Response, error) {
	if err != nil {
		if res != nil && res.Body != nil {
			res.Body.Close()
		}
		return nil, err
	}

	result := &Response{
		Closer: res.Body,
		Header: res.Header,
	}
	if err = c.errorMapper(res); err != nil {
		return result, err
	}

	err = validateSuccessfulResponse(c.codec, res, rc)
	if err != nil {
		res.Body.Close()
		return nil, err
	}

	switch res.StatusCode {
	case http.StatusOK:
		debug.Log("request OK, decoding response")

		sf := newSourceFactory(rc)
		if sf == nil {
			if rc != client.ResponseClassNoData {
				panic("nil Source for response that expected data")
			}
			// we don't expect any data. drain the response body and close it (compliant with golang's expectations
			// for http/1.1 keepalive support.
			defer res.Body.Close()
			_, err = io.Copy(ioutil.Discard, res.Body)
			return nil, err
		}

		result.Decoder = c.codec.NewDecoder(sf.NewSource(res.Body))

	case http.StatusAccepted:
		debug.Log("request Accepted")

		// noop; no decoder for these types of calls
		defer res.Body.Close()
		_, err = io.Copy(ioutil.Discard, res.Body)
		return nil, err

	default:
		debug.Log("unexpected HTTP status", res.StatusCode)

		defer res.Body.Close()
		io.Copy(ioutil.Discard, res.Body) // intentionally discard any error here
		return nil, ProtocolError(fmt.Sprintf("unexpected mesos HTTP response code: %d", res.StatusCode))
	}

	return result, nil
}

// Do is deprecated in favor of Send.
func (c *Client) Do(m encoding.Marshaler, opt ...RequestOpt) (res mesos.Response, err error) {
	return c.Send(client.RequestSingleton(m), client.ResponseClassAuto, opt...)
}

// Send sends a Call and returns (a) a Response (should be closed when finished) that
// contains a either a streaming or non-streaming Decoder from which callers can read
// objects from, and; (b) an error in case of failure. Callers are expected to *always*
// close a non-nil Response if one is returned. For operations which are successful but
// also for which there are no expected result objects the embedded Decoder will be nil.
// The provided ResponseClass determines whether the client implementation will attempt
// to decode a result as a single obeject or as an object stream. When working with
// versions of Mesos prior to v1.2.x callers MUST use ResponseClassAuto.
func (c *Client) Send(cr client.Request, rc client.ResponseClass, opt ...RequestOpt) (res mesos.Response, err error) {
	var (
		hreq *http.Request
		hres *http.Response
	)
	hreq, err = c.buildRequestFunc(cr, rc, opt...)
	if err == nil {
		hres, err = c.do(hreq)
		res, err = c.handleResponse(hres, rc, err)
	}
	return
}

// ErrorMapper returns am Opt that overrides the existing error mapping behavior of the client.
func ErrorMapper(em ErrorMapperFunc) Opt {
	return func(c *Client) Opt {
		old := c.errorMapper
		c.errorMapper = em
		return ErrorMapper(old)
	}
}

// Endpoint returns an Opt that sets a Client's URL.
func Endpoint(rawurl string) Opt {
	return func(c *Client) Opt {
		old := c.url
		c.url = rawurl
		return Endpoint(old)
	}
}

// WrapDoer returns an Opt that decorates a Client's DoFunc
func WrapDoer(f func(DoFunc) DoFunc) Opt {
	return func(c *Client) Opt {
		old := c.do
		c.do = f(c.do)
		return Do(old)
	}
}

// Do returns an Opt that sets a Client's DoFunc
func Do(do DoFunc) Opt {
	return func(c *Client) Opt {
		old := c.do
		c.do = do
		return Do(old)
	}
}

// Codec returns an Opt that sets a Client's Codec.
func Codec(codec encoding.Codec) Opt {
	return func(c *Client) Opt {
		old := c.codec
		c.codec = codec
		return Codec(old)
	}
}

// DefaultHeader returns an Opt that adds a header to an Client's headers.
func DefaultHeader(k, v string) Opt {
	return func(c *Client) Opt {
		old, found := c.header[k]
		old = append([]string{}, old...) // clone
		c.header.Add(k, v)
		return func(c *Client) Opt {
			if found {
				c.header[k] = old
			} else {
				c.header.Del(k)
			}
			return DefaultHeader(k, v)
		}
	}
}

// HandleResponse returns a functional config option to set the HTTP response handler of the client.
func HandleResponse(f ResponseHandler) Opt {
	return func(c *Client) Opt {
		old := c.handleResponse
		c.handleResponse = f
		return HandleResponse(old)
	}
}

// RequestOptions returns an Opt that applies the given set of options to every Client request.
func RequestOptions(opts ...RequestOpt) Opt {
	if len(opts) == 0 {
		return nil
	}
	return func(c *Client) Opt {
		old := append([]RequestOpt{}, c.requestOpts...)
		c.requestOpts = opts
		return RequestOptions(old...)
	}
}

// Header returns an RequestOpt that adds a header value to an HTTP requests's header.
func Header(k, v string) RequestOpt { return func(r *http.Request) { r.Header.Add(k, v) } }

// Close returns a RequestOpt that determines whether to close the underlying connection after sending the request.
func Close(b bool) RequestOpt { return func(r *http.Request) { r.Close = b } }

// Context returns a RequestOpt that sets the request's Context (ctx must be non-nil)
func Context(ctx context.Context) RequestOpt {
	return func(r *http.Request) {
		r2 := r.WithContext(ctx)
		*r = *r2
	}
}

type Config struct {
	client    *http.Client
	dialer    *net.Dialer
	transport *http.Transport
}

type ConfigOpt func(*Config)

// With returns a DoFunc that executes HTTP round-trips.
// The default implementation provides reasonable defaults for timeouts:
// keep-alive, connection, request/response read/write, and TLS handshake.
// Callers can customize configuration by specifying one or more ConfigOpt's.
func With(opt ...ConfigOpt) DoFunc {
	var (
		dialer = &net.Dialer{
			LocalAddr: &net.TCPAddr{IP: net.IPv4zero},
			KeepAlive: 30 * time.Second,
			Timeout:   5 * time.Second,
		}
		transport = &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			Dial:  dialer.Dial,
			ResponseHeaderTimeout: 5 * time.Second,
			TLSClientConfig:       &tls.Config{InsecureSkipVerify: false},
			TLSHandshakeTimeout:   5 * time.Second,
		}
		config = &Config{
			dialer:    dialer,
			transport: transport,
			client: &http.Client{
				Transport:     transport,
				CheckRedirect: noRedirect, // so we can actually see the 307 redirects
			},
		}
	)
	for _, o := range opt {
		if o != nil {
			o(config)
		}
	}
	return config.client.Do
}

// Timeout returns an ConfigOpt that sets a Config's response header timeout, tls handshake timeout,
// and dialer timeout.
func Timeout(d time.Duration) ConfigOpt {
	return func(c *Config) {
		c.transport.ResponseHeaderTimeout = d
		c.transport.TLSHandshakeTimeout = d
		c.dialer.Timeout = d
	}
}

// RoundTripper returns a ConfigOpt that sets a Config's round-tripper.
func RoundTripper(rt http.RoundTripper) ConfigOpt {
	return func(c *Config) {
		c.client.Transport = rt
	}
}

// TLSConfig returns a ConfigOpt that sets a Config's TLS configuration.
func TLSConfig(tc *tls.Config) ConfigOpt {
	return func(c *Config) {
		c.transport.TLSClientConfig = tc
	}
}

// Transport returns a ConfigOpt that allows tweaks of the default Config's http.Transport
func Transport(modifyTransport func(*http.Transport)) ConfigOpt {
	return func(c *Config) {
		if modifyTransport != nil {
			modifyTransport(c.transport)
		}
	}
}

// WrapRoundTripper allows a caller to customize a configuration's HTTP exchanger. Useful
// for authentication protocols that operate over stock HTTP.
func WrapRoundTripper(f func(http.RoundTripper) http.RoundTripper) ConfigOpt {
	return func(c *Config) {
		if f != nil {
			if rt := f(c.client.Transport); rt != nil {
				c.client.Transport = rt
			}
		}
	}
}

// HTTPRequestHelper wraps an http.Request and provides utility funcs to simplify code elsewhere
type HTTPRequestHelper struct {
	*http.Request
}

func (r *HTTPRequestHelper) withOptions(optsets ...RequestOpts) *HTTPRequestHelper {
	for _, opts := range optsets {
		opts.Apply(r.Request)
	}
	return r
}

func (r *HTTPRequestHelper) withHeaders(hh http.Header) *HTTPRequestHelper {
	for k, v := range hh {
		r.Header[k] = v
		debug.Log("request header " + k + ": " + v[0])
	}
	return r
}

func (r *HTTPRequestHelper) withHeader(key, value string) *HTTPRequestHelper {
	r.Header.Set(key, value)
	return r
}
