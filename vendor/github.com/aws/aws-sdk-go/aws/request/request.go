package request

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/internal/sdkio"
)

const (
	// ErrCodeSerialization is the serialization error code that is received
	// during protocol unmarshaling.
	ErrCodeSerialization = "SerializationError"

	// ErrCodeRead is an error that is returned during HTTP reads.
	ErrCodeRead = "ReadError"

	// ErrCodeResponseTimeout is the connection timeout error that is received
	// during body reads.
	ErrCodeResponseTimeout = "ResponseTimeout"

	// ErrCodeInvalidPresignExpire is returned when the expire time provided to
	// presign is invalid
	ErrCodeInvalidPresignExpire = "InvalidPresignExpireError"

	// CanceledErrorCode is the error code that will be returned by an
	// API request that was canceled. Requests given a aws.Context may
	// return this error when canceled.
	CanceledErrorCode = "RequestCanceled"

	// ErrCodeRequestError is an error preventing the SDK from continuing to
	// process the request.
	ErrCodeRequestError = "RequestError"
)

// A Request is the service request to be made.
type Request struct {
	Config     aws.Config
	ClientInfo metadata.ClientInfo
	Handlers   Handlers

	Retryer
	AttemptTime            time.Time
	Time                   time.Time
	Operation              *Operation
	HTTPRequest            *http.Request
	HTTPResponse           *http.Response
	Body                   io.ReadSeeker
	streamingBody          io.ReadCloser
	BodyStart              int64 // offset from beginning of Body that the request body starts
	Params                 interface{}
	Error                  error
	Data                   interface{}
	RequestID              string
	RetryCount             int
	Retryable              *bool
	RetryDelay             time.Duration
	NotHoist               bool
	SignedHeaderVals       http.Header
	LastSignedAt           time.Time
	DisableFollowRedirects bool

	// Additional API error codes that should be retried. IsErrorRetryable
	// will consider these codes in addition to its built in cases.
	RetryErrorCodes []string

	// Additional API error codes that should be retried with throttle backoff
	// delay. IsErrorThrottle will consider these codes in addition to its
	// built in cases.
	ThrottleErrorCodes []string

	// A value greater than 0 instructs the request to be signed as Presigned URL
	// You should not set this field directly. Instead use Request's
	// Presign or PresignRequest methods.
	ExpireTime time.Duration

	context aws.Context

	built bool

	// Need to persist an intermediate body between the input Body and HTTP
	// request body because the HTTP Client's transport can maintain a reference
	// to the HTTP request's body after the client has returned. This value is
	// safe to use concurrently and wrap the input Body for each HTTP request.
	safeBody *offsetReader
}

// An Operation is the service API operation to be made.
type Operation struct {
	Name       string
	HTTPMethod string
	HTTPPath   string
	*Paginator

	BeforePresignFn func(r *Request) error
}

// New returns a new Request pointer for the service API operation and
// parameters.
//
// A Retryer should be provided to direct how the request is retried. If
// Retryer is nil, a default no retry value will be used. You can use
// NoOpRetryer in the Client package to disable retry behavior directly.
//
// Params is any value of input parameters to be the request payload.
// Data is pointer value to an object which the request's response
// payload will be deserialized to.
func New(cfg aws.Config, clientInfo metadata.ClientInfo, handlers Handlers,
	retryer Retryer, operation *Operation, params interface{}, data interface{}) *Request {

	if retryer == nil {
		retryer = noOpRetryer{}
	}

	method := operation.HTTPMethod
	if method == "" {
		method = "POST"
	}

	httpReq, _ := http.NewRequest(method, "", nil)

	var err error
	httpReq.URL, err = url.Parse(clientInfo.Endpoint)
	if err != nil {
		httpReq.URL = &url.URL{}
		err = awserr.New("InvalidEndpointURL", "invalid endpoint uri", err)
	}

	if len(operation.HTTPPath) != 0 {
		opHTTPPath := operation.HTTPPath
		var opQueryString string
		if idx := strings.Index(opHTTPPath, "?"); idx >= 0 {
			opQueryString = opHTTPPath[idx+1:]
			opHTTPPath = opHTTPPath[:idx]
		}

		if strings.HasSuffix(httpReq.URL.Path, "/") && strings.HasPrefix(opHTTPPath, "/") {
			opHTTPPath = opHTTPPath[1:]
		}
		httpReq.URL.Path += opHTTPPath
		httpReq.URL.RawQuery = opQueryString
	}

	r := &Request{
		Config:     cfg,
		ClientInfo: clientInfo,
		Handlers:   handlers.Copy(),

		Retryer:     retryer,
		Time:        time.Now(),
		ExpireTime:  0,
		Operation:   operation,
		HTTPRequest: httpReq,
		Body:        nil,
		Params:      params,
		Error:       err,
		Data:        data,
	}
	r.SetBufferBody([]byte{})

	return r
}

// A Option is a functional option that can augment or modify a request when
// using a WithContext API operation method.
type Option func(*Request)

// WithGetResponseHeader builds a request Option which will retrieve a single
// header value from the HTTP Response. If there are multiple values for the
// header key use WithGetResponseHeaders instead to access the http.Header
// map directly. The passed in val pointer must be non-nil.
//
// This Option can be used multiple times with a single API operation.
//
//    var id2, versionID string
//    svc.PutObjectWithContext(ctx, params,
//        request.WithGetResponseHeader("x-amz-id-2", &id2),
//        request.WithGetResponseHeader("x-amz-version-id", &versionID),
//    )
func WithGetResponseHeader(key string, val *string) Option {
	return func(r *Request) {
		r.Handlers.Complete.PushBack(func(req *Request) {
			*val = req.HTTPResponse.Header.Get(key)
		})
	}
}

// WithGetResponseHeaders builds a request Option which will retrieve the
// headers from the HTTP response and assign them to the passed in headers
// variable. The passed in headers pointer must be non-nil.
//
//    var headers http.Header
//    svc.PutObjectWithContext(ctx, params, request.WithGetResponseHeaders(&headers))
func WithGetResponseHeaders(headers *http.Header) Option {
	return func(r *Request) {
		r.Handlers.Complete.PushBack(func(req *Request) {
			*headers = req.HTTPResponse.Header
		})
	}
}

// WithLogLevel is a request option that will set the request to use a specific
// log level when the request is made.
//
//     svc.PutObjectWithContext(ctx, params, request.WithLogLevel(aws.LogDebugWithHTTPBody)
func WithLogLevel(l aws.LogLevelType) Option {
	return func(r *Request) {
		r.Config.LogLevel = aws.LogLevel(l)
	}
}

// ApplyOptions will apply each option to the request calling them in the order
// the were provided.
func (r *Request) ApplyOptions(opts ...Option) {
	for _, opt := range opts {
		opt(r)
	}
}

// Context will always returns a non-nil context. If Request does not have a
// context aws.BackgroundContext will be returned.
func (r *Request) Context() aws.Context {
	if r.context != nil {
		return r.context
	}
	return aws.BackgroundContext()
}

// SetContext adds a Context to the current request that can be used to cancel
// a in-flight request. The Context value must not be nil, or this method will
// panic.
//
// Unlike http.Request.WithContext, SetContext does not return a copy of the
// Request. It is not safe to use use a single Request value for multiple
// requests. A new Request should be created for each API operation request.
//
// Go 1.6 and below:
// The http.Request's Cancel field will be set to the Done() value of
// the context. This will overwrite the Cancel field's value.
//
// Go 1.7 and above:
// The http.Request.WithContext will be used to set the context on the underlying
// http.Request. This will create a shallow copy of the http.Request. The SDK
// may create sub contexts in the future for nested requests such as retries.
func (r *Request) SetContext(ctx aws.Context) {
	if ctx == nil {
		panic("context cannot be nil")
	}
	setRequestContext(r, ctx)
}

// WillRetry returns if the request's can be retried.
func (r *Request) WillRetry() bool {
	if !aws.IsReaderSeekable(r.Body) && r.HTTPRequest.Body != NoBody {
		return false
	}
	return r.Error != nil && aws.BoolValue(r.Retryable) && r.RetryCount < r.MaxRetries()
}

func fmtAttemptCount(retryCount, maxRetries int) string {
	return fmt.Sprintf("attempt %v/%v", retryCount, maxRetries)
}

// ParamsFilled returns if the request's parameters have been populated
// and the parameters are valid. False is returned if no parameters are
// provided or invalid.
func (r *Request) ParamsFilled() bool {
	return r.Params != nil && reflect.ValueOf(r.Params).Elem().IsValid()
}

// DataFilled returns true if the request's data for response deserialization
// target has been set and is a valid. False is returned if data is not
// set, or is invalid.
func (r *Request) DataFilled() bool {
	return r.Data != nil && reflect.ValueOf(r.Data).Elem().IsValid()
}

// SetBufferBody will set the request's body bytes that will be sent to
// the service API.
func (r *Request) SetBufferBody(buf []byte) {
	r.SetReaderBody(bytes.NewReader(buf))
}

// SetStringBody sets the body of the request to be backed by a string.
func (r *Request) SetStringBody(s string) {
	r.SetReaderBody(strings.NewReader(s))
}

// SetReaderBody will set the request's body reader.
func (r *Request) SetReaderBody(reader io.ReadSeeker) {
	r.Body = reader

	if aws.IsReaderSeekable(reader) {
		var err error
		// Get the Bodies current offset so retries will start from the same
		// initial position.
		r.BodyStart, err = reader.Seek(0, sdkio.SeekCurrent)
		if err != nil {
			r.Error = awserr.New(ErrCodeSerialization,
				"failed to determine start of request body", err)
			return
		}
	}
	r.ResetBody()
}

// SetStreamingBody set the reader to be used for the request that will stream
// bytes to the server. Request's Body must not be set to any reader.
func (r *Request) SetStreamingBody(reader io.ReadCloser) {
	r.streamingBody = reader
	r.SetReaderBody(aws.ReadSeekCloser(reader))
}

// Presign returns the request's signed URL. Error will be returned
// if the signing fails. The expire parameter is only used for presigned Amazon
// S3 API requests. All other AWS services will use a fixed expiration
// time of 15 minutes.
//
// It is invalid to create a presigned URL with a expire duration 0 or less. An
// error is returned if expire duration is 0 or less.
func (r *Request) Presign(expire time.Duration) (string, error) {
	r = r.copy()

	// Presign requires all headers be hoisted. There is no way to retrieve
	// the signed headers not hoisted without this. Making the presigned URL
	// useless.
	r.NotHoist = false

	u, _, err := getPresignedURL(r, expire)
	return u, err
}

// PresignRequest behaves just like presign, with the addition of returning a
// set of headers that were signed. The expire parameter is only used for
// presigned Amazon S3 API requests. All other AWS services will use a fixed
// expiration time of 15 minutes.
//
// It is invalid to create a presigned URL with a expire duration 0 or less. An
// error is returned if expire duration is 0 or less.
//
// Returns the URL string for the API operation with signature in the query string,
// and the HTTP headers that were included in the signature. These headers must
// be included in any HTTP request made with the presigned URL.
//
// To prevent hoisting any headers to the query string set NotHoist to true on
// this Request value prior to calling PresignRequest.
func (r *Request) PresignRequest(expire time.Duration) (string, http.Header, error) {
	r = r.copy()
	return getPresignedURL(r, expire)
}

// IsPresigned returns true if the request represents a presigned API url.
func (r *Request) IsPresigned() bool {
	return r.ExpireTime != 0
}

func getPresignedURL(r *Request, expire time.Duration) (string, http.Header, error) {
	if expire <= 0 {
		return "", nil, awserr.New(
			ErrCodeInvalidPresignExpire,
			"presigned URL requires an expire duration greater than 0",
			nil,
		)
	}

	r.ExpireTime = expire

	if r.Operation.BeforePresignFn != nil {
		if err := r.Operation.BeforePresignFn(r); err != nil {
			return "", nil, err
		}
	}

	if err := r.Sign(); err != nil {
		return "", nil, err
	}

	return r.HTTPRequest.URL.String(), r.SignedHeaderVals, nil
}

const (
	notRetrying = "not retrying"
)

func debugLogReqError(r *Request, stage, retryStr string, err error) {
	if !r.Config.LogLevel.Matches(aws.LogDebugWithRequestErrors) {
		return
	}

	r.Config.Logger.Log(fmt.Sprintf("DEBUG: %s %s/%s failed, %s, error %v",
		stage, r.ClientInfo.ServiceName, r.Operation.Name, retryStr, err))
}

// Build will build the request's object so it can be signed and sent
// to the service. Build will also validate all the request's parameters.
// Any additional build Handlers set on this request will be run
// in the order they were set.
//
// The request will only be built once. Multiple calls to build will have
// no effect.
//
// If any Validate or Build errors occur the build will stop and the error
// which occurred will be returned.
func (r *Request) Build() error {
	if !r.built {
		r.Handlers.Validate.Run(r)
		if r.Error != nil {
			debugLogReqError(r, "Validate Request", notRetrying, r.Error)
			return r.Error
		}
		r.Handlers.Build.Run(r)
		if r.Error != nil {
			debugLogReqError(r, "Build Request", notRetrying, r.Error)
			return r.Error
		}
		r.built = true
	}

	return r.Error
}

// Sign will sign the request, returning error if errors are encountered.
//
// Sign will build the request prior to signing. All Sign Handlers will
// be executed in the order they were set.
func (r *Request) Sign() error {
	r.Build()
	if r.Error != nil {
		debugLogReqError(r, "Build Request", notRetrying, r.Error)
		return r.Error
	}

	SanitizeHostForHeader(r.HTTPRequest)

	r.Handlers.Sign.Run(r)
	return r.Error
}

func (r *Request) getNextRequestBody() (body io.ReadCloser, err error) {
	if r.streamingBody != nil {
		return r.streamingBody, nil
	}

	if r.safeBody != nil {
		r.safeBody.Close()
	}

	r.safeBody, err = newOffsetReader(r.Body, r.BodyStart)
	if err != nil {
		return nil, awserr.New(ErrCodeSerialization,
			"failed to get next request body reader", err)
	}

	// Go 1.8 tightened and clarified the rules code needs to use when building
	// requests with the http package. Go 1.8 removed the automatic detection
	// of if the Request.Body was empty, or actually had bytes in it. The SDK
	// always sets the Request.Body even if it is empty and should not actually
	// be sent. This is incorrect.
	//
	// Go 1.8 did add a http.NoBody value that the SDK can use to tell the http
	// client that the request really should be sent without a body. The
	// Request.Body cannot be set to nil, which is preferable, because the
	// field is exported and could introduce nil pointer dereferences for users
	// of the SDK if they used that field.
	//
	// Related golang/go#18257
	l, err := aws.SeekerLen(r.Body)
	if err != nil {
		return nil, awserr.New(ErrCodeSerialization,
			"failed to compute request body size", err)
	}

	if l == 0 {
		body = NoBody
	} else if l > 0 {
		body = r.safeBody
	} else {
		// Hack to prevent sending bodies for methods where the body
		// should be ignored by the server. Sending bodies on these
		// methods without an associated ContentLength will cause the
		// request to socket timeout because the server does not handle
		// Transfer-Encoding: chunked bodies for these methods.
		//
		// This would only happen if a aws.ReaderSeekerCloser was used with
		// a io.Reader that was not also an io.Seeker, or did not implement
		// Len() method.
		switch r.Operation.HTTPMethod {
		case "GET", "HEAD", "DELETE":
			body = NoBody
		default:
			body = r.safeBody
		}
	}

	return body, nil
}

// GetBody will return an io.ReadSeeker of the Request's underlying
// input body with a concurrency safe wrapper.
func (r *Request) GetBody() io.ReadSeeker {
	return r.safeBody
}

// Send will send the request, returning error if errors are encountered.
//
// Send will sign the request prior to sending. All Send Handlers will
// be executed in the order they were set.
//
// Canceling a request is non-deterministic. If a request has been canceled,
// then the transport will choose, randomly, one of the state channels during
// reads or getting the connection.
//
// readLoop() and getConn(req *Request, cm connectMethod)
// https://github.com/golang/go/blob/master/src/net/http/transport.go
//
// Send will not close the request.Request's body.
func (r *Request) Send() error {
	defer func() {
		// Regardless of success or failure of the request trigger the Complete
		// request handlers.
		r.Handlers.Complete.Run(r)
	}()

	if err := r.Error; err != nil {
		return err
	}

	for {
		r.Error = nil
		r.AttemptTime = time.Now()

		if err := r.Sign(); err != nil {
			debugLogReqError(r, "Sign Request", notRetrying, err)
			return err
		}

		if err := r.sendRequest(); err == nil {
			return nil
		}
		r.Handlers.Retry.Run(r)
		r.Handlers.AfterRetry.Run(r)

		if r.Error != nil || !aws.BoolValue(r.Retryable) {
			return r.Error
		}

		if err := r.prepareRetry(); err != nil {
			r.Error = err
			return err
		}
	}
}

func (r *Request) prepareRetry() error {
	if r.Config.LogLevel.Matches(aws.LogDebugWithRequestRetries) {
		r.Config.Logger.Log(fmt.Sprintf("DEBUG: Retrying Request %s/%s, attempt %d",
			r.ClientInfo.ServiceName, r.Operation.Name, r.RetryCount))
	}

	// The previous http.Request will have a reference to the r.Body
	// and the HTTP Client's Transport may still be reading from
	// the request's body even though the Client's Do returned.
	r.HTTPRequest = copyHTTPRequest(r.HTTPRequest, nil)
	r.ResetBody()
	if err := r.Error; err != nil {
		return awserr.New(ErrCodeSerialization,
			"failed to prepare body for retry", err)

	}

	// Closing response body to ensure that no response body is leaked
	// between retry attempts.
	if r.HTTPResponse != nil && r.HTTPResponse.Body != nil {
		r.HTTPResponse.Body.Close()
	}

	return nil
}

func (r *Request) sendRequest() (sendErr error) {
	defer r.Handlers.CompleteAttempt.Run(r)

	r.Retryable = nil
	r.Handlers.Send.Run(r)
	if r.Error != nil {
		debugLogReqError(r, "Send Request",
			fmtAttemptCount(r.RetryCount, r.MaxRetries()),
			r.Error)
		return r.Error
	}

	r.Handlers.UnmarshalMeta.Run(r)
	r.Handlers.ValidateResponse.Run(r)
	if r.Error != nil {
		r.Handlers.UnmarshalError.Run(r)
		debugLogReqError(r, "Validate Response",
			fmtAttemptCount(r.RetryCount, r.MaxRetries()),
			r.Error)
		return r.Error
	}

	r.Handlers.Unmarshal.Run(r)
	if r.Error != nil {
		debugLogReqError(r, "Unmarshal Response",
			fmtAttemptCount(r.RetryCount, r.MaxRetries()),
			r.Error)
		return r.Error
	}

	return nil
}

// copy will copy a request which will allow for local manipulation of the
// request.
func (r *Request) copy() *Request {
	req := &Request{}
	*req = *r
	req.Handlers = r.Handlers.Copy()
	op := *r.Operation
	req.Operation = &op
	return req
}

// AddToUserAgent adds the string to the end of the request's current user agent.
func AddToUserAgent(r *Request, s string) {
	curUA := r.HTTPRequest.Header.Get("User-Agent")
	if len(curUA) > 0 {
		s = curUA + " " + s
	}
	r.HTTPRequest.Header.Set("User-Agent", s)
}

// SanitizeHostForHeader removes default port from host and updates request.Host
func SanitizeHostForHeader(r *http.Request) {
	host := getHost(r)
	port := portOnly(host)
	if port != "" && isDefaultPort(r.URL.Scheme, port) {
		r.Host = stripPort(host)
	}
}

// Returns host from request
func getHost(r *http.Request) string {
	if r.Host != "" {
		return r.Host
	}

	if r.URL == nil {
		return ""
	}

	return r.URL.Host
}

// Hostname returns u.Host, without any port number.
//
// If Host is an IPv6 literal with a port number, Hostname returns the
// IPv6 literal without the square brackets. IPv6 literals may include
// a zone identifier.
//
// Copied from the Go 1.8 standard library (net/url)
func stripPort(hostport string) string {
	colon := strings.IndexByte(hostport, ':')
	if colon == -1 {
		return hostport
	}
	if i := strings.IndexByte(hostport, ']'); i != -1 {
		return strings.TrimPrefix(hostport[:i], "[")
	}
	return hostport[:colon]
}

// Port returns the port part of u.Host, without the leading colon.
// If u.Host doesn't contain a port, Port returns an empty string.
//
// Copied from the Go 1.8 standard library (net/url)
func portOnly(hostport string) string {
	colon := strings.IndexByte(hostport, ':')
	if colon == -1 {
		return ""
	}
	if i := strings.Index(hostport, "]:"); i != -1 {
		return hostport[i+len("]:"):]
	}
	if strings.Contains(hostport, "]") {
		return ""
	}
	return hostport[colon+len(":"):]
}

// Returns true if the specified URI is using the standard port
// (i.e. port 80 for HTTP URIs or 443 for HTTPS URIs)
func isDefaultPort(scheme, port string) bool {
	if port == "" {
		return true
	}

	lowerCaseScheme := strings.ToLower(scheme)
	if (lowerCaseScheme == "http" && port == "80") || (lowerCaseScheme == "https" && port == "443") {
		return true
	}

	return false
}
