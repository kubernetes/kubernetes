package request

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
)

// A Request is the service request to be made.
type Request struct {
	Config     aws.Config
	ClientInfo metadata.ClientInfo
	Handlers   Handlers

	Retryer
	Time             time.Time
	ExpireTime       time.Duration
	Operation        *Operation
	HTTPRequest      *http.Request
	HTTPResponse     *http.Response
	Body             io.ReadSeeker
	BodyStart        int64 // offset from beginning of Body that the request body starts
	Params           interface{}
	Error            error
	Data             interface{}
	RequestID        string
	RetryCount       int
	Retryable        *bool
	RetryDelay       time.Duration
	NotHoist         bool
	SignedHeaderVals http.Header
	LastSignedAt     time.Time

	built bool

	// Need to persist an intermideant body betweend the input Body and HTTP
	// request body because the HTTP Client's transport can maintain a reference
	// to the HTTP request's body after the client has returned. This value is
	// safe to use concurrently and rewraps the input Body for each HTTP request.
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

// Paginator keeps track of pagination configuration for an API operation.
type Paginator struct {
	InputTokens     []string
	OutputTokens    []string
	LimitToken      string
	TruncationToken string
}

// New returns a new Request pointer for the service API
// operation and parameters.
//
// Params is any value of input parameters to be the request payload.
// Data is pointer value to an object which the request's response
// payload will be deserialized to.
func New(cfg aws.Config, clientInfo metadata.ClientInfo, handlers Handlers,
	retryer Retryer, operation *Operation, params interface{}, data interface{}) *Request {

	method := operation.HTTPMethod
	if method == "" {
		method = "POST"
	}

	httpReq, _ := http.NewRequest(method, "", nil)

	var err error
	httpReq.URL, err = url.Parse(clientInfo.Endpoint + operation.HTTPPath)
	if err != nil {
		httpReq.URL = &url.URL{}
		err = awserr.New("InvalidEndpointURL", "invalid endpoint uri", err)
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

// WillRetry returns if the request's can be retried.
func (r *Request) WillRetry() bool {
	return r.Error != nil && aws.BoolValue(r.Retryable) && r.RetryCount < r.MaxRetries()
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
	r.ResetBody()
}

// Presign returns the request's signed URL. Error will be returned
// if the signing fails.
func (r *Request) Presign(expireTime time.Duration) (string, error) {
	r.ExpireTime = expireTime
	r.NotHoist = false

	if r.Operation.BeforePresignFn != nil {
		r = r.copy()
		err := r.Operation.BeforePresignFn(r)
		if err != nil {
			return "", err
		}
	}

	r.Sign()
	if r.Error != nil {
		return "", r.Error
	}
	return r.HTTPRequest.URL.String(), nil
}

// PresignRequest behaves just like presign, but hoists all headers and signs them.
// Also returns the signed hash back to the user
func (r *Request) PresignRequest(expireTime time.Duration) (string, http.Header, error) {
	r.ExpireTime = expireTime
	r.NotHoist = true
	r.Sign()
	if r.Error != nil {
		return "", nil, r.Error
	}
	return r.HTTPRequest.URL.String(), r.SignedHeaderVals, nil
}

func debugLogReqError(r *Request, stage string, retrying bool, err error) {
	if !r.Config.LogLevel.Matches(aws.LogDebugWithRequestErrors) {
		return
	}

	retryStr := "not retrying"
	if retrying {
		retryStr = "will retry"
	}

	r.Config.Logger.Log(fmt.Sprintf("DEBUG: %s %s/%s failed, %s, error %v",
		stage, r.ClientInfo.ServiceName, r.Operation.Name, retryStr, err))
}

// Build will build the request's object so it can be signed and sent
// to the service. Build will also validate all the request's parameters.
// Anny additional build Handlers set on this request will be run
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
			debugLogReqError(r, "Validate Request", false, r.Error)
			return r.Error
		}
		r.Handlers.Build.Run(r)
		if r.Error != nil {
			debugLogReqError(r, "Build Request", false, r.Error)
			return r.Error
		}
		r.built = true
	}

	return r.Error
}

// Sign will sign the request returning error if errors are encountered.
//
// Send will build the request prior to signing. All Sign Handlers will
// be executed in the order they were set.
func (r *Request) Sign() error {
	r.Build()
	if r.Error != nil {
		debugLogReqError(r, "Build Request", false, r.Error)
		return r.Error
	}

	r.Handlers.Sign.Run(r)
	return r.Error
}

// ResetBody rewinds the request body backto its starting position, and
// set's the HTTP Request body reference. When the body is read prior
// to being sent in the HTTP request it will need to be rewound.
func (r *Request) ResetBody() {
	if r.safeBody != nil {
		r.safeBody.Close()
	}

	r.safeBody = newOffsetReader(r.Body, r.BodyStart)

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
	l, err := computeBodyLength(r.Body)
	if err != nil {
		r.Error = awserr.New("SerializationError", "failed to compute request body size", err)
		return
	}

	if l == 0 {
		r.HTTPRequest.Body = noBodyReader
	} else if l > 0 {
		r.HTTPRequest.Body = r.safeBody
	} else {
		// Hack to prevent sending bodies for methods where the body
		// should be ignored by the server. Sending bodies on these
		// methods without an associated ContentLength will cause the
		// request to socket timeout because the server does not handle
		// Transfer-Encoding: chunked bodies for these methods.
		//
		// This would only happen if a aws.ReaderSeekerCloser was used with
		// a io.Reader that was not also an io.Seeker.
		switch r.Operation.HTTPMethod {
		case "GET", "HEAD", "DELETE":
			r.HTTPRequest.Body = noBodyReader
		default:
			r.HTTPRequest.Body = r.safeBody
		}
	}
}

// Attempts to compute the length of the body of the reader using the
// io.Seeker interface. If the value is not seekable because of being
// a ReaderSeekerCloser without an unerlying Seeker -1 will be returned.
// If no error occurs the length of the body will be returned.
func computeBodyLength(r io.ReadSeeker) (int64, error) {
	seekable := true
	// Determine if the seeker is actually seekable. ReaderSeekerCloser
	// hides the fact that a io.Readers might not actually be seekable.
	switch v := r.(type) {
	case aws.ReaderSeekerCloser:
		seekable = v.IsSeeker()
	case *aws.ReaderSeekerCloser:
		seekable = v.IsSeeker()
	}
	if !seekable {
		return -1, nil
	}

	curOffset, err := r.Seek(0, 1)
	if err != nil {
		return 0, err
	}

	endOffset, err := r.Seek(0, 2)
	if err != nil {
		return 0, err
	}

	_, err = r.Seek(curOffset, 0)
	if err != nil {
		return 0, err
	}

	return endOffset - curOffset, nil
}

// GetBody will return an io.ReadSeeker of the Request's underlying
// input body with a concurrency safe wrapper.
func (r *Request) GetBody() io.ReadSeeker {
	return r.safeBody
}

// Send will send the request returning error if errors are encountered.
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
	for {
		if aws.BoolValue(r.Retryable) {
			if r.Config.LogLevel.Matches(aws.LogDebugWithRequestRetries) {
				r.Config.Logger.Log(fmt.Sprintf("DEBUG: Retrying Request %s/%s, attempt %d",
					r.ClientInfo.ServiceName, r.Operation.Name, r.RetryCount))
			}

			// The previous http.Request will have a reference to the r.Body
			// and the HTTP Client's Transport may still be reading from
			// the request's body even though the Client's Do returned.
			r.HTTPRequest = copyHTTPRequest(r.HTTPRequest, nil)
			r.ResetBody()

			// Closing response body to ensure that no response body is leaked
			// between retry attempts.
			if r.HTTPResponse != nil && r.HTTPResponse.Body != nil {
				r.HTTPResponse.Body.Close()
			}
		}

		r.Sign()
		if r.Error != nil {
			return r.Error
		}

		r.Retryable = nil

		r.Handlers.Send.Run(r)
		if r.Error != nil {
			if !shouldRetryCancel(r) {
				return r.Error
			}

			err := r.Error
			r.Handlers.Retry.Run(r)
			r.Handlers.AfterRetry.Run(r)
			if r.Error != nil {
				debugLogReqError(r, "Send Request", false, r.Error)
				return r.Error
			}
			debugLogReqError(r, "Send Request", true, err)
			continue
		}
		r.Handlers.UnmarshalMeta.Run(r)
		r.Handlers.ValidateResponse.Run(r)
		if r.Error != nil {
			err := r.Error
			r.Handlers.UnmarshalError.Run(r)
			r.Handlers.Retry.Run(r)
			r.Handlers.AfterRetry.Run(r)
			if r.Error != nil {
				debugLogReqError(r, "Validate Response", false, r.Error)
				return r.Error
			}
			debugLogReqError(r, "Validate Response", true, err)
			continue
		}

		r.Handlers.Unmarshal.Run(r)
		if r.Error != nil {
			err := r.Error
			r.Handlers.Retry.Run(r)
			r.Handlers.AfterRetry.Run(r)
			if r.Error != nil {
				debugLogReqError(r, "Unmarshal Response", false, r.Error)
				return r.Error
			}
			debugLogReqError(r, "Unmarshal Response", true, err)
			continue
		}

		break
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

func shouldRetryCancel(r *Request) bool {
	awsErr, ok := r.Error.(awserr.Error)
	timeoutErr := false
	errStr := r.Error.Error()
	if ok {
		err := awsErr.OrigErr()
		netErr, netOK := err.(net.Error)
		timeoutErr = netOK && netErr.Temporary()
		if urlErr, ok := err.(*url.Error); !timeoutErr && ok {
			errStr = urlErr.Err.Error()
		}
	}

	// There can be two types of canceled errors here.
	// The first being a net.Error and the other being an error.
	// If the request was timed out, we want to continue the retry
	// process. Otherwise, return the canceled error.
	return timeoutErr ||
		(errStr != "net/http: request canceled" &&
			errStr != "net/http: request canceled while waiting for connection")

}
