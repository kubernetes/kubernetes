package request

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
)

// A Request is the service request to be made.
type Request struct {
	Config     aws.Config
	ClientInfo metadata.ClientInfo
	Handlers   Handlers

	Retryer
	Time         time.Time
	ExpireTime   time.Duration
	Operation    *Operation
	HTTPRequest  *http.Request
	HTTPResponse *http.Response
	Body         io.ReadSeeker
	BodyStart    int64 // offset from beginning of Body that the request body starts
	Params       interface{}
	Error        error
	Data         interface{}
	RequestID    string
	RetryCount   int
	Retryable    *bool
	RetryDelay   time.Duration

	built bool
}

// An Operation is the service API operation to be made.
type Operation struct {
	Name       string
	HTTPMethod string
	HTTPPath   string
	*Paginator
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
	p := operation.HTTPPath
	if p == "" {
		p = "/"
	}

	httpReq, _ := http.NewRequest(method, "", nil)
	httpReq.URL, _ = url.Parse(clientInfo.Endpoint + p)

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
		Error:       nil,
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
	r.HTTPRequest.Body = ioutil.NopCloser(reader)
	r.Body = reader
}

// Presign returns the request's signed URL. Error will be returned
// if the signing fails.
func (r *Request) Presign(expireTime time.Duration) (string, error) {
	r.ExpireTime = expireTime
	r.Sign()
	if r.Error != nil {
		return "", r.Error
	}
	return r.HTTPRequest.URL.String(), nil
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
		r.Error = nil
		r.Handlers.Validate.Run(r)
		if r.Error != nil {
			debugLogReqError(r, "Validate Request", false, r.Error)
			return r.Error
		}
		r.Handlers.Build.Run(r)
		r.built = true
	}

	return r.Error
}

// Sign will sign the request retuning error if errors are encountered.
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

// Send will send the request returning error if errors are encountered.
//
// Send will sign the request prior to sending. All Send Handlers will
// be executed in the order they were set.
func (r *Request) Send() error {
	for {
		r.Sign()
		if r.Error != nil {
			return r.Error
		}

		if aws.BoolValue(r.Retryable) {
			if r.Config.LogLevel.Matches(aws.LogDebugWithRequestRetries) {
				r.Config.Logger.Log(fmt.Sprintf("DEBUG: Retrying Request %s/%s, attempt %d",
					r.ClientInfo.ServiceName, r.Operation.Name, r.RetryCount))
			}

			// Re-seek the body back to the original point in for a retry so that
			// send will send the body's contents again in the upcoming request.
			r.Body.Seek(r.BodyStart, 0)
			r.HTTPRequest.Body = ioutil.NopCloser(r.Body)
		}
		r.Retryable = nil

		r.Handlers.Send.Run(r)
		if r.Error != nil {
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

// AddToUserAgent adds the string to the end of the request's current user agent.
func AddToUserAgent(r *Request, s string) {
	curUA := r.HTTPRequest.Header.Get("User-Agent")
	if len(curUA) > 0 {
		s = curUA + " " + s
	}
	r.HTTPRequest.Header.Set("User-Agent", s)
}
