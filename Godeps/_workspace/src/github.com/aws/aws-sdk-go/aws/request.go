package aws

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/awsutil"
)

// A Request is the service request to be made.
type Request struct {
	*Service
	Handlers     Handlers
	Time         time.Time
	ExpireTime   time.Duration
	Operation    *Operation
	HTTPRequest  *http.Request
	HTTPResponse *http.Response
	Body         io.ReadSeeker
	bodyStart    int64 // offset from beginning of Body that the request body starts
	Params       interface{}
	Error        error
	Data         interface{}
	RequestID    string
	RetryCount   uint
	Retryable    SettableBool
	RetryDelay   time.Duration

	built  bool
	signed bool
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

// NewRequest returns a new Request pointer for the service API
// operation and parameters.
//
// Params is any value of input parameters to be the request payload.
// Data is pointer value to an object which the request's response
// payload will be deserialized to.
func NewRequest(service *Service, operation *Operation, params interface{}, data interface{}) *Request {
	method := operation.HTTPMethod
	if method == "" {
		method = "POST"
	}
	p := operation.HTTPPath
	if p == "" {
		p = "/"
	}

	httpReq, _ := http.NewRequest(method, "", nil)
	httpReq.URL, _ = url.Parse(service.Endpoint + p)

	r := &Request{
		Service:     service,
		Handlers:    service.Handlers.copy(),
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
	return r.Error != nil && r.Retryable.Get() && r.RetryCount < r.Service.MaxRetries()
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
	if r.signed {
		return r.Error
	}

	r.Build()
	if r.Error != nil {
		return r.Error
	}

	r.Handlers.Sign.Run(r)
	r.signed = r.Error != nil
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

		if r.Retryable.Get() {
			// Re-seek the body back to the original point in for a retry so that
			// send will send the body's contents again in the upcoming request.
			r.Body.Seek(r.bodyStart, 0)
		}
		r.Retryable.Reset()

		r.Handlers.Send.Run(r)
		if r.Error != nil {
			r.Handlers.Retry.Run(r)
			r.Handlers.AfterRetry.Run(r)
			if r.Error != nil {
				return r.Error
			}
			continue
		}

		r.Handlers.UnmarshalMeta.Run(r)
		r.Handlers.ValidateResponse.Run(r)
		if r.Error != nil {
			r.Handlers.UnmarshalError.Run(r)
			r.Handlers.Retry.Run(r)
			r.Handlers.AfterRetry.Run(r)
			if r.Error != nil {
				return r.Error
			}
			continue
		}

		r.Handlers.Unmarshal.Run(r)
		if r.Error != nil {
			r.Handlers.Retry.Run(r)
			r.Handlers.AfterRetry.Run(r)
			if r.Error != nil {
				return r.Error
			}
			continue
		}

		break
	}

	return nil
}

// HasNextPage returns true if this request has more pages of data available.
func (r *Request) HasNextPage() bool {
	return r.nextPageTokens() != nil
}

// nextPageTokens returns the tokens to use when asking for the next page of
// data.
func (r *Request) nextPageTokens() []interface{} {
	if r.Operation.Paginator == nil {
		return nil
	}

	if r.Operation.TruncationToken != "" {
		tr := awsutil.ValuesAtAnyPath(r.Data, r.Operation.TruncationToken)
		if tr == nil || len(tr) == 0 {
			return nil
		}
		switch v := tr[0].(type) {
		case bool:
			if v == false {
				return nil
			}
		}
	}

	found := false
	tokens := make([]interface{}, len(r.Operation.OutputTokens))

	for i, outtok := range r.Operation.OutputTokens {
		v := awsutil.ValuesAtAnyPath(r.Data, outtok)
		if v != nil && len(v) > 0 {
			found = true
			tokens[i] = v[0]
		}
	}

	if found {
		return tokens
	}
	return nil
}

// NextPage returns a new Request that can be executed to return the next
// page of result data. Call .Send() on this request to execute it.
func (r *Request) NextPage() *Request {
	tokens := r.nextPageTokens()
	if tokens == nil {
		return nil
	}

	data := reflect.New(reflect.TypeOf(r.Data).Elem()).Interface()
	nr := NewRequest(r.Service, r.Operation, awsutil.CopyOf(r.Params), data)
	for i, intok := range nr.Operation.InputTokens {
		awsutil.SetValueAtAnyPath(nr.Params, intok, tokens[i])
	}
	return nr
}

// EachPage iterates over each page of a paginated request object. The fn
// parameter should be a function with the following sample signature:
//
//   func(page *T, lastPage bool) bool {
//       return true // return false to stop iterating
//   }
//
// Where "T" is the structure type matching the output structure of the given
// operation. For example, a request object generated by
// DynamoDB.ListTablesRequest() would expect to see dynamodb.ListTablesOutput
// as the structure "T". The lastPage value represents whether the page is
// the last page of data or not. The return value of this function should
// return true to keep iterating or false to stop.
func (r *Request) EachPage(fn func(data interface{}, isLastPage bool) (shouldContinue bool)) error {
	for page := r; page != nil; page = page.NextPage() {
		page.Send()
		shouldContinue := fn(page.Data, !page.HasNextPage())
		if page.Error != nil || !shouldContinue {
			return page.Error
		}
	}

	return nil
}
