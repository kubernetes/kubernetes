package restxml_test

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/restxml"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
	"github.com/aws/aws-sdk-go/private/util"
)

var _ bytes.Buffer // always import bytes
var _ http.Request
var _ json.Marshaler
var _ time.Time
var _ xmlutil.XMLNode
var _ xml.Attr
var _ = ioutil.Discard
var _ = util.Trim("")
var _ = url.Values{}
var _ = io.EOF
var _ = aws.String
var _ = fmt.Println
var _ = reflect.Value{}

func init() {
	protocol.RandReader = &awstesting.ZeroReader{}
}

// InputService1ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService1ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService1ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService1ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService1ProtocolTest client from just a session.
//     svc := inputservice1protocoltest.New(mySession)
//
//     // Create a InputService1ProtocolTest client with additional configuration
//     svc := inputservice1protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService1ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService1ProtocolTest {
	c := p.ClientConfig("inputservice1protocoltest", cfgs...)
	return newInputService1ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService1ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService1ProtocolTest {
	svc := &InputService1ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice1protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService1ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService1ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService1TestCaseOperation1 = "OperationName"

// InputService1TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService1TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService1TestCaseOperation1 for more information on using the InputService1TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService1TestCaseOperation1Request method.
//    req, resp := client.InputService1TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1Request(input *InputService1TestShapeInputService1TestCaseOperation2Input) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService1TestShapeInputService1TestCaseOperation2Input{}
	}

	output = &InputService1TestShapeInputService1TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService1TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService1TestCaseOperation1 for usage and error information.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1(input *InputService1TestShapeInputService1TestCaseOperation2Input) (*InputService1TestShapeInputService1TestCaseOperation1Output, error) {
	req, out := c.InputService1TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService1TestCaseOperation1WithContext is the same as InputService1TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService1TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1WithContext(ctx aws.Context, input *InputService1TestShapeInputService1TestCaseOperation2Input, opts ...request.Option) (*InputService1TestShapeInputService1TestCaseOperation1Output, error) {
	req, out := c.InputService1TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService1TestCaseOperation2 = "OperationName"

// InputService1TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService1TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService1TestCaseOperation2 for more information on using the InputService1TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService1TestCaseOperation2Request method.
//    req, resp := client.InputService1TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService1ProtocolTest) InputService1TestCaseOperation2Request(input *InputService1TestShapeInputService1TestCaseOperation2Input) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation2,
		HTTPMethod: "PUT",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService1TestShapeInputService1TestCaseOperation2Input{}
	}

	output = &InputService1TestShapeInputService1TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService1TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService1TestCaseOperation2 for usage and error information.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation2(input *InputService1TestShapeInputService1TestCaseOperation2Input) (*InputService1TestShapeInputService1TestCaseOperation2Output, error) {
	req, out := c.InputService1TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService1TestCaseOperation2WithContext is the same as InputService1TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService1TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation2WithContext(ctx aws.Context, input *InputService1TestShapeInputService1TestCaseOperation2Input, opts ...request.Option) (*InputService1TestShapeInputService1TestCaseOperation2Output, error) {
	req, out := c.InputService1TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService1TestCaseOperation3 = "OperationName"

// InputService1TestCaseOperation3Request generates a "aws/request.Request" representing the
// client's request for the InputService1TestCaseOperation3 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService1TestCaseOperation3 for more information on using the InputService1TestCaseOperation3
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService1TestCaseOperation3Request method.
//    req, resp := client.InputService1TestCaseOperation3Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService1ProtocolTest) InputService1TestCaseOperation3Request(input *InputService1TestShapeInputService1TestCaseOperation3Input) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation3Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation3,
		HTTPMethod: "GET",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService1TestShapeInputService1TestCaseOperation3Input{}
	}

	output = &InputService1TestShapeInputService1TestCaseOperation3Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService1TestCaseOperation3 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService1TestCaseOperation3 for usage and error information.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation3(input *InputService1TestShapeInputService1TestCaseOperation3Input) (*InputService1TestShapeInputService1TestCaseOperation3Output, error) {
	req, out := c.InputService1TestCaseOperation3Request(input)
	return out, req.Send()
}

// InputService1TestCaseOperation3WithContext is the same as InputService1TestCaseOperation3 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService1TestCaseOperation3 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation3WithContext(ctx aws.Context, input *InputService1TestShapeInputService1TestCaseOperation3Input, opts ...request.Option) (*InputService1TestShapeInputService1TestCaseOperation3Output, error) {
	req, out := c.InputService1TestCaseOperation3Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService1TestShapeInputService1TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService1TestShapeInputService1TestCaseOperation2Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Description *string `type:"string"`

	Name *string `type:"string"`
}

// SetDescription sets the Description field's value.
func (s *InputService1TestShapeInputService1TestCaseOperation2Input) SetDescription(v string) *InputService1TestShapeInputService1TestCaseOperation2Input {
	s.Description = &v
	return s
}

// SetName sets the Name field's value.
func (s *InputService1TestShapeInputService1TestCaseOperation2Input) SetName(v string) *InputService1TestShapeInputService1TestCaseOperation2Input {
	s.Name = &v
	return s
}

type InputService1TestShapeInputService1TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService1TestShapeInputService1TestCaseOperation3Input struct {
	_ struct{} `type:"structure"`
}

type InputService1TestShapeInputService1TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

// InputService2ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService2ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService2ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService2ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService2ProtocolTest client from just a session.
//     svc := inputservice2protocoltest.New(mySession)
//
//     // Create a InputService2ProtocolTest client with additional configuration
//     svc := inputservice2protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService2ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService2ProtocolTest {
	c := p.ClientConfig("inputservice2protocoltest", cfgs...)
	return newInputService2ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService2ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService2ProtocolTest {
	svc := &InputService2ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice2protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService2ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService2ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService2TestCaseOperation1 = "OperationName"

// InputService2TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService2TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService2TestCaseOperation1 for more information on using the InputService2TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService2TestCaseOperation1Request method.
//    req, resp := client.InputService2TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService2ProtocolTest) InputService2TestCaseOperation1Request(input *InputService2TestShapeInputService2TestCaseOperation1Input) (req *request.Request, output *InputService2TestShapeInputService2TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService2TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService2TestShapeInputService2TestCaseOperation1Input{}
	}

	output = &InputService2TestShapeInputService2TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService2TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService2TestCaseOperation1 for usage and error information.
func (c *InputService2ProtocolTest) InputService2TestCaseOperation1(input *InputService2TestShapeInputService2TestCaseOperation1Input) (*InputService2TestShapeInputService2TestCaseOperation1Output, error) {
	req, out := c.InputService2TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService2TestCaseOperation1WithContext is the same as InputService2TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService2TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService2ProtocolTest) InputService2TestCaseOperation1WithContext(ctx aws.Context, input *InputService2TestShapeInputService2TestCaseOperation1Input, opts ...request.Option) (*InputService2TestShapeInputService2TestCaseOperation1Output, error) {
	req, out := c.InputService2TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService2TestShapeInputService2TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	First *bool `type:"boolean"`

	Fourth *int64 `type:"integer"`

	Second *bool `type:"boolean"`

	Third *float64 `type:"float"`
}

// SetFirst sets the First field's value.
func (s *InputService2TestShapeInputService2TestCaseOperation1Input) SetFirst(v bool) *InputService2TestShapeInputService2TestCaseOperation1Input {
	s.First = &v
	return s
}

// SetFourth sets the Fourth field's value.
func (s *InputService2TestShapeInputService2TestCaseOperation1Input) SetFourth(v int64) *InputService2TestShapeInputService2TestCaseOperation1Input {
	s.Fourth = &v
	return s
}

// SetSecond sets the Second field's value.
func (s *InputService2TestShapeInputService2TestCaseOperation1Input) SetSecond(v bool) *InputService2TestShapeInputService2TestCaseOperation1Input {
	s.Second = &v
	return s
}

// SetThird sets the Third field's value.
func (s *InputService2TestShapeInputService2TestCaseOperation1Input) SetThird(v float64) *InputService2TestShapeInputService2TestCaseOperation1Input {
	s.Third = &v
	return s
}

type InputService2TestShapeInputService2TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService3ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService3ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService3ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService3ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService3ProtocolTest client from just a session.
//     svc := inputservice3protocoltest.New(mySession)
//
//     // Create a InputService3ProtocolTest client with additional configuration
//     svc := inputservice3protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService3ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService3ProtocolTest {
	c := p.ClientConfig("inputservice3protocoltest", cfgs...)
	return newInputService3ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService3ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService3ProtocolTest {
	svc := &InputService3ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice3protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService3ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService3ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService3TestCaseOperation1 = "OperationName"

// InputService3TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService3TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService3TestCaseOperation1 for more information on using the InputService3TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService3TestCaseOperation1Request method.
//    req, resp := client.InputService3TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService3ProtocolTest) InputService3TestCaseOperation1Request(input *InputService3TestShapeInputService3TestCaseOperation2Input) (req *request.Request, output *InputService3TestShapeInputService3TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService3TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService3TestShapeInputService3TestCaseOperation2Input{}
	}

	output = &InputService3TestShapeInputService3TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService3TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService3TestCaseOperation1 for usage and error information.
func (c *InputService3ProtocolTest) InputService3TestCaseOperation1(input *InputService3TestShapeInputService3TestCaseOperation2Input) (*InputService3TestShapeInputService3TestCaseOperation1Output, error) {
	req, out := c.InputService3TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService3TestCaseOperation1WithContext is the same as InputService3TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService3TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService3ProtocolTest) InputService3TestCaseOperation1WithContext(ctx aws.Context, input *InputService3TestShapeInputService3TestCaseOperation2Input, opts ...request.Option) (*InputService3TestShapeInputService3TestCaseOperation1Output, error) {
	req, out := c.InputService3TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService3TestCaseOperation2 = "OperationName"

// InputService3TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService3TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService3TestCaseOperation2 for more information on using the InputService3TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService3TestCaseOperation2Request method.
//    req, resp := client.InputService3TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService3ProtocolTest) InputService3TestCaseOperation2Request(input *InputService3TestShapeInputService3TestCaseOperation2Input) (req *request.Request, output *InputService3TestShapeInputService3TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService3TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService3TestShapeInputService3TestCaseOperation2Input{}
	}

	output = &InputService3TestShapeInputService3TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService3TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService3TestCaseOperation2 for usage and error information.
func (c *InputService3ProtocolTest) InputService3TestCaseOperation2(input *InputService3TestShapeInputService3TestCaseOperation2Input) (*InputService3TestShapeInputService3TestCaseOperation2Output, error) {
	req, out := c.InputService3TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService3TestCaseOperation2WithContext is the same as InputService3TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService3TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService3ProtocolTest) InputService3TestCaseOperation2WithContext(ctx aws.Context, input *InputService3TestShapeInputService3TestCaseOperation2Input, opts ...request.Option) (*InputService3TestShapeInputService3TestCaseOperation2Output, error) {
	req, out := c.InputService3TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService3TestShapeInputService3TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService3TestShapeInputService3TestCaseOperation2Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Description *string `type:"string"`

	SubStructure *InputService3TestShapeSubStructure `type:"structure"`
}

// SetDescription sets the Description field's value.
func (s *InputService3TestShapeInputService3TestCaseOperation2Input) SetDescription(v string) *InputService3TestShapeInputService3TestCaseOperation2Input {
	s.Description = &v
	return s
}

// SetSubStructure sets the SubStructure field's value.
func (s *InputService3TestShapeInputService3TestCaseOperation2Input) SetSubStructure(v *InputService3TestShapeSubStructure) *InputService3TestShapeInputService3TestCaseOperation2Input {
	s.SubStructure = v
	return s
}

type InputService3TestShapeInputService3TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService3TestShapeSubStructure struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Foo *string `type:"string"`
}

// SetBar sets the Bar field's value.
func (s *InputService3TestShapeSubStructure) SetBar(v string) *InputService3TestShapeSubStructure {
	s.Bar = &v
	return s
}

// SetFoo sets the Foo field's value.
func (s *InputService3TestShapeSubStructure) SetFoo(v string) *InputService3TestShapeSubStructure {
	s.Foo = &v
	return s
}

// InputService4ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService4ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService4ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService4ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService4ProtocolTest client from just a session.
//     svc := inputservice4protocoltest.New(mySession)
//
//     // Create a InputService4ProtocolTest client with additional configuration
//     svc := inputservice4protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService4ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService4ProtocolTest {
	c := p.ClientConfig("inputservice4protocoltest", cfgs...)
	return newInputService4ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService4ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService4ProtocolTest {
	svc := &InputService4ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice4protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService4ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService4ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService4TestCaseOperation1 = "OperationName"

// InputService4TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService4TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService4TestCaseOperation1 for more information on using the InputService4TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService4TestCaseOperation1Request method.
//    req, resp := client.InputService4TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService4ProtocolTest) InputService4TestCaseOperation1Request(input *InputService4TestShapeInputService4TestCaseOperation1Input) (req *request.Request, output *InputService4TestShapeInputService4TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService4TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService4TestShapeInputService4TestCaseOperation1Input{}
	}

	output = &InputService4TestShapeInputService4TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService4TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService4TestCaseOperation1 for usage and error information.
func (c *InputService4ProtocolTest) InputService4TestCaseOperation1(input *InputService4TestShapeInputService4TestCaseOperation1Input) (*InputService4TestShapeInputService4TestCaseOperation1Output, error) {
	req, out := c.InputService4TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService4TestCaseOperation1WithContext is the same as InputService4TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService4TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService4ProtocolTest) InputService4TestCaseOperation1WithContext(ctx aws.Context, input *InputService4TestShapeInputService4TestCaseOperation1Input, opts ...request.Option) (*InputService4TestShapeInputService4TestCaseOperation1Output, error) {
	req, out := c.InputService4TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService4TestShapeInputService4TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Description *string `type:"string"`

	SubStructure *InputService4TestShapeSubStructure `type:"structure"`
}

// SetDescription sets the Description field's value.
func (s *InputService4TestShapeInputService4TestCaseOperation1Input) SetDescription(v string) *InputService4TestShapeInputService4TestCaseOperation1Input {
	s.Description = &v
	return s
}

// SetSubStructure sets the SubStructure field's value.
func (s *InputService4TestShapeInputService4TestCaseOperation1Input) SetSubStructure(v *InputService4TestShapeSubStructure) *InputService4TestShapeInputService4TestCaseOperation1Input {
	s.SubStructure = v
	return s
}

type InputService4TestShapeInputService4TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService4TestShapeSubStructure struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Foo *string `type:"string"`
}

// SetBar sets the Bar field's value.
func (s *InputService4TestShapeSubStructure) SetBar(v string) *InputService4TestShapeSubStructure {
	s.Bar = &v
	return s
}

// SetFoo sets the Foo field's value.
func (s *InputService4TestShapeSubStructure) SetFoo(v string) *InputService4TestShapeSubStructure {
	s.Foo = &v
	return s
}

// InputService5ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService5ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService5ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService5ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService5ProtocolTest client from just a session.
//     svc := inputservice5protocoltest.New(mySession)
//
//     // Create a InputService5ProtocolTest client with additional configuration
//     svc := inputservice5protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService5ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService5ProtocolTest {
	c := p.ClientConfig("inputservice5protocoltest", cfgs...)
	return newInputService5ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService5ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService5ProtocolTest {
	svc := &InputService5ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice5protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService5ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService5ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService5TestCaseOperation1 = "OperationName"

// InputService5TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService5TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService5TestCaseOperation1 for more information on using the InputService5TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService5TestCaseOperation1Request method.
//    req, resp := client.InputService5TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1Request(input *InputService5TestShapeInputService5TestCaseOperation1Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService5TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation1Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService5TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService5TestCaseOperation1 for usage and error information.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1(input *InputService5TestShapeInputService5TestCaseOperation1Input) (*InputService5TestShapeInputService5TestCaseOperation1Output, error) {
	req, out := c.InputService5TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService5TestCaseOperation1WithContext is the same as InputService5TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService5TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation1Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation1Output, error) {
	req, out := c.InputService5TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService5TestShapeInputService5TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `type:"list"`
}

// SetListParam sets the ListParam field's value.
func (s *InputService5TestShapeInputService5TestCaseOperation1Input) SetListParam(v []*string) *InputService5TestShapeInputService5TestCaseOperation1Input {
	s.ListParam = v
	return s
}

type InputService5TestShapeInputService5TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService6ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService6ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService6ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService6ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService6ProtocolTest client from just a session.
//     svc := inputservice6protocoltest.New(mySession)
//
//     // Create a InputService6ProtocolTest client with additional configuration
//     svc := inputservice6protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService6ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService6ProtocolTest {
	c := p.ClientConfig("inputservice6protocoltest", cfgs...)
	return newInputService6ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService6ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService6ProtocolTest {
	svc := &InputService6ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice6protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService6ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService6ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService6TestCaseOperation1 = "OperationName"

// InputService6TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService6TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService6TestCaseOperation1 for more information on using the InputService6TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService6TestCaseOperation1Request method.
//    req, resp := client.InputService6TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService6ProtocolTest) InputService6TestCaseOperation1Request(input *InputService6TestShapeInputService6TestCaseOperation1Input) (req *request.Request, output *InputService6TestShapeInputService6TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService6TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService6TestShapeInputService6TestCaseOperation1Input{}
	}

	output = &InputService6TestShapeInputService6TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService6TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService6TestCaseOperation1 for usage and error information.
func (c *InputService6ProtocolTest) InputService6TestCaseOperation1(input *InputService6TestShapeInputService6TestCaseOperation1Input) (*InputService6TestShapeInputService6TestCaseOperation1Output, error) {
	req, out := c.InputService6TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService6TestCaseOperation1WithContext is the same as InputService6TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService6TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService6ProtocolTest) InputService6TestCaseOperation1WithContext(ctx aws.Context, input *InputService6TestShapeInputService6TestCaseOperation1Input, opts ...request.Option) (*InputService6TestShapeInputService6TestCaseOperation1Output, error) {
	req, out := c.InputService6TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService6TestShapeInputService6TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `locationName:"AlternateName" locationNameList:"NotMember" type:"list"`
}

// SetListParam sets the ListParam field's value.
func (s *InputService6TestShapeInputService6TestCaseOperation1Input) SetListParam(v []*string) *InputService6TestShapeInputService6TestCaseOperation1Input {
	s.ListParam = v
	return s
}

type InputService6TestShapeInputService6TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService7ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService7ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService7ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService7ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService7ProtocolTest client from just a session.
//     svc := inputservice7protocoltest.New(mySession)
//
//     // Create a InputService7ProtocolTest client with additional configuration
//     svc := inputservice7protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService7ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService7ProtocolTest {
	c := p.ClientConfig("inputservice7protocoltest", cfgs...)
	return newInputService7ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService7ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService7ProtocolTest {
	svc := &InputService7ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice7protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService7ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService7ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService7TestCaseOperation1 = "OperationName"

// InputService7TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService7TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService7TestCaseOperation1 for more information on using the InputService7TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService7TestCaseOperation1Request method.
//    req, resp := client.InputService7TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1Request(input *InputService7TestShapeInputService7TestCaseOperation1Input) (req *request.Request, output *InputService7TestShapeInputService7TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService7TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService7TestShapeInputService7TestCaseOperation1Input{}
	}

	output = &InputService7TestShapeInputService7TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService7TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService7TestCaseOperation1 for usage and error information.
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1(input *InputService7TestShapeInputService7TestCaseOperation1Input) (*InputService7TestShapeInputService7TestCaseOperation1Output, error) {
	req, out := c.InputService7TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService7TestCaseOperation1WithContext is the same as InputService7TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService7TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1WithContext(ctx aws.Context, input *InputService7TestShapeInputService7TestCaseOperation1Input, opts ...request.Option) (*InputService7TestShapeInputService7TestCaseOperation1Output, error) {
	req, out := c.InputService7TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService7TestShapeInputService7TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `type:"list" flattened:"true"`
}

// SetListParam sets the ListParam field's value.
func (s *InputService7TestShapeInputService7TestCaseOperation1Input) SetListParam(v []*string) *InputService7TestShapeInputService7TestCaseOperation1Input {
	s.ListParam = v
	return s
}

type InputService7TestShapeInputService7TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService8ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService8ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService8ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService8ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService8ProtocolTest client from just a session.
//     svc := inputservice8protocoltest.New(mySession)
//
//     // Create a InputService8ProtocolTest client with additional configuration
//     svc := inputservice8protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService8ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService8ProtocolTest {
	c := p.ClientConfig("inputservice8protocoltest", cfgs...)
	return newInputService8ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService8ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService8ProtocolTest {
	svc := &InputService8ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice8protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService8ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService8ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService8TestCaseOperation1 = "OperationName"

// InputService8TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService8TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService8TestCaseOperation1 for more information on using the InputService8TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService8TestCaseOperation1Request method.
//    req, resp := client.InputService8TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService8ProtocolTest) InputService8TestCaseOperation1Request(input *InputService8TestShapeInputService8TestCaseOperation1Input) (req *request.Request, output *InputService8TestShapeInputService8TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService8TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService8TestShapeInputService8TestCaseOperation1Input{}
	}

	output = &InputService8TestShapeInputService8TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService8TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService8TestCaseOperation1 for usage and error information.
func (c *InputService8ProtocolTest) InputService8TestCaseOperation1(input *InputService8TestShapeInputService8TestCaseOperation1Input) (*InputService8TestShapeInputService8TestCaseOperation1Output, error) {
	req, out := c.InputService8TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService8TestCaseOperation1WithContext is the same as InputService8TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService8TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService8ProtocolTest) InputService8TestCaseOperation1WithContext(ctx aws.Context, input *InputService8TestShapeInputService8TestCaseOperation1Input, opts ...request.Option) (*InputService8TestShapeInputService8TestCaseOperation1Output, error) {
	req, out := c.InputService8TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService8TestShapeInputService8TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `locationName:"item" type:"list" flattened:"true"`
}

// SetListParam sets the ListParam field's value.
func (s *InputService8TestShapeInputService8TestCaseOperation1Input) SetListParam(v []*string) *InputService8TestShapeInputService8TestCaseOperation1Input {
	s.ListParam = v
	return s
}

type InputService8TestShapeInputService8TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService9ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService9ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService9ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService9ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService9ProtocolTest client from just a session.
//     svc := inputservice9protocoltest.New(mySession)
//
//     // Create a InputService9ProtocolTest client with additional configuration
//     svc := inputservice9protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService9ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService9ProtocolTest {
	c := p.ClientConfig("inputservice9protocoltest", cfgs...)
	return newInputService9ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService9ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService9ProtocolTest {
	svc := &InputService9ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice9protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService9ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService9ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService9TestCaseOperation1 = "OperationName"

// InputService9TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService9TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService9TestCaseOperation1 for more information on using the InputService9TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService9TestCaseOperation1Request method.
//    req, resp := client.InputService9TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService9ProtocolTest) InputService9TestCaseOperation1Request(input *InputService9TestShapeInputService9TestCaseOperation1Input) (req *request.Request, output *InputService9TestShapeInputService9TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService9TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService9TestShapeInputService9TestCaseOperation1Input{}
	}

	output = &InputService9TestShapeInputService9TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService9TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService9TestCaseOperation1 for usage and error information.
func (c *InputService9ProtocolTest) InputService9TestCaseOperation1(input *InputService9TestShapeInputService9TestCaseOperation1Input) (*InputService9TestShapeInputService9TestCaseOperation1Output, error) {
	req, out := c.InputService9TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService9TestCaseOperation1WithContext is the same as InputService9TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService9TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService9ProtocolTest) InputService9TestCaseOperation1WithContext(ctx aws.Context, input *InputService9TestShapeInputService9TestCaseOperation1Input, opts ...request.Option) (*InputService9TestShapeInputService9TestCaseOperation1Output, error) {
	req, out := c.InputService9TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService9TestShapeInputService9TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*InputService9TestShapeSingleFieldStruct `locationName:"item" type:"list" flattened:"true"`
}

// SetListParam sets the ListParam field's value.
func (s *InputService9TestShapeInputService9TestCaseOperation1Input) SetListParam(v []*InputService9TestShapeSingleFieldStruct) *InputService9TestShapeInputService9TestCaseOperation1Input {
	s.ListParam = v
	return s
}

type InputService9TestShapeInputService9TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService9TestShapeSingleFieldStruct struct {
	_ struct{} `type:"structure"`

	Element *string `locationName:"value" type:"string"`
}

// SetElement sets the Element field's value.
func (s *InputService9TestShapeSingleFieldStruct) SetElement(v string) *InputService9TestShapeSingleFieldStruct {
	s.Element = &v
	return s
}

// InputService10ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService10ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService10ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService10ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService10ProtocolTest client from just a session.
//     svc := inputservice10protocoltest.New(mySession)
//
//     // Create a InputService10ProtocolTest client with additional configuration
//     svc := inputservice10protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService10ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService10ProtocolTest {
	c := p.ClientConfig("inputservice10protocoltest", cfgs...)
	return newInputService10ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService10ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService10ProtocolTest {
	svc := &InputService10ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice10protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService10ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService10ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService10TestCaseOperation1 = "OperationName"

// InputService10TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService10TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService10TestCaseOperation1 for more information on using the InputService10TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService10TestCaseOperation1Request method.
//    req, resp := client.InputService10TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService10ProtocolTest) InputService10TestCaseOperation1Request(input *InputService10TestShapeInputService10TestCaseOperation1Input) (req *request.Request, output *InputService10TestShapeInputService10TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService10TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService10TestShapeInputService10TestCaseOperation1Input{}
	}

	output = &InputService10TestShapeInputService10TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService10TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService10TestCaseOperation1 for usage and error information.
func (c *InputService10ProtocolTest) InputService10TestCaseOperation1(input *InputService10TestShapeInputService10TestCaseOperation1Input) (*InputService10TestShapeInputService10TestCaseOperation1Output, error) {
	req, out := c.InputService10TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService10TestCaseOperation1WithContext is the same as InputService10TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService10TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService10ProtocolTest) InputService10TestCaseOperation1WithContext(ctx aws.Context, input *InputService10TestShapeInputService10TestCaseOperation1Input, opts ...request.Option) (*InputService10TestShapeInputService10TestCaseOperation1Output, error) {
	req, out := c.InputService10TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService10TestShapeInputService10TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	StructureParam *InputService10TestShapeStructureShape `type:"structure"`
}

// SetStructureParam sets the StructureParam field's value.
func (s *InputService10TestShapeInputService10TestCaseOperation1Input) SetStructureParam(v *InputService10TestShapeStructureShape) *InputService10TestShapeInputService10TestCaseOperation1Input {
	s.StructureParam = v
	return s
}

type InputService10TestShapeInputService10TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService10TestShapeStructureShape struct {
	_ struct{} `type:"structure"`

	// B is automatically base64 encoded/decoded by the SDK.
	B []byte `locationName:"b" type:"blob"`

	T *time.Time `locationName:"t" type:"timestamp" timestampFormat:"iso8601"`
}

// SetB sets the B field's value.
func (s *InputService10TestShapeStructureShape) SetB(v []byte) *InputService10TestShapeStructureShape {
	s.B = v
	return s
}

// SetT sets the T field's value.
func (s *InputService10TestShapeStructureShape) SetT(v time.Time) *InputService10TestShapeStructureShape {
	s.T = &v
	return s
}

// InputService11ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService11ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService11ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService11ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService11ProtocolTest client from just a session.
//     svc := inputservice11protocoltest.New(mySession)
//
//     // Create a InputService11ProtocolTest client with additional configuration
//     svc := inputservice11protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService11ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService11ProtocolTest {
	c := p.ClientConfig("inputservice11protocoltest", cfgs...)
	return newInputService11ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService11ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService11ProtocolTest {
	svc := &InputService11ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice11protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService11ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService11ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService11TestCaseOperation1 = "OperationName"

// InputService11TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService11TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService11TestCaseOperation1 for more information on using the InputService11TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService11TestCaseOperation1Request method.
//    req, resp := client.InputService11TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService11ProtocolTest) InputService11TestCaseOperation1Request(input *InputService11TestShapeInputService11TestCaseOperation1Input) (req *request.Request, output *InputService11TestShapeInputService11TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService11TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService11TestShapeInputService11TestCaseOperation1Input{}
	}

	output = &InputService11TestShapeInputService11TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService11TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService11TestCaseOperation1 for usage and error information.
func (c *InputService11ProtocolTest) InputService11TestCaseOperation1(input *InputService11TestShapeInputService11TestCaseOperation1Input) (*InputService11TestShapeInputService11TestCaseOperation1Output, error) {
	req, out := c.InputService11TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService11TestCaseOperation1WithContext is the same as InputService11TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService11TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService11ProtocolTest) InputService11TestCaseOperation1WithContext(ctx aws.Context, input *InputService11TestShapeInputService11TestCaseOperation1Input, opts ...request.Option) (*InputService11TestShapeInputService11TestCaseOperation1Output, error) {
	req, out := c.InputService11TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService11TestShapeInputService11TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Foo map[string]*string `location:"headers" locationName:"x-foo-" type:"map"`
}

// SetFoo sets the Foo field's value.
func (s *InputService11TestShapeInputService11TestCaseOperation1Input) SetFoo(v map[string]*string) *InputService11TestShapeInputService11TestCaseOperation1Input {
	s.Foo = v
	return s
}

type InputService11TestShapeInputService11TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService12ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService12ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService12ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService12ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService12ProtocolTest client from just a session.
//     svc := inputservice12protocoltest.New(mySession)
//
//     // Create a InputService12ProtocolTest client with additional configuration
//     svc := inputservice12protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService12ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService12ProtocolTest {
	c := p.ClientConfig("inputservice12protocoltest", cfgs...)
	return newInputService12ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService12ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService12ProtocolTest {
	svc := &InputService12ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice12protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService12ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService12ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService12TestCaseOperation1 = "OperationName"

// InputService12TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService12TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService12TestCaseOperation1 for more information on using the InputService12TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService12TestCaseOperation1Request method.
//    req, resp := client.InputService12TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService12ProtocolTest) InputService12TestCaseOperation1Request(input *InputService12TestShapeInputService12TestCaseOperation1Input) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService12TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService12TestShapeInputService12TestCaseOperation1Input{}
	}

	output = &InputService12TestShapeInputService12TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService12TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService12TestCaseOperation1 for usage and error information.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation1(input *InputService12TestShapeInputService12TestCaseOperation1Input) (*InputService12TestShapeInputService12TestCaseOperation1Output, error) {
	req, out := c.InputService12TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService12TestCaseOperation1WithContext is the same as InputService12TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService12TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation1WithContext(ctx aws.Context, input *InputService12TestShapeInputService12TestCaseOperation1Input, opts ...request.Option) (*InputService12TestShapeInputService12TestCaseOperation1Output, error) {
	req, out := c.InputService12TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService12TestShapeInputService12TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	Items []*string `location:"querystring" locationName:"item" type:"list"`
}

// SetItems sets the Items field's value.
func (s *InputService12TestShapeInputService12TestCaseOperation1Input) SetItems(v []*string) *InputService12TestShapeInputService12TestCaseOperation1Input {
	s.Items = v
	return s
}

type InputService12TestShapeInputService12TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService13ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService13ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService13ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService13ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService13ProtocolTest client from just a session.
//     svc := inputservice13protocoltest.New(mySession)
//
//     // Create a InputService13ProtocolTest client with additional configuration
//     svc := inputservice13protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService13ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService13ProtocolTest {
	c := p.ClientConfig("inputservice13protocoltest", cfgs...)
	return newInputService13ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService13ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService13ProtocolTest {
	svc := &InputService13ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice13protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService13ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService13ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService13TestCaseOperation1 = "OperationName"

// InputService13TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService13TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService13TestCaseOperation1 for more information on using the InputService13TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService13TestCaseOperation1Request method.
//    req, resp := client.InputService13TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService13ProtocolTest) InputService13TestCaseOperation1Request(input *InputService13TestShapeInputService13TestCaseOperation1Input) (req *request.Request, output *InputService13TestShapeInputService13TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService13TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/2014-01-01/jobsByPipeline/{PipelineId}",
	}

	if input == nil {
		input = &InputService13TestShapeInputService13TestCaseOperation1Input{}
	}

	output = &InputService13TestShapeInputService13TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService13TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService13TestCaseOperation1 for usage and error information.
func (c *InputService13ProtocolTest) InputService13TestCaseOperation1(input *InputService13TestShapeInputService13TestCaseOperation1Input) (*InputService13TestShapeInputService13TestCaseOperation1Output, error) {
	req, out := c.InputService13TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService13TestCaseOperation1WithContext is the same as InputService13TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService13TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService13ProtocolTest) InputService13TestCaseOperation1WithContext(ctx aws.Context, input *InputService13TestShapeInputService13TestCaseOperation1Input, opts ...request.Option) (*InputService13TestShapeInputService13TestCaseOperation1Output, error) {
	req, out := c.InputService13TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService13TestShapeInputService13TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	PipelineId *string `location:"uri" type:"string"`

	QueryDoc map[string]*string `location:"querystring" type:"map"`
}

// SetPipelineId sets the PipelineId field's value.
func (s *InputService13TestShapeInputService13TestCaseOperation1Input) SetPipelineId(v string) *InputService13TestShapeInputService13TestCaseOperation1Input {
	s.PipelineId = &v
	return s
}

// SetQueryDoc sets the QueryDoc field's value.
func (s *InputService13TestShapeInputService13TestCaseOperation1Input) SetQueryDoc(v map[string]*string) *InputService13TestShapeInputService13TestCaseOperation1Input {
	s.QueryDoc = v
	return s
}

type InputService13TestShapeInputService13TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService14ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService14ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService14ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService14ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService14ProtocolTest client from just a session.
//     svc := inputservice14protocoltest.New(mySession)
//
//     // Create a InputService14ProtocolTest client with additional configuration
//     svc := inputservice14protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService14ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService14ProtocolTest {
	c := p.ClientConfig("inputservice14protocoltest", cfgs...)
	return newInputService14ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService14ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService14ProtocolTest {
	svc := &InputService14ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice14protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService14ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService14ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService14TestCaseOperation1 = "OperationName"

// InputService14TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService14TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService14TestCaseOperation1 for more information on using the InputService14TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService14TestCaseOperation1Request method.
//    req, resp := client.InputService14TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService14ProtocolTest) InputService14TestCaseOperation1Request(input *InputService14TestShapeInputService14TestCaseOperation1Input) (req *request.Request, output *InputService14TestShapeInputService14TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService14TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/2014-01-01/jobsByPipeline/{PipelineId}",
	}

	if input == nil {
		input = &InputService14TestShapeInputService14TestCaseOperation1Input{}
	}

	output = &InputService14TestShapeInputService14TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService14TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService14TestCaseOperation1 for usage and error information.
func (c *InputService14ProtocolTest) InputService14TestCaseOperation1(input *InputService14TestShapeInputService14TestCaseOperation1Input) (*InputService14TestShapeInputService14TestCaseOperation1Output, error) {
	req, out := c.InputService14TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService14TestCaseOperation1WithContext is the same as InputService14TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService14TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService14ProtocolTest) InputService14TestCaseOperation1WithContext(ctx aws.Context, input *InputService14TestShapeInputService14TestCaseOperation1Input, opts ...request.Option) (*InputService14TestShapeInputService14TestCaseOperation1Output, error) {
	req, out := c.InputService14TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService14TestShapeInputService14TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	PipelineId *string `location:"uri" type:"string"`

	QueryDoc map[string][]*string `location:"querystring" type:"map"`
}

// SetPipelineId sets the PipelineId field's value.
func (s *InputService14TestShapeInputService14TestCaseOperation1Input) SetPipelineId(v string) *InputService14TestShapeInputService14TestCaseOperation1Input {
	s.PipelineId = &v
	return s
}

// SetQueryDoc sets the QueryDoc field's value.
func (s *InputService14TestShapeInputService14TestCaseOperation1Input) SetQueryDoc(v map[string][]*string) *InputService14TestShapeInputService14TestCaseOperation1Input {
	s.QueryDoc = v
	return s
}

type InputService14TestShapeInputService14TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService15ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService15ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService15ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService15ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService15ProtocolTest client from just a session.
//     svc := inputservice15protocoltest.New(mySession)
//
//     // Create a InputService15ProtocolTest client with additional configuration
//     svc := inputservice15protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService15ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService15ProtocolTest {
	c := p.ClientConfig("inputservice15protocoltest", cfgs...)
	return newInputService15ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService15ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService15ProtocolTest {
	svc := &InputService15ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice15protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService15ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService15ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService15TestCaseOperation1 = "OperationName"

// InputService15TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService15TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService15TestCaseOperation1 for more information on using the InputService15TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService15TestCaseOperation1Request method.
//    req, resp := client.InputService15TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService15ProtocolTest) InputService15TestCaseOperation1Request(input *InputService15TestShapeInputService15TestCaseOperation2Input) (req *request.Request, output *InputService15TestShapeInputService15TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService15TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService15TestShapeInputService15TestCaseOperation2Input{}
	}

	output = &InputService15TestShapeInputService15TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService15TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService15TestCaseOperation1 for usage and error information.
func (c *InputService15ProtocolTest) InputService15TestCaseOperation1(input *InputService15TestShapeInputService15TestCaseOperation2Input) (*InputService15TestShapeInputService15TestCaseOperation1Output, error) {
	req, out := c.InputService15TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService15TestCaseOperation1WithContext is the same as InputService15TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService15TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService15ProtocolTest) InputService15TestCaseOperation1WithContext(ctx aws.Context, input *InputService15TestShapeInputService15TestCaseOperation2Input, opts ...request.Option) (*InputService15TestShapeInputService15TestCaseOperation1Output, error) {
	req, out := c.InputService15TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService15TestCaseOperation2 = "OperationName"

// InputService15TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService15TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService15TestCaseOperation2 for more information on using the InputService15TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService15TestCaseOperation2Request method.
//    req, resp := client.InputService15TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService15ProtocolTest) InputService15TestCaseOperation2Request(input *InputService15TestShapeInputService15TestCaseOperation2Input) (req *request.Request, output *InputService15TestShapeInputService15TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService15TestCaseOperation2,
		HTTPMethod: "GET",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService15TestShapeInputService15TestCaseOperation2Input{}
	}

	output = &InputService15TestShapeInputService15TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService15TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService15TestCaseOperation2 for usage and error information.
func (c *InputService15ProtocolTest) InputService15TestCaseOperation2(input *InputService15TestShapeInputService15TestCaseOperation2Input) (*InputService15TestShapeInputService15TestCaseOperation2Output, error) {
	req, out := c.InputService15TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService15TestCaseOperation2WithContext is the same as InputService15TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService15TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService15ProtocolTest) InputService15TestCaseOperation2WithContext(ctx aws.Context, input *InputService15TestShapeInputService15TestCaseOperation2Input, opts ...request.Option) (*InputService15TestShapeInputService15TestCaseOperation2Output, error) {
	req, out := c.InputService15TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService15TestShapeInputService15TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService15TestShapeInputService15TestCaseOperation2Input struct {
	_ struct{} `type:"structure"`

	BoolQuery *bool `location:"querystring" locationName:"bool-query" type:"boolean"`
}

// SetBoolQuery sets the BoolQuery field's value.
func (s *InputService15TestShapeInputService15TestCaseOperation2Input) SetBoolQuery(v bool) *InputService15TestShapeInputService15TestCaseOperation2Input {
	s.BoolQuery = &v
	return s
}

type InputService15TestShapeInputService15TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

// InputService16ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService16ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService16ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService16ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService16ProtocolTest client from just a session.
//     svc := inputservice16protocoltest.New(mySession)
//
//     // Create a InputService16ProtocolTest client with additional configuration
//     svc := inputservice16protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService16ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService16ProtocolTest {
	c := p.ClientConfig("inputservice16protocoltest", cfgs...)
	return newInputService16ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService16ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService16ProtocolTest {
	svc := &InputService16ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice16protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService16ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService16ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService16TestCaseOperation1 = "OperationName"

// InputService16TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService16TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService16TestCaseOperation1 for more information on using the InputService16TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService16TestCaseOperation1Request method.
//    req, resp := client.InputService16TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService16ProtocolTest) InputService16TestCaseOperation1Request(input *InputService16TestShapeInputService16TestCaseOperation1Input) (req *request.Request, output *InputService16TestShapeInputService16TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService16TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService16TestShapeInputService16TestCaseOperation1Input{}
	}

	output = &InputService16TestShapeInputService16TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService16TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService16TestCaseOperation1 for usage and error information.
func (c *InputService16ProtocolTest) InputService16TestCaseOperation1(input *InputService16TestShapeInputService16TestCaseOperation1Input) (*InputService16TestShapeInputService16TestCaseOperation1Output, error) {
	req, out := c.InputService16TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService16TestCaseOperation1WithContext is the same as InputService16TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService16TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService16ProtocolTest) InputService16TestCaseOperation1WithContext(ctx aws.Context, input *InputService16TestShapeInputService16TestCaseOperation1Input, opts ...request.Option) (*InputService16TestShapeInputService16TestCaseOperation1Output, error) {
	req, out := c.InputService16TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService16TestShapeInputService16TestCaseOperation1Input struct {
	_ struct{} `type:"structure" payload:"Foo"`

	Foo *string `locationName:"foo" type:"string"`
}

// SetFoo sets the Foo field's value.
func (s *InputService16TestShapeInputService16TestCaseOperation1Input) SetFoo(v string) *InputService16TestShapeInputService16TestCaseOperation1Input {
	s.Foo = &v
	return s
}

type InputService16TestShapeInputService16TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService17ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService17ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService17ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService17ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService17ProtocolTest client from just a session.
//     svc := inputservice17protocoltest.New(mySession)
//
//     // Create a InputService17ProtocolTest client with additional configuration
//     svc := inputservice17protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService17ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService17ProtocolTest {
	c := p.ClientConfig("inputservice17protocoltest", cfgs...)
	return newInputService17ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService17ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService17ProtocolTest {
	svc := &InputService17ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice17protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService17ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService17ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService17TestCaseOperation1 = "OperationName"

// InputService17TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService17TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService17TestCaseOperation1 for more information on using the InputService17TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService17TestCaseOperation1Request method.
//    req, resp := client.InputService17TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService17ProtocolTest) InputService17TestCaseOperation1Request(input *InputService17TestShapeInputService17TestCaseOperation2Input) (req *request.Request, output *InputService17TestShapeInputService17TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService17TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService17TestShapeInputService17TestCaseOperation2Input{}
	}

	output = &InputService17TestShapeInputService17TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService17TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService17TestCaseOperation1 for usage and error information.
func (c *InputService17ProtocolTest) InputService17TestCaseOperation1(input *InputService17TestShapeInputService17TestCaseOperation2Input) (*InputService17TestShapeInputService17TestCaseOperation1Output, error) {
	req, out := c.InputService17TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService17TestCaseOperation1WithContext is the same as InputService17TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService17TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService17ProtocolTest) InputService17TestCaseOperation1WithContext(ctx aws.Context, input *InputService17TestShapeInputService17TestCaseOperation2Input, opts ...request.Option) (*InputService17TestShapeInputService17TestCaseOperation1Output, error) {
	req, out := c.InputService17TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService17TestCaseOperation2 = "OperationName"

// InputService17TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService17TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService17TestCaseOperation2 for more information on using the InputService17TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService17TestCaseOperation2Request method.
//    req, resp := client.InputService17TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService17ProtocolTest) InputService17TestCaseOperation2Request(input *InputService17TestShapeInputService17TestCaseOperation2Input) (req *request.Request, output *InputService17TestShapeInputService17TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService17TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService17TestShapeInputService17TestCaseOperation2Input{}
	}

	output = &InputService17TestShapeInputService17TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService17TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService17TestCaseOperation2 for usage and error information.
func (c *InputService17ProtocolTest) InputService17TestCaseOperation2(input *InputService17TestShapeInputService17TestCaseOperation2Input) (*InputService17TestShapeInputService17TestCaseOperation2Output, error) {
	req, out := c.InputService17TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService17TestCaseOperation2WithContext is the same as InputService17TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService17TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService17ProtocolTest) InputService17TestCaseOperation2WithContext(ctx aws.Context, input *InputService17TestShapeInputService17TestCaseOperation2Input, opts ...request.Option) (*InputService17TestShapeInputService17TestCaseOperation2Output, error) {
	req, out := c.InputService17TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService17TestShapeInputService17TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService17TestShapeInputService17TestCaseOperation2Input struct {
	_ struct{} `type:"structure" payload:"Foo"`

	Foo []byte `locationName:"foo" type:"blob"`
}

// SetFoo sets the Foo field's value.
func (s *InputService17TestShapeInputService17TestCaseOperation2Input) SetFoo(v []byte) *InputService17TestShapeInputService17TestCaseOperation2Input {
	s.Foo = v
	return s
}

type InputService17TestShapeInputService17TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

// InputService18ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService18ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService18ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService18ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService18ProtocolTest client from just a session.
//     svc := inputservice18protocoltest.New(mySession)
//
//     // Create a InputService18ProtocolTest client with additional configuration
//     svc := inputservice18protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService18ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService18ProtocolTest {
	c := p.ClientConfig("inputservice18protocoltest", cfgs...)
	return newInputService18ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService18ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService18ProtocolTest {
	svc := &InputService18ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice18protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService18ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService18ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService18TestCaseOperation1 = "OperationName"

// InputService18TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService18TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService18TestCaseOperation1 for more information on using the InputService18TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService18TestCaseOperation1Request method.
//    req, resp := client.InputService18TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService18ProtocolTest) InputService18TestCaseOperation1Request(input *InputService18TestShapeInputService18TestCaseOperation4Input) (req *request.Request, output *InputService18TestShapeInputService18TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService18TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService18TestShapeInputService18TestCaseOperation4Input{}
	}

	output = &InputService18TestShapeInputService18TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService18TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService18TestCaseOperation1 for usage and error information.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation1(input *InputService18TestShapeInputService18TestCaseOperation4Input) (*InputService18TestShapeInputService18TestCaseOperation1Output, error) {
	req, out := c.InputService18TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService18TestCaseOperation1WithContext is the same as InputService18TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService18TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation1WithContext(ctx aws.Context, input *InputService18TestShapeInputService18TestCaseOperation4Input, opts ...request.Option) (*InputService18TestShapeInputService18TestCaseOperation1Output, error) {
	req, out := c.InputService18TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService18TestCaseOperation2 = "OperationName"

// InputService18TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService18TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService18TestCaseOperation2 for more information on using the InputService18TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService18TestCaseOperation2Request method.
//    req, resp := client.InputService18TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService18ProtocolTest) InputService18TestCaseOperation2Request(input *InputService18TestShapeInputService18TestCaseOperation4Input) (req *request.Request, output *InputService18TestShapeInputService18TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService18TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService18TestShapeInputService18TestCaseOperation4Input{}
	}

	output = &InputService18TestShapeInputService18TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService18TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService18TestCaseOperation2 for usage and error information.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation2(input *InputService18TestShapeInputService18TestCaseOperation4Input) (*InputService18TestShapeInputService18TestCaseOperation2Output, error) {
	req, out := c.InputService18TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService18TestCaseOperation2WithContext is the same as InputService18TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService18TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation2WithContext(ctx aws.Context, input *InputService18TestShapeInputService18TestCaseOperation4Input, opts ...request.Option) (*InputService18TestShapeInputService18TestCaseOperation2Output, error) {
	req, out := c.InputService18TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService18TestCaseOperation3 = "OperationName"

// InputService18TestCaseOperation3Request generates a "aws/request.Request" representing the
// client's request for the InputService18TestCaseOperation3 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService18TestCaseOperation3 for more information on using the InputService18TestCaseOperation3
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService18TestCaseOperation3Request method.
//    req, resp := client.InputService18TestCaseOperation3Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService18ProtocolTest) InputService18TestCaseOperation3Request(input *InputService18TestShapeInputService18TestCaseOperation4Input) (req *request.Request, output *InputService18TestShapeInputService18TestCaseOperation3Output) {
	op := &request.Operation{
		Name:       opInputService18TestCaseOperation3,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService18TestShapeInputService18TestCaseOperation4Input{}
	}

	output = &InputService18TestShapeInputService18TestCaseOperation3Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService18TestCaseOperation3 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService18TestCaseOperation3 for usage and error information.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation3(input *InputService18TestShapeInputService18TestCaseOperation4Input) (*InputService18TestShapeInputService18TestCaseOperation3Output, error) {
	req, out := c.InputService18TestCaseOperation3Request(input)
	return out, req.Send()
}

// InputService18TestCaseOperation3WithContext is the same as InputService18TestCaseOperation3 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService18TestCaseOperation3 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation3WithContext(ctx aws.Context, input *InputService18TestShapeInputService18TestCaseOperation4Input, opts ...request.Option) (*InputService18TestShapeInputService18TestCaseOperation3Output, error) {
	req, out := c.InputService18TestCaseOperation3Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService18TestCaseOperation4 = "OperationName"

// InputService18TestCaseOperation4Request generates a "aws/request.Request" representing the
// client's request for the InputService18TestCaseOperation4 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService18TestCaseOperation4 for more information on using the InputService18TestCaseOperation4
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService18TestCaseOperation4Request method.
//    req, resp := client.InputService18TestCaseOperation4Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService18ProtocolTest) InputService18TestCaseOperation4Request(input *InputService18TestShapeInputService18TestCaseOperation4Input) (req *request.Request, output *InputService18TestShapeInputService18TestCaseOperation4Output) {
	op := &request.Operation{
		Name:       opInputService18TestCaseOperation4,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService18TestShapeInputService18TestCaseOperation4Input{}
	}

	output = &InputService18TestShapeInputService18TestCaseOperation4Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService18TestCaseOperation4 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService18TestCaseOperation4 for usage and error information.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation4(input *InputService18TestShapeInputService18TestCaseOperation4Input) (*InputService18TestShapeInputService18TestCaseOperation4Output, error) {
	req, out := c.InputService18TestCaseOperation4Request(input)
	return out, req.Send()
}

// InputService18TestCaseOperation4WithContext is the same as InputService18TestCaseOperation4 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService18TestCaseOperation4 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation4WithContext(ctx aws.Context, input *InputService18TestShapeInputService18TestCaseOperation4Input, opts ...request.Option) (*InputService18TestShapeInputService18TestCaseOperation4Output, error) {
	req, out := c.InputService18TestCaseOperation4Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService18TestShapeFooShape struct {
	_ struct{} `locationName:"foo" type:"structure"`

	Baz *string `locationName:"baz" type:"string"`
}

// SetBaz sets the Baz field's value.
func (s *InputService18TestShapeFooShape) SetBaz(v string) *InputService18TestShapeFooShape {
	s.Baz = &v
	return s
}

type InputService18TestShapeInputService18TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService18TestShapeInputService18TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService18TestShapeInputService18TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService18TestShapeInputService18TestCaseOperation4Input struct {
	_ struct{} `type:"structure" payload:"Foo"`

	Foo *InputService18TestShapeFooShape `locationName:"foo" type:"structure"`
}

// SetFoo sets the Foo field's value.
func (s *InputService18TestShapeInputService18TestCaseOperation4Input) SetFoo(v *InputService18TestShapeFooShape) *InputService18TestShapeInputService18TestCaseOperation4Input {
	s.Foo = v
	return s
}

type InputService18TestShapeInputService18TestCaseOperation4Output struct {
	_ struct{} `type:"structure"`
}

// InputService19ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService19ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService19ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService19ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService19ProtocolTest client from just a session.
//     svc := inputservice19protocoltest.New(mySession)
//
//     // Create a InputService19ProtocolTest client with additional configuration
//     svc := inputservice19protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService19ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService19ProtocolTest {
	c := p.ClientConfig("inputservice19protocoltest", cfgs...)
	return newInputService19ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService19ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService19ProtocolTest {
	svc := &InputService19ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice19protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService19ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService19ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService19TestCaseOperation1 = "OperationName"

// InputService19TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService19TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService19TestCaseOperation1 for more information on using the InputService19TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService19TestCaseOperation1Request method.
//    req, resp := client.InputService19TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService19ProtocolTest) InputService19TestCaseOperation1Request(input *InputService19TestShapeInputService19TestCaseOperation1Input) (req *request.Request, output *InputService19TestShapeInputService19TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService19TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService19TestShapeInputService19TestCaseOperation1Input{}
	}

	output = &InputService19TestShapeInputService19TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService19TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService19TestCaseOperation1 for usage and error information.
func (c *InputService19ProtocolTest) InputService19TestCaseOperation1(input *InputService19TestShapeInputService19TestCaseOperation1Input) (*InputService19TestShapeInputService19TestCaseOperation1Output, error) {
	req, out := c.InputService19TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService19TestCaseOperation1WithContext is the same as InputService19TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService19TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService19ProtocolTest) InputService19TestCaseOperation1WithContext(ctx aws.Context, input *InputService19TestShapeInputService19TestCaseOperation1Input, opts ...request.Option) (*InputService19TestShapeInputService19TestCaseOperation1Output, error) {
	req, out := c.InputService19TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService19TestShapeGrant struct {
	_ struct{} `locationName:"Grant" type:"structure"`

	Grantee *InputService19TestShapeGrantee `type:"structure" xmlPrefix:"xsi" xmlURI:"http://www.w3.org/2001/XMLSchema-instance"`
}

// SetGrantee sets the Grantee field's value.
func (s *InputService19TestShapeGrant) SetGrantee(v *InputService19TestShapeGrantee) *InputService19TestShapeGrant {
	s.Grantee = v
	return s
}

type InputService19TestShapeGrantee struct {
	_ struct{} `type:"structure" xmlPrefix:"xsi" xmlURI:"http://www.w3.org/2001/XMLSchema-instance"`

	EmailAddress *string `type:"string"`

	Type *string `locationName:"xsi:type" type:"string" xmlAttribute:"true"`
}

// SetEmailAddress sets the EmailAddress field's value.
func (s *InputService19TestShapeGrantee) SetEmailAddress(v string) *InputService19TestShapeGrantee {
	s.EmailAddress = &v
	return s
}

// SetType sets the Type field's value.
func (s *InputService19TestShapeGrantee) SetType(v string) *InputService19TestShapeGrantee {
	s.Type = &v
	return s
}

type InputService19TestShapeInputService19TestCaseOperation1Input struct {
	_ struct{} `type:"structure" payload:"Grant"`

	Grant *InputService19TestShapeGrant `locationName:"Grant" type:"structure"`
}

// SetGrant sets the Grant field's value.
func (s *InputService19TestShapeInputService19TestCaseOperation1Input) SetGrant(v *InputService19TestShapeGrant) *InputService19TestShapeInputService19TestCaseOperation1Input {
	s.Grant = v
	return s
}

type InputService19TestShapeInputService19TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService20ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService20ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService20ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService20ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService20ProtocolTest client from just a session.
//     svc := inputservice20protocoltest.New(mySession)
//
//     // Create a InputService20ProtocolTest client with additional configuration
//     svc := inputservice20protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService20ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService20ProtocolTest {
	c := p.ClientConfig("inputservice20protocoltest", cfgs...)
	return newInputService20ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService20ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService20ProtocolTest {
	svc := &InputService20ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice20protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService20ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService20ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService20TestCaseOperation1 = "OperationName"

// InputService20TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService20TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService20TestCaseOperation1 for more information on using the InputService20TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService20TestCaseOperation1Request method.
//    req, resp := client.InputService20TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService20ProtocolTest) InputService20TestCaseOperation1Request(input *InputService20TestShapeInputService20TestCaseOperation1Input) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/{Bucket}/{Key+}",
	}

	if input == nil {
		input = &InputService20TestShapeInputService20TestCaseOperation1Input{}
	}

	output = &InputService20TestShapeInputService20TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService20TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService20TestCaseOperation1 for usage and error information.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation1(input *InputService20TestShapeInputService20TestCaseOperation1Input) (*InputService20TestShapeInputService20TestCaseOperation1Output, error) {
	req, out := c.InputService20TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService20TestCaseOperation1WithContext is the same as InputService20TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService20TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation1WithContext(ctx aws.Context, input *InputService20TestShapeInputService20TestCaseOperation1Input, opts ...request.Option) (*InputService20TestShapeInputService20TestCaseOperation1Output, error) {
	req, out := c.InputService20TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService20TestShapeInputService20TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	Bucket *string `location:"uri" type:"string"`

	Key *string `location:"uri" type:"string"`
}

// SetBucket sets the Bucket field's value.
func (s *InputService20TestShapeInputService20TestCaseOperation1Input) SetBucket(v string) *InputService20TestShapeInputService20TestCaseOperation1Input {
	s.Bucket = &v
	return s
}

// SetKey sets the Key field's value.
func (s *InputService20TestShapeInputService20TestCaseOperation1Input) SetKey(v string) *InputService20TestShapeInputService20TestCaseOperation1Input {
	s.Key = &v
	return s
}

type InputService20TestShapeInputService20TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService21ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService21ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService21ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService21ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService21ProtocolTest client from just a session.
//     svc := inputservice21protocoltest.New(mySession)
//
//     // Create a InputService21ProtocolTest client with additional configuration
//     svc := inputservice21protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService21ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService21ProtocolTest {
	c := p.ClientConfig("inputservice21protocoltest", cfgs...)
	return newInputService21ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService21ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService21ProtocolTest {
	svc := &InputService21ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice21protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService21ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService21ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService21TestCaseOperation1 = "OperationName"

// InputService21TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService21TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService21TestCaseOperation1 for more information on using the InputService21TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService21TestCaseOperation1Request method.
//    req, resp := client.InputService21TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService21ProtocolTest) InputService21TestCaseOperation1Request(input *InputService21TestShapeInputService21TestCaseOperation2Input) (req *request.Request, output *InputService21TestShapeInputService21TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService21TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService21TestShapeInputService21TestCaseOperation2Input{}
	}

	output = &InputService21TestShapeInputService21TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService21TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService21TestCaseOperation1 for usage and error information.
func (c *InputService21ProtocolTest) InputService21TestCaseOperation1(input *InputService21TestShapeInputService21TestCaseOperation2Input) (*InputService21TestShapeInputService21TestCaseOperation1Output, error) {
	req, out := c.InputService21TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService21TestCaseOperation1WithContext is the same as InputService21TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService21TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService21ProtocolTest) InputService21TestCaseOperation1WithContext(ctx aws.Context, input *InputService21TestShapeInputService21TestCaseOperation2Input, opts ...request.Option) (*InputService21TestShapeInputService21TestCaseOperation1Output, error) {
	req, out := c.InputService21TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService21TestCaseOperation2 = "OperationName"

// InputService21TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService21TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService21TestCaseOperation2 for more information on using the InputService21TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService21TestCaseOperation2Request method.
//    req, resp := client.InputService21TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService21ProtocolTest) InputService21TestCaseOperation2Request(input *InputService21TestShapeInputService21TestCaseOperation2Input) (req *request.Request, output *InputService21TestShapeInputService21TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService21TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/path?abc=mno",
	}

	if input == nil {
		input = &InputService21TestShapeInputService21TestCaseOperation2Input{}
	}

	output = &InputService21TestShapeInputService21TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService21TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService21TestCaseOperation2 for usage and error information.
func (c *InputService21ProtocolTest) InputService21TestCaseOperation2(input *InputService21TestShapeInputService21TestCaseOperation2Input) (*InputService21TestShapeInputService21TestCaseOperation2Output, error) {
	req, out := c.InputService21TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService21TestCaseOperation2WithContext is the same as InputService21TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService21TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService21ProtocolTest) InputService21TestCaseOperation2WithContext(ctx aws.Context, input *InputService21TestShapeInputService21TestCaseOperation2Input, opts ...request.Option) (*InputService21TestShapeInputService21TestCaseOperation2Output, error) {
	req, out := c.InputService21TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService21TestShapeInputService21TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService21TestShapeInputService21TestCaseOperation2Input struct {
	_ struct{} `type:"structure"`

	Foo *string `location:"querystring" locationName:"param-name" type:"string"`
}

// SetFoo sets the Foo field's value.
func (s *InputService21TestShapeInputService21TestCaseOperation2Input) SetFoo(v string) *InputService21TestShapeInputService21TestCaseOperation2Input {
	s.Foo = &v
	return s
}

type InputService21TestShapeInputService21TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

// InputService22ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService22ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService22ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService22ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService22ProtocolTest client from just a session.
//     svc := inputservice22protocoltest.New(mySession)
//
//     // Create a InputService22ProtocolTest client with additional configuration
//     svc := inputservice22protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService22ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService22ProtocolTest {
	c := p.ClientConfig("inputservice22protocoltest", cfgs...)
	return newInputService22ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService22ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService22ProtocolTest {
	svc := &InputService22ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice22protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService22ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService22ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService22TestCaseOperation1 = "OperationName"

// InputService22TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService22TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService22TestCaseOperation1 for more information on using the InputService22TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService22TestCaseOperation1Request method.
//    req, resp := client.InputService22TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService22ProtocolTest) InputService22TestCaseOperation1Request(input *InputService22TestShapeInputService22TestCaseOperation6Input) (req *request.Request, output *InputService22TestShapeInputService22TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService22TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService22TestShapeInputService22TestCaseOperation6Input{}
	}

	output = &InputService22TestShapeInputService22TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService22TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService22TestCaseOperation1 for usage and error information.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation1(input *InputService22TestShapeInputService22TestCaseOperation6Input) (*InputService22TestShapeInputService22TestCaseOperation1Output, error) {
	req, out := c.InputService22TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService22TestCaseOperation1WithContext is the same as InputService22TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService22TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation1WithContext(ctx aws.Context, input *InputService22TestShapeInputService22TestCaseOperation6Input, opts ...request.Option) (*InputService22TestShapeInputService22TestCaseOperation1Output, error) {
	req, out := c.InputService22TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService22TestCaseOperation2 = "OperationName"

// InputService22TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService22TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService22TestCaseOperation2 for more information on using the InputService22TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService22TestCaseOperation2Request method.
//    req, resp := client.InputService22TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService22ProtocolTest) InputService22TestCaseOperation2Request(input *InputService22TestShapeInputService22TestCaseOperation6Input) (req *request.Request, output *InputService22TestShapeInputService22TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService22TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService22TestShapeInputService22TestCaseOperation6Input{}
	}

	output = &InputService22TestShapeInputService22TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService22TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService22TestCaseOperation2 for usage and error information.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation2(input *InputService22TestShapeInputService22TestCaseOperation6Input) (*InputService22TestShapeInputService22TestCaseOperation2Output, error) {
	req, out := c.InputService22TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService22TestCaseOperation2WithContext is the same as InputService22TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService22TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation2WithContext(ctx aws.Context, input *InputService22TestShapeInputService22TestCaseOperation6Input, opts ...request.Option) (*InputService22TestShapeInputService22TestCaseOperation2Output, error) {
	req, out := c.InputService22TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService22TestCaseOperation3 = "OperationName"

// InputService22TestCaseOperation3Request generates a "aws/request.Request" representing the
// client's request for the InputService22TestCaseOperation3 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService22TestCaseOperation3 for more information on using the InputService22TestCaseOperation3
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService22TestCaseOperation3Request method.
//    req, resp := client.InputService22TestCaseOperation3Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService22ProtocolTest) InputService22TestCaseOperation3Request(input *InputService22TestShapeInputService22TestCaseOperation6Input) (req *request.Request, output *InputService22TestShapeInputService22TestCaseOperation3Output) {
	op := &request.Operation{
		Name:       opInputService22TestCaseOperation3,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService22TestShapeInputService22TestCaseOperation6Input{}
	}

	output = &InputService22TestShapeInputService22TestCaseOperation3Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService22TestCaseOperation3 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService22TestCaseOperation3 for usage and error information.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation3(input *InputService22TestShapeInputService22TestCaseOperation6Input) (*InputService22TestShapeInputService22TestCaseOperation3Output, error) {
	req, out := c.InputService22TestCaseOperation3Request(input)
	return out, req.Send()
}

// InputService22TestCaseOperation3WithContext is the same as InputService22TestCaseOperation3 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService22TestCaseOperation3 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation3WithContext(ctx aws.Context, input *InputService22TestShapeInputService22TestCaseOperation6Input, opts ...request.Option) (*InputService22TestShapeInputService22TestCaseOperation3Output, error) {
	req, out := c.InputService22TestCaseOperation3Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService22TestCaseOperation4 = "OperationName"

// InputService22TestCaseOperation4Request generates a "aws/request.Request" representing the
// client's request for the InputService22TestCaseOperation4 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService22TestCaseOperation4 for more information on using the InputService22TestCaseOperation4
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService22TestCaseOperation4Request method.
//    req, resp := client.InputService22TestCaseOperation4Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService22ProtocolTest) InputService22TestCaseOperation4Request(input *InputService22TestShapeInputService22TestCaseOperation6Input) (req *request.Request, output *InputService22TestShapeInputService22TestCaseOperation4Output) {
	op := &request.Operation{
		Name:       opInputService22TestCaseOperation4,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService22TestShapeInputService22TestCaseOperation6Input{}
	}

	output = &InputService22TestShapeInputService22TestCaseOperation4Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService22TestCaseOperation4 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService22TestCaseOperation4 for usage and error information.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation4(input *InputService22TestShapeInputService22TestCaseOperation6Input) (*InputService22TestShapeInputService22TestCaseOperation4Output, error) {
	req, out := c.InputService22TestCaseOperation4Request(input)
	return out, req.Send()
}

// InputService22TestCaseOperation4WithContext is the same as InputService22TestCaseOperation4 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService22TestCaseOperation4 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation4WithContext(ctx aws.Context, input *InputService22TestShapeInputService22TestCaseOperation6Input, opts ...request.Option) (*InputService22TestShapeInputService22TestCaseOperation4Output, error) {
	req, out := c.InputService22TestCaseOperation4Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService22TestCaseOperation5 = "OperationName"

// InputService22TestCaseOperation5Request generates a "aws/request.Request" representing the
// client's request for the InputService22TestCaseOperation5 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService22TestCaseOperation5 for more information on using the InputService22TestCaseOperation5
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService22TestCaseOperation5Request method.
//    req, resp := client.InputService22TestCaseOperation5Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService22ProtocolTest) InputService22TestCaseOperation5Request(input *InputService22TestShapeInputService22TestCaseOperation6Input) (req *request.Request, output *InputService22TestShapeInputService22TestCaseOperation5Output) {
	op := &request.Operation{
		Name:       opInputService22TestCaseOperation5,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService22TestShapeInputService22TestCaseOperation6Input{}
	}

	output = &InputService22TestShapeInputService22TestCaseOperation5Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService22TestCaseOperation5 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService22TestCaseOperation5 for usage and error information.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation5(input *InputService22TestShapeInputService22TestCaseOperation6Input) (*InputService22TestShapeInputService22TestCaseOperation5Output, error) {
	req, out := c.InputService22TestCaseOperation5Request(input)
	return out, req.Send()
}

// InputService22TestCaseOperation5WithContext is the same as InputService22TestCaseOperation5 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService22TestCaseOperation5 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation5WithContext(ctx aws.Context, input *InputService22TestShapeInputService22TestCaseOperation6Input, opts ...request.Option) (*InputService22TestShapeInputService22TestCaseOperation5Output, error) {
	req, out := c.InputService22TestCaseOperation5Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService22TestCaseOperation6 = "OperationName"

// InputService22TestCaseOperation6Request generates a "aws/request.Request" representing the
// client's request for the InputService22TestCaseOperation6 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService22TestCaseOperation6 for more information on using the InputService22TestCaseOperation6
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService22TestCaseOperation6Request method.
//    req, resp := client.InputService22TestCaseOperation6Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService22ProtocolTest) InputService22TestCaseOperation6Request(input *InputService22TestShapeInputService22TestCaseOperation6Input) (req *request.Request, output *InputService22TestShapeInputService22TestCaseOperation6Output) {
	op := &request.Operation{
		Name:       opInputService22TestCaseOperation6,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService22TestShapeInputService22TestCaseOperation6Input{}
	}

	output = &InputService22TestShapeInputService22TestCaseOperation6Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService22TestCaseOperation6 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService22TestCaseOperation6 for usage and error information.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation6(input *InputService22TestShapeInputService22TestCaseOperation6Input) (*InputService22TestShapeInputService22TestCaseOperation6Output, error) {
	req, out := c.InputService22TestCaseOperation6Request(input)
	return out, req.Send()
}

// InputService22TestCaseOperation6WithContext is the same as InputService22TestCaseOperation6 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService22TestCaseOperation6 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService22ProtocolTest) InputService22TestCaseOperation6WithContext(ctx aws.Context, input *InputService22TestShapeInputService22TestCaseOperation6Input, opts ...request.Option) (*InputService22TestShapeInputService22TestCaseOperation6Output, error) {
	req, out := c.InputService22TestCaseOperation6Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService22TestShapeInputService22TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService22TestShapeInputService22TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService22TestShapeInputService22TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService22TestShapeInputService22TestCaseOperation4Output struct {
	_ struct{} `type:"structure"`
}

type InputService22TestShapeInputService22TestCaseOperation5Output struct {
	_ struct{} `type:"structure"`
}

type InputService22TestShapeInputService22TestCaseOperation6Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	RecursiveStruct *InputService22TestShapeRecursiveStructType `type:"structure"`
}

// SetRecursiveStruct sets the RecursiveStruct field's value.
func (s *InputService22TestShapeInputService22TestCaseOperation6Input) SetRecursiveStruct(v *InputService22TestShapeRecursiveStructType) *InputService22TestShapeInputService22TestCaseOperation6Input {
	s.RecursiveStruct = v
	return s
}

type InputService22TestShapeInputService22TestCaseOperation6Output struct {
	_ struct{} `type:"structure"`
}

type InputService22TestShapeRecursiveStructType struct {
	_ struct{} `type:"structure"`

	NoRecurse *string `type:"string"`

	RecursiveList []*InputService22TestShapeRecursiveStructType `type:"list"`

	RecursiveMap map[string]*InputService22TestShapeRecursiveStructType `type:"map"`

	RecursiveStruct *InputService22TestShapeRecursiveStructType `type:"structure"`
}

// SetNoRecurse sets the NoRecurse field's value.
func (s *InputService22TestShapeRecursiveStructType) SetNoRecurse(v string) *InputService22TestShapeRecursiveStructType {
	s.NoRecurse = &v
	return s
}

// SetRecursiveList sets the RecursiveList field's value.
func (s *InputService22TestShapeRecursiveStructType) SetRecursiveList(v []*InputService22TestShapeRecursiveStructType) *InputService22TestShapeRecursiveStructType {
	s.RecursiveList = v
	return s
}

// SetRecursiveMap sets the RecursiveMap field's value.
func (s *InputService22TestShapeRecursiveStructType) SetRecursiveMap(v map[string]*InputService22TestShapeRecursiveStructType) *InputService22TestShapeRecursiveStructType {
	s.RecursiveMap = v
	return s
}

// SetRecursiveStruct sets the RecursiveStruct field's value.
func (s *InputService22TestShapeRecursiveStructType) SetRecursiveStruct(v *InputService22TestShapeRecursiveStructType) *InputService22TestShapeRecursiveStructType {
	s.RecursiveStruct = v
	return s
}

// InputService23ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService23ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService23ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService23ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService23ProtocolTest client from just a session.
//     svc := inputservice23protocoltest.New(mySession)
//
//     // Create a InputService23ProtocolTest client with additional configuration
//     svc := inputservice23protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService23ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService23ProtocolTest {
	c := p.ClientConfig("inputservice23protocoltest", cfgs...)
	return newInputService23ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService23ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService23ProtocolTest {
	svc := &InputService23ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice23protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService23ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService23ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService23TestCaseOperation1 = "OperationName"

// InputService23TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService23TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService23TestCaseOperation1 for more information on using the InputService23TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService23TestCaseOperation1Request method.
//    req, resp := client.InputService23TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService23ProtocolTest) InputService23TestCaseOperation1Request(input *InputService23TestShapeInputService23TestCaseOperation1Input) (req *request.Request, output *InputService23TestShapeInputService23TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService23TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService23TestShapeInputService23TestCaseOperation1Input{}
	}

	output = &InputService23TestShapeInputService23TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService23TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService23TestCaseOperation1 for usage and error information.
func (c *InputService23ProtocolTest) InputService23TestCaseOperation1(input *InputService23TestShapeInputService23TestCaseOperation1Input) (*InputService23TestShapeInputService23TestCaseOperation1Output, error) {
	req, out := c.InputService23TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService23TestCaseOperation1WithContext is the same as InputService23TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService23TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService23ProtocolTest) InputService23TestCaseOperation1WithContext(ctx aws.Context, input *InputService23TestShapeInputService23TestCaseOperation1Input, opts ...request.Option) (*InputService23TestShapeInputService23TestCaseOperation1Output, error) {
	req, out := c.InputService23TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService23TestShapeInputService23TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	TimeArgInHeader *time.Time `location:"header" locationName:"x-amz-timearg" type:"timestamp" timestampFormat:"rfc822"`
}

// SetTimeArgInHeader sets the TimeArgInHeader field's value.
func (s *InputService23TestShapeInputService23TestCaseOperation1Input) SetTimeArgInHeader(v time.Time) *InputService23TestShapeInputService23TestCaseOperation1Input {
	s.TimeArgInHeader = &v
	return s
}

type InputService23TestShapeInputService23TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

// InputService24ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// InputService24ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type InputService24ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the InputService24ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a InputService24ProtocolTest client from just a session.
//     svc := inputservice24protocoltest.New(mySession)
//
//     // Create a InputService24ProtocolTest client with additional configuration
//     svc := inputservice24protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewInputService24ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *InputService24ProtocolTest {
	c := p.ClientConfig("inputservice24protocoltest", cfgs...)
	return newInputService24ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService24ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *InputService24ProtocolTest {
	svc := &InputService24ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice24protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(restxml.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(restxml.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(restxml.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(restxml.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a InputService24ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService24ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService24TestCaseOperation1 = "OperationName"

// InputService24TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the InputService24TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService24TestCaseOperation1 for more information on using the InputService24TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService24TestCaseOperation1Request method.
//    req, resp := client.InputService24TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService24ProtocolTest) InputService24TestCaseOperation1Request(input *InputService24TestShapeInputService24TestCaseOperation2Input) (req *request.Request, output *InputService24TestShapeInputService24TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService24TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService24TestShapeInputService24TestCaseOperation2Input{}
	}

	output = &InputService24TestShapeInputService24TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService24TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService24TestCaseOperation1 for usage and error information.
func (c *InputService24ProtocolTest) InputService24TestCaseOperation1(input *InputService24TestShapeInputService24TestCaseOperation2Input) (*InputService24TestShapeInputService24TestCaseOperation1Output, error) {
	req, out := c.InputService24TestCaseOperation1Request(input)
	return out, req.Send()
}

// InputService24TestCaseOperation1WithContext is the same as InputService24TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService24TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService24ProtocolTest) InputService24TestCaseOperation1WithContext(ctx aws.Context, input *InputService24TestShapeInputService24TestCaseOperation2Input, opts ...request.Option) (*InputService24TestShapeInputService24TestCaseOperation1Output, error) {
	req, out := c.InputService24TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService24TestCaseOperation2 = "OperationName"

// InputService24TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService24TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService24TestCaseOperation2 for more information on using the InputService24TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService24TestCaseOperation2Request method.
//    req, resp := client.InputService24TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService24ProtocolTest) InputService24TestCaseOperation2Request(input *InputService24TestShapeInputService24TestCaseOperation2Input) (req *request.Request, output *InputService24TestShapeInputService24TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService24TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService24TestShapeInputService24TestCaseOperation2Input{}
	}

	output = &InputService24TestShapeInputService24TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(restxml.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService24TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService24TestCaseOperation2 for usage and error information.
func (c *InputService24ProtocolTest) InputService24TestCaseOperation2(input *InputService24TestShapeInputService24TestCaseOperation2Input) (*InputService24TestShapeInputService24TestCaseOperation2Output, error) {
	req, out := c.InputService24TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService24TestCaseOperation2WithContext is the same as InputService24TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService24TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService24ProtocolTest) InputService24TestCaseOperation2WithContext(ctx aws.Context, input *InputService24TestShapeInputService24TestCaseOperation2Input, opts ...request.Option) (*InputService24TestShapeInputService24TestCaseOperation2Output, error) {
	req, out := c.InputService24TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService24TestShapeInputService24TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService24TestShapeInputService24TestCaseOperation2Input struct {
	_ struct{} `type:"structure"`

	Token *string `type:"string" idempotencyToken:"true"`
}

// SetToken sets the Token field's value.
func (s *InputService24TestShapeInputService24TestCaseOperation2Input) SetToken(v string) *InputService24TestShapeInputService24TestCaseOperation2Input {
	s.Token = &v
	return s
}

type InputService24TestShapeInputService24TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

//
// Tests begin here
//

func TestInputService1ProtocolTestBasicXMLSerializationCase1(t *testing.T) {
	svc := NewInputService1ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService1TestShapeInputService1TestCaseOperation2Input{
		Description: aws.String("bar"),
		Name:        aws.String("foo"),
	}
	req, _ := svc.InputService1TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">bar</Description><Name xmlns="https://foo/">foo</Name></OperationRequest>`, util.Trim(string(body)), InputService1TestShapeInputService1TestCaseOperation2Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService1ProtocolTestBasicXMLSerializationCase2(t *testing.T) {
	svc := NewInputService1ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService1TestShapeInputService1TestCaseOperation2Input{
		Description: aws.String("bar"),
		Name:        aws.String("foo"),
	}
	req, _ := svc.InputService1TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">bar</Description><Name xmlns="https://foo/">foo</Name></OperationRequest>`, util.Trim(string(body)), InputService1TestShapeInputService1TestCaseOperation2Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService1ProtocolTestBasicXMLSerializationCase3(t *testing.T) {
	svc := NewInputService1ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService1TestShapeInputService1TestCaseOperation3Input{}
	req, _ := svc.InputService1TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService2ProtocolTestSerializeOtherScalarTypesCase1(t *testing.T) {
	svc := NewInputService2ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService2TestShapeInputService2TestCaseOperation1Input{
		First:  aws.Bool(true),
		Fourth: aws.Int64(3),
		Second: aws.Bool(false),
		Third:  aws.Float64(1.2),
	}
	req, _ := svc.InputService2TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><First xmlns="https://foo/">true</First><Fourth xmlns="https://foo/">3</Fourth><Second xmlns="https://foo/">false</Second><Third xmlns="https://foo/">1.2</Third></OperationRequest>`, util.Trim(string(body)), InputService2TestShapeInputService2TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService3ProtocolTestNestedStructuresCase1(t *testing.T) {
	svc := NewInputService3ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService3TestShapeInputService3TestCaseOperation2Input{
		Description: aws.String("baz"),
		SubStructure: &InputService3TestShapeSubStructure{
			Bar: aws.String("b"),
			Foo: aws.String("a"),
		},
	}
	req, _ := svc.InputService3TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">baz</Description><SubStructure xmlns="https://foo/"><Bar xmlns="https://foo/">b</Bar><Foo xmlns="https://foo/">a</Foo></SubStructure></OperationRequest>`, util.Trim(string(body)), InputService3TestShapeInputService3TestCaseOperation2Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService3ProtocolTestNestedStructuresCase2(t *testing.T) {
	svc := NewInputService3ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService3TestShapeInputService3TestCaseOperation2Input{
		Description: aws.String("baz"),
		SubStructure: &InputService3TestShapeSubStructure{
			Foo: aws.String("a"),
		},
	}
	req, _ := svc.InputService3TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">baz</Description><SubStructure xmlns="https://foo/"><Foo xmlns="https://foo/">a</Foo></SubStructure></OperationRequest>`, util.Trim(string(body)), InputService3TestShapeInputService3TestCaseOperation2Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService4ProtocolTestNestedStructuresCase1(t *testing.T) {
	svc := NewInputService4ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService4TestShapeInputService4TestCaseOperation1Input{
		Description:  aws.String("baz"),
		SubStructure: &InputService4TestShapeSubStructure{},
	}
	req, _ := svc.InputService4TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">baz</Description><SubStructure xmlns="https://foo/"></SubStructure></OperationRequest>`, util.Trim(string(body)), InputService4TestShapeInputService4TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService5ProtocolTestNonFlattenedListsCase1(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation1Input{
		ListParam: []*string{
			aws.String("one"),
			aws.String("two"),
			aws.String("three"),
		},
	}
	req, _ := svc.InputService5TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><ListParam xmlns="https://foo/"><member xmlns="https://foo/">one</member><member xmlns="https://foo/">two</member><member xmlns="https://foo/">three</member></ListParam></OperationRequest>`, util.Trim(string(body)), InputService5TestShapeInputService5TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService6ProtocolTestNonFlattenedListsWithLocationNameCase1(t *testing.T) {
	svc := NewInputService6ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService6TestShapeInputService6TestCaseOperation1Input{
		ListParam: []*string{
			aws.String("one"),
			aws.String("two"),
			aws.String("three"),
		},
	}
	req, _ := svc.InputService6TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><AlternateName xmlns="https://foo/"><NotMember xmlns="https://foo/">one</NotMember><NotMember xmlns="https://foo/">two</NotMember><NotMember xmlns="https://foo/">three</NotMember></AlternateName></OperationRequest>`, util.Trim(string(body)), InputService6TestShapeInputService6TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService7ProtocolTestFlattenedListsCase1(t *testing.T) {
	svc := NewInputService7ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService7TestShapeInputService7TestCaseOperation1Input{
		ListParam: []*string{
			aws.String("one"),
			aws.String("two"),
			aws.String("three"),
		},
	}
	req, _ := svc.InputService7TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><ListParam xmlns="https://foo/">one</ListParam><ListParam xmlns="https://foo/">two</ListParam><ListParam xmlns="https://foo/">three</ListParam></OperationRequest>`, util.Trim(string(body)), InputService7TestShapeInputService7TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService8ProtocolTestFlattenedListsWithLocationNameCase1(t *testing.T) {
	svc := NewInputService8ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService8TestShapeInputService8TestCaseOperation1Input{
		ListParam: []*string{
			aws.String("one"),
			aws.String("two"),
			aws.String("three"),
		},
	}
	req, _ := svc.InputService8TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><item xmlns="https://foo/">one</item><item xmlns="https://foo/">two</item><item xmlns="https://foo/">three</item></OperationRequest>`, util.Trim(string(body)), InputService8TestShapeInputService8TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService9ProtocolTestListOfStructuresCase1(t *testing.T) {
	svc := NewInputService9ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService9TestShapeInputService9TestCaseOperation1Input{
		ListParam: []*InputService9TestShapeSingleFieldStruct{
			{
				Element: aws.String("one"),
			},
			{
				Element: aws.String("two"),
			},
			{
				Element: aws.String("three"),
			},
		},
	}
	req, _ := svc.InputService9TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><item xmlns="https://foo/"><value xmlns="https://foo/">one</value></item><item xmlns="https://foo/"><value xmlns="https://foo/">two</value></item><item xmlns="https://foo/"><value xmlns="https://foo/">three</value></item></OperationRequest>`, util.Trim(string(body)), InputService9TestShapeInputService9TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService10ProtocolTestBlobAndTimestampShapesCase1(t *testing.T) {
	svc := NewInputService10ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService10TestShapeInputService10TestCaseOperation1Input{
		StructureParam: &InputService10TestShapeStructureShape{
			B: []byte("foo"),
			T: aws.Time(time.Unix(1422172800, 0)),
		},
	}
	req, _ := svc.InputService10TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><StructureParam xmlns="https://foo/"><b xmlns="https://foo/">Zm9v</b><t xmlns="https://foo/">2015-01-25T08:00:00Z</t></StructureParam></OperationRequest>`, util.Trim(string(body)), InputService10TestShapeInputService10TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService11ProtocolTestHeaderMapsCase1(t *testing.T) {
	svc := NewInputService11ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService11TestShapeInputService11TestCaseOperation1Input{
		Foo: map[string]*string{
			"a": aws.String("b"),
			"c": aws.String("d"),
		},
	}
	req, _ := svc.InputService11TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "b", r.Header.Get("x-foo-a"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "d", r.Header.Get("x-foo-c"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService12ProtocolTestQuerystringListOfStringsCase1(t *testing.T) {
	svc := NewInputService12ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService12TestShapeInputService12TestCaseOperation1Input{
		Items: []*string{
			aws.String("value1"),
			aws.String("value2"),
		},
	}
	req, _ := svc.InputService12TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/path?item=value1&item=value2", r.URL.String())

	// assert headers

}

func TestInputService13ProtocolTestStringToStringMapsInQuerystringCase1(t *testing.T) {
	svc := NewInputService13ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService13TestShapeInputService13TestCaseOperation1Input{
		PipelineId: aws.String("foo"),
		QueryDoc: map[string]*string{
			"bar":  aws.String("baz"),
			"fizz": aws.String("buzz"),
		},
	}
	req, _ := svc.InputService13TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/jobsByPipeline/foo?bar=baz&fizz=buzz", r.URL.String())

	// assert headers

}

func TestInputService14ProtocolTestStringToStringListMapsInQuerystringCase1(t *testing.T) {
	svc := NewInputService14ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService14TestShapeInputService14TestCaseOperation1Input{
		PipelineId: aws.String("id"),
		QueryDoc: map[string][]*string{
			"fizz": {
				aws.String("buzz"),
				aws.String("pop"),
			},
			"foo": {
				aws.String("bar"),
				aws.String("baz"),
			},
		},
	}
	req, _ := svc.InputService14TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/jobsByPipeline/id?foo=bar&foo=baz&fizz=buzz&fizz=pop", r.URL.String())

	// assert headers

}

func TestInputService15ProtocolTestBooleanInQuerystringCase1(t *testing.T) {
	svc := NewInputService15ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService15TestShapeInputService15TestCaseOperation2Input{
		BoolQuery: aws.Bool(true),
	}
	req, _ := svc.InputService15TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/path?bool-query=true", r.URL.String())

	// assert headers

}

func TestInputService15ProtocolTestBooleanInQuerystringCase2(t *testing.T) {
	svc := NewInputService15ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService15TestShapeInputService15TestCaseOperation2Input{
		BoolQuery: aws.Bool(false),
	}
	req, _ := svc.InputService15TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/path?bool-query=false", r.URL.String())

	// assert headers

}

func TestInputService16ProtocolTestStringPayloadCase1(t *testing.T) {
	svc := NewInputService16ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService16TestShapeInputService16TestCaseOperation1Input{
		Foo: aws.String("bar"),
	}
	req, _ := svc.InputService16TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	if e, a := "bar", util.Trim(string(body)); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService17ProtocolTestBlobPayloadCase1(t *testing.T) {
	svc := NewInputService17ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService17TestShapeInputService17TestCaseOperation2Input{
		Foo: []byte("bar"),
	}
	req, _ := svc.InputService17TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	if e, a := "bar", util.Trim(string(body)); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService17ProtocolTestBlobPayloadCase2(t *testing.T) {
	svc := NewInputService17ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService17TestShapeInputService17TestCaseOperation2Input{}
	req, _ := svc.InputService17TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService18ProtocolTestStructurePayloadCase1(t *testing.T) {
	svc := NewInputService18ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService18TestShapeInputService18TestCaseOperation4Input{
		Foo: &InputService18TestShapeFooShape{
			Baz: aws.String("bar"),
		},
	}
	req, _ := svc.InputService18TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<foo><baz>bar</baz></foo>`, util.Trim(string(body)), InputService18TestShapeInputService18TestCaseOperation4Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService18ProtocolTestStructurePayloadCase2(t *testing.T) {
	svc := NewInputService18ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService18TestShapeInputService18TestCaseOperation4Input{}
	req, _ := svc.InputService18TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService18ProtocolTestStructurePayloadCase3(t *testing.T) {
	svc := NewInputService18ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService18TestShapeInputService18TestCaseOperation4Input{
		Foo: &InputService18TestShapeFooShape{},
	}
	req, _ := svc.InputService18TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<foo></foo>`, util.Trim(string(body)), InputService18TestShapeInputService18TestCaseOperation4Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService18ProtocolTestStructurePayloadCase4(t *testing.T) {
	svc := NewInputService18ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService18TestShapeInputService18TestCaseOperation4Input{}
	req, _ := svc.InputService18TestCaseOperation4Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService19ProtocolTestXMLAttributeCase1(t *testing.T) {
	svc := NewInputService19ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService19TestShapeInputService19TestCaseOperation1Input{
		Grant: &InputService19TestShapeGrant{
			Grantee: &InputService19TestShapeGrantee{
				EmailAddress: aws.String("foo@example.com"),
				Type:         aws.String("CanonicalUser"),
			},
		},
	}
	req, _ := svc.InputService19TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<Grant xmlns:_xmlns="xmlns" _xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:XMLSchema-instance="http://www.w3.org/2001/XMLSchema-instance" XMLSchema-instance:type="CanonicalUser"><Grantee><EmailAddress>foo@example.com</EmailAddress></Grantee></Grant>`, util.Trim(string(body)), InputService19TestShapeInputService19TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestGreedyKeysCase1(t *testing.T) {
	svc := NewInputService20ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService20TestShapeInputService20TestCaseOperation1Input{
		Bucket: aws.String("my/bucket"),
		Key:    aws.String("testing /123"),
	}
	req, _ := svc.InputService20TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/my%2Fbucket/testing%20/123", r.URL.String())

	// assert headers

}

func TestInputService21ProtocolTestOmitsNullQueryParamsButSerializesEmptyStringsCase1(t *testing.T) {
	svc := NewInputService21ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService21TestShapeInputService21TestCaseOperation2Input{}
	req, _ := svc.InputService21TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService21ProtocolTestOmitsNullQueryParamsButSerializesEmptyStringsCase2(t *testing.T) {
	svc := NewInputService21ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService21TestShapeInputService21TestCaseOperation2Input{
		Foo: aws.String(""),
	}
	req, _ := svc.InputService21TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/path?abc=mno&param-name=", r.URL.String())

	// assert headers

}

func TestInputService22ProtocolTestRecursiveShapesCase1(t *testing.T) {
	svc := NewInputService22ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService22TestShapeInputService22TestCaseOperation6Input{
		RecursiveStruct: &InputService22TestShapeRecursiveStructType{
			NoRecurse: aws.String("foo"),
		},
	}
	req, _ := svc.InputService22TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService22TestShapeInputService22TestCaseOperation6Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService22ProtocolTestRecursiveShapesCase2(t *testing.T) {
	svc := NewInputService22ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService22TestShapeInputService22TestCaseOperation6Input{
		RecursiveStruct: &InputService22TestShapeRecursiveStructType{
			RecursiveStruct: &InputService22TestShapeRecursiveStructType{
				NoRecurse: aws.String("foo"),
			},
		},
	}
	req, _ := svc.InputService22TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></RecursiveStruct></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService22TestShapeInputService22TestCaseOperation6Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService22ProtocolTestRecursiveShapesCase3(t *testing.T) {
	svc := NewInputService22ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService22TestShapeInputService22TestCaseOperation6Input{
		RecursiveStruct: &InputService22TestShapeRecursiveStructType{
			RecursiveStruct: &InputService22TestShapeRecursiveStructType{
				RecursiveStruct: &InputService22TestShapeRecursiveStructType{
					RecursiveStruct: &InputService22TestShapeRecursiveStructType{
						NoRecurse: aws.String("foo"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService22TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></RecursiveStruct></RecursiveStruct></RecursiveStruct></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService22TestShapeInputService22TestCaseOperation6Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService22ProtocolTestRecursiveShapesCase4(t *testing.T) {
	svc := NewInputService22ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService22TestShapeInputService22TestCaseOperation6Input{
		RecursiveStruct: &InputService22TestShapeRecursiveStructType{
			RecursiveList: []*InputService22TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					NoRecurse: aws.String("bar"),
				},
			},
		},
	}
	req, _ := svc.InputService22TestCaseOperation4Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveList xmlns="https://foo/"><member xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></member><member xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></member></RecursiveList></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService22TestShapeInputService22TestCaseOperation6Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService22ProtocolTestRecursiveShapesCase5(t *testing.T) {
	svc := NewInputService22ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService22TestShapeInputService22TestCaseOperation6Input{
		RecursiveStruct: &InputService22TestShapeRecursiveStructType{
			RecursiveList: []*InputService22TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					RecursiveStruct: &InputService22TestShapeRecursiveStructType{
						NoRecurse: aws.String("bar"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService22TestCaseOperation5Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveList xmlns="https://foo/"><member xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></member><member xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></RecursiveStruct></member></RecursiveList></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService22TestShapeInputService22TestCaseOperation6Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService22ProtocolTestRecursiveShapesCase6(t *testing.T) {
	svc := NewInputService22ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService22TestShapeInputService22TestCaseOperation6Input{
		RecursiveStruct: &InputService22TestShapeRecursiveStructType{
			RecursiveMap: map[string]*InputService22TestShapeRecursiveStructType{
				"bar": {
					NoRecurse: aws.String("bar"),
				},
				"foo": {
					NoRecurse: aws.String("foo"),
				},
			},
		},
	}
	req, _ := svc.InputService22TestCaseOperation6Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService22TestShapeInputService22TestCaseOperation6Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService23ProtocolTestTimestampInHeaderCase1(t *testing.T) {
	svc := NewInputService23ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService23TestShapeInputService23TestCaseOperation1Input{
		TimeArgInHeader: aws.Time(time.Unix(1422172800, 0)),
	}
	req, _ := svc.InputService23TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers
	if e, a := "Sun, 25 Jan 2015 08:00:00 GMT", r.Header.Get("x-amz-timearg"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService24ProtocolTestIdempotencyTokenAutoFillCase1(t *testing.T) {
	svc := NewInputService24ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService24TestShapeInputService24TestCaseOperation2Input{
		Token: aws.String("abc123"),
	}
	req, _ := svc.InputService24TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<InputShape><Token>abc123</Token></InputShape>`, util.Trim(string(body)), InputService24TestShapeInputService24TestCaseOperation2Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService24ProtocolTestIdempotencyTokenAutoFillCase2(t *testing.T) {
	svc := NewInputService24ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService24TestShapeInputService24TestCaseOperation2Input{}
	req, _ := svc.InputService24TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<InputShape><Token>00000000-0000-4000-8000-000000000000</Token></InputShape>`, util.Trim(string(body)), InputService24TestShapeInputService24TestCaseOperation2Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}
