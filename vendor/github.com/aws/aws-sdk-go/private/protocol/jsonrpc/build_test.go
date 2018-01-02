package jsonrpc_test

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
	"github.com/aws/aws-sdk-go/private/protocol/jsonrpc"
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
				APIVersion:    "",
				JSONVersion:   "1.1",
				TargetPrefix:  "com.amazonaws.foo",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1Request(input *InputService1TestShapeInputService1TestCaseOperation1Input) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService1TestShapeInputService1TestCaseOperation1Input{}
	}

	output = &InputService1TestShapeInputService1TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1(input *InputService1TestShapeInputService1TestCaseOperation1Input) (*InputService1TestShapeInputService1TestCaseOperation1Output, error) {
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
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1WithContext(ctx aws.Context, input *InputService1TestShapeInputService1TestCaseOperation1Input, opts ...request.Option) (*InputService1TestShapeInputService1TestCaseOperation1Output, error) {
	req, out := c.InputService1TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService1TestShapeInputService1TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	Name *string `type:"string"`
}

// SetName sets the Name field's value.
func (s *InputService1TestShapeInputService1TestCaseOperation1Input) SetName(v string) *InputService1TestShapeInputService1TestCaseOperation1Input {
	s.Name = &v
	return s
}

type InputService1TestShapeInputService1TestCaseOperation1Output struct {
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
				APIVersion:    "",
				JSONVersion:   "1.1",
				TargetPrefix:  "com.amazonaws.foo",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
		Name:     opInputService2TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService2TestShapeInputService2TestCaseOperation1Input{}
	}

	output = &InputService2TestShapeInputService2TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
	_ struct{} `type:"structure"`

	TimeArg *time.Time `type:"timestamp" timestampFormat:"unix"`
}

// SetTimeArg sets the TimeArg field's value.
func (s *InputService2TestShapeInputService2TestCaseOperation1Input) SetTimeArg(v time.Time) *InputService2TestShapeInputService2TestCaseOperation1Input {
	s.TimeArg = &v
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
				APIVersion:    "",
				JSONVersion:   "1.1",
				TargetPrefix:  "com.amazonaws.foo",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
		Name:     opInputService3TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService3TestShapeInputService3TestCaseOperation2Input{}
	}

	output = &InputService3TestShapeInputService3TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
		Name:     opInputService3TestCaseOperation2,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService3TestShapeInputService3TestCaseOperation2Input{}
	}

	output = &InputService3TestShapeInputService3TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
	_ struct{} `type:"structure"`

	// BlobArg is automatically base64 encoded/decoded by the SDK.
	BlobArg []byte `type:"blob"`

	BlobMap map[string][]byte `type:"map"`
}

// SetBlobArg sets the BlobArg field's value.
func (s *InputService3TestShapeInputService3TestCaseOperation2Input) SetBlobArg(v []byte) *InputService3TestShapeInputService3TestCaseOperation2Input {
	s.BlobArg = v
	return s
}

// SetBlobMap sets the BlobMap field's value.
func (s *InputService3TestShapeInputService3TestCaseOperation2Input) SetBlobMap(v map[string][]byte) *InputService3TestShapeInputService3TestCaseOperation2Input {
	s.BlobMap = v
	return s
}

type InputService3TestShapeInputService3TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
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
				APIVersion:    "",
				JSONVersion:   "1.1",
				TargetPrefix:  "com.amazonaws.foo",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService4TestShapeInputService4TestCaseOperation1Input{}
	}

	output = &InputService4TestShapeInputService4TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
	_ struct{} `type:"structure"`

	ListParam [][]byte `type:"list"`
}

// SetListParam sets the ListParam field's value.
func (s *InputService4TestShapeInputService4TestCaseOperation1Input) SetListParam(v [][]byte) *InputService4TestShapeInputService4TestCaseOperation1Input {
	s.ListParam = v
	return s
}

type InputService4TestShapeInputService4TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
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
				APIVersion:    "",
				JSONVersion:   "1.1",
				TargetPrefix:  "com.amazonaws.foo",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1Request(input *InputService5TestShapeInputService5TestCaseOperation6Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opInputService5TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation6Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1(input *InputService5TestShapeInputService5TestCaseOperation6Input) (*InputService5TestShapeInputService5TestCaseOperation1Output, error) {
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
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation6Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation1Output, error) {
	req, out := c.InputService5TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService5TestCaseOperation2 = "OperationName"

// InputService5TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService5TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService5TestCaseOperation2 for more information on using the InputService5TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService5TestCaseOperation2Request method.
//    req, resp := client.InputService5TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService5ProtocolTest) InputService5TestCaseOperation2Request(input *InputService5TestShapeInputService5TestCaseOperation6Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation2Output) {
	op := &request.Operation{
		Name:     opInputService5TestCaseOperation2,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation6Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService5TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService5TestCaseOperation2 for usage and error information.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation2(input *InputService5TestShapeInputService5TestCaseOperation6Input) (*InputService5TestShapeInputService5TestCaseOperation2Output, error) {
	req, out := c.InputService5TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService5TestCaseOperation2WithContext is the same as InputService5TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService5TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation2WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation6Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation2Output, error) {
	req, out := c.InputService5TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService5TestCaseOperation3 = "OperationName"

// InputService5TestCaseOperation3Request generates a "aws/request.Request" representing the
// client's request for the InputService5TestCaseOperation3 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService5TestCaseOperation3 for more information on using the InputService5TestCaseOperation3
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService5TestCaseOperation3Request method.
//    req, resp := client.InputService5TestCaseOperation3Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService5ProtocolTest) InputService5TestCaseOperation3Request(input *InputService5TestShapeInputService5TestCaseOperation6Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation3Output) {
	op := &request.Operation{
		Name:     opInputService5TestCaseOperation3,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation6Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation3Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService5TestCaseOperation3 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService5TestCaseOperation3 for usage and error information.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation3(input *InputService5TestShapeInputService5TestCaseOperation6Input) (*InputService5TestShapeInputService5TestCaseOperation3Output, error) {
	req, out := c.InputService5TestCaseOperation3Request(input)
	return out, req.Send()
}

// InputService5TestCaseOperation3WithContext is the same as InputService5TestCaseOperation3 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService5TestCaseOperation3 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation3WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation6Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation3Output, error) {
	req, out := c.InputService5TestCaseOperation3Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService5TestCaseOperation4 = "OperationName"

// InputService5TestCaseOperation4Request generates a "aws/request.Request" representing the
// client's request for the InputService5TestCaseOperation4 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService5TestCaseOperation4 for more information on using the InputService5TestCaseOperation4
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService5TestCaseOperation4Request method.
//    req, resp := client.InputService5TestCaseOperation4Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService5ProtocolTest) InputService5TestCaseOperation4Request(input *InputService5TestShapeInputService5TestCaseOperation6Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation4Output) {
	op := &request.Operation{
		Name:     opInputService5TestCaseOperation4,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation6Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation4Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService5TestCaseOperation4 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService5TestCaseOperation4 for usage and error information.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation4(input *InputService5TestShapeInputService5TestCaseOperation6Input) (*InputService5TestShapeInputService5TestCaseOperation4Output, error) {
	req, out := c.InputService5TestCaseOperation4Request(input)
	return out, req.Send()
}

// InputService5TestCaseOperation4WithContext is the same as InputService5TestCaseOperation4 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService5TestCaseOperation4 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation4WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation6Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation4Output, error) {
	req, out := c.InputService5TestCaseOperation4Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService5TestCaseOperation5 = "OperationName"

// InputService5TestCaseOperation5Request generates a "aws/request.Request" representing the
// client's request for the InputService5TestCaseOperation5 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService5TestCaseOperation5 for more information on using the InputService5TestCaseOperation5
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService5TestCaseOperation5Request method.
//    req, resp := client.InputService5TestCaseOperation5Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService5ProtocolTest) InputService5TestCaseOperation5Request(input *InputService5TestShapeInputService5TestCaseOperation6Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation5Output) {
	op := &request.Operation{
		Name:     opInputService5TestCaseOperation5,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation6Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation5Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService5TestCaseOperation5 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService5TestCaseOperation5 for usage and error information.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation5(input *InputService5TestShapeInputService5TestCaseOperation6Input) (*InputService5TestShapeInputService5TestCaseOperation5Output, error) {
	req, out := c.InputService5TestCaseOperation5Request(input)
	return out, req.Send()
}

// InputService5TestCaseOperation5WithContext is the same as InputService5TestCaseOperation5 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService5TestCaseOperation5 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation5WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation6Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation5Output, error) {
	req, out := c.InputService5TestCaseOperation5Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService5TestCaseOperation6 = "OperationName"

// InputService5TestCaseOperation6Request generates a "aws/request.Request" representing the
// client's request for the InputService5TestCaseOperation6 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService5TestCaseOperation6 for more information on using the InputService5TestCaseOperation6
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService5TestCaseOperation6Request method.
//    req, resp := client.InputService5TestCaseOperation6Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService5ProtocolTest) InputService5TestCaseOperation6Request(input *InputService5TestShapeInputService5TestCaseOperation6Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation6Output) {
	op := &request.Operation{
		Name:     opInputService5TestCaseOperation6,
		HTTPPath: "/",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation6Input{}
	}

	output = &InputService5TestShapeInputService5TestCaseOperation6Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService5TestCaseOperation6 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService5TestCaseOperation6 for usage and error information.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation6(input *InputService5TestShapeInputService5TestCaseOperation6Input) (*InputService5TestShapeInputService5TestCaseOperation6Output, error) {
	req, out := c.InputService5TestCaseOperation6Request(input)
	return out, req.Send()
}

// InputService5TestCaseOperation6WithContext is the same as InputService5TestCaseOperation6 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService5TestCaseOperation6 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation6WithContext(ctx aws.Context, input *InputService5TestShapeInputService5TestCaseOperation6Input, opts ...request.Option) (*InputService5TestShapeInputService5TestCaseOperation6Output, error) {
	req, out := c.InputService5TestCaseOperation6Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService5TestShapeInputService5TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService5TestShapeInputService5TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService5TestShapeInputService5TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService5TestShapeInputService5TestCaseOperation4Output struct {
	_ struct{} `type:"structure"`
}

type InputService5TestShapeInputService5TestCaseOperation5Output struct {
	_ struct{} `type:"structure"`
}

type InputService5TestShapeInputService5TestCaseOperation6Input struct {
	_ struct{} `type:"structure"`

	RecursiveStruct *InputService5TestShapeRecursiveStructType `type:"structure"`
}

// SetRecursiveStruct sets the RecursiveStruct field's value.
func (s *InputService5TestShapeInputService5TestCaseOperation6Input) SetRecursiveStruct(v *InputService5TestShapeRecursiveStructType) *InputService5TestShapeInputService5TestCaseOperation6Input {
	s.RecursiveStruct = v
	return s
}

type InputService5TestShapeInputService5TestCaseOperation6Output struct {
	_ struct{} `type:"structure"`
}

type InputService5TestShapeRecursiveStructType struct {
	_ struct{} `type:"structure"`

	NoRecurse *string `type:"string"`

	RecursiveList []*InputService5TestShapeRecursiveStructType `type:"list"`

	RecursiveMap map[string]*InputService5TestShapeRecursiveStructType `type:"map"`

	RecursiveStruct *InputService5TestShapeRecursiveStructType `type:"structure"`
}

// SetNoRecurse sets the NoRecurse field's value.
func (s *InputService5TestShapeRecursiveStructType) SetNoRecurse(v string) *InputService5TestShapeRecursiveStructType {
	s.NoRecurse = &v
	return s
}

// SetRecursiveList sets the RecursiveList field's value.
func (s *InputService5TestShapeRecursiveStructType) SetRecursiveList(v []*InputService5TestShapeRecursiveStructType) *InputService5TestShapeRecursiveStructType {
	s.RecursiveList = v
	return s
}

// SetRecursiveMap sets the RecursiveMap field's value.
func (s *InputService5TestShapeRecursiveStructType) SetRecursiveMap(v map[string]*InputService5TestShapeRecursiveStructType) *InputService5TestShapeRecursiveStructType {
	s.RecursiveMap = v
	return s
}

// SetRecursiveStruct sets the RecursiveStruct field's value.
func (s *InputService5TestShapeRecursiveStructType) SetRecursiveStruct(v *InputService5TestShapeRecursiveStructType) *InputService5TestShapeRecursiveStructType {
	s.RecursiveStruct = v
	return s
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
				APIVersion:    "",
				JSONVersion:   "1.1",
				TargetPrefix:  "com.amazonaws.foo",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBackNamed(v4.SignRequestHandler)
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService6TestShapeInputService6TestCaseOperation1Input{}
	}

	output = &InputService6TestShapeInputService6TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
	_ struct{} `type:"structure"`

	Map map[string]*string `type:"map"`
}

// SetMap sets the Map field's value.
func (s *InputService6TestShapeInputService6TestCaseOperation1Input) SetMap(v map[string]*string) *InputService6TestShapeInputService6TestCaseOperation1Input {
	s.Map = v
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
	svc.Handlers.Build.PushBackNamed(jsonrpc.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(jsonrpc.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(jsonrpc.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(jsonrpc.UnmarshalErrorHandler)

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
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1Request(input *InputService7TestShapeInputService7TestCaseOperation2Input) (req *request.Request, output *InputService7TestShapeInputService7TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService7TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService7TestShapeInputService7TestCaseOperation2Input{}
	}

	output = &InputService7TestShapeInputService7TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
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
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1(input *InputService7TestShapeInputService7TestCaseOperation2Input) (*InputService7TestShapeInputService7TestCaseOperation1Output, error) {
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
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1WithContext(ctx aws.Context, input *InputService7TestShapeInputService7TestCaseOperation2Input, opts ...request.Option) (*InputService7TestShapeInputService7TestCaseOperation1Output, error) {
	req, out := c.InputService7TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opInputService7TestCaseOperation2 = "OperationName"

// InputService7TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the InputService7TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See InputService7TestCaseOperation2 for more information on using the InputService7TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the InputService7TestCaseOperation2Request method.
//    req, resp := client.InputService7TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *InputService7ProtocolTest) InputService7TestCaseOperation2Request(input *InputService7TestShapeInputService7TestCaseOperation2Input) (req *request.Request, output *InputService7TestShapeInputService7TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService7TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService7TestShapeInputService7TestCaseOperation2Input{}
	}

	output = &InputService7TestShapeInputService7TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	req.Handlers.Unmarshal.Remove(jsonrpc.UnmarshalHandler)
	req.Handlers.Unmarshal.PushBackNamed(protocol.UnmarshalDiscardBodyHandler)
	return
}

// InputService7TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation InputService7TestCaseOperation2 for usage and error information.
func (c *InputService7ProtocolTest) InputService7TestCaseOperation2(input *InputService7TestShapeInputService7TestCaseOperation2Input) (*InputService7TestShapeInputService7TestCaseOperation2Output, error) {
	req, out := c.InputService7TestCaseOperation2Request(input)
	return out, req.Send()
}

// InputService7TestCaseOperation2WithContext is the same as InputService7TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See InputService7TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *InputService7ProtocolTest) InputService7TestCaseOperation2WithContext(ctx aws.Context, input *InputService7TestShapeInputService7TestCaseOperation2Input, opts ...request.Option) (*InputService7TestShapeInputService7TestCaseOperation2Output, error) {
	req, out := c.InputService7TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type InputService7TestShapeInputService7TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService7TestShapeInputService7TestCaseOperation2Input struct {
	_ struct{} `type:"structure"`

	Token *string `type:"string" idempotencyToken:"true"`
}

// SetToken sets the Token field's value.
func (s *InputService7TestShapeInputService7TestCaseOperation2Input) SetToken(v string) *InputService7TestShapeInputService7TestCaseOperation2Input {
	s.Token = &v
	return s
}

type InputService7TestShapeInputService7TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

//
// Tests begin here
//

func TestInputService1ProtocolTestScalarMembersCase1(t *testing.T) {
	svc := NewInputService1ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService1TestShapeInputService1TestCaseOperation1Input{
		Name: aws.String("myname"),
	}
	req, _ := svc.InputService1TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"Name":"myname"}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService2ProtocolTestTimestampValuesCase1(t *testing.T) {
	svc := NewInputService2ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService2TestShapeInputService2TestCaseOperation1Input{
		TimeArg: aws.Time(time.Unix(1422172800, 0)),
	}
	req, _ := svc.InputService2TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"TimeArg":1422172800}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService3ProtocolTestBase64EncodedBlobsCase1(t *testing.T) {
	svc := NewInputService3ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService3TestShapeInputService3TestCaseOperation2Input{
		BlobArg: []byte("foo"),
	}
	req, _ := svc.InputService3TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"BlobArg":"Zm9v"}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService3ProtocolTestBase64EncodedBlobsCase2(t *testing.T) {
	svc := NewInputService3ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService3TestShapeInputService3TestCaseOperation2Input{
		BlobMap: map[string][]byte{
			"key1": []byte("foo"),
			"key2": []byte("bar"),
		},
	}
	req, _ := svc.InputService3TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"BlobMap":{"key1":"Zm9v","key2":"YmFy"}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService4ProtocolTestNestedBlobsCase1(t *testing.T) {
	svc := NewInputService4ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService4TestShapeInputService4TestCaseOperation1Input{
		ListParam: [][]byte{
			[]byte("foo"),
			[]byte("bar"),
		},
	}
	req, _ := svc.InputService4TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"ListParam":["Zm9v","YmFy"]}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService5ProtocolTestRecursiveShapesCase1(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation6Input{
		RecursiveStruct: &InputService5TestShapeRecursiveStructType{
			NoRecurse: aws.String("foo"),
		},
	}
	req, _ := svc.InputService5TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"RecursiveStruct":{"NoRecurse":"foo"}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService5ProtocolTestRecursiveShapesCase2(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation6Input{
		RecursiveStruct: &InputService5TestShapeRecursiveStructType{
			RecursiveStruct: &InputService5TestShapeRecursiveStructType{
				NoRecurse: aws.String("foo"),
			},
		},
	}
	req, _ := svc.InputService5TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"RecursiveStruct":{"RecursiveStruct":{"NoRecurse":"foo"}}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService5ProtocolTestRecursiveShapesCase3(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation6Input{
		RecursiveStruct: &InputService5TestShapeRecursiveStructType{
			RecursiveStruct: &InputService5TestShapeRecursiveStructType{
				RecursiveStruct: &InputService5TestShapeRecursiveStructType{
					RecursiveStruct: &InputService5TestShapeRecursiveStructType{
						NoRecurse: aws.String("foo"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService5TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"RecursiveStruct":{"RecursiveStruct":{"RecursiveStruct":{"RecursiveStruct":{"NoRecurse":"foo"}}}}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService5ProtocolTestRecursiveShapesCase4(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation6Input{
		RecursiveStruct: &InputService5TestShapeRecursiveStructType{
			RecursiveList: []*InputService5TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					NoRecurse: aws.String("bar"),
				},
			},
		},
	}
	req, _ := svc.InputService5TestCaseOperation4Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"RecursiveStruct":{"RecursiveList":[{"NoRecurse":"foo"},{"NoRecurse":"bar"}]}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService5ProtocolTestRecursiveShapesCase5(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation6Input{
		RecursiveStruct: &InputService5TestShapeRecursiveStructType{
			RecursiveList: []*InputService5TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					RecursiveStruct: &InputService5TestShapeRecursiveStructType{
						NoRecurse: aws.String("bar"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService5TestCaseOperation5Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"RecursiveStruct":{"RecursiveList":[{"NoRecurse":"foo"},{"RecursiveStruct":{"NoRecurse":"bar"}}]}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService5ProtocolTestRecursiveShapesCase6(t *testing.T) {
	svc := NewInputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService5TestShapeInputService5TestCaseOperation6Input{
		RecursiveStruct: &InputService5TestShapeRecursiveStructType{
			RecursiveMap: map[string]*InputService5TestShapeRecursiveStructType{
				"bar": {
					NoRecurse: aws.String("bar"),
				},
				"foo": {
					NoRecurse: aws.String("foo"),
				},
			},
		},
	}
	req, _ := svc.InputService5TestCaseOperation6Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"RecursiveStruct":{"RecursiveMap":{"foo":{"NoRecurse":"foo"},"bar":{"NoRecurse":"bar"}}}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService6ProtocolTestEmptyMapsCase1(t *testing.T) {
	svc := NewInputService6ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService6TestShapeInputService6TestCaseOperation1Input{
		Map: map[string]*string{},
	}
	req, _ := svc.InputService6TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"Map":{}}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	if e, a := "application/x-amz-json-1.1", r.Header.Get("Content-Type"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}
	if e, a := "com.amazonaws.foo.OperationName", r.Header.Get("X-Amz-Target"); e != a {
		t.Errorf("expect %v to be %v", e, a)
	}

}

func TestInputService7ProtocolTestIdempotencyTokenAutoFillCase1(t *testing.T) {
	svc := NewInputService7ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService7TestShapeInputService7TestCaseOperation2Input{
		Token: aws.String("abc123"),
	}
	req, _ := svc.InputService7TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"Token":"abc123"}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService7ProtocolTestIdempotencyTokenAutoFillCase2(t *testing.T) {
	svc := NewInputService7ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})
	input := &InputService7TestShapeInputService7TestCaseOperation2Input{}
	req, _ := svc.InputService7TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	jsonrpc.Build(req)
	if req.Error != nil {
		t.Errorf("expect no error, got %v", req.Error)
	}

	// assert body
	if r.Body == nil {
		t.Errorf("expect body not to be nil")
	}
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertJSON(t, `{"Token":"00000000-0000-4000-8000-000000000000"}`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}
