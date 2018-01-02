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

// OutputService1ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService1ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService1ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService1ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService1ProtocolTest client from just a session.
//     svc := outputservice1protocoltest.New(mySession)
//
//     // Create a OutputService1ProtocolTest client with additional configuration
//     svc := outputservice1protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService1ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService1ProtocolTest {
	c := p.ClientConfig("outputservice1protocoltest", cfgs...)
	return newOutputService1ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService1ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService1ProtocolTest {
	svc := &OutputService1ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice1protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "",
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

// newRequest creates a new request for a OutputService1ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService1ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService1TestCaseOperation1 = "OperationName"

// OutputService1TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService1TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService1TestCaseOperation1 for more information on using the OutputService1TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService1TestCaseOperation1Request method.
//    req, resp := client.OutputService1TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService1ProtocolTest) OutputService1TestCaseOperation1Request(input *OutputService1TestShapeOutputService1TestCaseOperation1Input) (req *request.Request, output *OutputService1TestShapeOutputService1TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService1TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService1TestShapeOutputService1TestCaseOperation1Input{}
	}

	output = &OutputService1TestShapeOutputService1TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService1TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService1TestCaseOperation1 for usage and error information.
func (c *OutputService1ProtocolTest) OutputService1TestCaseOperation1(input *OutputService1TestShapeOutputService1TestCaseOperation1Input) (*OutputService1TestShapeOutputService1TestCaseOperation1Output, error) {
	req, out := c.OutputService1TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService1TestCaseOperation1WithContext is the same as OutputService1TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService1TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService1ProtocolTest) OutputService1TestCaseOperation1WithContext(ctx aws.Context, input *OutputService1TestShapeOutputService1TestCaseOperation1Input, opts ...request.Option) (*OutputService1TestShapeOutputService1TestCaseOperation1Output, error) {
	req, out := c.OutputService1TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService1TestShapeOutputService1TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService1TestShapeOutputService1TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Char *string `type:"character"`

	Double *float64 `type:"double"`

	FalseBool *bool `type:"boolean"`

	Float *float64 `type:"float"`

	Long *int64 `type:"long"`

	Num *int64 `type:"integer"`

	Str *string `type:"string"`

	TrueBool *bool `type:"boolean"`
}

// SetChar sets the Char field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetChar(v string) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Char = &v
	return s
}

// SetDouble sets the Double field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetDouble(v float64) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Double = &v
	return s
}

// SetFalseBool sets the FalseBool field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetFalseBool(v bool) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.FalseBool = &v
	return s
}

// SetFloat sets the Float field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetFloat(v float64) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Float = &v
	return s
}

// SetLong sets the Long field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetLong(v int64) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Long = &v
	return s
}

// SetNum sets the Num field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetNum(v int64) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Num = &v
	return s
}

// SetStr sets the Str field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetStr(v string) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Str = &v
	return s
}

// SetTrueBool sets the TrueBool field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetTrueBool(v bool) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.TrueBool = &v
	return s
}

// OutputService2ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService2ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService2ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService2ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService2ProtocolTest client from just a session.
//     svc := outputservice2protocoltest.New(mySession)
//
//     // Create a OutputService2ProtocolTest client with additional configuration
//     svc := outputservice2protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService2ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService2ProtocolTest {
	c := p.ClientConfig("outputservice2protocoltest", cfgs...)
	return newOutputService2ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService2ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService2ProtocolTest {
	svc := &OutputService2ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice2protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "",
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

// newRequest creates a new request for a OutputService2ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService2ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService2TestCaseOperation1 = "OperationName"

// OutputService2TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService2TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService2TestCaseOperation1 for more information on using the OutputService2TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService2TestCaseOperation1Request method.
//    req, resp := client.OutputService2TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService2ProtocolTest) OutputService2TestCaseOperation1Request(input *OutputService2TestShapeOutputService2TestCaseOperation1Input) (req *request.Request, output *OutputService2TestShapeOutputService2TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService2TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService2TestShapeOutputService2TestCaseOperation1Input{}
	}

	output = &OutputService2TestShapeOutputService2TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService2TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService2TestCaseOperation1 for usage and error information.
func (c *OutputService2ProtocolTest) OutputService2TestCaseOperation1(input *OutputService2TestShapeOutputService2TestCaseOperation1Input) (*OutputService2TestShapeOutputService2TestCaseOperation1Output, error) {
	req, out := c.OutputService2TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService2TestCaseOperation1WithContext is the same as OutputService2TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService2TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService2ProtocolTest) OutputService2TestCaseOperation1WithContext(ctx aws.Context, input *OutputService2TestShapeOutputService2TestCaseOperation1Input, opts ...request.Option) (*OutputService2TestShapeOutputService2TestCaseOperation1Output, error) {
	req, out := c.OutputService2TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService2TestShapeBlobContainer struct {
	_ struct{} `type:"structure"`

	// Foo is automatically base64 encoded/decoded by the SDK.
	Foo []byte `locationName:"foo" type:"blob"`
}

// SetFoo sets the Foo field's value.
func (s *OutputService2TestShapeBlobContainer) SetFoo(v []byte) *OutputService2TestShapeBlobContainer {
	s.Foo = v
	return s
}

type OutputService2TestShapeOutputService2TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService2TestShapeOutputService2TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	// BlobMember is automatically base64 encoded/decoded by the SDK.
	BlobMember []byte `type:"blob"`

	StructMember *OutputService2TestShapeBlobContainer `type:"structure"`
}

// SetBlobMember sets the BlobMember field's value.
func (s *OutputService2TestShapeOutputService2TestCaseOperation1Output) SetBlobMember(v []byte) *OutputService2TestShapeOutputService2TestCaseOperation1Output {
	s.BlobMember = v
	return s
}

// SetStructMember sets the StructMember field's value.
func (s *OutputService2TestShapeOutputService2TestCaseOperation1Output) SetStructMember(v *OutputService2TestShapeBlobContainer) *OutputService2TestShapeOutputService2TestCaseOperation1Output {
	s.StructMember = v
	return s
}

// OutputService3ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService3ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService3ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService3ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService3ProtocolTest client from just a session.
//     svc := outputservice3protocoltest.New(mySession)
//
//     // Create a OutputService3ProtocolTest client with additional configuration
//     svc := outputservice3protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService3ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService3ProtocolTest {
	c := p.ClientConfig("outputservice3protocoltest", cfgs...)
	return newOutputService3ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService3ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService3ProtocolTest {
	svc := &OutputService3ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice3protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "",
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

// newRequest creates a new request for a OutputService3ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService3ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService3TestCaseOperation1 = "OperationName"

// OutputService3TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService3TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService3TestCaseOperation1 for more information on using the OutputService3TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService3TestCaseOperation1Request method.
//    req, resp := client.OutputService3TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService3ProtocolTest) OutputService3TestCaseOperation1Request(input *OutputService3TestShapeOutputService3TestCaseOperation1Input) (req *request.Request, output *OutputService3TestShapeOutputService3TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService3TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService3TestShapeOutputService3TestCaseOperation1Input{}
	}

	output = &OutputService3TestShapeOutputService3TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService3TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService3TestCaseOperation1 for usage and error information.
func (c *OutputService3ProtocolTest) OutputService3TestCaseOperation1(input *OutputService3TestShapeOutputService3TestCaseOperation1Input) (*OutputService3TestShapeOutputService3TestCaseOperation1Output, error) {
	req, out := c.OutputService3TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService3TestCaseOperation1WithContext is the same as OutputService3TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService3TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService3ProtocolTest) OutputService3TestCaseOperation1WithContext(ctx aws.Context, input *OutputService3TestShapeOutputService3TestCaseOperation1Input, opts ...request.Option) (*OutputService3TestShapeOutputService3TestCaseOperation1Output, error) {
	req, out := c.OutputService3TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService3TestShapeOutputService3TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService3TestShapeOutputService3TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	StructMember *OutputService3TestShapeTimeContainer `type:"structure"`

	TimeMember *time.Time `type:"timestamp" timestampFormat:"unix"`
}

// SetStructMember sets the StructMember field's value.
func (s *OutputService3TestShapeOutputService3TestCaseOperation1Output) SetStructMember(v *OutputService3TestShapeTimeContainer) *OutputService3TestShapeOutputService3TestCaseOperation1Output {
	s.StructMember = v
	return s
}

// SetTimeMember sets the TimeMember field's value.
func (s *OutputService3TestShapeOutputService3TestCaseOperation1Output) SetTimeMember(v time.Time) *OutputService3TestShapeOutputService3TestCaseOperation1Output {
	s.TimeMember = &v
	return s
}

type OutputService3TestShapeTimeContainer struct {
	_ struct{} `type:"structure"`

	Foo *time.Time `locationName:"foo" type:"timestamp" timestampFormat:"unix"`
}

// SetFoo sets the Foo field's value.
func (s *OutputService3TestShapeTimeContainer) SetFoo(v time.Time) *OutputService3TestShapeTimeContainer {
	s.Foo = &v
	return s
}

// OutputService4ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService4ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService4ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService4ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService4ProtocolTest client from just a session.
//     svc := outputservice4protocoltest.New(mySession)
//
//     // Create a OutputService4ProtocolTest client with additional configuration
//     svc := outputservice4protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService4ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService4ProtocolTest {
	c := p.ClientConfig("outputservice4protocoltest", cfgs...)
	return newOutputService4ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService4ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService4ProtocolTest {
	svc := &OutputService4ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice4protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "",
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

// newRequest creates a new request for a OutputService4ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService4ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService4TestCaseOperation1 = "OperationName"

// OutputService4TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService4TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService4TestCaseOperation1 for more information on using the OutputService4TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService4TestCaseOperation1Request method.
//    req, resp := client.OutputService4TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1Request(input *OutputService4TestShapeOutputService4TestCaseOperation1Input) (req *request.Request, output *OutputService4TestShapeOutputService4TestCaseOperation2Output) {
	op := &request.Operation{
		Name:     opOutputService4TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService4TestShapeOutputService4TestCaseOperation1Input{}
	}

	output = &OutputService4TestShapeOutputService4TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService4TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService4TestCaseOperation1 for usage and error information.
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1(input *OutputService4TestShapeOutputService4TestCaseOperation1Input) (*OutputService4TestShapeOutputService4TestCaseOperation2Output, error) {
	req, out := c.OutputService4TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService4TestCaseOperation1WithContext is the same as OutputService4TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService4TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1WithContext(ctx aws.Context, input *OutputService4TestShapeOutputService4TestCaseOperation1Input, opts ...request.Option) (*OutputService4TestShapeOutputService4TestCaseOperation2Output, error) {
	req, out := c.OutputService4TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

const opOutputService4TestCaseOperation2 = "OperationName"

// OutputService4TestCaseOperation2Request generates a "aws/request.Request" representing the
// client's request for the OutputService4TestCaseOperation2 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService4TestCaseOperation2 for more information on using the OutputService4TestCaseOperation2
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService4TestCaseOperation2Request method.
//    req, resp := client.OutputService4TestCaseOperation2Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation2Request(input *OutputService4TestShapeOutputService4TestCaseOperation2Input) (req *request.Request, output *OutputService4TestShapeOutputService4TestCaseOperation2Output) {
	op := &request.Operation{
		Name:     opOutputService4TestCaseOperation2,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService4TestShapeOutputService4TestCaseOperation2Input{}
	}

	output = &OutputService4TestShapeOutputService4TestCaseOperation2Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService4TestCaseOperation2 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService4TestCaseOperation2 for usage and error information.
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation2(input *OutputService4TestShapeOutputService4TestCaseOperation2Input) (*OutputService4TestShapeOutputService4TestCaseOperation2Output, error) {
	req, out := c.OutputService4TestCaseOperation2Request(input)
	return out, req.Send()
}

// OutputService4TestCaseOperation2WithContext is the same as OutputService4TestCaseOperation2 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService4TestCaseOperation2 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation2WithContext(ctx aws.Context, input *OutputService4TestShapeOutputService4TestCaseOperation2Input, opts ...request.Option) (*OutputService4TestShapeOutputService4TestCaseOperation2Output, error) {
	req, out := c.OutputService4TestCaseOperation2Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService4TestShapeOutputService4TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService4TestShapeOutputService4TestCaseOperation2Input struct {
	_ struct{} `type:"structure"`
}

type OutputService4TestShapeOutputService4TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`

	ListMember []*string `type:"list"`

	ListMemberMap []map[string]*string `type:"list"`

	ListMemberStruct []*OutputService4TestShapeStructType `type:"list"`
}

// SetListMember sets the ListMember field's value.
func (s *OutputService4TestShapeOutputService4TestCaseOperation2Output) SetListMember(v []*string) *OutputService4TestShapeOutputService4TestCaseOperation2Output {
	s.ListMember = v
	return s
}

// SetListMemberMap sets the ListMemberMap field's value.
func (s *OutputService4TestShapeOutputService4TestCaseOperation2Output) SetListMemberMap(v []map[string]*string) *OutputService4TestShapeOutputService4TestCaseOperation2Output {
	s.ListMemberMap = v
	return s
}

// SetListMemberStruct sets the ListMemberStruct field's value.
func (s *OutputService4TestShapeOutputService4TestCaseOperation2Output) SetListMemberStruct(v []*OutputService4TestShapeStructType) *OutputService4TestShapeOutputService4TestCaseOperation2Output {
	s.ListMemberStruct = v
	return s
}

type OutputService4TestShapeStructType struct {
	_ struct{} `type:"structure"`
}

// OutputService5ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService5ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService5ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService5ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService5ProtocolTest client from just a session.
//     svc := outputservice5protocoltest.New(mySession)
//
//     // Create a OutputService5ProtocolTest client with additional configuration
//     svc := outputservice5protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService5ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService5ProtocolTest {
	c := p.ClientConfig("outputservice5protocoltest", cfgs...)
	return newOutputService5ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService5ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService5ProtocolTest {
	svc := &OutputService5ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice5protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "",
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

// newRequest creates a new request for a OutputService5ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService5ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService5TestCaseOperation1 = "OperationName"

// OutputService5TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService5TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService5TestCaseOperation1 for more information on using the OutputService5TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService5TestCaseOperation1Request method.
//    req, resp := client.OutputService5TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService5ProtocolTest) OutputService5TestCaseOperation1Request(input *OutputService5TestShapeOutputService5TestCaseOperation1Input) (req *request.Request, output *OutputService5TestShapeOutputService5TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService5TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService5TestShapeOutputService5TestCaseOperation1Input{}
	}

	output = &OutputService5TestShapeOutputService5TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService5TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService5TestCaseOperation1 for usage and error information.
func (c *OutputService5ProtocolTest) OutputService5TestCaseOperation1(input *OutputService5TestShapeOutputService5TestCaseOperation1Input) (*OutputService5TestShapeOutputService5TestCaseOperation1Output, error) {
	req, out := c.OutputService5TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService5TestCaseOperation1WithContext is the same as OutputService5TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService5TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService5ProtocolTest) OutputService5TestCaseOperation1WithContext(ctx aws.Context, input *OutputService5TestShapeOutputService5TestCaseOperation1Input, opts ...request.Option) (*OutputService5TestShapeOutputService5TestCaseOperation1Output, error) {
	req, out := c.OutputService5TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService5TestShapeOutputService5TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService5TestShapeOutputService5TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	MapMember map[string][]*int64 `type:"map"`
}

// SetMapMember sets the MapMember field's value.
func (s *OutputService5TestShapeOutputService5TestCaseOperation1Output) SetMapMember(v map[string][]*int64) *OutputService5TestShapeOutputService5TestCaseOperation1Output {
	s.MapMember = v
	return s
}

// OutputService6ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService6ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService6ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService6ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService6ProtocolTest client from just a session.
//     svc := outputservice6protocoltest.New(mySession)
//
//     // Create a OutputService6ProtocolTest client with additional configuration
//     svc := outputservice6protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService6ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService6ProtocolTest {
	c := p.ClientConfig("outputservice6protocoltest", cfgs...)
	return newOutputService6ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService6ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService6ProtocolTest {
	svc := &OutputService6ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice6protocoltest",
				SigningName:   signingName,
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "",
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

// newRequest creates a new request for a OutputService6ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService6ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService6TestCaseOperation1 = "OperationName"

// OutputService6TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService6TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService6TestCaseOperation1 for more information on using the OutputService6TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService6TestCaseOperation1Request method.
//    req, resp := client.OutputService6TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService6ProtocolTest) OutputService6TestCaseOperation1Request(input *OutputService6TestShapeOutputService6TestCaseOperation1Input) (req *request.Request, output *OutputService6TestShapeOutputService6TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService6TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService6TestShapeOutputService6TestCaseOperation1Input{}
	}

	output = &OutputService6TestShapeOutputService6TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService6TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService6TestCaseOperation1 for usage and error information.
func (c *OutputService6ProtocolTest) OutputService6TestCaseOperation1(input *OutputService6TestShapeOutputService6TestCaseOperation1Input) (*OutputService6TestShapeOutputService6TestCaseOperation1Output, error) {
	req, out := c.OutputService6TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService6TestCaseOperation1WithContext is the same as OutputService6TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService6TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService6ProtocolTest) OutputService6TestCaseOperation1WithContext(ctx aws.Context, input *OutputService6TestShapeOutputService6TestCaseOperation1Input, opts ...request.Option) (*OutputService6TestShapeOutputService6TestCaseOperation1Output, error) {
	req, out := c.OutputService6TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService6TestShapeOutputService6TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService6TestShapeOutputService6TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	StrType *string `type:"string"`
}

// SetStrType sets the StrType field's value.
func (s *OutputService6TestShapeOutputService6TestCaseOperation1Output) SetStrType(v string) *OutputService6TestShapeOutputService6TestCaseOperation1Output {
	s.StrType = &v
	return s
}

//
// Tests begin here
//

func TestOutputService1ProtocolTestScalarMembersCase1(t *testing.T) {
	svc := NewOutputService1ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"Str\": \"myname\", \"Num\": 123, \"FalseBool\": false, \"TrueBool\": true, \"Float\": 1.2, \"Double\": 1.3, \"Long\": 200, \"Char\": \"a\"}"))
	req, out := svc.OutputService1TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "a", *out.Char; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := 1.3, *out.Double; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := false, *out.FalseBool; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := 1.2, *out.Float; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := int64(200), *out.Long; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := int64(123), *out.Num; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "myname", *out.Str; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := true, *out.TrueBool; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService2ProtocolTestBlobMembersCase1(t *testing.T) {
	svc := NewOutputService2ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"BlobMember\": \"aGkh\", \"StructMember\": {\"foo\": \"dGhlcmUh\"}}"))
	req, out := svc.OutputService2TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "hi!", string(out.BlobMember); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "there!", string(out.StructMember.Foo); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService3ProtocolTestTimestampMembersCase1(t *testing.T) {
	svc := NewOutputService3ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"TimeMember\": 1398796238, \"StructMember\": {\"foo\": 1398796238}}"))
	req, out := svc.OutputService3TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := time.Unix(1.398796238e+09, 0).UTC().String(), out.StructMember.Foo.String(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := time.Unix(1.398796238e+09, 0).UTC().String(), out.TimeMember.String(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService4ProtocolTestListsCase1(t *testing.T) {
	svc := NewOutputService4ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"ListMember\": [\"a\", \"b\"]}"))
	req, out := svc.OutputService4TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "a", *out.ListMember[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "b", *out.ListMember[1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService4ProtocolTestListsCase2(t *testing.T) {
	svc := NewOutputService4ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"ListMember\": [\"a\", null], \"ListMemberMap\": [{}, null, null, {}], \"ListMemberStruct\": [{}, null, null, {}]}"))
	req, out := svc.OutputService4TestCaseOperation2Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "a", *out.ListMember[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e := out.ListMember[1]; e != nil {
		t.Errorf("expect nil, got %v", e)
	}
	if e := out.ListMemberMap[1]; e != nil {
		t.Errorf("expect nil, got %v", e)
	}
	if e := out.ListMemberMap[2]; e != nil {
		t.Errorf("expect nil, got %v", e)
	}
	if e := out.ListMemberStruct[1]; e != nil {
		t.Errorf("expect nil, got %v", e)
	}
	if e := out.ListMemberStruct[2]; e != nil {
		t.Errorf("expect nil, got %v", e)
	}

}

func TestOutputService5ProtocolTestMapsCase1(t *testing.T) {
	svc := NewOutputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"MapMember\": {\"a\": [1, 2], \"b\": [3, 4]}}"))
	req, out := svc.OutputService5TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := int64(1), *out.MapMember["a"][0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := int64(2), *out.MapMember["a"][1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := int64(3), *out.MapMember["b"][0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := int64(4), *out.MapMember["b"][1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService6ProtocolTestIgnoresExtraDataCase1(t *testing.T) {
	svc := NewOutputService6ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("{\"foo\": \"bar\"}"))
	req, out := svc.OutputService6TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	jsonrpc.UnmarshalMeta(req)
	jsonrpc.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}

}
