package query_test

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
	"github.com/aws/aws-sdk-go/private/protocol/query"
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

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

	Num *int64 `locationName:"FooNum" type:"integer"`

	Str *string `type:"string"`

	Timestamp *time.Time `type:"timestamp" timestampFormat:"iso8601"`

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

// SetTimestamp sets the Timestamp field's value.
func (s *OutputService1TestShapeOutputService1TestCaseOperation1Output) SetTimestamp(v time.Time) *OutputService1TestShapeOutputService1TestCaseOperation1Output {
	s.Timestamp = &v
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

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

type OutputService2TestShapeOutputService2TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService2TestShapeOutputService2TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Num *int64 `type:"integer"`

	Str *string `type:"string"`
}

// SetNum sets the Num field's value.
func (s *OutputService2TestShapeOutputService2TestCaseOperation1Output) SetNum(v int64) *OutputService2TestShapeOutputService2TestCaseOperation1Output {
	s.Num = &v
	return s
}

// SetStr sets the Str field's value.
func (s *OutputService2TestShapeOutputService2TestCaseOperation1Output) SetStr(v string) *OutputService2TestShapeOutputService2TestCaseOperation1Output {
	s.Str = &v
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

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

	// Blob is automatically base64 encoded/decoded by the SDK.
	Blob []byte `type:"blob"`
}

// SetBlob sets the Blob field's value.
func (s *OutputService3TestShapeOutputService3TestCaseOperation1Output) SetBlob(v []byte) *OutputService3TestShapeOutputService3TestCaseOperation1Output {
	s.Blob = v
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

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
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1Request(input *OutputService4TestShapeOutputService4TestCaseOperation1Input) (req *request.Request, output *OutputService4TestShapeOutputService4TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService4TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService4TestShapeOutputService4TestCaseOperation1Input{}
	}

	output = &OutputService4TestShapeOutputService4TestCaseOperation1Output{}
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
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1(input *OutputService4TestShapeOutputService4TestCaseOperation1Input) (*OutputService4TestShapeOutputService4TestCaseOperation1Output, error) {
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
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1WithContext(ctx aws.Context, input *OutputService4TestShapeOutputService4TestCaseOperation1Input, opts ...request.Option) (*OutputService4TestShapeOutputService4TestCaseOperation1Output, error) {
	req, out := c.OutputService4TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService4TestShapeOutputService4TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService4TestShapeOutputService4TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	ListMember []*string `type:"list"`
}

// SetListMember sets the ListMember field's value.
func (s *OutputService4TestShapeOutputService4TestCaseOperation1Output) SetListMember(v []*string) *OutputService4TestShapeOutputService4TestCaseOperation1Output {
	s.ListMember = v
	return s
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

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

	ListMember []*string `locationNameList:"item" type:"list"`
}

// SetListMember sets the ListMember field's value.
func (s *OutputService5TestShapeOutputService5TestCaseOperation1Output) SetListMember(v []*string) *OutputService5TestShapeOutputService5TestCaseOperation1Output {
	s.ListMember = v
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

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

	ListMember []*string `type:"list" flattened:"true"`
}

// SetListMember sets the ListMember field's value.
func (s *OutputService6TestShapeOutputService6TestCaseOperation1Output) SetListMember(v []*string) *OutputService6TestShapeOutputService6TestCaseOperation1Output {
	s.ListMember = v
	return s
}

// OutputService7ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService7ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService7ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService7ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService7ProtocolTest client from just a session.
//     svc := outputservice7protocoltest.New(mySession)
//
//     // Create a OutputService7ProtocolTest client with additional configuration
//     svc := outputservice7protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService7ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService7ProtocolTest {
	c := p.ClientConfig("outputservice7protocoltest", cfgs...)
	return newOutputService7ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService7ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService7ProtocolTest {
	svc := &OutputService7ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice7protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService7ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService7ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService7TestCaseOperation1 = "OperationName"

// OutputService7TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService7TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService7TestCaseOperation1 for more information on using the OutputService7TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService7TestCaseOperation1Request method.
//    req, resp := client.OutputService7TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService7ProtocolTest) OutputService7TestCaseOperation1Request(input *OutputService7TestShapeOutputService7TestCaseOperation1Input) (req *request.Request, output *OutputService7TestShapeOutputService7TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService7TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService7TestShapeOutputService7TestCaseOperation1Input{}
	}

	output = &OutputService7TestShapeOutputService7TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService7TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService7TestCaseOperation1 for usage and error information.
func (c *OutputService7ProtocolTest) OutputService7TestCaseOperation1(input *OutputService7TestShapeOutputService7TestCaseOperation1Input) (*OutputService7TestShapeOutputService7TestCaseOperation1Output, error) {
	req, out := c.OutputService7TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService7TestCaseOperation1WithContext is the same as OutputService7TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService7TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService7ProtocolTest) OutputService7TestCaseOperation1WithContext(ctx aws.Context, input *OutputService7TestShapeOutputService7TestCaseOperation1Input, opts ...request.Option) (*OutputService7TestShapeOutputService7TestCaseOperation1Output, error) {
	req, out := c.OutputService7TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService7TestShapeOutputService7TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService7TestShapeOutputService7TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	ListMember []*string `type:"list" flattened:"true"`
}

// SetListMember sets the ListMember field's value.
func (s *OutputService7TestShapeOutputService7TestCaseOperation1Output) SetListMember(v []*string) *OutputService7TestShapeOutputService7TestCaseOperation1Output {
	s.ListMember = v
	return s
}

// OutputService8ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService8ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService8ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService8ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService8ProtocolTest client from just a session.
//     svc := outputservice8protocoltest.New(mySession)
//
//     // Create a OutputService8ProtocolTest client with additional configuration
//     svc := outputservice8protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService8ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService8ProtocolTest {
	c := p.ClientConfig("outputservice8protocoltest", cfgs...)
	return newOutputService8ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService8ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService8ProtocolTest {
	svc := &OutputService8ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice8protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService8ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService8ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService8TestCaseOperation1 = "OperationName"

// OutputService8TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService8TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService8TestCaseOperation1 for more information on using the OutputService8TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService8TestCaseOperation1Request method.
//    req, resp := client.OutputService8TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService8ProtocolTest) OutputService8TestCaseOperation1Request(input *OutputService8TestShapeOutputService8TestCaseOperation1Input) (req *request.Request, output *OutputService8TestShapeOutputService8TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService8TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService8TestShapeOutputService8TestCaseOperation1Input{}
	}

	output = &OutputService8TestShapeOutputService8TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService8TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService8TestCaseOperation1 for usage and error information.
func (c *OutputService8ProtocolTest) OutputService8TestCaseOperation1(input *OutputService8TestShapeOutputService8TestCaseOperation1Input) (*OutputService8TestShapeOutputService8TestCaseOperation1Output, error) {
	req, out := c.OutputService8TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService8TestCaseOperation1WithContext is the same as OutputService8TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService8TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService8ProtocolTest) OutputService8TestCaseOperation1WithContext(ctx aws.Context, input *OutputService8TestShapeOutputService8TestCaseOperation1Input, opts ...request.Option) (*OutputService8TestShapeOutputService8TestCaseOperation1Output, error) {
	req, out := c.OutputService8TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService8TestShapeOutputService8TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService8TestShapeOutputService8TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	List []*OutputService8TestShapeStructureShape `type:"list"`
}

// SetList sets the List field's value.
func (s *OutputService8TestShapeOutputService8TestCaseOperation1Output) SetList(v []*OutputService8TestShapeStructureShape) *OutputService8TestShapeOutputService8TestCaseOperation1Output {
	s.List = v
	return s
}

type OutputService8TestShapeStructureShape struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Baz *string `type:"string"`

	Foo *string `type:"string"`
}

// SetBar sets the Bar field's value.
func (s *OutputService8TestShapeStructureShape) SetBar(v string) *OutputService8TestShapeStructureShape {
	s.Bar = &v
	return s
}

// SetBaz sets the Baz field's value.
func (s *OutputService8TestShapeStructureShape) SetBaz(v string) *OutputService8TestShapeStructureShape {
	s.Baz = &v
	return s
}

// SetFoo sets the Foo field's value.
func (s *OutputService8TestShapeStructureShape) SetFoo(v string) *OutputService8TestShapeStructureShape {
	s.Foo = &v
	return s
}

// OutputService9ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService9ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService9ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService9ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService9ProtocolTest client from just a session.
//     svc := outputservice9protocoltest.New(mySession)
//
//     // Create a OutputService9ProtocolTest client with additional configuration
//     svc := outputservice9protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService9ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService9ProtocolTest {
	c := p.ClientConfig("outputservice9protocoltest", cfgs...)
	return newOutputService9ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService9ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService9ProtocolTest {
	svc := &OutputService9ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice9protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService9ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService9ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService9TestCaseOperation1 = "OperationName"

// OutputService9TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService9TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService9TestCaseOperation1 for more information on using the OutputService9TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService9TestCaseOperation1Request method.
//    req, resp := client.OutputService9TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService9ProtocolTest) OutputService9TestCaseOperation1Request(input *OutputService9TestShapeOutputService9TestCaseOperation1Input) (req *request.Request, output *OutputService9TestShapeOutputService9TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService9TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService9TestShapeOutputService9TestCaseOperation1Input{}
	}

	output = &OutputService9TestShapeOutputService9TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService9TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService9TestCaseOperation1 for usage and error information.
func (c *OutputService9ProtocolTest) OutputService9TestCaseOperation1(input *OutputService9TestShapeOutputService9TestCaseOperation1Input) (*OutputService9TestShapeOutputService9TestCaseOperation1Output, error) {
	req, out := c.OutputService9TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService9TestCaseOperation1WithContext is the same as OutputService9TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService9TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService9ProtocolTest) OutputService9TestCaseOperation1WithContext(ctx aws.Context, input *OutputService9TestShapeOutputService9TestCaseOperation1Input, opts ...request.Option) (*OutputService9TestShapeOutputService9TestCaseOperation1Output, error) {
	req, out := c.OutputService9TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService9TestShapeOutputService9TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService9TestShapeOutputService9TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	List []*OutputService9TestShapeStructureShape `type:"list" flattened:"true"`
}

// SetList sets the List field's value.
func (s *OutputService9TestShapeOutputService9TestCaseOperation1Output) SetList(v []*OutputService9TestShapeStructureShape) *OutputService9TestShapeOutputService9TestCaseOperation1Output {
	s.List = v
	return s
}

type OutputService9TestShapeStructureShape struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Baz *string `type:"string"`

	Foo *string `type:"string"`
}

// SetBar sets the Bar field's value.
func (s *OutputService9TestShapeStructureShape) SetBar(v string) *OutputService9TestShapeStructureShape {
	s.Bar = &v
	return s
}

// SetBaz sets the Baz field's value.
func (s *OutputService9TestShapeStructureShape) SetBaz(v string) *OutputService9TestShapeStructureShape {
	s.Baz = &v
	return s
}

// SetFoo sets the Foo field's value.
func (s *OutputService9TestShapeStructureShape) SetFoo(v string) *OutputService9TestShapeStructureShape {
	s.Foo = &v
	return s
}

// OutputService10ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService10ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService10ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService10ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService10ProtocolTest client from just a session.
//     svc := outputservice10protocoltest.New(mySession)
//
//     // Create a OutputService10ProtocolTest client with additional configuration
//     svc := outputservice10protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService10ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService10ProtocolTest {
	c := p.ClientConfig("outputservice10protocoltest", cfgs...)
	return newOutputService10ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService10ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService10ProtocolTest {
	svc := &OutputService10ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice10protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService10ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService10ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService10TestCaseOperation1 = "OperationName"

// OutputService10TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService10TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService10TestCaseOperation1 for more information on using the OutputService10TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService10TestCaseOperation1Request method.
//    req, resp := client.OutputService10TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService10ProtocolTest) OutputService10TestCaseOperation1Request(input *OutputService10TestShapeOutputService10TestCaseOperation1Input) (req *request.Request, output *OutputService10TestShapeOutputService10TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService10TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService10TestShapeOutputService10TestCaseOperation1Input{}
	}

	output = &OutputService10TestShapeOutputService10TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService10TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService10TestCaseOperation1 for usage and error information.
func (c *OutputService10ProtocolTest) OutputService10TestCaseOperation1(input *OutputService10TestShapeOutputService10TestCaseOperation1Input) (*OutputService10TestShapeOutputService10TestCaseOperation1Output, error) {
	req, out := c.OutputService10TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService10TestCaseOperation1WithContext is the same as OutputService10TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService10TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService10ProtocolTest) OutputService10TestCaseOperation1WithContext(ctx aws.Context, input *OutputService10TestShapeOutputService10TestCaseOperation1Input, opts ...request.Option) (*OutputService10TestShapeOutputService10TestCaseOperation1Output, error) {
	req, out := c.OutputService10TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService10TestShapeOutputService10TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService10TestShapeOutputService10TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	List []*string `locationNameList:"NamedList" type:"list" flattened:"true"`
}

// SetList sets the List field's value.
func (s *OutputService10TestShapeOutputService10TestCaseOperation1Output) SetList(v []*string) *OutputService10TestShapeOutputService10TestCaseOperation1Output {
	s.List = v
	return s
}

// OutputService11ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService11ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService11ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService11ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService11ProtocolTest client from just a session.
//     svc := outputservice11protocoltest.New(mySession)
//
//     // Create a OutputService11ProtocolTest client with additional configuration
//     svc := outputservice11protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService11ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService11ProtocolTest {
	c := p.ClientConfig("outputservice11protocoltest", cfgs...)
	return newOutputService11ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService11ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService11ProtocolTest {
	svc := &OutputService11ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice11protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService11ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService11ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService11TestCaseOperation1 = "OperationName"

// OutputService11TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService11TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService11TestCaseOperation1 for more information on using the OutputService11TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService11TestCaseOperation1Request method.
//    req, resp := client.OutputService11TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService11ProtocolTest) OutputService11TestCaseOperation1Request(input *OutputService11TestShapeOutputService11TestCaseOperation1Input) (req *request.Request, output *OutputService11TestShapeOutputService11TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService11TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService11TestShapeOutputService11TestCaseOperation1Input{}
	}

	output = &OutputService11TestShapeOutputService11TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService11TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService11TestCaseOperation1 for usage and error information.
func (c *OutputService11ProtocolTest) OutputService11TestCaseOperation1(input *OutputService11TestShapeOutputService11TestCaseOperation1Input) (*OutputService11TestShapeOutputService11TestCaseOperation1Output, error) {
	req, out := c.OutputService11TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService11TestCaseOperation1WithContext is the same as OutputService11TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService11TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService11ProtocolTest) OutputService11TestCaseOperation1WithContext(ctx aws.Context, input *OutputService11TestShapeOutputService11TestCaseOperation1Input, opts ...request.Option) (*OutputService11TestShapeOutputService11TestCaseOperation1Output, error) {
	req, out := c.OutputService11TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService11TestShapeOutputService11TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService11TestShapeOutputService11TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Map map[string]*OutputService11TestShapeStructType `type:"map"`
}

// SetMap sets the Map field's value.
func (s *OutputService11TestShapeOutputService11TestCaseOperation1Output) SetMap(v map[string]*OutputService11TestShapeStructType) *OutputService11TestShapeOutputService11TestCaseOperation1Output {
	s.Map = v
	return s
}

type OutputService11TestShapeStructType struct {
	_ struct{} `type:"structure"`

	Foo *string `locationName:"foo" type:"string"`
}

// SetFoo sets the Foo field's value.
func (s *OutputService11TestShapeStructType) SetFoo(v string) *OutputService11TestShapeStructType {
	s.Foo = &v
	return s
}

// OutputService12ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService12ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService12ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService12ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService12ProtocolTest client from just a session.
//     svc := outputservice12protocoltest.New(mySession)
//
//     // Create a OutputService12ProtocolTest client with additional configuration
//     svc := outputservice12protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService12ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService12ProtocolTest {
	c := p.ClientConfig("outputservice12protocoltest", cfgs...)
	return newOutputService12ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService12ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService12ProtocolTest {
	svc := &OutputService12ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice12protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService12ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService12ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService12TestCaseOperation1 = "OperationName"

// OutputService12TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService12TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService12TestCaseOperation1 for more information on using the OutputService12TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService12TestCaseOperation1Request method.
//    req, resp := client.OutputService12TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService12ProtocolTest) OutputService12TestCaseOperation1Request(input *OutputService12TestShapeOutputService12TestCaseOperation1Input) (req *request.Request, output *OutputService12TestShapeOutputService12TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService12TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService12TestShapeOutputService12TestCaseOperation1Input{}
	}

	output = &OutputService12TestShapeOutputService12TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService12TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService12TestCaseOperation1 for usage and error information.
func (c *OutputService12ProtocolTest) OutputService12TestCaseOperation1(input *OutputService12TestShapeOutputService12TestCaseOperation1Input) (*OutputService12TestShapeOutputService12TestCaseOperation1Output, error) {
	req, out := c.OutputService12TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService12TestCaseOperation1WithContext is the same as OutputService12TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService12TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService12ProtocolTest) OutputService12TestCaseOperation1WithContext(ctx aws.Context, input *OutputService12TestShapeOutputService12TestCaseOperation1Input, opts ...request.Option) (*OutputService12TestShapeOutputService12TestCaseOperation1Output, error) {
	req, out := c.OutputService12TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService12TestShapeOutputService12TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService12TestShapeOutputService12TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Map map[string]*string `type:"map" flattened:"true"`
}

// SetMap sets the Map field's value.
func (s *OutputService12TestShapeOutputService12TestCaseOperation1Output) SetMap(v map[string]*string) *OutputService12TestShapeOutputService12TestCaseOperation1Output {
	s.Map = v
	return s
}

// OutputService13ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService13ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService13ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService13ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService13ProtocolTest client from just a session.
//     svc := outputservice13protocoltest.New(mySession)
//
//     // Create a OutputService13ProtocolTest client with additional configuration
//     svc := outputservice13protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService13ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService13ProtocolTest {
	c := p.ClientConfig("outputservice13protocoltest", cfgs...)
	return newOutputService13ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService13ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService13ProtocolTest {
	svc := &OutputService13ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice13protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService13ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService13ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService13TestCaseOperation1 = "OperationName"

// OutputService13TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService13TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService13TestCaseOperation1 for more information on using the OutputService13TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService13TestCaseOperation1Request method.
//    req, resp := client.OutputService13TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService13ProtocolTest) OutputService13TestCaseOperation1Request(input *OutputService13TestShapeOutputService13TestCaseOperation1Input) (req *request.Request, output *OutputService13TestShapeOutputService13TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService13TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService13TestShapeOutputService13TestCaseOperation1Input{}
	}

	output = &OutputService13TestShapeOutputService13TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService13TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService13TestCaseOperation1 for usage and error information.
func (c *OutputService13ProtocolTest) OutputService13TestCaseOperation1(input *OutputService13TestShapeOutputService13TestCaseOperation1Input) (*OutputService13TestShapeOutputService13TestCaseOperation1Output, error) {
	req, out := c.OutputService13TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService13TestCaseOperation1WithContext is the same as OutputService13TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService13TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService13ProtocolTest) OutputService13TestCaseOperation1WithContext(ctx aws.Context, input *OutputService13TestShapeOutputService13TestCaseOperation1Input, opts ...request.Option) (*OutputService13TestShapeOutputService13TestCaseOperation1Output, error) {
	req, out := c.OutputService13TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService13TestShapeOutputService13TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService13TestShapeOutputService13TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Map map[string]*string `locationName:"Attribute" locationNameKey:"Name" locationNameValue:"Value" type:"map" flattened:"true"`
}

// SetMap sets the Map field's value.
func (s *OutputService13TestShapeOutputService13TestCaseOperation1Output) SetMap(v map[string]*string) *OutputService13TestShapeOutputService13TestCaseOperation1Output {
	s.Map = v
	return s
}

// OutputService14ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService14ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService14ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService14ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService14ProtocolTest client from just a session.
//     svc := outputservice14protocoltest.New(mySession)
//
//     // Create a OutputService14ProtocolTest client with additional configuration
//     svc := outputservice14protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService14ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService14ProtocolTest {
	c := p.ClientConfig("outputservice14protocoltest", cfgs...)
	return newOutputService14ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService14ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService14ProtocolTest {
	svc := &OutputService14ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice14protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService14ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService14ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService14TestCaseOperation1 = "OperationName"

// OutputService14TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService14TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService14TestCaseOperation1 for more information on using the OutputService14TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService14TestCaseOperation1Request method.
//    req, resp := client.OutputService14TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService14ProtocolTest) OutputService14TestCaseOperation1Request(input *OutputService14TestShapeOutputService14TestCaseOperation1Input) (req *request.Request, output *OutputService14TestShapeOutputService14TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService14TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService14TestShapeOutputService14TestCaseOperation1Input{}
	}

	output = &OutputService14TestShapeOutputService14TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService14TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService14TestCaseOperation1 for usage and error information.
func (c *OutputService14ProtocolTest) OutputService14TestCaseOperation1(input *OutputService14TestShapeOutputService14TestCaseOperation1Input) (*OutputService14TestShapeOutputService14TestCaseOperation1Output, error) {
	req, out := c.OutputService14TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService14TestCaseOperation1WithContext is the same as OutputService14TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService14TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService14ProtocolTest) OutputService14TestCaseOperation1WithContext(ctx aws.Context, input *OutputService14TestShapeOutputService14TestCaseOperation1Input, opts ...request.Option) (*OutputService14TestShapeOutputService14TestCaseOperation1Output, error) {
	req, out := c.OutputService14TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService14TestShapeOutputService14TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService14TestShapeOutputService14TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Map map[string]*string `locationNameKey:"foo" locationNameValue:"bar" type:"map" flattened:"true"`
}

// SetMap sets the Map field's value.
func (s *OutputService14TestShapeOutputService14TestCaseOperation1Output) SetMap(v map[string]*string) *OutputService14TestShapeOutputService14TestCaseOperation1Output {
	s.Map = v
	return s
}

// OutputService15ProtocolTest provides the API operation methods for making requests to
// . See this package's package overview docs
// for details on the service.
//
// OutputService15ProtocolTest methods are safe to use concurrently. It is not safe to
// modify mutate any of the struct's properties though.
type OutputService15ProtocolTest struct {
	*client.Client
}

// New creates a new instance of the OutputService15ProtocolTest client with a session.
// If additional configuration is needed for the client instance use the optional
// aws.Config parameter to add your extra config.
//
// Example:
//     // Create a OutputService15ProtocolTest client from just a session.
//     svc := outputservice15protocoltest.New(mySession)
//
//     // Create a OutputService15ProtocolTest client with additional configuration
//     svc := outputservice15protocoltest.New(mySession, aws.NewConfig().WithRegion("us-west-2"))
func NewOutputService15ProtocolTest(p client.ConfigProvider, cfgs ...*aws.Config) *OutputService15ProtocolTest {
	c := p.ClientConfig("outputservice15protocoltest", cfgs...)
	return newOutputService15ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion, c.SigningName)
}

// newClient creates, initializes and returns a new service client instance.
func newOutputService15ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion, signingName string) *OutputService15ProtocolTest {
	svc := &OutputService15ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "outputservice15protocoltest",
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
	svc.Handlers.Build.PushBackNamed(query.BuildHandler)
	svc.Handlers.Unmarshal.PushBackNamed(query.UnmarshalHandler)
	svc.Handlers.UnmarshalMeta.PushBackNamed(query.UnmarshalMetaHandler)
	svc.Handlers.UnmarshalError.PushBackNamed(query.UnmarshalErrorHandler)

	return svc
}

// newRequest creates a new request for a OutputService15ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService15ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService15TestCaseOperation1 = "OperationName"

// OutputService15TestCaseOperation1Request generates a "aws/request.Request" representing the
// client's request for the OutputService15TestCaseOperation1 operation. The "output" return
// value will be populated with the request's response once the request complets
// successfuly.
//
// Use "Send" method on the returned Request to send the API call to the service.
// the "output" return value is not valid until after Send returns without error.
//
// See OutputService15TestCaseOperation1 for more information on using the OutputService15TestCaseOperation1
// API call, and error handling.
//
// This method is useful when you want to inject custom logic or configuration
// into the SDK's request lifecycle. Such as custom headers, or retry logic.
//
//
//    // Example sending a request using the OutputService15TestCaseOperation1Request method.
//    req, resp := client.OutputService15TestCaseOperation1Request(params)
//
//    err := req.Send()
//    if err == nil { // resp is now filled
//        fmt.Println(resp)
//    }
func (c *OutputService15ProtocolTest) OutputService15TestCaseOperation1Request(input *OutputService15TestShapeOutputService15TestCaseOperation1Input) (req *request.Request, output *OutputService15TestShapeOutputService15TestCaseOperation1Output) {
	op := &request.Operation{
		Name:     opOutputService15TestCaseOperation1,
		HTTPPath: "/",
	}

	if input == nil {
		input = &OutputService15TestShapeOutputService15TestCaseOperation1Input{}
	}

	output = &OutputService15TestShapeOutputService15TestCaseOperation1Output{}
	req = c.newRequest(op, input, output)
	return
}

// OutputService15TestCaseOperation1 API operation for .
//
// Returns awserr.Error for service API and SDK errors. Use runtime type assertions
// with awserr.Error's Code and Message methods to get detailed information about
// the error.
//
// See the AWS API reference guide for 's
// API operation OutputService15TestCaseOperation1 for usage and error information.
func (c *OutputService15ProtocolTest) OutputService15TestCaseOperation1(input *OutputService15TestShapeOutputService15TestCaseOperation1Input) (*OutputService15TestShapeOutputService15TestCaseOperation1Output, error) {
	req, out := c.OutputService15TestCaseOperation1Request(input)
	return out, req.Send()
}

// OutputService15TestCaseOperation1WithContext is the same as OutputService15TestCaseOperation1 with the addition of
// the ability to pass a context and additional request options.
//
// See OutputService15TestCaseOperation1 for details on how to use this API operation.
//
// The context must be non-nil and will be used for request cancellation. If
// the context is nil a panic will occur. In the future the SDK may create
// sub-contexts for http.Requests. See https://golang.org/pkg/context/
// for more information on using Contexts.
func (c *OutputService15ProtocolTest) OutputService15TestCaseOperation1WithContext(ctx aws.Context, input *OutputService15TestShapeOutputService15TestCaseOperation1Input, opts ...request.Option) (*OutputService15TestShapeOutputService15TestCaseOperation1Output, error) {
	req, out := c.OutputService15TestCaseOperation1Request(input)
	req.SetContext(ctx)
	req.ApplyOptions(opts...)
	return out, req.Send()
}

type OutputService15TestShapeOutputService15TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`
}

type OutputService15TestShapeOutputService15TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`

	Foo *string `type:"string"`
}

// SetFoo sets the Foo field's value.
func (s *OutputService15TestShapeOutputService15TestCaseOperation1Output) SetFoo(v string) *OutputService15TestShapeOutputService15TestCaseOperation1Output {
	s.Foo = &v
	return s
}

//
// Tests begin here
//

func TestOutputService1ProtocolTestScalarMembersCase1(t *testing.T) {
	svc := NewOutputService1ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Str>myname</Str><FooNum>123</FooNum><FalseBool>false</FalseBool><TrueBool>true</TrueBool><Float>1.2</Float><Double>1.3</Double><Long>200</Long><Char>a</Char><Timestamp>2015-01-25T08:00:00Z</Timestamp></OperationNameResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService1TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
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
	if e, a := time.Unix(1.4221728e+09, 0).UTC().String(), out.Timestamp.String(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := true, *out.TrueBool; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService2ProtocolTestNotAllMembersInResponseCase1(t *testing.T) {
	svc := NewOutputService2ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Str>myname</Str></OperationNameResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService2TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "myname", *out.Str; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService3ProtocolTestBlobCase1(t *testing.T) {
	svc := NewOutputService3ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Blob>dmFsdWU=</Blob></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService3TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "value", string(out.Blob); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService4ProtocolTestListsCase1(t *testing.T) {
	svc := NewOutputService4ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember><member>abc</member><member>123</member></ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService4TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "abc", *out.ListMember[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "123", *out.ListMember[1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService5ProtocolTestListWithCustomMemberNameCase1(t *testing.T) {
	svc := NewOutputService5ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember><item>abc</item><item>123</item></ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService5TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "abc", *out.ListMember[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "123", *out.ListMember[1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService6ProtocolTestFlattenedListCase1(t *testing.T) {
	svc := NewOutputService6ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember>abc</ListMember><ListMember>123</ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService6TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "abc", *out.ListMember[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "123", *out.ListMember[1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService7ProtocolTestFlattenedSingleElementListCase1(t *testing.T) {
	svc := NewOutputService7ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember>abc</ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService7TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "abc", *out.ListMember[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService8ProtocolTestListOfStructuresCase1(t *testing.T) {
	svc := NewOutputService8ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08/\"><OperationNameResult><List><member><Foo>firstfoo</Foo><Bar>firstbar</Bar><Baz>firstbaz</Baz></member><member><Foo>secondfoo</Foo><Bar>secondbar</Bar><Baz>secondbaz</Baz></member></List></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService8TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "firstbar", *out.List[0].Bar; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "firstbaz", *out.List[0].Baz; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "firstfoo", *out.List[0].Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secondbar", *out.List[1].Bar; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secondbaz", *out.List[1].Baz; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secondfoo", *out.List[1].Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService9ProtocolTestFlattenedListOfStructuresCase1(t *testing.T) {
	svc := NewOutputService9ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08/\"><OperationNameResult><List><Foo>firstfoo</Foo><Bar>firstbar</Bar><Baz>firstbaz</Baz></List><List><Foo>secondfoo</Foo><Bar>secondbar</Bar><Baz>secondbaz</Baz></List></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService9TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "firstbar", *out.List[0].Bar; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "firstbaz", *out.List[0].Baz; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "firstfoo", *out.List[0].Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secondbar", *out.List[1].Bar; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secondbaz", *out.List[1].Baz; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secondfoo", *out.List[1].Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService10ProtocolTestFlattenedListWithLocationNameCase1(t *testing.T) {
	svc := NewOutputService10ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08/\"><OperationNameResult><NamedList>a</NamedList><NamedList>b</NamedList></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService10TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "a", *out.List[0]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "b", *out.List[1]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService11ProtocolTestNormalMapCase1(t *testing.T) {
	svc := NewOutputService11ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08\"><OperationNameResult><Map><entry><key>qux</key><value><foo>bar</foo></value></entry><entry><key>baz</key><value><foo>bam</foo></value></entry></Map></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService11TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "bam", *out.Map["baz"].Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "bar", *out.Map["qux"].Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService12ProtocolTestFlattenedMapCase1(t *testing.T) {
	svc := NewOutputService12ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Map><key>qux</key><value>bar</value></Map><Map><key>baz</key><value>bam</value></Map></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService12TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "bam", *out.Map["baz"]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "bar", *out.Map["qux"]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService13ProtocolTestFlattenedMapInShapeDefinitionCase1(t *testing.T) {
	svc := NewOutputService13ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Attribute><Name>qux</Name><Value>bar</Value></Attribute></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService13TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "bar", *out.Map["qux"]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService14ProtocolTestNamedMapCase1(t *testing.T) {
	svc := NewOutputService14ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Map><foo>qux</foo><bar>bar</bar></Map><Map><foo>baz</foo><bar>bam</bar></Map></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService14TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "bam", *out.Map["baz"]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "bar", *out.Map["qux"]; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}

func TestOutputService15ProtocolTestEmptyStringCase1(t *testing.T) {
	svc := NewOutputService15ProtocolTest(unit.Session, &aws.Config{Endpoint: aws.String("https://test")})

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Foo/></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService15TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	if req.Error != nil {
		t.Errorf("expect not error, got %v", req.Error)
	}

	// assert response
	if out == nil {
		t.Errorf("expect not to be nil")
	}
	if e, a := "", *out.Foo; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

}
