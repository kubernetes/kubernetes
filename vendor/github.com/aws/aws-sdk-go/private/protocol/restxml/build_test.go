package restxml_test

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/private/protocol/restxml"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
	"github.com/aws/aws-sdk-go/private/signer/v4"
	"github.com/aws/aws-sdk-go/private/util"
	"github.com/stretchr/testify/assert"
)

var _ bytes.Buffer // always import bytes
var _ http.Request
var _ json.Marshaler
var _ time.Time
var _ xmlutil.XMLNode
var _ xml.Attr
var _ = awstesting.GenerateAssertions
var _ = ioutil.Discard
var _ = util.Trim("")
var _ = url.Values{}
var _ = io.EOF
var _ = aws.String

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService1ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService1ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService1ProtocolTest {
	svc := &InputService1ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice1protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService1ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService1ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService1TestCaseOperation1 = "OperationName"

// InputService1TestCaseOperation1Request generates a request for the InputService1TestCaseOperation1 operation.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation1Request(input *InputService1TestShapeInputShape) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService1TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService1TestShapeInputService1TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService1ProtocolTest) InputService1TestCaseOperation1(input *InputService1TestShapeInputShape) (*InputService1TestShapeInputService1TestCaseOperation1Output, error) {
	req, out := c.InputService1TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService1TestCaseOperation2 = "OperationName"

// InputService1TestCaseOperation2Request generates a request for the InputService1TestCaseOperation2 operation.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation2Request(input *InputService1TestShapeInputShape) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation2,
		HTTPMethod: "PUT",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService1TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService1TestShapeInputService1TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService1ProtocolTest) InputService1TestCaseOperation2(input *InputService1TestShapeInputShape) (*InputService1TestShapeInputService1TestCaseOperation2Output, error) {
	req, out := c.InputService1TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

const opInputService1TestCaseOperation3 = "OperationName"

// InputService1TestCaseOperation3Request generates a request for the InputService1TestCaseOperation3 operation.
func (c *InputService1ProtocolTest) InputService1TestCaseOperation3Request(input *InputService1TestShapeInputService1TestCaseOperation3Input) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation3Output) {
	op := &request.Operation{
		Name:       opInputService1TestCaseOperation3,
		HTTPMethod: "GET",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService1TestShapeInputService1TestCaseOperation3Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService1TestShapeInputService1TestCaseOperation3Output{}
	req.Data = output
	return
}

func (c *InputService1ProtocolTest) InputService1TestCaseOperation3(input *InputService1TestShapeInputService1TestCaseOperation3Input) (*InputService1TestShapeInputService1TestCaseOperation3Output, error) {
	req, out := c.InputService1TestCaseOperation3Request(input)
	err := req.Send()
	return out, err
}

type InputService1TestShapeInputService1TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
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

type InputService1TestShapeInputShape struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Description *string `type:"string"`

	Name *string `type:"string"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService2ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService2ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService2ProtocolTest {
	svc := &InputService2ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice2protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService2ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService2ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService2TestCaseOperation1 = "OperationName"

// InputService2TestCaseOperation1Request generates a request for the InputService2TestCaseOperation1 operation.
func (c *InputService2ProtocolTest) InputService2TestCaseOperation1Request(input *InputService2TestShapeInputService2TestCaseOperation1Input) (req *request.Request, output *InputService2TestShapeInputService2TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService2TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService2TestShapeInputService2TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService2TestShapeInputService2TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService2ProtocolTest) InputService2TestCaseOperation1(input *InputService2TestShapeInputService2TestCaseOperation1Input) (*InputService2TestShapeInputService2TestCaseOperation1Output, error) {
	req, out := c.InputService2TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService2TestShapeInputService2TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	First *bool `type:"boolean"`

	Fourth *int64 `type:"integer"`

	Second *bool `type:"boolean"`

	Third *float64 `type:"float"`
}

type InputService2TestShapeInputService2TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService3ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService3ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService3ProtocolTest {
	svc := &InputService3ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice3protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService3ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService3ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService3TestCaseOperation1 = "OperationName"

// InputService3TestCaseOperation1Request generates a request for the InputService3TestCaseOperation1 operation.
func (c *InputService3ProtocolTest) InputService3TestCaseOperation1Request(input *InputService3TestShapeInputShape) (req *request.Request, output *InputService3TestShapeInputService3TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService3TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService3TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService3TestShapeInputService3TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService3ProtocolTest) InputService3TestCaseOperation1(input *InputService3TestShapeInputShape) (*InputService3TestShapeInputService3TestCaseOperation1Output, error) {
	req, out := c.InputService3TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService3TestCaseOperation2 = "OperationName"

// InputService3TestCaseOperation2Request generates a request for the InputService3TestCaseOperation2 operation.
func (c *InputService3ProtocolTest) InputService3TestCaseOperation2Request(input *InputService3TestShapeInputShape) (req *request.Request, output *InputService3TestShapeInputService3TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService3TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService3TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService3TestShapeInputService3TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService3ProtocolTest) InputService3TestCaseOperation2(input *InputService3TestShapeInputShape) (*InputService3TestShapeInputService3TestCaseOperation2Output, error) {
	req, out := c.InputService3TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

type InputService3TestShapeInputService3TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService3TestShapeInputService3TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService3TestShapeInputShape struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Description *string `type:"string"`

	SubStructure *InputService3TestShapeSubStructure `type:"structure"`
}

type InputService3TestShapeSubStructure struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Foo *string `type:"string"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService4ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService4ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService4ProtocolTest {
	svc := &InputService4ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice4protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService4ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService4ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService4TestCaseOperation1 = "OperationName"

// InputService4TestCaseOperation1Request generates a request for the InputService4TestCaseOperation1 operation.
func (c *InputService4ProtocolTest) InputService4TestCaseOperation1Request(input *InputService4TestShapeInputService4TestCaseOperation1Input) (req *request.Request, output *InputService4TestShapeInputService4TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService4TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService4TestShapeInputService4TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService4TestShapeInputService4TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService4ProtocolTest) InputService4TestCaseOperation1(input *InputService4TestShapeInputService4TestCaseOperation1Input) (*InputService4TestShapeInputService4TestCaseOperation1Output, error) {
	req, out := c.InputService4TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService4TestShapeInputService4TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Description *string `type:"string"`

	SubStructure *InputService4TestShapeSubStructure `type:"structure"`
}

type InputService4TestShapeInputService4TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService4TestShapeSubStructure struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Foo *string `type:"string"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService5ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService5ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService5ProtocolTest {
	svc := &InputService5ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice5protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService5ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService5ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService5TestCaseOperation1 = "OperationName"

// InputService5TestCaseOperation1Request generates a request for the InputService5TestCaseOperation1 operation.
func (c *InputService5ProtocolTest) InputService5TestCaseOperation1Request(input *InputService5TestShapeInputService5TestCaseOperation1Input) (req *request.Request, output *InputService5TestShapeInputService5TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService5TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService5TestShapeInputService5TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService5TestShapeInputService5TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService5ProtocolTest) InputService5TestCaseOperation1(input *InputService5TestShapeInputService5TestCaseOperation1Input) (*InputService5TestShapeInputService5TestCaseOperation1Output, error) {
	req, out := c.InputService5TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService5TestShapeInputService5TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `type:"list"`
}

type InputService5TestShapeInputService5TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService6ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService6ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService6ProtocolTest {
	svc := &InputService6ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice6protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService6ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService6ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService6TestCaseOperation1 = "OperationName"

// InputService6TestCaseOperation1Request generates a request for the InputService6TestCaseOperation1 operation.
func (c *InputService6ProtocolTest) InputService6TestCaseOperation1Request(input *InputService6TestShapeInputService6TestCaseOperation1Input) (req *request.Request, output *InputService6TestShapeInputService6TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService6TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService6TestShapeInputService6TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService6TestShapeInputService6TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService6ProtocolTest) InputService6TestCaseOperation1(input *InputService6TestShapeInputService6TestCaseOperation1Input) (*InputService6TestShapeInputService6TestCaseOperation1Output, error) {
	req, out := c.InputService6TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService6TestShapeInputService6TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `locationName:"AlternateName" locationNameList:"NotMember" type:"list"`
}

type InputService6TestShapeInputService6TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService7ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService7ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService7ProtocolTest {
	svc := &InputService7ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice7protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService7ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService7ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService7TestCaseOperation1 = "OperationName"

// InputService7TestCaseOperation1Request generates a request for the InputService7TestCaseOperation1 operation.
func (c *InputService7ProtocolTest) InputService7TestCaseOperation1Request(input *InputService7TestShapeInputService7TestCaseOperation1Input) (req *request.Request, output *InputService7TestShapeInputService7TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService7TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService7TestShapeInputService7TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService7TestShapeInputService7TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService7ProtocolTest) InputService7TestCaseOperation1(input *InputService7TestShapeInputService7TestCaseOperation1Input) (*InputService7TestShapeInputService7TestCaseOperation1Output, error) {
	req, out := c.InputService7TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService7TestShapeInputService7TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `type:"list" flattened:"true"`
}

type InputService7TestShapeInputService7TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService8ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService8ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService8ProtocolTest {
	svc := &InputService8ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice8protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService8ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService8ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService8TestCaseOperation1 = "OperationName"

// InputService8TestCaseOperation1Request generates a request for the InputService8TestCaseOperation1 operation.
func (c *InputService8ProtocolTest) InputService8TestCaseOperation1Request(input *InputService8TestShapeInputService8TestCaseOperation1Input) (req *request.Request, output *InputService8TestShapeInputService8TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService8TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService8TestShapeInputService8TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService8TestShapeInputService8TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService8ProtocolTest) InputService8TestCaseOperation1(input *InputService8TestShapeInputService8TestCaseOperation1Input) (*InputService8TestShapeInputService8TestCaseOperation1Output, error) {
	req, out := c.InputService8TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService8TestShapeInputService8TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*string `locationName:"item" type:"list" flattened:"true"`
}

type InputService8TestShapeInputService8TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService9ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService9ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService9ProtocolTest {
	svc := &InputService9ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice9protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService9ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService9ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService9TestCaseOperation1 = "OperationName"

// InputService9TestCaseOperation1Request generates a request for the InputService9TestCaseOperation1 operation.
func (c *InputService9ProtocolTest) InputService9TestCaseOperation1Request(input *InputService9TestShapeInputService9TestCaseOperation1Input) (req *request.Request, output *InputService9TestShapeInputService9TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService9TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService9TestShapeInputService9TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService9TestShapeInputService9TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService9ProtocolTest) InputService9TestCaseOperation1(input *InputService9TestShapeInputService9TestCaseOperation1Input) (*InputService9TestShapeInputService9TestCaseOperation1Output, error) {
	req, out := c.InputService9TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService9TestShapeInputService9TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	ListParam []*InputService9TestShapeSingleFieldStruct `locationName:"item" type:"list" flattened:"true"`
}

type InputService9TestShapeInputService9TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService9TestShapeSingleFieldStruct struct {
	_ struct{} `type:"structure"`

	Element *string `locationName:"value" type:"string"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService10ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService10ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService10ProtocolTest {
	svc := &InputService10ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice10protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService10ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService10ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService10TestCaseOperation1 = "OperationName"

// InputService10TestCaseOperation1Request generates a request for the InputService10TestCaseOperation1 operation.
func (c *InputService10ProtocolTest) InputService10TestCaseOperation1Request(input *InputService10TestShapeInputService10TestCaseOperation1Input) (req *request.Request, output *InputService10TestShapeInputService10TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService10TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/2014-01-01/hostedzone",
	}

	if input == nil {
		input = &InputService10TestShapeInputService10TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService10TestShapeInputService10TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService10ProtocolTest) InputService10TestCaseOperation1(input *InputService10TestShapeInputService10TestCaseOperation1Input) (*InputService10TestShapeInputService10TestCaseOperation1Output, error) {
	req, out := c.InputService10TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService10TestShapeInputService10TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	StructureParam *InputService10TestShapeStructureShape `type:"structure"`
}

type InputService10TestShapeInputService10TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService10TestShapeStructureShape struct {
	_ struct{} `type:"structure"`

	B []byte `locationName:"b" type:"blob"`

	T *time.Time `locationName:"t" type:"timestamp" timestampFormat:"iso8601"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService11ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService11ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService11ProtocolTest {
	svc := &InputService11ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice11protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService11ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService11ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService11TestCaseOperation1 = "OperationName"

// InputService11TestCaseOperation1Request generates a request for the InputService11TestCaseOperation1 operation.
func (c *InputService11ProtocolTest) InputService11TestCaseOperation1Request(input *InputService11TestShapeInputService11TestCaseOperation1Input) (req *request.Request, output *InputService11TestShapeInputService11TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService11TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService11TestShapeInputService11TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService11TestShapeInputService11TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService11ProtocolTest) InputService11TestCaseOperation1(input *InputService11TestShapeInputService11TestCaseOperation1Input) (*InputService11TestShapeInputService11TestCaseOperation1Output, error) {
	req, out := c.InputService11TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService11TestShapeInputService11TestCaseOperation1Input struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	Foo map[string]*string `location:"headers" locationName:"x-foo-" type:"map"`
}

type InputService11TestShapeInputService11TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService12ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService12ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService12ProtocolTest {
	svc := &InputService12ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice12protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService12ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService12ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService12TestCaseOperation1 = "OperationName"

// InputService12TestCaseOperation1Request generates a request for the InputService12TestCaseOperation1 operation.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation1Request(input *InputService12TestShapeInputService12TestCaseOperation1Input) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService12TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/2014-01-01/jobsByPipeline/{PipelineId}",
	}

	if input == nil {
		input = &InputService12TestShapeInputService12TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation1(input *InputService12TestShapeInputService12TestCaseOperation1Input) (*InputService12TestShapeInputService12TestCaseOperation1Output, error) {
	req, out := c.InputService12TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService12TestShapeInputService12TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	PipelineId *string `location:"uri" type:"string"`

	QueryDoc map[string]*string `location:"querystring" type:"map"`
}

type InputService12TestShapeInputService12TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService13ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService13ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService13ProtocolTest {
	svc := &InputService13ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice13protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService13ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService13ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService13TestCaseOperation1 = "OperationName"

// InputService13TestCaseOperation1Request generates a request for the InputService13TestCaseOperation1 operation.
func (c *InputService13ProtocolTest) InputService13TestCaseOperation1Request(input *InputService13TestShapeInputService13TestCaseOperation1Input) (req *request.Request, output *InputService13TestShapeInputService13TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService13TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/2014-01-01/jobsByPipeline/{PipelineId}",
	}

	if input == nil {
		input = &InputService13TestShapeInputService13TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService13TestShapeInputService13TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService13ProtocolTest) InputService13TestCaseOperation1(input *InputService13TestShapeInputService13TestCaseOperation1Input) (*InputService13TestShapeInputService13TestCaseOperation1Output, error) {
	req, out := c.InputService13TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService13TestShapeInputService13TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	PipelineId *string `location:"uri" type:"string"`

	QueryDoc map[string][]*string `location:"querystring" type:"map"`
}

type InputService13TestShapeInputService13TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService14ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService14ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService14ProtocolTest {
	svc := &InputService14ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice14protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService14ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService14ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService14TestCaseOperation1 = "OperationName"

// InputService14TestCaseOperation1Request generates a request for the InputService14TestCaseOperation1 operation.
func (c *InputService14ProtocolTest) InputService14TestCaseOperation1Request(input *InputService14TestShapeInputService14TestCaseOperation1Input) (req *request.Request, output *InputService14TestShapeInputService14TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService14TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService14TestShapeInputService14TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService14TestShapeInputService14TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService14ProtocolTest) InputService14TestCaseOperation1(input *InputService14TestShapeInputService14TestCaseOperation1Input) (*InputService14TestShapeInputService14TestCaseOperation1Output, error) {
	req, out := c.InputService14TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService14TestShapeInputService14TestCaseOperation1Input struct {
	_ struct{} `type:"structure" payload:"Foo"`

	Foo *string `locationName:"foo" type:"string"`
}

type InputService14TestShapeInputService14TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService15ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService15ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService15ProtocolTest {
	svc := &InputService15ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice15protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService15ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService15ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService15TestCaseOperation1 = "OperationName"

// InputService15TestCaseOperation1Request generates a request for the InputService15TestCaseOperation1 operation.
func (c *InputService15ProtocolTest) InputService15TestCaseOperation1Request(input *InputService15TestShapeInputShape) (req *request.Request, output *InputService15TestShapeInputService15TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService15TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService15TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService15TestShapeInputService15TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService15ProtocolTest) InputService15TestCaseOperation1(input *InputService15TestShapeInputShape) (*InputService15TestShapeInputService15TestCaseOperation1Output, error) {
	req, out := c.InputService15TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService15TestCaseOperation2 = "OperationName"

// InputService15TestCaseOperation2Request generates a request for the InputService15TestCaseOperation2 operation.
func (c *InputService15ProtocolTest) InputService15TestCaseOperation2Request(input *InputService15TestShapeInputShape) (req *request.Request, output *InputService15TestShapeInputService15TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService15TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService15TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService15TestShapeInputService15TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService15ProtocolTest) InputService15TestCaseOperation2(input *InputService15TestShapeInputShape) (*InputService15TestShapeInputService15TestCaseOperation2Output, error) {
	req, out := c.InputService15TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

type InputService15TestShapeInputService15TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService15TestShapeInputService15TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService15TestShapeInputShape struct {
	_ struct{} `type:"structure" payload:"Foo"`

	Foo []byte `locationName:"foo" type:"blob"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService16ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService16ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService16ProtocolTest {
	svc := &InputService16ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice16protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService16ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService16ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService16TestCaseOperation1 = "OperationName"

// InputService16TestCaseOperation1Request generates a request for the InputService16TestCaseOperation1 operation.
func (c *InputService16ProtocolTest) InputService16TestCaseOperation1Request(input *InputService16TestShapeInputShape) (req *request.Request, output *InputService16TestShapeInputService16TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService16TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService16TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService16TestShapeInputService16TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService16ProtocolTest) InputService16TestCaseOperation1(input *InputService16TestShapeInputShape) (*InputService16TestShapeInputService16TestCaseOperation1Output, error) {
	req, out := c.InputService16TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService16TestCaseOperation2 = "OperationName"

// InputService16TestCaseOperation2Request generates a request for the InputService16TestCaseOperation2 operation.
func (c *InputService16ProtocolTest) InputService16TestCaseOperation2Request(input *InputService16TestShapeInputShape) (req *request.Request, output *InputService16TestShapeInputService16TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService16TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService16TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService16TestShapeInputService16TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService16ProtocolTest) InputService16TestCaseOperation2(input *InputService16TestShapeInputShape) (*InputService16TestShapeInputService16TestCaseOperation2Output, error) {
	req, out := c.InputService16TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

const opInputService16TestCaseOperation3 = "OperationName"

// InputService16TestCaseOperation3Request generates a request for the InputService16TestCaseOperation3 operation.
func (c *InputService16ProtocolTest) InputService16TestCaseOperation3Request(input *InputService16TestShapeInputShape) (req *request.Request, output *InputService16TestShapeInputService16TestCaseOperation3Output) {
	op := &request.Operation{
		Name:       opInputService16TestCaseOperation3,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService16TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService16TestShapeInputService16TestCaseOperation3Output{}
	req.Data = output
	return
}

func (c *InputService16ProtocolTest) InputService16TestCaseOperation3(input *InputService16TestShapeInputShape) (*InputService16TestShapeInputService16TestCaseOperation3Output, error) {
	req, out := c.InputService16TestCaseOperation3Request(input)
	err := req.Send()
	return out, err
}

const opInputService16TestCaseOperation4 = "OperationName"

// InputService16TestCaseOperation4Request generates a request for the InputService16TestCaseOperation4 operation.
func (c *InputService16ProtocolTest) InputService16TestCaseOperation4Request(input *InputService16TestShapeInputShape) (req *request.Request, output *InputService16TestShapeInputService16TestCaseOperation4Output) {
	op := &request.Operation{
		Name:       opInputService16TestCaseOperation4,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService16TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService16TestShapeInputService16TestCaseOperation4Output{}
	req.Data = output
	return
}

func (c *InputService16ProtocolTest) InputService16TestCaseOperation4(input *InputService16TestShapeInputShape) (*InputService16TestShapeInputService16TestCaseOperation4Output, error) {
	req, out := c.InputService16TestCaseOperation4Request(input)
	err := req.Send()
	return out, err
}

type InputService16TestShapeFooShape struct {
	_ struct{} `locationName:"foo" type:"structure"`

	Baz *string `locationName:"baz" type:"string"`
}

type InputService16TestShapeInputService16TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService16TestShapeInputService16TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService16TestShapeInputService16TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService16TestShapeInputService16TestCaseOperation4Output struct {
	_ struct{} `type:"structure"`
}

type InputService16TestShapeInputShape struct {
	_ struct{} `type:"structure" payload:"Foo"`

	Foo *InputService16TestShapeFooShape `locationName:"foo" type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService17ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService17ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService17ProtocolTest {
	svc := &InputService17ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice17protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService17ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService17ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService17TestCaseOperation1 = "OperationName"

// InputService17TestCaseOperation1Request generates a request for the InputService17TestCaseOperation1 operation.
func (c *InputService17ProtocolTest) InputService17TestCaseOperation1Request(input *InputService17TestShapeInputService17TestCaseOperation1Input) (req *request.Request, output *InputService17TestShapeInputService17TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService17TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &InputService17TestShapeInputService17TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService17TestShapeInputService17TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService17ProtocolTest) InputService17TestCaseOperation1(input *InputService17TestShapeInputService17TestCaseOperation1Input) (*InputService17TestShapeInputService17TestCaseOperation1Output, error) {
	req, out := c.InputService17TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService17TestShapeGrant struct {
	_ struct{} `locationName:"Grant" type:"structure"`

	Grantee *InputService17TestShapeGrantee `type:"structure"`
}

type InputService17TestShapeGrantee struct {
	_ struct{} `type:"structure" xmlPrefix:"xsi" xmlURI:"http://www.w3.org/2001/XMLSchema-instance"`

	EmailAddress *string `type:"string"`

	Type *string `locationName:"xsi:type" type:"string" xmlAttribute:"true"`
}

type InputService17TestShapeInputService17TestCaseOperation1Input struct {
	_ struct{} `type:"structure" payload:"Grant"`

	Grant *InputService17TestShapeGrant `locationName:"Grant" type:"structure"`
}

type InputService17TestShapeInputService17TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService18ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService18ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService18ProtocolTest {
	svc := &InputService18ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice18protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService18ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService18ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService18TestCaseOperation1 = "OperationName"

// InputService18TestCaseOperation1Request generates a request for the InputService18TestCaseOperation1 operation.
func (c *InputService18ProtocolTest) InputService18TestCaseOperation1Request(input *InputService18TestShapeInputService18TestCaseOperation1Input) (req *request.Request, output *InputService18TestShapeInputService18TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService18TestCaseOperation1,
		HTTPMethod: "GET",
		HTTPPath:   "/{Bucket}/{Key+}",
	}

	if input == nil {
		input = &InputService18TestShapeInputService18TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService18TestShapeInputService18TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService18ProtocolTest) InputService18TestCaseOperation1(input *InputService18TestShapeInputService18TestCaseOperation1Input) (*InputService18TestShapeInputService18TestCaseOperation1Output, error) {
	req, out := c.InputService18TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService18TestShapeInputService18TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	Bucket *string `location:"uri" type:"string"`

	Key *string `location:"uri" type:"string"`
}

type InputService18TestShapeInputService18TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService19ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService19ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService19ProtocolTest {
	svc := &InputService19ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice19protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService19ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService19ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService19TestCaseOperation1 = "OperationName"

// InputService19TestCaseOperation1Request generates a request for the InputService19TestCaseOperation1 operation.
func (c *InputService19ProtocolTest) InputService19TestCaseOperation1Request(input *InputService19TestShapeInputShape) (req *request.Request, output *InputService19TestShapeInputService19TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService19TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService19TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService19TestShapeInputService19TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService19ProtocolTest) InputService19TestCaseOperation1(input *InputService19TestShapeInputShape) (*InputService19TestShapeInputService19TestCaseOperation1Output, error) {
	req, out := c.InputService19TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService19TestCaseOperation2 = "OperationName"

// InputService19TestCaseOperation2Request generates a request for the InputService19TestCaseOperation2 operation.
func (c *InputService19ProtocolTest) InputService19TestCaseOperation2Request(input *InputService19TestShapeInputShape) (req *request.Request, output *InputService19TestShapeInputService19TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService19TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/path?abc=mno",
	}

	if input == nil {
		input = &InputService19TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService19TestShapeInputService19TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService19ProtocolTest) InputService19TestCaseOperation2(input *InputService19TestShapeInputShape) (*InputService19TestShapeInputService19TestCaseOperation2Output, error) {
	req, out := c.InputService19TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

type InputService19TestShapeInputService19TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService19TestShapeInputService19TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService19TestShapeInputShape struct {
	_ struct{} `type:"structure"`

	Foo *string `location:"querystring" locationName:"param-name" type:"string"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService20ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService20ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService20ProtocolTest {
	svc := &InputService20ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice20protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService20ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService20ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService20TestCaseOperation1 = "OperationName"

// InputService20TestCaseOperation1Request generates a request for the InputService20TestCaseOperation1 operation.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation1Request(input *InputService20TestShapeInputShape) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService20TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService20TestShapeInputService20TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService20ProtocolTest) InputService20TestCaseOperation1(input *InputService20TestShapeInputShape) (*InputService20TestShapeInputService20TestCaseOperation1Output, error) {
	req, out := c.InputService20TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService20TestCaseOperation2 = "OperationName"

// InputService20TestCaseOperation2Request generates a request for the InputService20TestCaseOperation2 operation.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation2Request(input *InputService20TestShapeInputShape) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation2Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation2,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService20TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService20TestShapeInputService20TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService20ProtocolTest) InputService20TestCaseOperation2(input *InputService20TestShapeInputShape) (*InputService20TestShapeInputService20TestCaseOperation2Output, error) {
	req, out := c.InputService20TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

const opInputService20TestCaseOperation3 = "OperationName"

// InputService20TestCaseOperation3Request generates a request for the InputService20TestCaseOperation3 operation.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation3Request(input *InputService20TestShapeInputShape) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation3Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation3,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService20TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService20TestShapeInputService20TestCaseOperation3Output{}
	req.Data = output
	return
}

func (c *InputService20ProtocolTest) InputService20TestCaseOperation3(input *InputService20TestShapeInputShape) (*InputService20TestShapeInputService20TestCaseOperation3Output, error) {
	req, out := c.InputService20TestCaseOperation3Request(input)
	err := req.Send()
	return out, err
}

const opInputService20TestCaseOperation4 = "OperationName"

// InputService20TestCaseOperation4Request generates a request for the InputService20TestCaseOperation4 operation.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation4Request(input *InputService20TestShapeInputShape) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation4Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation4,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService20TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService20TestShapeInputService20TestCaseOperation4Output{}
	req.Data = output
	return
}

func (c *InputService20ProtocolTest) InputService20TestCaseOperation4(input *InputService20TestShapeInputShape) (*InputService20TestShapeInputService20TestCaseOperation4Output, error) {
	req, out := c.InputService20TestCaseOperation4Request(input)
	err := req.Send()
	return out, err
}

const opInputService20TestCaseOperation5 = "OperationName"

// InputService20TestCaseOperation5Request generates a request for the InputService20TestCaseOperation5 operation.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation5Request(input *InputService20TestShapeInputShape) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation5Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation5,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService20TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService20TestShapeInputService20TestCaseOperation5Output{}
	req.Data = output
	return
}

func (c *InputService20ProtocolTest) InputService20TestCaseOperation5(input *InputService20TestShapeInputShape) (*InputService20TestShapeInputService20TestCaseOperation5Output, error) {
	req, out := c.InputService20TestCaseOperation5Request(input)
	err := req.Send()
	return out, err
}

const opInputService20TestCaseOperation6 = "OperationName"

// InputService20TestCaseOperation6Request generates a request for the InputService20TestCaseOperation6 operation.
func (c *InputService20ProtocolTest) InputService20TestCaseOperation6Request(input *InputService20TestShapeInputShape) (req *request.Request, output *InputService20TestShapeInputService20TestCaseOperation6Output) {
	op := &request.Operation{
		Name:       opInputService20TestCaseOperation6,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService20TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService20TestShapeInputService20TestCaseOperation6Output{}
	req.Data = output
	return
}

func (c *InputService20ProtocolTest) InputService20TestCaseOperation6(input *InputService20TestShapeInputShape) (*InputService20TestShapeInputService20TestCaseOperation6Output, error) {
	req, out := c.InputService20TestCaseOperation6Request(input)
	err := req.Send()
	return out, err
}

type InputService20TestShapeInputService20TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService20TestShapeInputService20TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService20TestShapeInputService20TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService20TestShapeInputService20TestCaseOperation4Output struct {
	_ struct{} `type:"structure"`
}

type InputService20TestShapeInputService20TestCaseOperation5Output struct {
	_ struct{} `type:"structure"`
}

type InputService20TestShapeInputService20TestCaseOperation6Output struct {
	_ struct{} `type:"structure"`
}

type InputService20TestShapeInputShape struct {
	_ struct{} `locationName:"OperationRequest" type:"structure" xmlURI:"https://foo/"`

	RecursiveStruct *InputService20TestShapeRecursiveStructType `type:"structure"`
}

type InputService20TestShapeRecursiveStructType struct {
	_ struct{} `type:"structure"`

	NoRecurse *string `type:"string"`

	RecursiveList []*InputService20TestShapeRecursiveStructType `type:"list"`

	RecursiveMap map[string]*InputService20TestShapeRecursiveStructType `type:"map"`

	RecursiveStruct *InputService20TestShapeRecursiveStructType `type:"structure"`
}

//The service client's operations are safe to be used concurrently.
// It is not safe to mutate any of the client's properties though.
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
	return newInputService21ProtocolTestClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// newClient creates, initializes and returns a new service client instance.
func newInputService21ProtocolTestClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string) *InputService21ProtocolTest {
	svc := &InputService21ProtocolTest{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName:   "inputservice21protocoltest",
				SigningRegion: signingRegion,
				Endpoint:      endpoint,
				APIVersion:    "2014-01-01",
			},
			handlers,
		),
	}

	// Handlers
	svc.Handlers.Sign.PushBack(v4.Sign)
	svc.Handlers.Build.PushBack(restxml.Build)
	svc.Handlers.Unmarshal.PushBack(restxml.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(restxml.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(restxml.UnmarshalError)

	return svc
}

// newRequest creates a new request for a InputService21ProtocolTest operation and runs any
// custom request initialization.
func (c *InputService21ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opInputService21TestCaseOperation1 = "OperationName"

// InputService21TestCaseOperation1Request generates a request for the InputService21TestCaseOperation1 operation.
func (c *InputService21ProtocolTest) InputService21TestCaseOperation1Request(input *InputService21TestShapeInputService21TestCaseOperation1Input) (req *request.Request, output *InputService21TestShapeInputService21TestCaseOperation1Output) {
	op := &request.Operation{
		Name:       opInputService21TestCaseOperation1,
		HTTPMethod: "POST",
		HTTPPath:   "/path",
	}

	if input == nil {
		input = &InputService21TestShapeInputService21TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService21TestShapeInputService21TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService21ProtocolTest) InputService21TestCaseOperation1(input *InputService21TestShapeInputService21TestCaseOperation1Input) (*InputService21TestShapeInputService21TestCaseOperation1Output, error) {
	req, out := c.InputService21TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type InputService21TestShapeInputService21TestCaseOperation1Input struct {
	_ struct{} `type:"structure"`

	TimeArgInHeader *time.Time `location:"header" locationName:"x-amz-timearg" type:"timestamp" timestampFormat:"rfc822"`
}

type InputService21TestShapeInputService21TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

//
// Tests begin here
//

func TestInputService1ProtocolTestBasicXMLSerializationCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService1ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService1TestShapeInputShape{
		Description: aws.String("bar"),
		Name:        aws.String("foo"),
	}
	req, _ := svc.InputService1TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">bar</Description><Name xmlns="https://foo/">foo</Name></OperationRequest>`, util.Trim(string(body)), InputService1TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService1ProtocolTestBasicXMLSerializationCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService1ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService1TestShapeInputShape{
		Description: aws.String("bar"),
		Name:        aws.String("foo"),
	}
	req, _ := svc.InputService1TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">bar</Description><Name xmlns="https://foo/">foo</Name></OperationRequest>`, util.Trim(string(body)), InputService1TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService1ProtocolTestBasicXMLSerializationCase3(t *testing.T) {
	sess := session.New()
	svc := NewInputService1ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService1TestShapeInputService1TestCaseOperation3Input{}
	req, _ := svc.InputService1TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService2ProtocolTestSerializeOtherScalarTypesCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService2ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><First xmlns="https://foo/">true</First><Fourth xmlns="https://foo/">3</Fourth><Second xmlns="https://foo/">false</Second><Third xmlns="https://foo/">1.2</Third></OperationRequest>`, util.Trim(string(body)), InputService2TestShapeInputService2TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService3ProtocolTestNestedStructuresCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService3ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService3TestShapeInputShape{
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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">baz</Description><SubStructure xmlns="https://foo/"><Bar xmlns="https://foo/">b</Bar><Foo xmlns="https://foo/">a</Foo></SubStructure></OperationRequest>`, util.Trim(string(body)), InputService3TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService3ProtocolTestNestedStructuresCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService3ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService3TestShapeInputShape{
		Description: aws.String("baz"),
		SubStructure: &InputService3TestShapeSubStructure{
			Foo: aws.String("a"),
		},
	}
	req, _ := svc.InputService3TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">baz</Description><SubStructure xmlns="https://foo/"><Foo xmlns="https://foo/">a</Foo></SubStructure></OperationRequest>`, util.Trim(string(body)), InputService3TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService4ProtocolTestNestedStructuresCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService4ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService4TestShapeInputService4TestCaseOperation1Input{
		Description:  aws.String("baz"),
		SubStructure: &InputService4TestShapeSubStructure{},
	}
	req, _ := svc.InputService4TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><Description xmlns="https://foo/">baz</Description><SubStructure xmlns="https://foo/"></SubStructure></OperationRequest>`, util.Trim(string(body)), InputService4TestShapeInputService4TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService5ProtocolTestNonFlattenedListsCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService5ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><ListParam xmlns="https://foo/"><member xmlns="https://foo/">one</member><member xmlns="https://foo/">two</member><member xmlns="https://foo/">three</member></ListParam></OperationRequest>`, util.Trim(string(body)), InputService5TestShapeInputService5TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService6ProtocolTestNonFlattenedListsWithLocationNameCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService6ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><AlternateName xmlns="https://foo/"><NotMember xmlns="https://foo/">one</NotMember><NotMember xmlns="https://foo/">two</NotMember><NotMember xmlns="https://foo/">three</NotMember></AlternateName></OperationRequest>`, util.Trim(string(body)), InputService6TestShapeInputService6TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService7ProtocolTestFlattenedListsCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService7ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><ListParam xmlns="https://foo/">one</ListParam><ListParam xmlns="https://foo/">two</ListParam><ListParam xmlns="https://foo/">three</ListParam></OperationRequest>`, util.Trim(string(body)), InputService7TestShapeInputService7TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService8ProtocolTestFlattenedListsWithLocationNameCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService8ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><item xmlns="https://foo/">one</item><item xmlns="https://foo/">two</item><item xmlns="https://foo/">three</item></OperationRequest>`, util.Trim(string(body)), InputService8TestShapeInputService8TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService9ProtocolTestListOfStructuresCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService9ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><item xmlns="https://foo/"><value xmlns="https://foo/">one</value></item><item xmlns="https://foo/"><value xmlns="https://foo/">two</value></item><item xmlns="https://foo/"><value xmlns="https://foo/">three</value></item></OperationRequest>`, util.Trim(string(body)), InputService9TestShapeInputService9TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService10ProtocolTestBlobAndTimestampShapesCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService10ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><StructureParam xmlns="https://foo/"><b xmlns="https://foo/">Zm9v</b><t xmlns="https://foo/">2015-01-25T08:00:00Z</t></StructureParam></OperationRequest>`, util.Trim(string(body)), InputService10TestShapeInputService10TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/hostedzone", r.URL.String())

	// assert headers

}

func TestInputService11ProtocolTestHeaderMapsCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService11ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

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
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers
	assert.Equal(t, "b", r.Header.Get("x-foo-a"))
	assert.Equal(t, "d", r.Header.Get("x-foo-c"))

}

func TestInputService12ProtocolTestStringToStringMapsInQuerystringCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputService12TestCaseOperation1Input{
		PipelineId: aws.String("foo"),
		QueryDoc: map[string]*string{
			"bar":  aws.String("baz"),
			"fizz": aws.String("buzz"),
		},
	}
	req, _ := svc.InputService12TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/jobsByPipeline/foo?bar=baz&fizz=buzz", r.URL.String())

	// assert headers

}

func TestInputService13ProtocolTestStringToStringListMapsInQuerystringCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService13ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService13TestShapeInputService13TestCaseOperation1Input{
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
	req, _ := svc.InputService13TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/2014-01-01/jobsByPipeline/id?foo=bar&foo=baz&fizz=buzz&fizz=pop", r.URL.String())

	// assert headers

}

func TestInputService14ProtocolTestStringPayloadCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService14ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService14TestShapeInputService14TestCaseOperation1Input{
		Foo: aws.String("bar"),
	}
	req, _ := svc.InputService14TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	assert.Equal(t, `bar`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService15ProtocolTestBlobPayloadCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService15ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService15TestShapeInputShape{
		Foo: []byte("bar"),
	}
	req, _ := svc.InputService15TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	assert.Equal(t, `bar`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService15ProtocolTestBlobPayloadCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService15ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService15TestShapeInputShape{}
	req, _ := svc.InputService15TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService16ProtocolTestStructurePayloadCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService16ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService16TestShapeInputShape{
		Foo: &InputService16TestShapeFooShape{
			Baz: aws.String("bar"),
		},
	}
	req, _ := svc.InputService16TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<foo><baz>bar</baz></foo>`, util.Trim(string(body)), InputService16TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService16ProtocolTestStructurePayloadCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService16ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService16TestShapeInputShape{}
	req, _ := svc.InputService16TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService16ProtocolTestStructurePayloadCase3(t *testing.T) {
	sess := session.New()
	svc := NewInputService16ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService16TestShapeInputShape{
		Foo: &InputService16TestShapeFooShape{},
	}
	req, _ := svc.InputService16TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<foo></foo>`, util.Trim(string(body)), InputService16TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService16ProtocolTestStructurePayloadCase4(t *testing.T) {
	sess := session.New()
	svc := NewInputService16ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService16TestShapeInputShape{}
	req, _ := svc.InputService16TestCaseOperation4Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService17ProtocolTestXMLAttributeCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService17ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService17TestShapeInputService17TestCaseOperation1Input{
		Grant: &InputService17TestShapeGrant{
			Grantee: &InputService17TestShapeGrantee{
				EmailAddress: aws.String("foo@example.com"),
				Type:         aws.String("CanonicalUser"),
			},
		},
	}
	req, _ := svc.InputService17TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<Grant xmlns:_xmlns="xmlns" _xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:XMLSchema-instance="http://www.w3.org/2001/XMLSchema-instance" XMLSchema-instance:type="CanonicalUser"><Grantee><EmailAddress>foo@example.com</EmailAddress></Grantee></Grant>`, util.Trim(string(body)), InputService17TestShapeInputService17TestCaseOperation1Input{})

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService18ProtocolTestGreedyKeysCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService18ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService18TestShapeInputService18TestCaseOperation1Input{
		Bucket: aws.String("my/bucket"),
		Key:    aws.String("testing /123"),
	}
	req, _ := svc.InputService18TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/my%2Fbucket/testing%20/123", r.URL.String())

	// assert headers

}

func TestInputService19ProtocolTestOmitsNullQueryParamsButSerializesEmptyStringsCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService19ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService19TestShapeInputShape{}
	req, _ := svc.InputService19TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService19ProtocolTestOmitsNullQueryParamsButSerializesEmptyStringsCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService19ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService19TestShapeInputShape{
		Foo: aws.String(""),
	}
	req, _ := svc.InputService19TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/path?abc=mno&param-name=", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestRecursiveShapesCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService20ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService20TestShapeInputShape{
		RecursiveStruct: &InputService20TestShapeRecursiveStructType{
			NoRecurse: aws.String("foo"),
		},
	}
	req, _ := svc.InputService20TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService20TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestRecursiveShapesCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService20ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService20TestShapeInputShape{
		RecursiveStruct: &InputService20TestShapeRecursiveStructType{
			RecursiveStruct: &InputService20TestShapeRecursiveStructType{
				NoRecurse: aws.String("foo"),
			},
		},
	}
	req, _ := svc.InputService20TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></RecursiveStruct></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService20TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestRecursiveShapesCase3(t *testing.T) {
	sess := session.New()
	svc := NewInputService20ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService20TestShapeInputShape{
		RecursiveStruct: &InputService20TestShapeRecursiveStructType{
			RecursiveStruct: &InputService20TestShapeRecursiveStructType{
				RecursiveStruct: &InputService20TestShapeRecursiveStructType{
					RecursiveStruct: &InputService20TestShapeRecursiveStructType{
						NoRecurse: aws.String("foo"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService20TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></RecursiveStruct></RecursiveStruct></RecursiveStruct></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService20TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestRecursiveShapesCase4(t *testing.T) {
	sess := session.New()
	svc := NewInputService20ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService20TestShapeInputShape{
		RecursiveStruct: &InputService20TestShapeRecursiveStructType{
			RecursiveList: []*InputService20TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					NoRecurse: aws.String("bar"),
				},
			},
		},
	}
	req, _ := svc.InputService20TestCaseOperation4Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveList xmlns="https://foo/"><member xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></member><member xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></member></RecursiveList></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService20TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestRecursiveShapesCase5(t *testing.T) {
	sess := session.New()
	svc := NewInputService20ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService20TestShapeInputShape{
		RecursiveStruct: &InputService20TestShapeRecursiveStructType{
			RecursiveList: []*InputService20TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					RecursiveStruct: &InputService20TestShapeRecursiveStructType{
						NoRecurse: aws.String("bar"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService20TestCaseOperation5Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveList xmlns="https://foo/"><member xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></member><member xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></RecursiveStruct></member></RecursiveList></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService20TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService20ProtocolTestRecursiveShapesCase6(t *testing.T) {
	sess := session.New()
	svc := NewInputService20ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService20TestShapeInputShape{
		RecursiveStruct: &InputService20TestShapeRecursiveStructType{
			RecursiveMap: map[string]*InputService20TestShapeRecursiveStructType{
				"bar": {
					NoRecurse: aws.String("bar"),
				},
				"foo": {
					NoRecurse: aws.String("foo"),
				},
			},
		},
	}
	req, _ := svc.InputService20TestCaseOperation6Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body := util.SortXML(r.Body)
	awstesting.AssertXML(t, `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`, util.Trim(string(body)), InputService20TestShapeInputShape{})

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers

}

func TestInputService21ProtocolTestTimestampInHeaderCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService21ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService21TestShapeInputService21TestCaseOperation1Input{
		TimeArgInHeader: aws.Time(time.Unix(1422172800, 0)),
	}
	req, _ := svc.InputService21TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	restxml.Build(req)
	assert.NoError(t, req.Error)

	// assert URL
	awstesting.AssertURL(t, "https://test/path", r.URL.String())

	// assert headers
	assert.Equal(t, "Sun, 25 Jan 2015 08:00:00 GMT", r.Header.Get("x-amz-timearg"))

}
