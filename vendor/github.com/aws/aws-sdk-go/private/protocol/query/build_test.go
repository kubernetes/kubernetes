package query_test

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
	"github.com/aws/aws-sdk-go/private/protocol/query"
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService1TestCaseOperation1,
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
		Name: opInputService1TestCaseOperation2,
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
func (c *InputService1ProtocolTest) InputService1TestCaseOperation3Request(input *InputService1TestShapeInputShape) (req *request.Request, output *InputService1TestShapeInputService1TestCaseOperation3Output) {
	op := &request.Operation{
		Name: opInputService1TestCaseOperation3,
	}

	if input == nil {
		input = &InputService1TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService1TestShapeInputService1TestCaseOperation3Output{}
	req.Data = output
	return
}

func (c *InputService1ProtocolTest) InputService1TestCaseOperation3(input *InputService1TestShapeInputShape) (*InputService1TestShapeInputService1TestCaseOperation3Output, error) {
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

type InputService1TestShapeInputService1TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService1TestShapeInputShape struct {
	_ struct{} `type:"structure"`

	Bar *string `type:"string"`

	Baz *bool `type:"boolean"`

	Foo *string `type:"string"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService2TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	StructArg *InputService2TestShapeStructType `type:"structure"`
}

type InputService2TestShapeInputService2TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService2TestShapeStructType struct {
	_ struct{} `type:"structure"`

	ScalarArg *string `type:"string"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService3TestCaseOperation1,
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
		Name: opInputService3TestCaseOperation2,
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
	_ struct{} `type:"structure"`

	ListArg []*string `type:"list"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
func (c *InputService4ProtocolTest) InputService4TestCaseOperation1Request(input *InputService4TestShapeInputShape) (req *request.Request, output *InputService4TestShapeInputService4TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opInputService4TestCaseOperation1,
	}

	if input == nil {
		input = &InputService4TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService4TestShapeInputService4TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService4ProtocolTest) InputService4TestCaseOperation1(input *InputService4TestShapeInputShape) (*InputService4TestShapeInputService4TestCaseOperation1Output, error) {
	req, out := c.InputService4TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService4TestCaseOperation2 = "OperationName"

// InputService4TestCaseOperation2Request generates a request for the InputService4TestCaseOperation2 operation.
func (c *InputService4ProtocolTest) InputService4TestCaseOperation2Request(input *InputService4TestShapeInputShape) (req *request.Request, output *InputService4TestShapeInputService4TestCaseOperation2Output) {
	op := &request.Operation{
		Name: opInputService4TestCaseOperation2,
	}

	if input == nil {
		input = &InputService4TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService4TestShapeInputService4TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService4ProtocolTest) InputService4TestCaseOperation2(input *InputService4TestShapeInputShape) (*InputService4TestShapeInputService4TestCaseOperation2Output, error) {
	req, out := c.InputService4TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

type InputService4TestShapeInputService4TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService4TestShapeInputService4TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService4TestShapeInputShape struct {
	_ struct{} `type:"structure"`

	ListArg []*string `type:"list" flattened:"true"`

	NamedListArg []*string `locationNameList:"Foo" type:"list" flattened:"true"`

	ScalarArg *string `type:"string"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService5TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	MapArg map[string]*string `type:"map" flattened:"true"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService6TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	ListArg []*string `locationNameList:"item" type:"list"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService7TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	ListArg []*string `locationNameList:"ListArgLocation" type:"list" flattened:"true"`

	ScalarArg *string `type:"string"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService8TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	MapArg map[string]*string `type:"map"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService9TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	MapArg map[string]*string `locationNameKey:"TheKey" locationNameValue:"TheValue" type:"map"`
}

type InputService9TestShapeInputService9TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService10TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	BlobArg []byte `type:"blob"`
}

type InputService10TestShapeInputService10TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
		Name: opInputService11TestCaseOperation1,
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
	_ struct{} `type:"structure"`

	TimeArg *time.Time `type:"timestamp" timestampFormat:"iso8601"`
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
	svc.Handlers.Build.PushBack(query.Build)
	svc.Handlers.Unmarshal.PushBack(query.Unmarshal)
	svc.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	svc.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

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
func (c *InputService12ProtocolTest) InputService12TestCaseOperation1Request(input *InputService12TestShapeInputShape) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opInputService12TestCaseOperation1,
	}

	if input == nil {
		input = &InputService12TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation1(input *InputService12TestShapeInputShape) (*InputService12TestShapeInputService12TestCaseOperation1Output, error) {
	req, out := c.InputService12TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

const opInputService12TestCaseOperation2 = "OperationName"

// InputService12TestCaseOperation2Request generates a request for the InputService12TestCaseOperation2 operation.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation2Request(input *InputService12TestShapeInputShape) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation2Output) {
	op := &request.Operation{
		Name: opInputService12TestCaseOperation2,
	}

	if input == nil {
		input = &InputService12TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation2Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation2(input *InputService12TestShapeInputShape) (*InputService12TestShapeInputService12TestCaseOperation2Output, error) {
	req, out := c.InputService12TestCaseOperation2Request(input)
	err := req.Send()
	return out, err
}

const opInputService12TestCaseOperation3 = "OperationName"

// InputService12TestCaseOperation3Request generates a request for the InputService12TestCaseOperation3 operation.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation3Request(input *InputService12TestShapeInputShape) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation3Output) {
	op := &request.Operation{
		Name: opInputService12TestCaseOperation3,
	}

	if input == nil {
		input = &InputService12TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation3Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation3(input *InputService12TestShapeInputShape) (*InputService12TestShapeInputService12TestCaseOperation3Output, error) {
	req, out := c.InputService12TestCaseOperation3Request(input)
	err := req.Send()
	return out, err
}

const opInputService12TestCaseOperation4 = "OperationName"

// InputService12TestCaseOperation4Request generates a request for the InputService12TestCaseOperation4 operation.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation4Request(input *InputService12TestShapeInputShape) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation4Output) {
	op := &request.Operation{
		Name: opInputService12TestCaseOperation4,
	}

	if input == nil {
		input = &InputService12TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation4Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation4(input *InputService12TestShapeInputShape) (*InputService12TestShapeInputService12TestCaseOperation4Output, error) {
	req, out := c.InputService12TestCaseOperation4Request(input)
	err := req.Send()
	return out, err
}

const opInputService12TestCaseOperation5 = "OperationName"

// InputService12TestCaseOperation5Request generates a request for the InputService12TestCaseOperation5 operation.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation5Request(input *InputService12TestShapeInputShape) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation5Output) {
	op := &request.Operation{
		Name: opInputService12TestCaseOperation5,
	}

	if input == nil {
		input = &InputService12TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation5Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation5(input *InputService12TestShapeInputShape) (*InputService12TestShapeInputService12TestCaseOperation5Output, error) {
	req, out := c.InputService12TestCaseOperation5Request(input)
	err := req.Send()
	return out, err
}

const opInputService12TestCaseOperation6 = "OperationName"

// InputService12TestCaseOperation6Request generates a request for the InputService12TestCaseOperation6 operation.
func (c *InputService12ProtocolTest) InputService12TestCaseOperation6Request(input *InputService12TestShapeInputShape) (req *request.Request, output *InputService12TestShapeInputService12TestCaseOperation6Output) {
	op := &request.Operation{
		Name: opInputService12TestCaseOperation6,
	}

	if input == nil {
		input = &InputService12TestShapeInputShape{}
	}

	req = c.newRequest(op, input, output)
	output = &InputService12TestShapeInputService12TestCaseOperation6Output{}
	req.Data = output
	return
}

func (c *InputService12ProtocolTest) InputService12TestCaseOperation6(input *InputService12TestShapeInputShape) (*InputService12TestShapeInputService12TestCaseOperation6Output, error) {
	req, out := c.InputService12TestCaseOperation6Request(input)
	err := req.Send()
	return out, err
}

type InputService12TestShapeInputService12TestCaseOperation1Output struct {
	_ struct{} `type:"structure"`
}

type InputService12TestShapeInputService12TestCaseOperation2Output struct {
	_ struct{} `type:"structure"`
}

type InputService12TestShapeInputService12TestCaseOperation3Output struct {
	_ struct{} `type:"structure"`
}

type InputService12TestShapeInputService12TestCaseOperation4Output struct {
	_ struct{} `type:"structure"`
}

type InputService12TestShapeInputService12TestCaseOperation5Output struct {
	_ struct{} `type:"structure"`
}

type InputService12TestShapeInputService12TestCaseOperation6Output struct {
	_ struct{} `type:"structure"`
}

type InputService12TestShapeInputShape struct {
	_ struct{} `type:"structure"`

	RecursiveStruct *InputService12TestShapeRecursiveStructType `type:"structure"`
}

type InputService12TestShapeRecursiveStructType struct {
	_ struct{} `type:"structure"`

	NoRecurse *string `type:"string"`

	RecursiveList []*InputService12TestShapeRecursiveStructType `type:"list"`

	RecursiveMap map[string]*InputService12TestShapeRecursiveStructType `type:"map"`

	RecursiveStruct *InputService12TestShapeRecursiveStructType `type:"structure"`
}

//
// Tests begin here
//

func TestInputService1ProtocolTestScalarMembersCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService1ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService1TestShapeInputShape{
		Bar: aws.String("val2"),
		Foo: aws.String("val1"),
	}
	req, _ := svc.InputService1TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&Bar=val2&Foo=val1&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService1ProtocolTestScalarMembersCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService1ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService1TestShapeInputShape{
		Baz: aws.Bool(true),
	}
	req, _ := svc.InputService1TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&Baz=true&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService1ProtocolTestScalarMembersCase3(t *testing.T) {
	sess := session.New()
	svc := NewInputService1ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService1TestShapeInputShape{
		Baz: aws.Bool(false),
	}
	req, _ := svc.InputService1TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&Baz=false&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService2ProtocolTestNestedStructureMembersCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService2ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService2TestShapeInputService2TestCaseOperation1Input{
		StructArg: &InputService2TestShapeStructType{
			ScalarArg: aws.String("foo"),
		},
	}
	req, _ := svc.InputService2TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&StructArg.ScalarArg=foo&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService3ProtocolTestListTypesCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService3ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService3TestShapeInputShape{
		ListArg: []*string{
			aws.String("foo"),
			aws.String("bar"),
			aws.String("baz"),
		},
	}
	req, _ := svc.InputService3TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&ListArg.member.1=foo&ListArg.member.2=bar&ListArg.member.3=baz&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService3ProtocolTestListTypesCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService3ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService3TestShapeInputShape{
		ListArg: []*string{},
	}
	req, _ := svc.InputService3TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&ListArg=&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService4ProtocolTestFlattenedListCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService4ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService4TestShapeInputShape{
		ListArg: []*string{
			aws.String("a"),
			aws.String("b"),
			aws.String("c"),
		},
		ScalarArg: aws.String("foo"),
	}
	req, _ := svc.InputService4TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&ListArg.1=a&ListArg.2=b&ListArg.3=c&ScalarArg=foo&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService4ProtocolTestFlattenedListCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService4ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService4TestShapeInputShape{
		NamedListArg: []*string{
			aws.String("a"),
		},
	}
	req, _ := svc.InputService4TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&Foo.1=a&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService5ProtocolTestSerializeFlattenedMapTypeCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService5ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService5TestShapeInputService5TestCaseOperation1Input{
		MapArg: map[string]*string{
			"key1": aws.String("val1"),
			"key2": aws.String("val2"),
		},
	}
	req, _ := svc.InputService5TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&MapArg.1.key=key1&MapArg.1.value=val1&MapArg.2.key=key2&MapArg.2.value=val2&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService6ProtocolTestNonFlattenedListWithLocationNameCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService6ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService6TestShapeInputService6TestCaseOperation1Input{
		ListArg: []*string{
			aws.String("a"),
			aws.String("b"),
			aws.String("c"),
		},
	}
	req, _ := svc.InputService6TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&ListArg.item.1=a&ListArg.item.2=b&ListArg.item.3=c&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService7ProtocolTestFlattenedListWithLocationNameCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService7ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService7TestShapeInputService7TestCaseOperation1Input{
		ListArg: []*string{
			aws.String("a"),
			aws.String("b"),
			aws.String("c"),
		},
		ScalarArg: aws.String("foo"),
	}
	req, _ := svc.InputService7TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&ListArgLocation.1=a&ListArgLocation.2=b&ListArgLocation.3=c&ScalarArg=foo&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService8ProtocolTestSerializeMapTypeCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService8ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService8TestShapeInputService8TestCaseOperation1Input{
		MapArg: map[string]*string{
			"key1": aws.String("val1"),
			"key2": aws.String("val2"),
		},
	}
	req, _ := svc.InputService8TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&MapArg.entry.1.key=key1&MapArg.entry.1.value=val1&MapArg.entry.2.key=key2&MapArg.entry.2.value=val2&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService9ProtocolTestSerializeMapTypeWithLocationNameCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService9ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService9TestShapeInputService9TestCaseOperation1Input{
		MapArg: map[string]*string{
			"key1": aws.String("val1"),
			"key2": aws.String("val2"),
		},
	}
	req, _ := svc.InputService9TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&MapArg.entry.1.TheKey=key1&MapArg.entry.1.TheValue=val1&MapArg.entry.2.TheKey=key2&MapArg.entry.2.TheValue=val2&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService10ProtocolTestBase64EncodedBlobsCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService10ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService10TestShapeInputService10TestCaseOperation1Input{
		BlobArg: []byte("foo"),
	}
	req, _ := svc.InputService10TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&BlobArg=Zm9v&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService11ProtocolTestTimestampValuesCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService11ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService11TestShapeInputService11TestCaseOperation1Input{
		TimeArg: aws.Time(time.Unix(1422172800, 0)),
	}
	req, _ := svc.InputService11TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&TimeArg=2015-01-25T08%3A00%3A00Z&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService12ProtocolTestRecursiveShapesCase1(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputShape{
		RecursiveStruct: &InputService12TestShapeRecursiveStructType{
			NoRecurse: aws.String("foo"),
		},
	}
	req, _ := svc.InputService12TestCaseOperation1Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&RecursiveStruct.NoRecurse=foo&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService12ProtocolTestRecursiveShapesCase2(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputShape{
		RecursiveStruct: &InputService12TestShapeRecursiveStructType{
			RecursiveStruct: &InputService12TestShapeRecursiveStructType{
				NoRecurse: aws.String("foo"),
			},
		},
	}
	req, _ := svc.InputService12TestCaseOperation2Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&RecursiveStruct.RecursiveStruct.NoRecurse=foo&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService12ProtocolTestRecursiveShapesCase3(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputShape{
		RecursiveStruct: &InputService12TestShapeRecursiveStructType{
			RecursiveStruct: &InputService12TestShapeRecursiveStructType{
				RecursiveStruct: &InputService12TestShapeRecursiveStructType{
					RecursiveStruct: &InputService12TestShapeRecursiveStructType{
						NoRecurse: aws.String("foo"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService12TestCaseOperation3Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&RecursiveStruct.RecursiveStruct.RecursiveStruct.RecursiveStruct.NoRecurse=foo&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService12ProtocolTestRecursiveShapesCase4(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputShape{
		RecursiveStruct: &InputService12TestShapeRecursiveStructType{
			RecursiveList: []*InputService12TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					NoRecurse: aws.String("bar"),
				},
			},
		},
	}
	req, _ := svc.InputService12TestCaseOperation4Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&RecursiveStruct.RecursiveList.member.1.NoRecurse=foo&RecursiveStruct.RecursiveList.member.2.NoRecurse=bar&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService12ProtocolTestRecursiveShapesCase5(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputShape{
		RecursiveStruct: &InputService12TestShapeRecursiveStructType{
			RecursiveList: []*InputService12TestShapeRecursiveStructType{
				{
					NoRecurse: aws.String("foo"),
				},
				{
					RecursiveStruct: &InputService12TestShapeRecursiveStructType{
						NoRecurse: aws.String("bar"),
					},
				},
			},
		},
	}
	req, _ := svc.InputService12TestCaseOperation5Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&RecursiveStruct.RecursiveList.member.1.NoRecurse=foo&RecursiveStruct.RecursiveList.member.2.RecursiveStruct.NoRecurse=bar&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}

func TestInputService12ProtocolTestRecursiveShapesCase6(t *testing.T) {
	sess := session.New()
	svc := NewInputService12ProtocolTest(sess, &aws.Config{Endpoint: aws.String("https://test")})

	input := &InputService12TestShapeInputShape{
		RecursiveStruct: &InputService12TestShapeRecursiveStructType{
			RecursiveMap: map[string]*InputService12TestShapeRecursiveStructType{
				"bar": {
					NoRecurse: aws.String("bar"),
				},
				"foo": {
					NoRecurse: aws.String("foo"),
				},
			},
		},
	}
	req, _ := svc.InputService12TestCaseOperation6Request(input)
	r := req.HTTPRequest

	// build request
	query.Build(req)
	assert.NoError(t, req.Error)

	// assert body
	assert.NotNil(t, r.Body)
	body, _ := ioutil.ReadAll(r.Body)
	awstesting.AssertQuery(t, `Action=OperationName&RecursiveStruct.RecursiveMap.entry.1.key=foo&RecursiveStruct.RecursiveMap.entry.1.value.NoRecurse=foo&RecursiveStruct.RecursiveMap.entry.2.key=bar&RecursiveStruct.RecursiveMap.entry.2.value.NoRecurse=bar&Version=2014-01-01`, util.Trim(string(body)))

	// assert URL
	awstesting.AssertURL(t, "https://test/", r.URL.String())

	// assert headers

}
