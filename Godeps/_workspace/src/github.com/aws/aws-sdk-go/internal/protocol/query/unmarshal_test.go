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
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/service"
	"github.com/aws/aws-sdk-go/aws/service/serviceinfo"
	"github.com/aws/aws-sdk-go/internal/protocol/query"
	"github.com/aws/aws-sdk-go/internal/protocol/xml/xmlutil"
	"github.com/aws/aws-sdk-go/internal/signer/v4"
	"github.com/aws/aws-sdk-go/internal/util"
	"github.com/stretchr/testify/assert"
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

type OutputService1ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService1ProtocolTest client.
func NewOutputService1ProtocolTest(config *aws.Config) *OutputService1ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice1protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService1ProtocolTest{service}
}

// newRequest creates a new request for a OutputService1ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService1ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService1TestCaseOperation1 = "OperationName"

// OutputService1TestCaseOperation1Request generates a request for the OutputService1TestCaseOperation1 operation.
func (c *OutputService1ProtocolTest) OutputService1TestCaseOperation1Request(input *OutputService1TestShapeOutputService1TestCaseOperation1Input) (req *request.Request, output *OutputService1TestShapeOutputService1TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService1TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService1TestShapeOutputService1TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService1TestShapeOutputService1TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService1ProtocolTest) OutputService1TestCaseOperation1(input *OutputService1TestShapeOutputService1TestCaseOperation1Input) (*OutputService1TestShapeOutputService1TestCaseOperation1Output, error) {
	req, out := c.OutputService1TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService1TestShapeOutputService1TestCaseOperation1Input struct {
	metadataOutputService1TestShapeOutputService1TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService1TestShapeOutputService1TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService1TestShapeOutputService1TestCaseOperation1Output struct {
	Char *string `type:"character"`

	Double *float64 `type:"double"`

	FalseBool *bool `type:"boolean"`

	Float *float64 `type:"float"`

	Long *int64 `type:"long"`

	Num *int64 `locationName:"FooNum" type:"integer"`

	Str *string `type:"string"`

	Timestamp *time.Time `type:"timestamp" timestampFormat:"iso8601"`

	TrueBool *bool `type:"boolean"`

	metadataOutputService1TestShapeOutputService1TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService1TestShapeOutputService1TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService2ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService2ProtocolTest client.
func NewOutputService2ProtocolTest(config *aws.Config) *OutputService2ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice2protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService2ProtocolTest{service}
}

// newRequest creates a new request for a OutputService2ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService2ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService2TestCaseOperation1 = "OperationName"

// OutputService2TestCaseOperation1Request generates a request for the OutputService2TestCaseOperation1 operation.
func (c *OutputService2ProtocolTest) OutputService2TestCaseOperation1Request(input *OutputService2TestShapeOutputService2TestCaseOperation1Input) (req *request.Request, output *OutputService2TestShapeOutputService2TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService2TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService2TestShapeOutputService2TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService2TestShapeOutputService2TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService2ProtocolTest) OutputService2TestCaseOperation1(input *OutputService2TestShapeOutputService2TestCaseOperation1Input) (*OutputService2TestShapeOutputService2TestCaseOperation1Output, error) {
	req, out := c.OutputService2TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService2TestShapeOutputService2TestCaseOperation1Input struct {
	metadataOutputService2TestShapeOutputService2TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService2TestShapeOutputService2TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService2TestShapeOutputService2TestCaseOperation1Output struct {
	Num *int64 `type:"integer"`

	Str *string `type:"string"`

	metadataOutputService2TestShapeOutputService2TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService2TestShapeOutputService2TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService3ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService3ProtocolTest client.
func NewOutputService3ProtocolTest(config *aws.Config) *OutputService3ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice3protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService3ProtocolTest{service}
}

// newRequest creates a new request for a OutputService3ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService3ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService3TestCaseOperation1 = "OperationName"

// OutputService3TestCaseOperation1Request generates a request for the OutputService3TestCaseOperation1 operation.
func (c *OutputService3ProtocolTest) OutputService3TestCaseOperation1Request(input *OutputService3TestShapeOutputService3TestCaseOperation1Input) (req *request.Request, output *OutputService3TestShapeOutputService3TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService3TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService3TestShapeOutputService3TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService3TestShapeOutputService3TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService3ProtocolTest) OutputService3TestCaseOperation1(input *OutputService3TestShapeOutputService3TestCaseOperation1Input) (*OutputService3TestShapeOutputService3TestCaseOperation1Output, error) {
	req, out := c.OutputService3TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService3TestShapeOutputService3TestCaseOperation1Input struct {
	metadataOutputService3TestShapeOutputService3TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService3TestShapeOutputService3TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService3TestShapeOutputService3TestCaseOperation1Output struct {
	Blob []byte `type:"blob"`

	metadataOutputService3TestShapeOutputService3TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService3TestShapeOutputService3TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService4ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService4ProtocolTest client.
func NewOutputService4ProtocolTest(config *aws.Config) *OutputService4ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice4protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService4ProtocolTest{service}
}

// newRequest creates a new request for a OutputService4ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService4ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService4TestCaseOperation1 = "OperationName"

// OutputService4TestCaseOperation1Request generates a request for the OutputService4TestCaseOperation1 operation.
func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1Request(input *OutputService4TestShapeOutputService4TestCaseOperation1Input) (req *request.Request, output *OutputService4TestShapeOutputService4TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService4TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService4TestShapeOutputService4TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService4TestShapeOutputService4TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService4ProtocolTest) OutputService4TestCaseOperation1(input *OutputService4TestShapeOutputService4TestCaseOperation1Input) (*OutputService4TestShapeOutputService4TestCaseOperation1Output, error) {
	req, out := c.OutputService4TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService4TestShapeOutputService4TestCaseOperation1Input struct {
	metadataOutputService4TestShapeOutputService4TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService4TestShapeOutputService4TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService4TestShapeOutputService4TestCaseOperation1Output struct {
	ListMember []*string `type:"list"`

	metadataOutputService4TestShapeOutputService4TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService4TestShapeOutputService4TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService5ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService5ProtocolTest client.
func NewOutputService5ProtocolTest(config *aws.Config) *OutputService5ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice5protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService5ProtocolTest{service}
}

// newRequest creates a new request for a OutputService5ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService5ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService5TestCaseOperation1 = "OperationName"

// OutputService5TestCaseOperation1Request generates a request for the OutputService5TestCaseOperation1 operation.
func (c *OutputService5ProtocolTest) OutputService5TestCaseOperation1Request(input *OutputService5TestShapeOutputService5TestCaseOperation1Input) (req *request.Request, output *OutputService5TestShapeOutputService5TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService5TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService5TestShapeOutputService5TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService5TestShapeOutputService5TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService5ProtocolTest) OutputService5TestCaseOperation1(input *OutputService5TestShapeOutputService5TestCaseOperation1Input) (*OutputService5TestShapeOutputService5TestCaseOperation1Output, error) {
	req, out := c.OutputService5TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService5TestShapeOutputService5TestCaseOperation1Input struct {
	metadataOutputService5TestShapeOutputService5TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService5TestShapeOutputService5TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService5TestShapeOutputService5TestCaseOperation1Output struct {
	ListMember []*string `locationNameList:"item" type:"list"`

	metadataOutputService5TestShapeOutputService5TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService5TestShapeOutputService5TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService6ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService6ProtocolTest client.
func NewOutputService6ProtocolTest(config *aws.Config) *OutputService6ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice6protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService6ProtocolTest{service}
}

// newRequest creates a new request for a OutputService6ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService6ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService6TestCaseOperation1 = "OperationName"

// OutputService6TestCaseOperation1Request generates a request for the OutputService6TestCaseOperation1 operation.
func (c *OutputService6ProtocolTest) OutputService6TestCaseOperation1Request(input *OutputService6TestShapeOutputService6TestCaseOperation1Input) (req *request.Request, output *OutputService6TestShapeOutputService6TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService6TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService6TestShapeOutputService6TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService6TestShapeOutputService6TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService6ProtocolTest) OutputService6TestCaseOperation1(input *OutputService6TestShapeOutputService6TestCaseOperation1Input) (*OutputService6TestShapeOutputService6TestCaseOperation1Output, error) {
	req, out := c.OutputService6TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService6TestShapeOutputService6TestCaseOperation1Input struct {
	metadataOutputService6TestShapeOutputService6TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService6TestShapeOutputService6TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService6TestShapeOutputService6TestCaseOperation1Output struct {
	ListMember []*string `type:"list" flattened:"true"`

	metadataOutputService6TestShapeOutputService6TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService6TestShapeOutputService6TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService7ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService7ProtocolTest client.
func NewOutputService7ProtocolTest(config *aws.Config) *OutputService7ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice7protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService7ProtocolTest{service}
}

// newRequest creates a new request for a OutputService7ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService7ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService7TestCaseOperation1 = "OperationName"

// OutputService7TestCaseOperation1Request generates a request for the OutputService7TestCaseOperation1 operation.
func (c *OutputService7ProtocolTest) OutputService7TestCaseOperation1Request(input *OutputService7TestShapeOutputService7TestCaseOperation1Input) (req *request.Request, output *OutputService7TestShapeOutputService7TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService7TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService7TestShapeOutputService7TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService7TestShapeOutputService7TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService7ProtocolTest) OutputService7TestCaseOperation1(input *OutputService7TestShapeOutputService7TestCaseOperation1Input) (*OutputService7TestShapeOutputService7TestCaseOperation1Output, error) {
	req, out := c.OutputService7TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService7TestShapeOutputService7TestCaseOperation1Input struct {
	metadataOutputService7TestShapeOutputService7TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService7TestShapeOutputService7TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService7TestShapeOutputService7TestCaseOperation1Output struct {
	ListMember []*string `type:"list" flattened:"true"`

	metadataOutputService7TestShapeOutputService7TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService7TestShapeOutputService7TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService8ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService8ProtocolTest client.
func NewOutputService8ProtocolTest(config *aws.Config) *OutputService8ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice8protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService8ProtocolTest{service}
}

// newRequest creates a new request for a OutputService8ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService8ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService8TestCaseOperation1 = "OperationName"

// OutputService8TestCaseOperation1Request generates a request for the OutputService8TestCaseOperation1 operation.
func (c *OutputService8ProtocolTest) OutputService8TestCaseOperation1Request(input *OutputService8TestShapeOutputService8TestCaseOperation1Input) (req *request.Request, output *OutputService8TestShapeOutputService8TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService8TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService8TestShapeOutputService8TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService8TestShapeOutputService8TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService8ProtocolTest) OutputService8TestCaseOperation1(input *OutputService8TestShapeOutputService8TestCaseOperation1Input) (*OutputService8TestShapeOutputService8TestCaseOperation1Output, error) {
	req, out := c.OutputService8TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService8TestShapeOutputService8TestCaseOperation1Input struct {
	metadataOutputService8TestShapeOutputService8TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService8TestShapeOutputService8TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService8TestShapeOutputService8TestCaseOperation1Output struct {
	List []*OutputService8TestShapeStructureShape `type:"list"`

	metadataOutputService8TestShapeOutputService8TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService8TestShapeOutputService8TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService8TestShapeStructureShape struct {
	Bar *string `type:"string"`

	Baz *string `type:"string"`

	Foo *string `type:"string"`

	metadataOutputService8TestShapeStructureShape `json:"-" xml:"-"`
}

type metadataOutputService8TestShapeStructureShape struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService9ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService9ProtocolTest client.
func NewOutputService9ProtocolTest(config *aws.Config) *OutputService9ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice9protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService9ProtocolTest{service}
}

// newRequest creates a new request for a OutputService9ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService9ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService9TestCaseOperation1 = "OperationName"

// OutputService9TestCaseOperation1Request generates a request for the OutputService9TestCaseOperation1 operation.
func (c *OutputService9ProtocolTest) OutputService9TestCaseOperation1Request(input *OutputService9TestShapeOutputService9TestCaseOperation1Input) (req *request.Request, output *OutputService9TestShapeOutputService9TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService9TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService9TestShapeOutputService9TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService9TestShapeOutputService9TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService9ProtocolTest) OutputService9TestCaseOperation1(input *OutputService9TestShapeOutputService9TestCaseOperation1Input) (*OutputService9TestShapeOutputService9TestCaseOperation1Output, error) {
	req, out := c.OutputService9TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService9TestShapeOutputService9TestCaseOperation1Input struct {
	metadataOutputService9TestShapeOutputService9TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService9TestShapeOutputService9TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService9TestShapeOutputService9TestCaseOperation1Output struct {
	List []*OutputService9TestShapeStructureShape `type:"list" flattened:"true"`

	metadataOutputService9TestShapeOutputService9TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService9TestShapeOutputService9TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService9TestShapeStructureShape struct {
	Bar *string `type:"string"`

	Baz *string `type:"string"`

	Foo *string `type:"string"`

	metadataOutputService9TestShapeStructureShape `json:"-" xml:"-"`
}

type metadataOutputService9TestShapeStructureShape struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService10ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService10ProtocolTest client.
func NewOutputService10ProtocolTest(config *aws.Config) *OutputService10ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice10protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService10ProtocolTest{service}
}

// newRequest creates a new request for a OutputService10ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService10ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService10TestCaseOperation1 = "OperationName"

// OutputService10TestCaseOperation1Request generates a request for the OutputService10TestCaseOperation1 operation.
func (c *OutputService10ProtocolTest) OutputService10TestCaseOperation1Request(input *OutputService10TestShapeOutputService10TestCaseOperation1Input) (req *request.Request, output *OutputService10TestShapeOutputService10TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService10TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService10TestShapeOutputService10TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService10TestShapeOutputService10TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService10ProtocolTest) OutputService10TestCaseOperation1(input *OutputService10TestShapeOutputService10TestCaseOperation1Input) (*OutputService10TestShapeOutputService10TestCaseOperation1Output, error) {
	req, out := c.OutputService10TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService10TestShapeOutputService10TestCaseOperation1Input struct {
	metadataOutputService10TestShapeOutputService10TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService10TestShapeOutputService10TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService10TestShapeOutputService10TestCaseOperation1Output struct {
	List []*string `locationNameList:"NamedList" type:"list" flattened:"true"`

	metadataOutputService10TestShapeOutputService10TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService10TestShapeOutputService10TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService11ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService11ProtocolTest client.
func NewOutputService11ProtocolTest(config *aws.Config) *OutputService11ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice11protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService11ProtocolTest{service}
}

// newRequest creates a new request for a OutputService11ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService11ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService11TestCaseOperation1 = "OperationName"

// OutputService11TestCaseOperation1Request generates a request for the OutputService11TestCaseOperation1 operation.
func (c *OutputService11ProtocolTest) OutputService11TestCaseOperation1Request(input *OutputService11TestShapeOutputService11TestCaseOperation1Input) (req *request.Request, output *OutputService11TestShapeOutputService11TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService11TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService11TestShapeOutputService11TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService11TestShapeOutputService11TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService11ProtocolTest) OutputService11TestCaseOperation1(input *OutputService11TestShapeOutputService11TestCaseOperation1Input) (*OutputService11TestShapeOutputService11TestCaseOperation1Output, error) {
	req, out := c.OutputService11TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService11TestShapeOutputService11TestCaseOperation1Input struct {
	metadataOutputService11TestShapeOutputService11TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService11TestShapeOutputService11TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService11TestShapeOutputService11TestCaseOperation1Output struct {
	Map map[string]*OutputService11TestShapeStructType `type:"map"`

	metadataOutputService11TestShapeOutputService11TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService11TestShapeOutputService11TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService11TestShapeStructType struct {
	Foo *string `locationName:"foo" type:"string"`

	metadataOutputService11TestShapeStructType `json:"-" xml:"-"`
}

type metadataOutputService11TestShapeStructType struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService12ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService12ProtocolTest client.
func NewOutputService12ProtocolTest(config *aws.Config) *OutputService12ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice12protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService12ProtocolTest{service}
}

// newRequest creates a new request for a OutputService12ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService12ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService12TestCaseOperation1 = "OperationName"

// OutputService12TestCaseOperation1Request generates a request for the OutputService12TestCaseOperation1 operation.
func (c *OutputService12ProtocolTest) OutputService12TestCaseOperation1Request(input *OutputService12TestShapeOutputService12TestCaseOperation1Input) (req *request.Request, output *OutputService12TestShapeOutputService12TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService12TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService12TestShapeOutputService12TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService12TestShapeOutputService12TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService12ProtocolTest) OutputService12TestCaseOperation1(input *OutputService12TestShapeOutputService12TestCaseOperation1Input) (*OutputService12TestShapeOutputService12TestCaseOperation1Output, error) {
	req, out := c.OutputService12TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService12TestShapeOutputService12TestCaseOperation1Input struct {
	metadataOutputService12TestShapeOutputService12TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService12TestShapeOutputService12TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService12TestShapeOutputService12TestCaseOperation1Output struct {
	Map map[string]*string `type:"map" flattened:"true"`

	metadataOutputService12TestShapeOutputService12TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService12TestShapeOutputService12TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService13ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService13ProtocolTest client.
func NewOutputService13ProtocolTest(config *aws.Config) *OutputService13ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice13protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService13ProtocolTest{service}
}

// newRequest creates a new request for a OutputService13ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService13ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService13TestCaseOperation1 = "OperationName"

// OutputService13TestCaseOperation1Request generates a request for the OutputService13TestCaseOperation1 operation.
func (c *OutputService13ProtocolTest) OutputService13TestCaseOperation1Request(input *OutputService13TestShapeOutputService13TestCaseOperation1Input) (req *request.Request, output *OutputService13TestShapeOutputService13TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService13TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService13TestShapeOutputService13TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService13TestShapeOutputService13TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService13ProtocolTest) OutputService13TestCaseOperation1(input *OutputService13TestShapeOutputService13TestCaseOperation1Input) (*OutputService13TestShapeOutputService13TestCaseOperation1Output, error) {
	req, out := c.OutputService13TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService13TestShapeOutputService13TestCaseOperation1Input struct {
	metadataOutputService13TestShapeOutputService13TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService13TestShapeOutputService13TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService13TestShapeOutputService13TestCaseOperation1Output struct {
	Map map[string]*string `locationName:"Attribute" locationNameKey:"Name" locationNameValue:"Value" type:"map" flattened:"true"`

	metadataOutputService13TestShapeOutputService13TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService13TestShapeOutputService13TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService14ProtocolTest struct {
	*service.Service
}

// New returns a new OutputService14ProtocolTest client.
func NewOutputService14ProtocolTest(config *aws.Config) *OutputService14ProtocolTest {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      defaults.DefaultConfig.Merge(config),
			ServiceName: "outputservice14protocoltest",
			APIVersion:  "",
		},
	}
	service.Initialize()

	// Handlers
	service.Handlers.Sign.PushBack(v4.Sign)
	service.Handlers.Build.PushBack(query.Build)
	service.Handlers.Unmarshal.PushBack(query.Unmarshal)
	service.Handlers.UnmarshalMeta.PushBack(query.UnmarshalMeta)
	service.Handlers.UnmarshalError.PushBack(query.UnmarshalError)

	return &OutputService14ProtocolTest{service}
}

// newRequest creates a new request for a OutputService14ProtocolTest operation and runs any
// custom request initialization.
func (c *OutputService14ProtocolTest) newRequest(op *request.Operation, params, data interface{}) *request.Request {
	req := c.NewRequest(op, params, data)

	return req
}

const opOutputService14TestCaseOperation1 = "OperationName"

// OutputService14TestCaseOperation1Request generates a request for the OutputService14TestCaseOperation1 operation.
func (c *OutputService14ProtocolTest) OutputService14TestCaseOperation1Request(input *OutputService14TestShapeOutputService14TestCaseOperation1Input) (req *request.Request, output *OutputService14TestShapeOutputService14TestCaseOperation1Output) {
	op := &request.Operation{
		Name: opOutputService14TestCaseOperation1,
	}

	if input == nil {
		input = &OutputService14TestShapeOutputService14TestCaseOperation1Input{}
	}

	req = c.newRequest(op, input, output)
	output = &OutputService14TestShapeOutputService14TestCaseOperation1Output{}
	req.Data = output
	return
}

func (c *OutputService14ProtocolTest) OutputService14TestCaseOperation1(input *OutputService14TestShapeOutputService14TestCaseOperation1Input) (*OutputService14TestShapeOutputService14TestCaseOperation1Output, error) {
	req, out := c.OutputService14TestCaseOperation1Request(input)
	err := req.Send()
	return out, err
}

type OutputService14TestShapeOutputService14TestCaseOperation1Input struct {
	metadataOutputService14TestShapeOutputService14TestCaseOperation1Input `json:"-" xml:"-"`
}

type metadataOutputService14TestShapeOutputService14TestCaseOperation1Input struct {
	SDKShapeTraits bool `type:"structure"`
}

type OutputService14TestShapeOutputService14TestCaseOperation1Output struct {
	Map map[string]*string `locationNameKey:"foo" locationNameValue:"bar" type:"map" flattened:"true"`

	metadataOutputService14TestShapeOutputService14TestCaseOperation1Output `json:"-" xml:"-"`
}

type metadataOutputService14TestShapeOutputService14TestCaseOperation1Output struct {
	SDKShapeTraits bool `type:"structure"`
}

//
// Tests begin here
//

func TestOutputService1ProtocolTestScalarMembersCase1(t *testing.T) {
	svc := NewOutputService1ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Str>myname</Str><FooNum>123</FooNum><FalseBool>false</FalseBool><TrueBool>true</TrueBool><Float>1.2</Float><Double>1.3</Double><Long>200</Long><Char>a</Char><Timestamp>2015-01-25T08:00:00Z</Timestamp></OperationNameResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService1TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "a", *out.Char)
	assert.Equal(t, 1.3, *out.Double)
	assert.Equal(t, false, *out.FalseBool)
	assert.Equal(t, 1.2, *out.Float)
	assert.Equal(t, int64(200), *out.Long)
	assert.Equal(t, int64(123), *out.Num)
	assert.Equal(t, "myname", *out.Str)
	assert.Equal(t, time.Unix(1.4221728e+09, 0).UTC().String(), out.Timestamp.String())
	assert.Equal(t, true, *out.TrueBool)

}

func TestOutputService2ProtocolTestNotAllMembersInResponseCase1(t *testing.T) {
	svc := NewOutputService2ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Str>myname</Str></OperationNameResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService2TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "myname", *out.Str)

}

func TestOutputService3ProtocolTestBlobCase1(t *testing.T) {
	svc := NewOutputService3ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Blob>dmFsdWU=</Blob></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService3TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "value", string(out.Blob))

}

func TestOutputService4ProtocolTestListsCase1(t *testing.T) {
	svc := NewOutputService4ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember><member>abc</member><member>123</member></ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService4TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "abc", *out.ListMember[0])
	assert.Equal(t, "123", *out.ListMember[1])

}

func TestOutputService5ProtocolTestListWithCustomMemberNameCase1(t *testing.T) {
	svc := NewOutputService5ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember><item>abc</item><item>123</item></ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService5TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "abc", *out.ListMember[0])
	assert.Equal(t, "123", *out.ListMember[1])

}

func TestOutputService6ProtocolTestFlattenedListCase1(t *testing.T) {
	svc := NewOutputService6ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember>abc</ListMember><ListMember>123</ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService6TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "abc", *out.ListMember[0])
	assert.Equal(t, "123", *out.ListMember[1])

}

func TestOutputService7ProtocolTestFlattenedSingleElementListCase1(t *testing.T) {
	svc := NewOutputService7ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><ListMember>abc</ListMember></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService7TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "abc", *out.ListMember[0])

}

func TestOutputService8ProtocolTestListOfStructuresCase1(t *testing.T) {
	svc := NewOutputService8ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08/\"><OperationNameResult><List><member><Foo>firstfoo</Foo><Bar>firstbar</Bar><Baz>firstbaz</Baz></member><member><Foo>secondfoo</Foo><Bar>secondbar</Bar><Baz>secondbaz</Baz></member></List></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService8TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "firstbar", *out.List[0].Bar)
	assert.Equal(t, "firstbaz", *out.List[0].Baz)
	assert.Equal(t, "firstfoo", *out.List[0].Foo)
	assert.Equal(t, "secondbar", *out.List[1].Bar)
	assert.Equal(t, "secondbaz", *out.List[1].Baz)
	assert.Equal(t, "secondfoo", *out.List[1].Foo)

}

func TestOutputService9ProtocolTestFlattenedListOfStructuresCase1(t *testing.T) {
	svc := NewOutputService9ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08/\"><OperationNameResult><List><Foo>firstfoo</Foo><Bar>firstbar</Bar><Baz>firstbaz</Baz></List><List><Foo>secondfoo</Foo><Bar>secondbar</Bar><Baz>secondbaz</Baz></List></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService9TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "firstbar", *out.List[0].Bar)
	assert.Equal(t, "firstbaz", *out.List[0].Baz)
	assert.Equal(t, "firstfoo", *out.List[0].Foo)
	assert.Equal(t, "secondbar", *out.List[1].Bar)
	assert.Equal(t, "secondbaz", *out.List[1].Baz)
	assert.Equal(t, "secondfoo", *out.List[1].Foo)

}

func TestOutputService10ProtocolTestFlattenedListWithLocationNameCase1(t *testing.T) {
	svc := NewOutputService10ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08/\"><OperationNameResult><NamedList>a</NamedList><NamedList>b</NamedList></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService10TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "a", *out.List[0])
	assert.Equal(t, "b", *out.List[1])

}

func TestOutputService11ProtocolTestNormalMapCase1(t *testing.T) {
	svc := NewOutputService11ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse xmlns=\"https://service.amazonaws.com/doc/2010-05-08\"><OperationNameResult><Map><entry><key>qux</key><value><foo>bar</foo></value></entry><entry><key>baz</key><value><foo>bam</foo></value></entry></Map></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService11TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "bam", *out.Map["baz"].Foo)
	assert.Equal(t, "bar", *out.Map["qux"].Foo)

}

func TestOutputService12ProtocolTestFlattenedMapCase1(t *testing.T) {
	svc := NewOutputService12ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Map><key>qux</key><value>bar</value></Map><Map><key>baz</key><value>bam</value></Map></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService12TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "bam", *out.Map["baz"])
	assert.Equal(t, "bar", *out.Map["qux"])

}

func TestOutputService13ProtocolTestFlattenedMapInShapeDefinitionCase1(t *testing.T) {
	svc := NewOutputService13ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Attribute><Name>qux</Name><Value>bar</Value></Attribute></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService13TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "bar", *out.Map["qux"])

}

func TestOutputService14ProtocolTestNamedMapCase1(t *testing.T) {
	svc := NewOutputService14ProtocolTest(nil)

	buf := bytes.NewReader([]byte("<OperationNameResponse><OperationNameResult><Map><foo>qux</foo><bar>bar</bar></Map><Map><foo>baz</foo><bar>bam</bar></Map></OperationNameResult><ResponseMetadata><RequestId>requestid</RequestId></ResponseMetadata></OperationNameResponse>"))
	req, out := svc.OutputService14TestCaseOperation1Request(nil)
	req.HTTPResponse = &http.Response{StatusCode: 200, Body: ioutil.NopCloser(buf), Header: http.Header{}}

	// set headers

	// unmarshal response
	query.UnmarshalMeta(req)
	query.Unmarshal(req)
	assert.NoError(t, req.Error)

	// assert response
	assert.NotNil(t, out) // ensure out variable is used
	assert.Equal(t, "bam", *out.Map["baz"])
	assert.Equal(t, "bar", *out.Map["qux"])

}
