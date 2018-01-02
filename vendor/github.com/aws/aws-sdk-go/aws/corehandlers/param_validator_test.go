package corehandlers_test

import (
	"fmt"
	"testing"
	"reflect"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/kinesis"
)

var testSvc = func() *client.Client {
	s := &client.Client{
		Config: aws.Config{},
		ClientInfo: metadata.ClientInfo{
			ServiceName: "mock-service",
			APIVersion:  "2015-01-01",
		},
	}
	return s
}()

type StructShape struct {
	_ struct{} `type:"structure"`

	RequiredList   []*ConditionalStructShape          `required:"true"`
	RequiredMap    map[string]*ConditionalStructShape `required:"true"`
	RequiredBool   *bool                              `required:"true"`
	OptionalStruct *ConditionalStructShape

	hiddenParameter *string
}

func (s *StructShape) Validate() error {
	invalidParams := request.ErrInvalidParams{Context: "StructShape"}
	if s.RequiredList == nil {
		invalidParams.Add(request.NewErrParamRequired("RequiredList"))
	}
	if s.RequiredMap == nil {
		invalidParams.Add(request.NewErrParamRequired("RequiredMap"))
	}
	if s.RequiredBool == nil {
		invalidParams.Add(request.NewErrParamRequired("RequiredBool"))
	}
	if s.RequiredList != nil {
		for i, v := range s.RequiredList {
			if v == nil {
				continue
			}
			if err := v.Validate(); err != nil {
				invalidParams.AddNested(fmt.Sprintf("%s[%v]", "RequiredList", i), err.(request.ErrInvalidParams))
			}
		}
	}
	if s.RequiredMap != nil {
		for i, v := range s.RequiredMap {
			if v == nil {
				continue
			}
			if err := v.Validate(); err != nil {
				invalidParams.AddNested(fmt.Sprintf("%s[%v]", "RequiredMap", i), err.(request.ErrInvalidParams))
			}
		}
	}
	if s.OptionalStruct != nil {
		if err := s.OptionalStruct.Validate(); err != nil {
			invalidParams.AddNested("OptionalStruct", err.(request.ErrInvalidParams))
		}
	}

	if invalidParams.Len() > 0 {
		return invalidParams
	}
	return nil
}

type ConditionalStructShape struct {
	_ struct{} `type:"structure"`

	Name *string `required:"true"`
}

func (s *ConditionalStructShape) Validate() error {
	invalidParams := request.ErrInvalidParams{Context: "ConditionalStructShape"}
	if s.Name == nil {
		invalidParams.Add(request.NewErrParamRequired("Name"))
	}

	if invalidParams.Len() > 0 {
		return invalidParams
	}
	return nil
}

func TestNoErrors(t *testing.T) {
	input := &StructShape{
		RequiredList: []*ConditionalStructShape{},
		RequiredMap: map[string]*ConditionalStructShape{
			"key1": {Name: aws.String("Name")},
			"key2": {Name: aws.String("Name")},
		},
		RequiredBool:   aws.Bool(true),
		OptionalStruct: &ConditionalStructShape{Name: aws.String("Name")},
	}

	req := testSvc.NewRequest(&request.Operation{}, input, nil)
	corehandlers.ValidateParametersHandler.Fn(req)
	if req.Error != nil {
		t.Fatalf("expect no error, got %v", req.Error)
	}
}

func TestMissingRequiredParameters(t *testing.T) {
	input := &StructShape{}
	req := testSvc.NewRequest(&request.Operation{}, input, nil)
	corehandlers.ValidateParametersHandler.Fn(req)

	if req.Error == nil {
		t.Fatalf("expect error")
	}
	if e, a := "InvalidParameter", req.Error.(awserr.Error).Code(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "3 validation error(s) found.", req.Error.(awserr.Error).Message(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	errs := req.Error.(awserr.BatchedErrors).OrigErrs()
	if e, a := 3, len(errs); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ParamRequiredError: missing required field, StructShape.RequiredList.", errs[0].Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ParamRequiredError: missing required field, StructShape.RequiredMap.", errs[1].Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ParamRequiredError: missing required field, StructShape.RequiredBool.", errs[2].Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	if e, a := "InvalidParameter: 3 validation error(s) found.\n- missing required field, StructShape.RequiredList.\n- missing required field, StructShape.RequiredMap.\n- missing required field, StructShape.RequiredBool.\n", req.Error.Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestNestedMissingRequiredParameters(t *testing.T) {
	input := &StructShape{
		RequiredList: []*ConditionalStructShape{{}},
		RequiredMap: map[string]*ConditionalStructShape{
			"key1": {Name: aws.String("Name")},
			"key2": {},
		},
		RequiredBool:   aws.Bool(true),
		OptionalStruct: &ConditionalStructShape{},
	}

	req := testSvc.NewRequest(&request.Operation{}, input, nil)
	corehandlers.ValidateParametersHandler.Fn(req)

	if req.Error == nil {
		t.Fatalf("expect error")
	}
	if e, a := "InvalidParameter", req.Error.(awserr.Error).Code(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "3 validation error(s) found.", req.Error.(awserr.Error).Message(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}

	errs := req.Error.(awserr.BatchedErrors).OrigErrs()
	if e, a := 3, len(errs); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ParamRequiredError: missing required field, StructShape.RequiredList[0].Name.", errs[0].Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ParamRequiredError: missing required field, StructShape.RequiredMap[key2].Name.", errs[1].Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "ParamRequiredError: missing required field, StructShape.OptionalStruct.Name.", errs[2].Error(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

type testInput struct {
	StringField *string           `min:"5"`
	ListField   []string          `min:"3"`
	MapField    map[string]string `min:"4"`
}

func (s testInput) Validate() error {
	invalidParams := request.ErrInvalidParams{Context: "testInput"}
	if s.StringField != nil && len(*s.StringField) < 5 {
		invalidParams.Add(request.NewErrParamMinLen("StringField", 5))
	}
	if s.ListField != nil && len(s.ListField) < 3 {
		invalidParams.Add(request.NewErrParamMinLen("ListField", 3))
	}
	if s.MapField != nil && len(s.MapField) < 4 {
		invalidParams.Add(request.NewErrParamMinLen("MapField", 4))
	}

	if invalidParams.Len() > 0 {
		return invalidParams
	}
	return nil
}

var testsFieldMin = []struct {
	err awserr.Error
	in  testInput
}{
	{
		err: func() awserr.Error {
			invalidParams := request.ErrInvalidParams{Context: "testInput"}
			invalidParams.Add(request.NewErrParamMinLen("StringField", 5))
			return invalidParams
		}(),
		in: testInput{StringField: aws.String("abcd")},
	},
	{
		err: func() awserr.Error {
			invalidParams := request.ErrInvalidParams{Context: "testInput"}
			invalidParams.Add(request.NewErrParamMinLen("StringField", 5))
			invalidParams.Add(request.NewErrParamMinLen("ListField", 3))
			return invalidParams
		}(),
		in: testInput{StringField: aws.String("abcd"), ListField: []string{"a", "b"}},
	},
	{
		err: func() awserr.Error {
			invalidParams := request.ErrInvalidParams{Context: "testInput"}
			invalidParams.Add(request.NewErrParamMinLen("StringField", 5))
			invalidParams.Add(request.NewErrParamMinLen("ListField", 3))
			invalidParams.Add(request.NewErrParamMinLen("MapField", 4))
			return invalidParams
		}(),
		in: testInput{StringField: aws.String("abcd"), ListField: []string{"a", "b"}, MapField: map[string]string{"a": "a", "b": "b"}},
	},
	{
		err: nil,
		in: testInput{StringField: aws.String("abcde"),
			ListField: []string{"a", "b", "c"}, MapField: map[string]string{"a": "a", "b": "b", "c": "c", "d": "d"}},
	},
}

func TestValidateFieldMinParameter(t *testing.T) {
	for i, c := range testsFieldMin {
		req := testSvc.NewRequest(&request.Operation{}, &c.in, nil)
		corehandlers.ValidateParametersHandler.Fn(req)

		if e, a := c.err, req.Error; !reflect.DeepEqual(e,a) {
			t.Errorf("%d, expect %v, got %v", i, e, a)
		}
	}
}

func BenchmarkValidateAny(b *testing.B) {
	input := &kinesis.PutRecordsInput{
		StreamName: aws.String("stream"),
	}
	for i := 0; i < 100; i++ {
		record := &kinesis.PutRecordsRequestEntry{
			Data:         make([]byte, 10000),
			PartitionKey: aws.String("partition"),
		}
		input.Records = append(input.Records, record)
	}

	req, _ := kinesis.New(unit.Session).PutRecordsRequest(input)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		corehandlers.ValidateParametersHandler.Fn(req)
		if err := req.Error; err != nil {
			b.Fatalf("validation failed: %v", err)
		}
	}
}
