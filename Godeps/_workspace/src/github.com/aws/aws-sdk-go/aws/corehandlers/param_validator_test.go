package corehandlers_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/service"
	"github.com/aws/aws-sdk-go/aws/service/serviceinfo"
)

var testSvc = func() *service.Service {
	s := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      &aws.Config{},
			ServiceName: "mock-service",
			APIVersion:  "2015-01-01",
		},
	}
	return s
}()

type StructShape struct {
	RequiredList   []*ConditionalStructShape          `required:"true"`
	RequiredMap    map[string]*ConditionalStructShape `required:"true"`
	RequiredBool   *bool                              `required:"true"`
	OptionalStruct *ConditionalStructShape

	hiddenParameter *string

	metadataStructureShape
}

type metadataStructureShape struct {
	SDKShapeTraits bool
}

type ConditionalStructShape struct {
	Name           *string `required:"true"`
	SDKShapeTraits bool
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
	assert.NoError(t, req.Error)
}

func TestMissingRequiredParameters(t *testing.T) {
	input := &StructShape{}
	req := testSvc.NewRequest(&request.Operation{}, input, nil)
	corehandlers.ValidateParametersHandler.Fn(req)

	assert.Error(t, req.Error)
	assert.Equal(t, "InvalidParameter", req.Error.(awserr.Error).Code())
	assert.Equal(t, "3 validation errors:\n- missing required parameter: RequiredList\n- missing required parameter: RequiredMap\n- missing required parameter: RequiredBool", req.Error.(awserr.Error).Message())
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

	assert.Error(t, req.Error)
	assert.Equal(t, "InvalidParameter", req.Error.(awserr.Error).Code())
	assert.Equal(t, "3 validation errors:\n- missing required parameter: RequiredList[0].Name\n- missing required parameter: RequiredMap[\"key2\"].Name\n- missing required parameter: OptionalStruct.Name", req.Error.(awserr.Error).Message())

}
