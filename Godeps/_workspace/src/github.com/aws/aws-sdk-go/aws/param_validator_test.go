package aws_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/stretchr/testify/assert"
)

var service = func() *aws.Service {
	s := &aws.Service{
		Config:      &aws.Config{},
		ServiceName: "mock-service",
		APIVersion:  "2015-01-01",
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
			"key1": &ConditionalStructShape{Name: aws.String("Name")},
			"key2": &ConditionalStructShape{Name: aws.String("Name")},
		},
		RequiredBool:   aws.Boolean(true),
		OptionalStruct: &ConditionalStructShape{Name: aws.String("Name")},
	}

	req := aws.NewRequest(service, &aws.Operation{}, input, nil)
	aws.ValidateParameters(req)
	assert.NoError(t, req.Error)
}

func TestMissingRequiredParameters(t *testing.T) {
	input := &StructShape{}
	req := aws.NewRequest(service, &aws.Operation{}, input, nil)
	aws.ValidateParameters(req)

	assert.Error(t, req.Error)
	assert.Equal(t, "InvalidParameter", req.Error.(awserr.Error).Code())
	assert.Equal(t, "3 validation errors:\n- missing required parameter: RequiredList\n- missing required parameter: RequiredMap\n- missing required parameter: RequiredBool", req.Error.(awserr.Error).Message())
}

func TestNestedMissingRequiredParameters(t *testing.T) {
	input := &StructShape{
		RequiredList: []*ConditionalStructShape{&ConditionalStructShape{}},
		RequiredMap: map[string]*ConditionalStructShape{
			"key1": &ConditionalStructShape{Name: aws.String("Name")},
			"key2": &ConditionalStructShape{},
		},
		RequiredBool:   aws.Boolean(true),
		OptionalStruct: &ConditionalStructShape{},
	}

	req := aws.NewRequest(service, &aws.Operation{}, input, nil)
	aws.ValidateParameters(req)

	assert.Error(t, req.Error)
	assert.Equal(t, "InvalidParameter", req.Error.(awserr.Error).Code())
	assert.Equal(t, "3 validation errors:\n- missing required parameter: RequiredList[0].Name\n- missing required parameter: RequiredMap[\"key2\"].Name\n- missing required parameter: OptionalStruct.Name", req.Error.(awserr.Error).Message())

}
