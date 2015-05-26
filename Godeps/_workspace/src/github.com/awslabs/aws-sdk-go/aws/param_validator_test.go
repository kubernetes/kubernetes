package aws_test

import (
	"testing"

	"github.com/awslabs/aws-sdk-go/aws"
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
	RequiredList   []*ConditionalStructShape           `required:"true"`
	RequiredMap    *map[string]*ConditionalStructShape `required:"true"`
	RequiredBool   *bool                               `required:"true"`
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
		RequiredMap: &map[string]*ConditionalStructShape{
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
	err := aws.Error(req.Error)

	assert.Error(t, err)
	assert.Equal(t, "InvalidParameter", err.Code)
	assert.Equal(t, "3 validation errors:\n- missing required parameter: RequiredList\n- missing required parameter: RequiredMap\n- missing required parameter: RequiredBool", err.Message)
}

func TestNestedMissingRequiredParameters(t *testing.T) {
	input := &StructShape{
		RequiredList: []*ConditionalStructShape{&ConditionalStructShape{}},
		RequiredMap: &map[string]*ConditionalStructShape{
			"key1": &ConditionalStructShape{Name: aws.String("Name")},
			"key2": &ConditionalStructShape{},
		},
		RequiredBool:   aws.Boolean(true),
		OptionalStruct: &ConditionalStructShape{},
	}

	req := aws.NewRequest(service, &aws.Operation{}, input, nil)
	aws.ValidateParameters(req)
	err := aws.Error(req.Error)

	assert.Error(t, err)
	assert.Equal(t, "InvalidParameter", err.Code)
	assert.Equal(t, "3 validation errors:\n- missing required parameter: RequiredList[0].Name\n- missing required parameter: RequiredMap[\"key2\"].Name\n- missing required parameter: OptionalStruct.Name", err.Message)

}
