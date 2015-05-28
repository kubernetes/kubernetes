/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validation

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
)

type InvalidTypeError struct {
	ExpectedKind reflect.Kind
	ObservedKind reflect.Kind
	FieldName    string
}

func (i *InvalidTypeError) Error() string {
	return fmt.Sprintf("expected type %s, for field %s, got %s", i.ExpectedKind.String(), i.FieldName, i.ObservedKind.String())
}

func NewInvalidTypeError(expected reflect.Kind, observed reflect.Kind, fieldName string) error {
	return &InvalidTypeError{expected, observed, fieldName}
}

// Schema is an interface that knows how to validate an API object serialized to a byte array.
type Schema interface {
	ValidateBytes(data []byte) error
}

type NullSchema struct{}

func (NullSchema) ValidateBytes(data []byte) error { return nil }

type SwaggerSchema struct {
	api swagger.ApiDeclaration
}

func NewSwaggerSchemaFromBytes(data []byte) (Schema, error) {
	schema := &SwaggerSchema{}
	err := json.Unmarshal(data, &schema.api)
	if err != nil {
		return nil, err
	}
	return schema, nil
}

func (s *SwaggerSchema) ValidateBytes(data []byte) error {
	var obj interface{}
	out, err := yaml.ToJSON(data)
	if err != nil {
		return err
	}
	data = out
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	fields, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("error in unmarshaling data %s", string(data))
	}
	apiVersion := fields["apiVersion"].(string)
	kind := fields["kind"].(string)
	return s.ValidateObject(obj, apiVersion, "", apiVersion+"."+kind)
}

func (s *SwaggerSchema) ValidateObject(obj interface{}, apiVersion, fieldName, typeName string) error {
	models := s.api.Models
	// TODO: handle required fields here too.
	model, ok := models[typeName]
	if !ok {
		return fmt.Errorf("couldn't find type: %s", typeName)
	}
	properties := model.Properties
	if len(properties) == 0 {
		// The object does not have any sub-fields.
		return nil
	}
	fields, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected object of type map[string]interface{} as value of %s field", fieldName)
	}
	if len(fieldName) > 0 {
		fieldName = fieldName + "."
	}
	for key, value := range fields {
		details, ok := properties[key]
		if !ok {
			glog.Infof("unknown field: %s", key)
			// Some properties can be missing because of
			// https://github.com/GoogleCloudPlatform/kubernetes/issues/6842.
			glog.Info("this may be a false alarm, see https://github.com/GoogleCloudPlatform/kubernetes/issues/6842")
			continue
		}
		if details.Type == nil && details.Ref == nil {
			return fmt.Errorf("could not find the type of %s from object: %v", key, details)
		}
		var fieldType string
		if details.Type != nil {
			fieldType = *details.Type
		} else {
			fieldType = *details.Ref
		}
		if value == nil {
			glog.V(2).Infof("Skipping nil field: %s", key)
			continue
		}
		err := s.validateField(value, apiVersion, fieldName+key, fieldType, &details)
		if err != nil {
			glog.Errorf("Validation failed for: %s, %v", key, value)
			return err
		}
	}
	return nil
}

func (s *SwaggerSchema) validateField(value interface{}, apiVersion, fieldName, fieldType string, fieldDetails *swagger.ModelProperty) error {
	if strings.HasPrefix(fieldType, apiVersion) {
		return s.ValidateObject(value, apiVersion, fieldName, fieldType)
	}
	switch fieldType {
	case "string":
		// Be loose about what we accept for 'string' since we use IntOrString in a couple of places
		_, isString := value.(string)
		_, isNumber := value.(float64)
		_, isInteger := value.(int)
		if !isString && !isNumber && !isInteger {
			return NewInvalidTypeError(reflect.String, reflect.TypeOf(value).Kind(), fieldName)
		}
	case "array":
		arr, ok := value.([]interface{})
		if !ok {
			return NewInvalidTypeError(reflect.Array, reflect.TypeOf(value).Kind(), fieldName)
		}
		var arrType string
		glog.Infof("field detail %v", fieldDetails)
		if fieldDetails.Items.Ref == nil && fieldDetails.Items.Type == nil {
			return NewInvalidTypeError(reflect.Array, reflect.TypeOf(value).Kind(), fieldName)
		}
		if fieldDetails.Items.Ref != nil {
			arrType = *fieldDetails.Items.Ref
		} else {
			arrType = *fieldDetails.Items.Type
		}
		for ix := range arr {
			err := s.validateField(arr[ix], apiVersion, fmt.Sprintf("%s[%d]", fieldName, ix), arrType, nil)
			if err != nil {
				return err
			}
		}
	case "uint64":
	case "int64":
	case "integer":
		_, isNumber := value.(float64)
		_, isInteger := value.(int)
		if !isNumber && !isInteger {
			return NewInvalidTypeError(reflect.Int, reflect.TypeOf(value).Kind(), fieldName)
		}
	case "float64":
		if _, ok := value.(float64); !ok {
			return NewInvalidTypeError(reflect.Float64, reflect.TypeOf(value).Kind(), fieldName)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return NewInvalidTypeError(reflect.Bool, reflect.TypeOf(value).Kind(), fieldName)
		}
	case "any":
	default:
		return fmt.Errorf("unexpected type: %v", fieldType)
	}
	return nil
}
