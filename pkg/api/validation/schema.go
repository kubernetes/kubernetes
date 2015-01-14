/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
	"gopkg.in/yaml.v2"
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
	err := yaml.Unmarshal(data, &obj)
	if err != nil {
		return err
	}
	fields := obj.(map[interface{}]interface{})
	apiVersion := fields["apiVersion"].(string)
	kind := fields["kind"].(string)
	return s.ValidateObject(obj, apiVersion, "", apiVersion+"."+kind)
}

func (s *SwaggerSchema) ValidateObject(obj interface{}, apiVersion, fieldName, typeName string) error {
	models := s.api.Models
	// TODO: handle required fields here too.
	model, ok := models[typeName]
	if !ok {
		glog.V(2).Infof("couldn't find type: %s, skipping validation", typeName)
		return nil
	}
	properties := model.Properties
	fields := obj.(map[interface{}]interface{})
	if len(fieldName) > 0 {
		fieldName = fieldName + "."
	}
	for key, value := range fields {
		details, ok := properties[key.(string)]
		if !ok {
			glog.V(2).Infof("couldn't find properties for %s, skipping", key)
			continue
		}
		fieldType := *details.Type
		if value == nil {
			glog.V(2).Infof("Skipping nil field: %s", key)
			continue
		}
		err := s.validateField(value, apiVersion, fieldName+key.(string), fieldType, &details)
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
		arrType := *fieldDetails.Items[0].Ref
		for ix := range arr {
			err := s.validateField(arr[ix], apiVersion, fmt.Sprintf("%s[%d]", fieldName, ix), arrType, nil)
			if err != nil {
				return err
			}
		}
	case "uint64":
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
	default:
		return fmt.Errorf("unexpected type: %v", fieldType)
	}
	return nil
}
