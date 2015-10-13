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
	"regexp"
	"strings"

	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/util/errors"
	errs "k8s.io/kubernetes/pkg/util/fielderrors"
	"k8s.io/kubernetes/pkg/util/yaml"
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

// validateList unpack a list and validate every item in the list.
// It return nil if every item is ok.
// Otherwise it return an error list contain errors of every item.
func (s *SwaggerSchema) validateList(obj map[string]interface{}) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	items, exists := obj["items"]
	if !exists {
		return append(allErrs, fmt.Errorf("no items field in %#v", obj))
	}
	itemList, ok := items.([]interface{})
	if !ok {
		return append(allErrs, fmt.Errorf("items isn't a slice"))
	}
	for i, item := range itemList {
		fields, ok := item.(map[string]interface{})
		if !ok {
			allErrs = append(allErrs, fmt.Errorf("items[%d] isn't a map[string]interface{}", i))
			continue
		}
		groupVersion := fields["apiVersion"]
		if groupVersion == nil {
			allErrs = append(allErrs, fmt.Errorf("items[%d].apiVersion not set", i))
			continue
		}
		itemVersion, ok := groupVersion.(string)
		if !ok {
			allErrs = append(allErrs, fmt.Errorf("items[%d].apiVersion isn't string type", i))
			continue
		}
		if len(itemVersion) == 0 {
			allErrs = append(allErrs, fmt.Errorf("items[%d].apiVersion is empty", i))
		}
		kind := fields["kind"]
		if kind == nil {
			allErrs = append(allErrs, fmt.Errorf("items[%d].kind not set", i))
			continue
		}
		itemKind, ok := kind.(string)
		if !ok {
			allErrs = append(allErrs, fmt.Errorf("items[%d].kind isn't string type", i))
			continue
		}
		if len(itemKind) == 0 {
			allErrs = append(allErrs, fmt.Errorf("items[%d].kind is empty", i))
		}
		version := apiutil.GetVersion(itemVersion)
		errs := s.ValidateObject(item, "", version+"."+itemKind)
		if len(errs) >= 1 {
			allErrs = append(allErrs, errs...)
		}
	}
	return allErrs
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
	groupVersion := fields["apiVersion"]
	if groupVersion == nil {
		return fmt.Errorf("apiVersion not set")
	}
	if _, ok := groupVersion.(string); !ok {
		return fmt.Errorf("apiVersion isn't string type")
	}
	kind := fields["kind"]
	if kind == nil {
		return fmt.Errorf("kind not set")
	}
	if _, ok := kind.(string); !ok {
		return fmt.Errorf("kind isn't string type")
	}
	if strings.HasSuffix(kind.(string), "List") {
		return errors.NewAggregate(s.validateList(fields))
	}
	version := apiutil.GetVersion(groupVersion.(string))
	allErrs := s.ValidateObject(obj, "", version+"."+kind.(string))
	if len(allErrs) == 1 {
		return allErrs[0]
	}
	return errors.NewAggregate(allErrs)
}

func (s *SwaggerSchema) ValidateObject(obj interface{}, fieldName, typeName string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	models := s.api.Models
	model, ok := models.At(typeName)
	if !ok {
		return append(allErrs, fmt.Errorf("couldn't find type: %s", typeName))
	}
	properties := model.Properties
	if len(properties.List) == 0 {
		// The object does not have any sub-fields.
		return nil
	}
	fields, ok := obj.(map[string]interface{})
	if !ok {
		return append(allErrs, fmt.Errorf("field %s: expected object of type map[string]interface{}, but the actual type is %T", fieldName, obj))
	}
	if len(fieldName) > 0 {
		fieldName = fieldName + "."
	}
	// handle required fields
	for _, requiredKey := range model.Required {
		if _, ok := fields[requiredKey]; !ok {
			allErrs = append(allErrs, fmt.Errorf("field %s: is required", requiredKey))
		}
	}
	for key, value := range fields {
		details, ok := properties.At(key)
		if !ok {
			allErrs = append(allErrs, fmt.Errorf("found invalid field %s for %s", key, typeName))
			continue
		}
		if details.Type == nil && details.Ref == nil {
			allErrs = append(allErrs, fmt.Errorf("could not find the type of %s from object: %v", key, details))
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
		errs := s.validateField(value, fieldName+key, fieldType, &details)
		if len(errs) > 0 {
			allErrs = append(allErrs, errs...)
		}
	}
	return allErrs
}

// This matches type name in the swagger spec, such as "v1.Binding".
var versionRegexp = regexp.MustCompile(`^v.+\..*`)

func (s *SwaggerSchema) validateField(value interface{}, fieldName, fieldType string, fieldDetails *swagger.ModelProperty) errs.ValidationErrorList {
	// TODO: caesarxuchao: because we have multiple group/versions and objects
	// may reference objects in other group, the commented out way of checking
	// if a filedType is a type defined by us is outdated. We use a hacky way
	// for now.
	// TODO: the type name in the swagger spec is something like "v1.Binding",
	// and the "v1" is generated from the package name, not the groupVersion of
	// the type. We need to fix go-restful to embed the group name in the type
	// name, otherwise we couldn't handle identically named types in different
	// groups correctly.
	if versionRegexp.MatchString(fieldType) {
		// if strings.HasPrefix(fieldType, apiVersion) {
		return s.ValidateObject(value, fieldName, fieldType)
	}
	allErrs := errs.ValidationErrorList{}
	switch fieldType {
	case "string":
		// Be loose about what we accept for 'string' since we use IntOrString in a couple of places
		_, isString := value.(string)
		_, isNumber := value.(float64)
		_, isInteger := value.(int)
		if !isString && !isNumber && !isInteger {
			return append(allErrs, NewInvalidTypeError(reflect.String, reflect.TypeOf(value).Kind(), fieldName))
		}
	case "array":
		arr, ok := value.([]interface{})
		if !ok {
			return append(allErrs, NewInvalidTypeError(reflect.Array, reflect.TypeOf(value).Kind(), fieldName))
		}
		var arrType string
		if fieldDetails.Items.Ref == nil && fieldDetails.Items.Type == nil {
			return append(allErrs, NewInvalidTypeError(reflect.Array, reflect.TypeOf(value).Kind(), fieldName))
		}
		if fieldDetails.Items.Ref != nil {
			arrType = *fieldDetails.Items.Ref
		} else {
			arrType = *fieldDetails.Items.Type
		}
		for ix := range arr {
			errs := s.validateField(arr[ix], fmt.Sprintf("%s[%d]", fieldName, ix), arrType, nil)
			if len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			}
		}
	case "uint64":
	case "int64":
	case "integer":
		_, isNumber := value.(float64)
		_, isInteger := value.(int)
		if !isNumber && !isInteger {
			return append(allErrs, NewInvalidTypeError(reflect.Int, reflect.TypeOf(value).Kind(), fieldName))
		}
	case "float64":
		if _, ok := value.(float64); !ok {
			return append(allErrs, NewInvalidTypeError(reflect.Float64, reflect.TypeOf(value).Kind(), fieldName))
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return append(allErrs, NewInvalidTypeError(reflect.Bool, reflect.TypeOf(value).Kind(), fieldName))
		}
	case "any":
	default:
		return append(allErrs, fmt.Errorf("unexpected type: %v", fieldType))
	}
	return allErrs
}
