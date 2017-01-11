/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/emicklei/go-restful/swagger"
	ejson "github.com/exponent-io/jsonpath"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/yaml"
	apiutil "k8s.io/kubernetes/pkg/api/util"
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

// TypeNotFoundError is returned when specified type
// can not found in schema
type TypeNotFoundError string

func (tnfe TypeNotFoundError) Error() string {
	return fmt.Sprintf("couldn't find type: %s", string(tnfe))
}

// Schema is an interface that knows how to validate an API object serialized to a byte array.
type Schema interface {
	ValidateBytes(data []byte) error
}

type NullSchema struct{}

func (NullSchema) ValidateBytes(data []byte) error { return nil }

type NoDoubleKeySchema struct{}

func (NoDoubleKeySchema) ValidateBytes(data []byte) error {
	var list []error = nil
	if err := validateNoDuplicateKeys(data, "metadata", "labels"); err != nil {
		list = append(list, err)
	}
	if err := validateNoDuplicateKeys(data, "metadata", "annotations"); err != nil {
		list = append(list, err)
	}
	return utilerrors.NewAggregate(list)
}

func validateNoDuplicateKeys(data []byte, path ...string) error {
	r := ejson.NewDecoder(bytes.NewReader(data))
	// This is Go being unfriendly. The 'path ...string' comes in as a
	// []string, and SeekTo takes ...interface{}, so we can't just pass
	// the path straight in, we have to copy it.  *sigh*
	ifacePath := []interface{}{}
	for ix := range path {
		ifacePath = append(ifacePath, path[ix])
	}
	found, err := r.SeekTo(ifacePath...)
	if err != nil {
		return err
	}
	if !found {
		return nil
	}
	seen := map[string]bool{}
	for {
		tok, err := r.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case json.Delim:
			if t.String() == "}" {
				return nil
			}
		case ejson.KeyString:
			if seen[string(t)] {
				return fmt.Errorf("duplicate key: %s", string(t))
			} else {
				seen[string(t)] = true
			}
		}
	}
}

type ConjunctiveSchema []Schema

func (c ConjunctiveSchema) ValidateBytes(data []byte) error {
	var list []error = nil
	schemas := []Schema(c)
	for ix := range schemas {
		if err := schemas[ix].ValidateBytes(data); err != nil {
			list = append(list, err)
		}
	}
	return utilerrors.NewAggregate(list)
}

type SwaggerSchema struct {
	api      swagger.ApiDeclaration
	delegate Schema // For delegating to other api groups
}

func NewSwaggerSchemaFromBytes(data []byte, factory Schema) (Schema, error) {
	schema := &SwaggerSchema{}
	err := json.Unmarshal(data, &schema.api)
	if err != nil {
		return nil, err
	}
	schema.delegate = factory
	return schema, nil
}

// validateList unpacks a list and validate every item in the list.
// It return nil if every item is ok.
// Otherwise it return an error list contain errors of every item.
func (s *SwaggerSchema) validateList(obj map[string]interface{}) []error {
	items, exists := obj["items"]
	if !exists {
		return []error{fmt.Errorf("no items field in %#v", obj)}
	}
	return s.validateItems(items)
}

func (s *SwaggerSchema) validateItems(items interface{}) []error {
	allErrs := []error{}
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
		return utilerrors.NewAggregate(s.validateList(fields))
	}
	version := apiutil.GetVersion(groupVersion.(string))
	allErrs := s.ValidateObject(obj, "", version+"."+kind.(string))
	if len(allErrs) == 1 {
		return allErrs[0]
	}
	return utilerrors.NewAggregate(allErrs)
}

func (s *SwaggerSchema) ValidateObject(obj interface{}, fieldName, typeName string) []error {
	allErrs := []error{}
	models := s.api.Models
	model, ok := models.At(typeName)

	// Verify the api version matches.  This is required for nested types with differing api versions because
	// s.api only has schema for 1 api version (the parent object type's version).
	// e.g. an extensions/v1beta1 Template embedding a /v1 Service requires the schema for the extensions/v1beta1
	// api to delegate to the schema for the /v1 api.
	// Only do this for !ok objects so that cross ApiVersion vendored types take precedence.
	if !ok && s.delegate != nil {
		fields, mapOk := obj.(map[string]interface{})
		if !mapOk {
			return append(allErrs, fmt.Errorf("field %s: expected object of type map[string]interface{}, but the actual type is %T", fieldName, obj))
		}
		if delegated, err := s.delegateIfDifferentApiVersion(&unstructured.Unstructured{Object: fields}); delegated {
			if err != nil {
				allErrs = append(allErrs, err)
			}
			return allErrs
		}
	}

	if !ok {
		return append(allErrs, TypeNotFoundError(typeName))
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

		// Special case for runtime.RawExtension and runtime.Objects because they always fail to validate
		// This is because the actual values will be of some sub-type (e.g. Deployment) not the expected
		// super-type (RawExtension)
		if s.isGenericArray(details) {
			errs := s.validateItems(value)
			if len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			}
			continue
		}
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

// delegateIfDifferentApiVersion delegates the validation of an object if its ApiGroup does not match the
// current SwaggerSchema.
// First return value is true if the validation was delegated (by a different ApiGroup SwaggerSchema)
// Second return value is the result of the delegated validation if performed.
func (s *SwaggerSchema) delegateIfDifferentApiVersion(obj *unstructured.Unstructured) (bool, error) {
	// Never delegate objects in the same ApiVersion or we will get infinite recursion
	if !s.isDifferentApiVersion(obj) {
		return false, nil
	}

	// Convert the object back into bytes so that we can pass it to the ValidateBytes function
	m, err := json.Marshal(obj.Object)
	if err != nil {
		return true, err
	}

	// Delegate validation of this object to the correct SwaggerSchema for its ApiGroup
	return true, s.delegate.ValidateBytes(m)
}

// isDifferentApiVersion Returns true if obj lives in a different ApiVersion than the SwaggerSchema does.
// The SwaggerSchema will not be able to process objects in different ApiVersions unless they are vendored.
func (s *SwaggerSchema) isDifferentApiVersion(obj *unstructured.Unstructured) bool {
	groupVersion := obj.GetAPIVersion()
	return len(groupVersion) > 0 && s.api.ApiVersion != groupVersion
}

// isGenericArray Returns true if p is an array of generic Objects - either RawExtension or Object.
func (s *SwaggerSchema) isGenericArray(p swagger.ModelProperty) bool {
	return p.DataTypeFields.Type != nil &&
		*p.DataTypeFields.Type == "array" &&
		p.Items != nil &&
		p.Items.Ref != nil &&
		(*p.Items.Ref == "runtime.RawExtension" || *p.Items.Ref == "runtime.Object")
}

// This matches type name in the swagger spec, such as "v1.Binding".
var versionRegexp = regexp.MustCompile(`^(v.+|unversioned)\..*`)

func (s *SwaggerSchema) validateField(value interface{}, fieldName, fieldType string, fieldDetails *swagger.ModelProperty) []error {
	allErrs := []error{}
	if reflect.TypeOf(value) == nil {
		return append(allErrs, fmt.Errorf("unexpected nil value for field %v", fieldName))
	}
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
	// API servers before release 1.3 produce swagger spec with `type: "any"` as the fallback type, while newer servers produce spec with `type: "object"`.
	// We have both here so that kubectl can work with both old and new api servers.
	case "object":
	case "any":
	default:
		return append(allErrs, fmt.Errorf("unexpected type: %v", fieldType))
	}
	return allErrs
}
