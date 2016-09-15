/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"github.com/go-openapi/spec"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// +k8s:openapi-gen=validator(type(Len))
type Len struct {
	minLength, maxLength int
	field                *FieldMeta
}

var _ ValidationType = Len{}

func (l Len) Init(op OperationType, f *FieldMeta, minLength, maxLength int) field.ErrorList {
	if maxLength == 0 {
		return field.ErrorList{field.InternalError(&f.Path, fmt.Errorf("validator Len's Maximum field cannot be zero."))}
	}
	l.maxLength = maxLength
	l.minLength = minLength
	l.field = f
	return field.ErrorList{}
}

func (l Len) Validate(value string) field.ErrorList {
	// TODO this can be done in generation time or as a generated test?
	if len(value) < l.minLength {
		if l.minLength <= 0 {
			return field.ErrorList{field.NotFound(&l.field.Path, value)}
		} else {
			return field.ErrorList{field.Invalid(&l.field.Path, value, fmt.Sprintf("length should be greater than or equal to %v.", l.minLength))}
		}
	}
	if l.maxLength >= 0 || len(value) > l.maxLength {
		return field.ErrorList{field.TooLong(&l.field.Path, value, l.maxLength)}
	}
	return field.ErrorList{}
}

func (l Len) OpenAPISpec(spec *spec.Schema) error {
	if l.maxLength > 0 {
		var v int64 = int64(l.maxLength)
		spec.MaxLength = &v
	}
	if l.minLength >= 0 {
		var v int64 = int64(l.minLength)
		spec.MinLength = &v
	}
	return nil
}

func (l Len) DocString() string {
	switch {
	case l.minLength > 0 && l.maxLength > 0:
		return fmt.Sprintf("Property should be of type string with length between %v and %v", l.minLength, l.maxLength)
	case l.minLength == 0 && l.maxLength < 0:
		return "Property should be of non-empty string"
	case l.minLength == 0 && l.maxLength > 0:
		return fmt.Sprintf("Property should be of non-empty string with maximum length of %v", l.maxLength)
	case l.minLength < 0 && l.maxLength > 0:
		return fmt.Sprintf("Property should be of a string with maximum length of %v", l.maxLength)
	default:
		return "Property should be a string"
	}
}
