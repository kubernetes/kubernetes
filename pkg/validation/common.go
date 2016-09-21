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
	"github.com/go-openapi/spec"
	"k8s.io/kubernetes/pkg/util/validation/field"
	"reflect"
)

type FieldMeta struct {
	*field.Path
	Type string // only if it is builtin
}

type OperationType int

const (
	CREATE OperationType = iota
	UPDATE
	DELETE
)

type ValidationType interface {

	// These methods do not have a set signature and the signiture would be varies between different types.
	// Generated code would be aware of this and given the right parameter and usage of the type, there would
	// be no compilation error. Semi-signatures here are only for reference.
	//
	// Init(OperationType, field meta of current field, any literal parameter or field meta(s) of reference fields)
	// Validate(field value(s) in their original type) field.ErrorList

	OpenAPISpec(spec *spec.Schema) error
	DocString() string
}

type Validator interface {
	Validate(meta *FieldMeta, op OperationType) field.ErrorList
}

func Validate(object interface{}, op OperationType) field.ErrorList {
	v, ok := object.(Validator)
	if !ok {
		return field.ErrorList{}
	}
	return v.Validate(&FieldMeta{Path: field.NewPath(reflect.TypeOf(object).Name())}, op)
}
