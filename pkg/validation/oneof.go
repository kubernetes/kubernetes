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
	"strings"
)

// +k8s:openapi-gen=validator(type(OneOf))
type OneOf struct {
	fields      []*field.Path
	common      *field.Path
	allNamesCsv string
}

var _ ValidationType = Len{}

func (l OneOf) Init(op OperationType, fields ...*FieldMeta) field.ErrorList {
	ret := field.ErrorList{}
	if len(fields) > 0 {
		for _, v := range fields {
			if v.Type != "string" {
				ret = append(ret, field.InternalError(v.Path, fmt.Errorf("Invalid type %v", v.Type)))
			}
		}
	} else {
		ret = append(ret, &field.Error{
			Type:     field.ErrorTypeInternal,
			Field:    "",
			BadValue: nil,
			Detail:   "No field provided to OneOf validatior",
		})
	}
	l.fields = []*field.Path{}
	for _, f := range fields {
		l.fields = append(l.fields, f.Path)
	}
	l.common = field.CommonParent(l.fields)
	allNames := []string{}
	for _, v := range l.fields {
		allNames = append(allNames, v.StringRelative(l.common))
	}
	l.allNamesCsv = strings.Join(allNames, ",")
	return ret
}

func (l OneOf) Validate(values ...interface{}) field.ErrorList {
	all := []int{}
	for i, v := range values {
		s := v.(string)
		if s != "" {
			all = append(all, i)
		}
	}

	if len(all) == 0 {
		return field.ErrorList{}
	}

	allWithValue := []string{}
	for _, i := range all {
		allWithValue = append(allWithValue, l.fields[i].StringRelative(l.common))
	}
	return field.ErrorList{&field.Error{
		Type:     field.ErrorTypeInvalid,
		Field:    l.common.String(),
		BadValue: nil,
		Detail:   fmt.Sprintf("Only one of %v fields should have non-empty values.", strings.Join(allWithValue, ",")),
	},
	}
}

func (l OneOf) OpenAPISpec(spec *spec.Schema) error {
	// spec.OneOf
	return nil
}

func (l OneOf) DocString() string {
	return fmt.Sprintf("Type should set only one of these fields: %v", l.allNamesCsv)
}
