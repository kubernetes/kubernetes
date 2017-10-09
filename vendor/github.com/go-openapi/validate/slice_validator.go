// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import (
	"fmt"
	"reflect"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

type schemaSliceValidator struct {
	Path            string
	In              string
	MaxItems        *int64
	MinItems        *int64
	UniqueItems     bool
	AdditionalItems *spec.SchemaOrBool
	Items           *spec.SchemaOrArray
	Root            interface{}
	KnownFormats    strfmt.Registry
}

func (s *schemaSliceValidator) SetPath(path string) {
	s.Path = path
}

func (s *schemaSliceValidator) Applies(source interface{}, kind reflect.Kind) bool {
	_, ok := source.(*spec.Schema)
	r := ok && kind == reflect.Slice
	return r
}

func (s *schemaSliceValidator) Validate(data interface{}) *Result {
	result := new(Result)
	if data == nil {
		return result
	}
	val := reflect.ValueOf(data)
	size := val.Len()

	if s.Items != nil && s.Items.Schema != nil {
		validator := NewSchemaValidator(s.Items.Schema, s.Root, s.Path, s.KnownFormats)
		for i := 0; i < size; i++ {
			validator.SetPath(fmt.Sprintf("%s.%d", s.Path, i))
			value := val.Index(i)
			result.Merge(validator.Validate(value.Interface()))
		}
	}

	itemsSize := int64(0)
	if s.Items != nil && len(s.Items.Schemas) > 0 {
		itemsSize = int64(len(s.Items.Schemas))
		for i := int64(0); i < itemsSize; i++ {
			validator := NewSchemaValidator(&s.Items.Schemas[i], s.Root, fmt.Sprintf("%s.%d", s.Path, i), s.KnownFormats)
			if val.Len() <= int(i) {
				break
			}
			result.Merge(validator.Validate(val.Index(int(i)).Interface()))
		}

	}
	if s.AdditionalItems != nil && itemsSize < int64(size) {
		if s.Items != nil && len(s.Items.Schemas) > 0 && !s.AdditionalItems.Allows {
			result.AddErrors(errors.New(422, "array doesn't allow for additional items"))
		}
		if s.AdditionalItems.Schema != nil {
			for i := itemsSize; i < (int64(size)-itemsSize)+1; i++ {
				validator := NewSchemaValidator(s.AdditionalItems.Schema, s.Root, fmt.Sprintf("%s.%d", s.Path, i), s.KnownFormats)
				result.Merge(validator.Validate(val.Index(int(i)).Interface()))
			}
		}
	}

	if s.MinItems != nil {
		if err := MinItems(s.Path, s.In, int64(size), *s.MinItems); err != nil {
			result.AddErrors(err)
		}
	}
	if s.MaxItems != nil {
		if err := MaxItems(s.Path, s.In, int64(size), *s.MaxItems); err != nil {
			result.AddErrors(err)
		}
	}
	if s.UniqueItems {
		if err := UniqueItems(s.Path, s.In, val.Interface()); err != nil {
			result.AddErrors(err)
		}
	}
	result.Inc()
	return result
}
