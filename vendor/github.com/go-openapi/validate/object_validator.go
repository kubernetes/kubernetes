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
	"reflect"
	"regexp"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

type objectValidator struct {
	Path                 string
	In                   string
	MaxProperties        *int64
	MinProperties        *int64
	Required             []string
	Properties           map[string]spec.Schema
	AdditionalProperties *spec.SchemaOrBool
	PatternProperties    map[string]spec.Schema
	Root                 interface{}
	KnownFormats         strfmt.Registry
}

func (o *objectValidator) SetPath(path string) {
	o.Path = path
}

func (o *objectValidator) Applies(source interface{}, kind reflect.Kind) bool {
	// TODO: this should also work for structs
	// there is a problem in the type validator where it will be unhappy about null values
	// so that requires more testing
	r := reflect.TypeOf(source) == specSchemaType && (kind == reflect.Map || kind == reflect.Struct)
	//fmt.Printf("object validator for %q applies %t for %T (kind: %v)\n", o.Path, r, source, kind)
	return r
}

func (o *objectValidator) Validate(data interface{}) *Result {
	val := data.(map[string]interface{})
	numKeys := int64(len(val))

	if o.MinProperties != nil && numKeys < *o.MinProperties {
		return sErr(errors.TooFewProperties(o.Path, o.In, *o.MinProperties))
	}
	if o.MaxProperties != nil && numKeys > *o.MaxProperties {
		return sErr(errors.TooManyProperties(o.Path, o.In, *o.MaxProperties))
	}

	res := new(Result)
	if len(o.Required) > 0 {
		for _, k := range o.Required {
			if _, ok := val[k]; !ok {
				res.AddErrors(errors.Required(o.Path+"."+k, o.In))
				continue
			}
		}
	}

	if o.AdditionalProperties != nil && !o.AdditionalProperties.Allows {
		for k := range val {
			_, regularProperty := o.Properties[k]
			matched := false

			for pk := range o.PatternProperties {
				if matches, _ := regexp.MatchString(pk, k); matches {
					matched = true
					break
				}
			}
			if !regularProperty && k != "$schema" && k != "id" && !matched {
				res.AddErrors(errors.PropertyNotAllowed(o.Path, o.In, k))
			}
		}
	} else {
		for key, value := range val {
			_, regularProperty := o.Properties[key]
			matched, succeededOnce, _ := o.validatePatternProperty(key, value, res)
			if !(regularProperty || matched || succeededOnce) {
				if o.AdditionalProperties != nil && o.AdditionalProperties.Schema != nil {
					res.Merge(NewSchemaValidator(o.AdditionalProperties.Schema, o.Root, o.Path+"."+key, o.KnownFormats).Validate(value))
				} else if regularProperty && !(matched || succeededOnce) {
					res.AddErrors(errors.FailedAllPatternProperties(o.Path, o.In, key))
				}
			}
		}
	}

	for pName, pSchema := range o.Properties {
		rName := pName
		if o.Path != "" {
			rName = o.Path + "." + pName
		}

		if v, ok := val[pName]; ok {
			res.Merge(NewSchemaValidator(&pSchema, o.Root, rName, o.KnownFormats).Validate(v))
		}
	}

	for key, value := range val {
		_, regularProperty := o.Properties[key]
		matched, succeededOnce, patterns := o.validatePatternProperty(key, value, res)
		if !regularProperty && (matched || succeededOnce) {
			for _, pName := range patterns {
				if v, ok := o.PatternProperties[pName]; ok {
					res.Merge(NewSchemaValidator(&v, o.Root, o.Path+"."+key, o.KnownFormats).Validate(value))
				}
			}
		}
	}
	return res
}

func (o *objectValidator) validatePatternProperty(key string, value interface{}, result *Result) (bool, bool, []string) {
	matched := false
	succeededOnce := false
	var patterns []string

	for k, schema := range o.PatternProperties {
		if match, _ := regexp.MatchString(k, key); match {
			patterns = append(patterns, k)
			matched = true
			validator := NewSchemaValidator(&schema, o.Root, o.Path+"."+key, o.KnownFormats)

			res := validator.Validate(value)
			result.Merge(res)
			if res.IsValid() {
				succeededOnce = true
			}
		}
	}

	if succeededOnce {
		result.Inc()
	}

	return matched, succeededOnce, patterns
}
