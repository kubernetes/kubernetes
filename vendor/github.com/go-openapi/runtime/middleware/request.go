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

package middleware

import (
	"net/http"
	"reflect"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

// RequestBinder binds and validates the data from a http request
type untypedRequestBinder struct {
	Spec         *spec.Swagger
	Parameters   map[string]spec.Parameter
	Formats      strfmt.Registry
	paramBinders map[string]*untypedParamBinder
}

// NewRequestBinder creates a new binder for reading a request.
func newUntypedRequestBinder(parameters map[string]spec.Parameter, spec *spec.Swagger, formats strfmt.Registry) *untypedRequestBinder {
	binders := make(map[string]*untypedParamBinder)
	for fieldName, param := range parameters {
		binders[fieldName] = newUntypedParamBinder(param, spec, formats)
	}
	return &untypedRequestBinder{
		Parameters:   parameters,
		paramBinders: binders,
		Spec:         spec,
		Formats:      formats,
	}
}

// Bind perform the databinding and validation
func (o *untypedRequestBinder) Bind(request *http.Request, routeParams RouteParams, consumer runtime.Consumer, data interface{}) error {
	val := reflect.Indirect(reflect.ValueOf(data))
	isMap := val.Kind() == reflect.Map
	var result []error

	for fieldName, param := range o.Parameters {
		binder := o.paramBinders[fieldName]

		var target reflect.Value
		if !isMap {
			binder.Name = fieldName
			target = val.FieldByName(fieldName)
		}

		if isMap {
			tpe := binder.Type()
			if tpe == nil {
				if param.Schema.Type.Contains("array") {
					tpe = reflect.TypeOf([]interface{}{})
				} else {
					tpe = reflect.TypeOf(map[string]interface{}{})
				}
			}
			target = reflect.Indirect(reflect.New(tpe))

		}

		if !target.IsValid() {
			result = append(result, errors.New(500, "parameter name %q is an unknown field", binder.Name))
			continue
		}

		if err := binder.Bind(request, routeParams, consumer, target); err != nil {
			result = append(result, err)
			continue
		}

		if binder.validator != nil {
			rr := binder.validator.Validate(target.Interface())
			if rr != nil && rr.HasErrors() {
				result = append(result, rr.AsError())
			}
		}

		if isMap {
			val.SetMapIndex(reflect.ValueOf(param.Name), target)
		}
	}

	if len(result) > 0 {
		return errors.CompositeValidationError(result...)
	}

	return nil
}
