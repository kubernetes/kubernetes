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

	"k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// valueValidator validates the values it applies to.
type valueValidator interface {
	// SetPath sets the exact path of the validator prior to calling Validate.
	// The exact path contains the map keys and array indices to locate the
	// value to be validated from the root data element.
	SetPath(path string)
	// Applies returns true if the validator applies to the valueKind
	// from source. Validate will be called if and only if Applies returns true.
	Applies(source interface{}, valueKind reflect.Kind) bool
	// Validate validates the value.
	Validate(value interface{}) *Result
}

type basicCommonValidator struct {
	Path    string
	In      string
	Default interface{}
	Enum    []interface{}
}

func (b *basicCommonValidator) SetPath(path string) {
	b.Path = path
}

func (b *basicCommonValidator) Applies(source interface{}, kind reflect.Kind) bool {
	switch source.(type) {
	case *spec.Schema:
		return true
	}
	return false
}

func (b *basicCommonValidator) Validate(data interface{}) (res *Result) {
	if len(b.Enum) > 0 {
		for _, enumValue := range b.Enum {
			actualType := reflect.TypeOf(enumValue)
			if actualType != nil { // Safeguard
				expectedValue := reflect.ValueOf(data)
				if expectedValue.IsValid() && expectedValue.Type().ConvertibleTo(actualType) {
					if reflect.DeepEqual(expectedValue.Convert(actualType).Interface(), enumValue) {
						return nil
					}
				}
			}
		}
		return errorHelp.sErr(errors.EnumFail(b.Path, b.In, data, b.Enum))
	}
	return nil
}

type numberValidator struct {
	Path             string
	In               string
	Default          interface{}
	MultipleOf       *float64
	Maximum          *float64
	ExclusiveMaximum bool
	Minimum          *float64
	ExclusiveMinimum bool
	// Allows for more accurate behavior regarding integers
	Type   string
	Format string
}

func (n *numberValidator) SetPath(path string) {
	n.Path = path
}

func (n *numberValidator) Applies(source interface{}, kind reflect.Kind) bool {
	switch source.(type) {
	case *spec.Schema:
		isInt := kind >= reflect.Int && kind <= reflect.Uint64
		isFloat := kind == reflect.Float32 || kind == reflect.Float64
		r := isInt || isFloat
		debugLog("schema props validator for %q applies %t for %T (kind: %v) isInt=%t, isFloat=%t\n", n.Path, r, source, kind, isInt, isFloat)
		return r
	}
	debugLog("schema props validator for %q applies %t for %T (kind: %v)\n", n.Path, false, source, kind)
	return false
}

// Validate provides a validator for generic JSON numbers,
//
// By default, numbers are internally represented as float64.
// Formats float, or float32 may alter this behavior by mapping to float32.
// A special validation process is followed for integers, with optional "format":
// this is an attempt to provide a validation with native types.
//
// NOTE: since the constraint specified (boundary, multipleOf) is unmarshalled
// as float64, loss of information remains possible (e.g. on very large integers).
//
// Since this value directly comes from the unmarshalling, it is not possible
// at this stage of processing to check further and guarantee the correctness of such values.
//
// Normally, the JSON Number.MAX_SAFE_INTEGER (resp. Number.MIN_SAFE_INTEGER)
// would check we do not get such a loss.
//
// If this is the case, replace AddErrors() by AddWarnings() and IsValid() by !HasWarnings().
//
// TODO: consider replacing boundary check errors by simple warnings.
//
// TODO: default boundaries with MAX_SAFE_INTEGER are not checked (specific to json.Number?)
func (n *numberValidator) Validate(val interface{}) *Result {
	res := new(Result)

	resMultiple := new(Result)
	resMinimum := new(Result)
	resMaximum := new(Result)

	// Used only to attempt to validate constraint on value,
	// even though value or constraint specified do not match type and format
	data := valueHelp.asFloat64(val)

	// Is the provided value within the range of the specified numeric type and format?
	res.AddErrors(IsValueValidAgainstRange(val, n.Type, n.Format, "Checked", n.Path))

	// nolint: dupl
	if n.MultipleOf != nil {
		// Is the constraint specifier within the range of the specific numeric type and format?
		resMultiple.AddErrors(IsValueValidAgainstRange(*n.MultipleOf, n.Type, n.Format, "MultipleOf", n.Path))
		if resMultiple.IsValid() {
			// Constraint validated with compatible types
			if err := MultipleOfNativeType(n.Path, n.In, val, *n.MultipleOf); err != nil {
				resMultiple.Merge(errorHelp.sErr(err))
			}
		} else {
			// Constraint nevertheless validated, converted as general number
			if err := MultipleOf(n.Path, n.In, data, *n.MultipleOf); err != nil {
				resMultiple.Merge(errorHelp.sErr(err))
			}
		}
	}

	// nolint: dupl
	if n.Maximum != nil {
		// Is the constraint specifier within the range of the specific numeric type and format?
		resMaximum.AddErrors(IsValueValidAgainstRange(*n.Maximum, n.Type, n.Format, "Maximum boundary", n.Path))
		if resMaximum.IsValid() {
			// Constraint validated with compatible types
			if err := MaximumNativeType(n.Path, n.In, val, *n.Maximum, n.ExclusiveMaximum); err != nil {
				resMaximum.Merge(errorHelp.sErr(err))
			}
		} else {
			// Constraint nevertheless validated, converted as general number
			if err := Maximum(n.Path, n.In, data, *n.Maximum, n.ExclusiveMaximum); err != nil {
				resMaximum.Merge(errorHelp.sErr(err))
			}
		}
	}

	// nolint: dupl
	if n.Minimum != nil {
		// Is the constraint specifier within the range of the specific numeric type and format?
		resMinimum.AddErrors(IsValueValidAgainstRange(*n.Minimum, n.Type, n.Format, "Minimum boundary", n.Path))
		if resMinimum.IsValid() {
			// Constraint validated with compatible types
			if err := MinimumNativeType(n.Path, n.In, val, *n.Minimum, n.ExclusiveMinimum); err != nil {
				resMinimum.Merge(errorHelp.sErr(err))
			}
		} else {
			// Constraint nevertheless validated, converted as general number
			if err := Minimum(n.Path, n.In, data, *n.Minimum, n.ExclusiveMinimum); err != nil {
				resMinimum.Merge(errorHelp.sErr(err))
			}
		}
	}
	res.Merge(resMultiple, resMinimum, resMaximum)
	res.Inc()
	return res
}

type stringValidator struct {
	MaxLength *int64
	MinLength *int64
	Pattern   string
	Path      string
	In        string
}

func (s *stringValidator) SetPath(path string) {
	s.Path = path
}

func (s *stringValidator) Applies(source interface{}, kind reflect.Kind) bool {
	switch source.(type) {
	case *spec.Schema:
		r := kind == reflect.String
		debugLog("string validator for %q applies %t for %T (kind: %v)\n", s.Path, r, source, kind)
		return r
	}
	debugLog("string validator for %q applies %t for %T (kind: %v)\n", s.Path, false, source, kind)
	return false
}

func (s *stringValidator) Validate(val interface{}) *Result {
	data, ok := val.(string)
	if !ok {
		return errorHelp.sErr(errors.InvalidType(s.Path, s.In, stringType, val))
	}

	if s.MaxLength != nil {
		if err := MaxLength(s.Path, s.In, data, *s.MaxLength); err != nil {
			return errorHelp.sErr(err)
		}
	}

	if s.MinLength != nil {
		if err := MinLength(s.Path, s.In, data, *s.MinLength); err != nil {
			return errorHelp.sErr(err)
		}
	}

	if s.Pattern != "" {
		if err := Pattern(s.Path, s.In, data, s.Pattern); err != nil {
			return errorHelp.sErr(err)
		}
	}
	return nil
}
