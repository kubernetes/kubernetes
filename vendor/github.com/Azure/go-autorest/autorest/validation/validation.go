/*
Package validation provides methods for validating parameter value using reflection.
*/
package validation

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
)

// Constraint stores constraint name, target field name
// Rule and chain validations.
type Constraint struct {

	// Target field name for validation.
	Target string

	// Constraint name e.g. minLength, MaxLength, Pattern, etc.
	Name string

	// Rule for constraint e.g. greater than 10, less than 5 etc.
	Rule interface{}

	// Chain Validations for struct type
	Chain []Constraint
}

// Validation stores parameter-wise validation.
type Validation struct {
	TargetValue interface{}
	Constraints []Constraint
}

// Constraint list
const (
	Empty            = "Empty"
	Null             = "Null"
	ReadOnly         = "ReadOnly"
	Pattern          = "Pattern"
	MaxLength        = "MaxLength"
	MinLength        = "MinLength"
	MaxItems         = "MaxItems"
	MinItems         = "MinItems"
	MultipleOf       = "MultipleOf"
	UniqueItems      = "UniqueItems"
	InclusiveMaximum = "InclusiveMaximum"
	ExclusiveMaximum = "ExclusiveMaximum"
	ExclusiveMinimum = "ExclusiveMinimum"
	InclusiveMinimum = "InclusiveMinimum"
)

// Validate method validates constraints on parameter
// passed in validation array.
func Validate(m []Validation) error {
	for _, item := range m {
		v := reflect.ValueOf(item.TargetValue)
		for _, constraint := range item.Constraints {
			var err error
			switch v.Kind() {
			case reflect.Ptr:
				err = validatePtr(v, constraint)
			case reflect.String:
				err = validateString(v, constraint)
			case reflect.Struct:
				err = validateStruct(v, constraint)
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				err = validateInt(v, constraint)
			case reflect.Float32, reflect.Float64:
				err = validateFloat(v, constraint)
			case reflect.Array, reflect.Slice, reflect.Map:
				err = validateArrayMap(v, constraint)
			default:
				err = createError(v, constraint, fmt.Sprintf("unknown type %v", v.Kind()))
			}

			if err != nil {
				return err
			}
		}
	}
	return nil
}

func validateStruct(x reflect.Value, v Constraint, name ...string) error {
	//Get field name from target name which is in format a.b.c
	s := strings.Split(v.Target, ".")
	f := x.FieldByName(s[len(s)-1])
	if isZero(f) {
		return createError(x, v, fmt.Sprintf("field %q doesn't exist", v.Target))
	}

	return Validate([]Validation{
		{
			TargetValue: getInterfaceValue(f),
			Constraints: []Constraint{v},
		},
	})
}

func validatePtr(x reflect.Value, v Constraint) error {
	if v.Name == ReadOnly {
		if !x.IsNil() {
			return createError(x.Elem(), v, "readonly parameter; must send as nil or empty in request")
		}
		return nil
	}
	if x.IsNil() {
		return checkNil(x, v)
	}
	if v.Chain != nil {
		return Validate([]Validation{
			{
				TargetValue: getInterfaceValue(x.Elem()),
				Constraints: v.Chain,
			},
		})
	}
	return nil
}

func validateInt(x reflect.Value, v Constraint) error {
	i := x.Int()
	r, ok := v.Rule.(int)
	if !ok {
		return createError(x, v, fmt.Sprintf("rule must be integer value for %v constraint; got: %v", v.Name, v.Rule))
	}
	switch v.Name {
	case MultipleOf:
		if i%int64(r) != 0 {
			return createError(x, v, fmt.Sprintf("value must be a multiple of %v", r))
		}
	case ExclusiveMinimum:
		if i <= int64(r) {
			return createError(x, v, fmt.Sprintf("value must be greater than %v", r))
		}
	case ExclusiveMaximum:
		if i >= int64(r) {
			return createError(x, v, fmt.Sprintf("value must be less than %v", r))
		}
	case InclusiveMinimum:
		if i < int64(r) {
			return createError(x, v, fmt.Sprintf("value must be greater than or equal to %v", r))
		}
	case InclusiveMaximum:
		if i > int64(r) {
			return createError(x, v, fmt.Sprintf("value must be less than or equal to %v", r))
		}
	default:
		return createError(x, v, fmt.Sprintf("constraint %v is not applicable for type integer", v.Name))
	}
	return nil
}

func validateFloat(x reflect.Value, v Constraint) error {
	f := x.Float()
	r, ok := v.Rule.(float64)
	if !ok {
		return createError(x, v, fmt.Sprintf("rule must be float value for %v constraint; got: %v", v.Name, v.Rule))
	}
	switch v.Name {
	case ExclusiveMinimum:
		if f <= r {
			return createError(x, v, fmt.Sprintf("value must be greater than %v", r))
		}
	case ExclusiveMaximum:
		if f >= r {
			return createError(x, v, fmt.Sprintf("value must be less than %v", r))
		}
	case InclusiveMinimum:
		if f < r {
			return createError(x, v, fmt.Sprintf("value must be greater than or equal to %v", r))
		}
	case InclusiveMaximum:
		if f > r {
			return createError(x, v, fmt.Sprintf("value must be less than or equal to %v", r))
		}
	default:
		return createError(x, v, fmt.Sprintf("constraint %s is not applicable for type float", v.Name))
	}
	return nil
}

func validateString(x reflect.Value, v Constraint) error {
	s := x.String()
	switch v.Name {
	case Empty:
		if len(s) == 0 {
			return checkEmpty(x, v)
		}
	case Pattern:
		reg, err := regexp.Compile(v.Rule.(string))
		if err != nil {
			return createError(x, v, err.Error())
		}
		if !reg.MatchString(s) {
			return createError(x, v, fmt.Sprintf("value doesn't match pattern %v", v.Rule))
		}
	case MaxLength:
		if _, ok := v.Rule.(int); !ok {
			return createError(x, v, fmt.Sprintf("rule must be integer value for %v constraint; got: %v", v.Name, v.Rule))
		}
		if len(s) > v.Rule.(int) {
			return createError(x, v, fmt.Sprintf("value length must be less than or equal to %v", v.Rule))
		}
	case MinLength:
		if _, ok := v.Rule.(int); !ok {
			return createError(x, v, fmt.Sprintf("rule must be integer value for %v constraint; got: %v", v.Name, v.Rule))
		}
		if len(s) < v.Rule.(int) {
			return createError(x, v, fmt.Sprintf("value length must be greater than or equal to %v", v.Rule))
		}
	case ReadOnly:
		if len(s) > 0 {
			return createError(reflect.ValueOf(s), v, "readonly parameter; must send as nil or empty in request")
		}
	default:
		return createError(x, v, fmt.Sprintf("constraint %s is not applicable to string type", v.Name))
	}

	if v.Chain != nil {
		return Validate([]Validation{
			{
				TargetValue: getInterfaceValue(x),
				Constraints: v.Chain,
			},
		})
	}
	return nil
}

func validateArrayMap(x reflect.Value, v Constraint) error {
	switch v.Name {
	case Null:
		if x.IsNil() {
			return checkNil(x, v)
		}
	case Empty:
		if x.IsNil() || x.Len() == 0 {
			return checkEmpty(x, v)
		}
	case MaxItems:
		if _, ok := v.Rule.(int); !ok {
			return createError(x, v, fmt.Sprintf("rule must be integer for %v constraint; got: %v", v.Name, v.Rule))
		}
		if x.Len() > v.Rule.(int) {
			return createError(x, v, fmt.Sprintf("maximum item limit is %v; got: %v", v.Rule, x.Len()))
		}
	case MinItems:
		if _, ok := v.Rule.(int); !ok {
			return createError(x, v, fmt.Sprintf("rule must be integer for %v constraint; got: %v", v.Name, v.Rule))
		}
		if x.Len() < v.Rule.(int) {
			return createError(x, v, fmt.Sprintf("minimum item limit is %v; got: %v", v.Rule, x.Len()))
		}
	case UniqueItems:
		if x.Kind() == reflect.Array || x.Kind() == reflect.Slice {
			if !checkForUniqueInArray(x) {
				return createError(x, v, fmt.Sprintf("all items in parameter %q must be unique; got:%v", v.Target, x))
			}
		} else if x.Kind() == reflect.Map {
			if !checkForUniqueInMap(x) {
				return createError(x, v, fmt.Sprintf("all items in parameter %q must be unique; got:%v", v.Target, x))
			}
		} else {
			return createError(x, v, fmt.Sprintf("type must be array, slice or map for constraint %v; got: %v", v.Name, x.Kind()))
		}
	case ReadOnly:
		if x.Len() != 0 {
			return createError(x, v, "readonly parameter; must send as nil or empty in request")
		}
	case Pattern:
		reg, err := regexp.Compile(v.Rule.(string))
		if err != nil {
			return createError(x, v, err.Error())
		}
		keys := x.MapKeys()
		for _, k := range keys {
			if !reg.MatchString(k.String()) {
				return createError(k, v, fmt.Sprintf("map key doesn't match pattern %v", v.Rule))
			}
		}
	default:
		return createError(x, v, fmt.Sprintf("constraint %v is not applicable to array, slice and map type", v.Name))
	}

	if v.Chain != nil {
		return Validate([]Validation{
			{
				TargetValue: getInterfaceValue(x),
				Constraints: v.Chain,
			},
		})
	}
	return nil
}

func checkNil(x reflect.Value, v Constraint) error {
	if _, ok := v.Rule.(bool); !ok {
		return createError(x, v, fmt.Sprintf("rule must be bool value for %v constraint; got: %v", v.Name, v.Rule))
	}
	if v.Rule.(bool) {
		return createError(x, v, "value can not be null; required parameter")
	}
	return nil
}

func checkEmpty(x reflect.Value, v Constraint) error {
	if _, ok := v.Rule.(bool); !ok {
		return createError(x, v, fmt.Sprintf("rule must be bool value for %v constraint; got: %v", v.Name, v.Rule))
	}

	if v.Rule.(bool) {
		return createError(x, v, "value can not be null or empty; required parameter")
	}
	return nil
}

func checkForUniqueInArray(x reflect.Value) bool {
	if x == reflect.Zero(reflect.TypeOf(x)) || x.Len() == 0 {
		return false
	}
	arrOfInterface := make([]interface{}, x.Len())

	for i := 0; i < x.Len(); i++ {
		arrOfInterface[i] = x.Index(i).Interface()
	}

	m := make(map[interface{}]bool)
	for _, val := range arrOfInterface {
		if m[val] {
			return false
		}
		m[val] = true
	}
	return true
}

func checkForUniqueInMap(x reflect.Value) bool {
	if x == reflect.Zero(reflect.TypeOf(x)) || x.Len() == 0 {
		return false
	}
	mapOfInterface := make(map[interface{}]interface{}, x.Len())

	keys := x.MapKeys()
	for _, k := range keys {
		mapOfInterface[k.Interface()] = x.MapIndex(k).Interface()
	}

	m := make(map[interface{}]bool)
	for _, val := range mapOfInterface {
		if m[val] {
			return false
		}
		m[val] = true
	}
	return true
}

func getInterfaceValue(x reflect.Value) interface{} {
	if x.Kind() == reflect.Invalid {
		return nil
	}
	return x.Interface()
}

func isZero(x interface{}) bool {
	return x == reflect.Zero(reflect.TypeOf(x)).Interface()
}

func createError(x reflect.Value, v Constraint, err string) error {
	return fmt.Errorf("autorest/validation: validation failed: parameter=%s constraint=%s value=%#v details: %s",
		v.Target, v.Name, getInterfaceValue(x), err)
}

// NewErrorWithValidationError appends package type and method name in
// validation error.
//
// Deprecated: Please use validation.NewError() instead.
func NewErrorWithValidationError(err error, packageType, method string) error {
	return NewError(packageType, method, err.Error())
}
