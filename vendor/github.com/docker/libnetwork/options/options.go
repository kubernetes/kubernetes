// Package options provides a way to pass unstructured sets of options to a
// component expecting a strongly-typed configuration structure.
package options

import (
	"fmt"
	"reflect"
)

// NoSuchFieldError is the error returned when the generic parameters hold a
// value for a field absent from the destination structure.
type NoSuchFieldError struct {
	Field string
	Type  string
}

func (e NoSuchFieldError) Error() string {
	return fmt.Sprintf("no field %q in type %q", e.Field, e.Type)
}

// CannotSetFieldError is the error returned when the generic parameters hold a
// value for a field that cannot be set in the destination structure.
type CannotSetFieldError struct {
	Field string
	Type  string
}

func (e CannotSetFieldError) Error() string {
	return fmt.Sprintf("cannot set field %q of type %q", e.Field, e.Type)
}

// TypeMismatchError is the error returned when the type of the generic value
// for a field mismatches the type of the destination structure.
type TypeMismatchError struct {
	Field      string
	ExpectType string
	ActualType string
}

func (e TypeMismatchError) Error() string {
	return fmt.Sprintf("type mismatch, field %s require type %v, actual type %v", e.Field, e.ExpectType, e.ActualType)
}

// Generic is a basic type to store arbitrary settings.
type Generic map[string]interface{}

// NewGeneric returns a new Generic instance.
func NewGeneric() Generic {
	return make(Generic)
}

// GenerateFromModel takes the generic options, and tries to build a new
// instance of the model's type by matching keys from the generic options to
// fields in the model.
//
// The return value is of the same type than the model (including a potential
// pointer qualifier).
func GenerateFromModel(options Generic, model interface{}) (interface{}, error) {
	modType := reflect.TypeOf(model)

	// If the model is of pointer type, we need to dereference for New.
	resType := reflect.TypeOf(model)
	if modType.Kind() == reflect.Ptr {
		resType = resType.Elem()
	}

	// Populate the result structure with the generic layout content.
	res := reflect.New(resType)
	for name, value := range options {
		field := res.Elem().FieldByName(name)
		if !field.IsValid() {
			return nil, NoSuchFieldError{name, resType.String()}
		}
		if !field.CanSet() {
			return nil, CannotSetFieldError{name, resType.String()}
		}
		if reflect.TypeOf(value) != field.Type() {
			return nil, TypeMismatchError{name, field.Type().String(), reflect.TypeOf(value).String()}
		}
		field.Set(reflect.ValueOf(value))
	}

	// If the model is not of pointer type, return content of the result.
	if modType.Kind() == reflect.Ptr {
		return res.Interface(), nil
	}
	return res.Elem().Interface(), nil
}
