package corehandlers

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// ValidateParameters is a request handler to validate the input parameters.
// Validating parameters only has meaning if done prior to the request being sent.
var ValidateParametersHandler = request.NamedHandler{"core.ValidateParametersHandler", func(r *request.Request) {
	if r.ParamsFilled() {
		v := validator{errors: []string{}}
		v.validateAny(reflect.ValueOf(r.Params), "")

		if count := len(v.errors); count > 0 {
			format := "%d validation errors:\n- %s"
			msg := fmt.Sprintf(format, count, strings.Join(v.errors, "\n- "))
			r.Error = awserr.New("InvalidParameter", msg, nil)
		}
	}
}}

// A validator validates values. Collects validations errors which occurs.
type validator struct {
	errors []string
}

// validateAny will validate any struct, slice or map type. All validations
// are also performed recursively for nested types.
func (v *validator) validateAny(value reflect.Value, path string) {
	value = reflect.Indirect(value)
	if !value.IsValid() {
		return
	}

	switch value.Kind() {
	case reflect.Struct:
		v.validateStruct(value, path)
	case reflect.Slice:
		for i := 0; i < value.Len(); i++ {
			v.validateAny(value.Index(i), path+fmt.Sprintf("[%d]", i))
		}
	case reflect.Map:
		for _, n := range value.MapKeys() {
			v.validateAny(value.MapIndex(n), path+fmt.Sprintf("[%q]", n.String()))
		}
	}
}

// validateStruct will validate the struct value's fields. If the structure has
// nested types those types will be validated also.
func (v *validator) validateStruct(value reflect.Value, path string) {
	prefix := "."
	if path == "" {
		prefix = ""
	}

	for i := 0; i < value.Type().NumField(); i++ {
		f := value.Type().Field(i)
		if strings.ToLower(f.Name[0:1]) == f.Name[0:1] {
			continue
		}
		fvalue := value.FieldByName(f.Name)

		err := validateField(f, fvalue, validateFieldRequired, validateFieldMin)
		if err != nil {
			v.errors = append(v.errors, fmt.Sprintf("%s: %s", err.Error(), path+prefix+f.Name))
			continue
		}

		v.validateAny(fvalue, path+prefix+f.Name)
	}
}

type validatorFunc func(f reflect.StructField, fvalue reflect.Value) error

func validateField(f reflect.StructField, fvalue reflect.Value, funcs ...validatorFunc) error {
	for _, fn := range funcs {
		if err := fn(f, fvalue); err != nil {
			return err
		}
	}
	return nil
}

// Validates that a field has a valid value provided for required fields.
func validateFieldRequired(f reflect.StructField, fvalue reflect.Value) error {
	if f.Tag.Get("required") == "" {
		return nil
	}

	switch fvalue.Kind() {
	case reflect.Ptr, reflect.Slice, reflect.Map:
		if fvalue.IsNil() {
			return fmt.Errorf("missing required parameter")
		}
	default:
		if !fvalue.IsValid() {
			return fmt.Errorf("missing required parameter")
		}
	}
	return nil
}

// Validates that if a value is provided for a field, that value must be at
// least a minimum length.
func validateFieldMin(f reflect.StructField, fvalue reflect.Value) error {
	minStr := f.Tag.Get("min")
	if minStr == "" {
		return nil
	}
	min, _ := strconv.ParseInt(minStr, 10, 64)

	kind := fvalue.Kind()
	if kind == reflect.Ptr {
		if fvalue.IsNil() {
			return nil
		}
		fvalue = fvalue.Elem()
	}

	switch fvalue.Kind() {
	case reflect.String:
		if int64(fvalue.Len()) < min {
			return fmt.Errorf("field too short, minimum length %d", min)
		}
	case reflect.Slice, reflect.Map:
		if fvalue.IsNil() {
			return nil
		}
		if int64(fvalue.Len()) < min {
			return fmt.Errorf("field too short, minimum length %d", min)
		}

		// TODO min can also apply to number minimum value.

	}
	return nil
}
