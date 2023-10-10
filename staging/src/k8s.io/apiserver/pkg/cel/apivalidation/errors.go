package apivalidation

import (
	"encoding/json"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	openapierrors "k8s.io/kube-openapi/pkg/validation/errors"
)

// Copied from staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/validation/validation.go
func kubeOpenAPIErrorToFieldError(fldPath *field.Path, err error) *field.Error {
	switch err := err.(type) {
	case *field.Error:
		if err == nil {
			return nil
		}
		if err.Field == "<nil>" {
			err.Field = fldPath.String()
		}
		return err
	case *openapierrors.Validation:
		if err == nil {
			return nil
		}
		errPath := fldPath
		if len(err.Name) > 0 && err.Name != "." {
			errPath = errPath.Child(strings.TrimPrefix(err.Name, "."))
		}

		switch err.Code() {
		case openapierrors.RequiredFailCode:
			return field.Required(errPath, "")

		case openapierrors.EnumFailCode:
			values := []string{}
			for _, allowedValue := range err.Values {
				if s, ok := allowedValue.(string); ok {
					values = append(values, s)
				} else {
					allowedJSON, _ := json.Marshal(allowedValue)
					values = append(values, string(allowedJSON))
				}
			}
			return field.NotSupported(errPath, err.Value, values)

		case openapierrors.TooLongFailCode:
			value := interface{}("")
			if err.Value != nil {
				value = err.Value
			}
			max := int64(-1)
			if i, ok := err.Valid.(int64); ok {
				max = i
			}
			return field.TooLongMaxLength(errPath, value, int(max))

		case openapierrors.MaxItemsFailCode:
			actual := int64(-1)
			if i, ok := err.Value.(int64); ok {
				actual = i
			}
			max := int64(-1)
			if i, ok := err.Valid.(int64); ok {
				max = i
			}
			return field.TooMany(errPath, int(actual), int(max))

		case openapierrors.TooManyPropertiesCode:
			actual := int64(-1)
			if i, ok := err.Value.(int64); ok {
				actual = i
			}
			max := int64(-1)
			if i, ok := err.Valid.(int64); ok {
				max = i
			}
			return field.TooMany(errPath, int(actual), int(max))

		case openapierrors.InvalidTypeCode:
			value := interface{}("")
			if err.Value != nil {
				value = err.Value
			}
			return field.TypeInvalid(errPath, value, err.Error())

		default:
			value := interface{}("")
			if err.Value != nil {
				value = err.Value
			}
			return field.Invalid(errPath, value, err.Error())
		}

	default:
		return field.Invalid(fldPath, "", err.Error())
	}
}
