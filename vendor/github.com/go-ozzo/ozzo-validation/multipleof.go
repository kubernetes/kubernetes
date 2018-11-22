package validation

import (
	"errors"
	"fmt"
	"reflect"
)

func MultipleOf(threshold interface{}) *multipleOfRule {
	return &multipleOfRule{
		threshold,
		fmt.Sprintf("must be multiple of %v", threshold),
	}
}

type multipleOfRule struct {
	threshold interface{}
	message   string
}

// Error sets the error message for the rule.
func (r *multipleOfRule) Error(message string) *multipleOfRule {
	r.message = message
	return r
}


func (r *multipleOfRule) Validate(value interface{}) error {

	rv := reflect.ValueOf(r.threshold)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v, err := ToInt(value)
		if err != nil {
			return err
		}
		if v%rv.Int() == 0 {
			return nil
		}

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		v, err := ToUint(value)
		if err != nil {
			return err
		}

		if v%rv.Uint() == 0 {
			return nil
		}
	default:
		return fmt.Errorf("type not supported: %v", rv.Type())
	}

	return errors.New(r.message)
}
