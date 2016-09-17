package mapstructure

import (
	"errors"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// typedDecodeHook takes a raw DecodeHookFunc (an interface{}) and turns
// it into the proper DecodeHookFunc type, such as DecodeHookFuncType.
func typedDecodeHook(h DecodeHookFunc) DecodeHookFunc {
	// Create variables here so we can reference them with the reflect pkg
	var f1 DecodeHookFuncType
	var f2 DecodeHookFuncKind

	// Fill in the variables into this interface and the rest is done
	// automatically using the reflect package.
	potential := []interface{}{f1, f2}

	v := reflect.ValueOf(h)
	vt := v.Type()
	for _, raw := range potential {
		pt := reflect.ValueOf(raw).Type()
		if vt.ConvertibleTo(pt) {
			return v.Convert(pt).Interface()
		}
	}

	return nil
}

// DecodeHookExec executes the given decode hook. This should be used
// since it'll naturally degrade to the older backwards compatible DecodeHookFunc
// that took reflect.Kind instead of reflect.Type.
func DecodeHookExec(
	raw DecodeHookFunc,
	from reflect.Type, to reflect.Type,
	data interface{}) (interface{}, error) {
	// Build our arguments that reflect expects
	argVals := make([]reflect.Value, 3)
	argVals[0] = reflect.ValueOf(from)
	argVals[1] = reflect.ValueOf(to)
	argVals[2] = reflect.ValueOf(data)

	switch f := typedDecodeHook(raw).(type) {
	case DecodeHookFuncType:
		return f(from, to, data)
	case DecodeHookFuncKind:
		return f(from.Kind(), to.Kind(), data)
	default:
		return nil, errors.New("invalid decode hook signature")
	}
}

// ComposeDecodeHookFunc creates a single DecodeHookFunc that
// automatically composes multiple DecodeHookFuncs.
//
// The composed funcs are called in order, with the result of the
// previous transformation.
func ComposeDecodeHookFunc(fs ...DecodeHookFunc) DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data interface{}) (interface{}, error) {
		var err error
		for _, f1 := range fs {
			data, err = DecodeHookExec(f1, f, t, data)
			if err != nil {
				return nil, err
			}

			// Modify the from kind to be correct with the new data
			f = nil
			if val := reflect.ValueOf(data); val.IsValid() {
				f = val.Type()
			}
		}

		return data, nil
	}
}

// StringToSliceHookFunc returns a DecodeHookFunc that converts
// string to []string by splitting on the given sep.
func StringToSliceHookFunc(sep string) DecodeHookFunc {
	return func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		if f != reflect.String || t != reflect.Slice {
			return data, nil
		}

		raw := data.(string)
		if raw == "" {
			return []string{}, nil
		}

		return strings.Split(raw, sep), nil
	}
}

// StringToTimeDurationHookFunc returns a DecodeHookFunc that converts
// strings to time.Duration.
func StringToTimeDurationHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data interface{}) (interface{}, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(time.Duration(5)) {
			return data, nil
		}

		// Convert it by parsing
		return time.ParseDuration(data.(string))
	}
}

func WeaklyTypedHook(
	f reflect.Kind,
	t reflect.Kind,
	data interface{}) (interface{}, error) {
	dataVal := reflect.ValueOf(data)
	switch t {
	case reflect.String:
		switch f {
		case reflect.Bool:
			if dataVal.Bool() {
				return "1", nil
			} else {
				return "0", nil
			}
		case reflect.Float32:
			return strconv.FormatFloat(dataVal.Float(), 'f', -1, 64), nil
		case reflect.Int:
			return strconv.FormatInt(dataVal.Int(), 10), nil
		case reflect.Slice:
			dataType := dataVal.Type()
			elemKind := dataType.Elem().Kind()
			if elemKind == reflect.Uint8 {
				return string(dataVal.Interface().([]uint8)), nil
			}
		case reflect.Uint:
			return strconv.FormatUint(dataVal.Uint(), 10), nil
		}
	}

	return data, nil
}
