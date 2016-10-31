package godo

import (
	"bytes"
	"fmt"
	"reflect"
)

var timestampType = reflect.TypeOf(Timestamp{})

// Stringify attempts to create a string representation of DigitalOcean types
func Stringify(message interface{}) string {
	var buf bytes.Buffer
	v := reflect.ValueOf(message)
	stringifyValue(&buf, v)
	return buf.String()
}

// stringifyValue was graciously cargoculted from the goprotubuf library
func stringifyValue(w *bytes.Buffer, val reflect.Value) {
	if val.Kind() == reflect.Ptr && val.IsNil() {
		_, _ = w.Write([]byte("<nil>"))
		return
	}

	v := reflect.Indirect(val)

	switch v.Kind() {
	case reflect.String:
		fmt.Fprintf(w, `"%s"`, v)
	case reflect.Slice:
		_, _ = w.Write([]byte{'['})
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				_, _ = w.Write([]byte{' '})
			}

			stringifyValue(w, v.Index(i))
		}

		_, _ = w.Write([]byte{']'})
		return
	case reflect.Struct:
		if v.Type().Name() != "" {
			_, _ = w.Write([]byte(v.Type().String()))
		}

		// special handling of Timestamp values
		if v.Type() == timestampType {
			fmt.Fprintf(w, "{%s}", v.Interface())
			return
		}

		_, _ = w.Write([]byte{'{'})

		var sep bool
		for i := 0; i < v.NumField(); i++ {
			fv := v.Field(i)
			if fv.Kind() == reflect.Ptr && fv.IsNil() {
				continue
			}
			if fv.Kind() == reflect.Slice && fv.IsNil() {
				continue
			}

			if sep {
				_, _ = w.Write([]byte(", "))
			} else {
				sep = true
			}

			_, _ = w.Write([]byte(v.Type().Field(i).Name))
			_, _ = w.Write([]byte{':'})
			stringifyValue(w, fv)
		}

		_, _ = w.Write([]byte{'}'})
	default:
		if v.CanInterface() {
			fmt.Fprint(w, v.Interface())
		}
	}
}
