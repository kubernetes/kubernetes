package gophercloud

import (
	"fmt"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// MaybeString takes a string that might be a zero-value, and either returns a
// pointer to its address or a nil value (i.e. empty pointer). This is useful
// for converting zero values in options structs when the end-user hasn't
// defined values. Those zero values need to be nil in order for the JSON
// serialization to ignore them.
func MaybeString(original string) *string {
	if original != "" {
		return &original
	}
	return nil
}

// MaybeInt takes an int that might be a zero-value, and either returns a
// pointer to its address or a nil value (i.e. empty pointer).
func MaybeInt(original int) *int {
	if original != 0 {
		return &original
	}
	return nil
}

var t time.Time

func isZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Func, reflect.Map, reflect.Slice:
		return v.IsNil()
	case reflect.Array:
		z := true
		for i := 0; i < v.Len(); i++ {
			z = z && isZero(v.Index(i))
		}
		return z
	case reflect.Struct:
		if v.Type() == reflect.TypeOf(t) {
			if v.Interface().(time.Time).IsZero() {
				return true
			}
			return false
		}
		z := true
		for i := 0; i < v.NumField(); i++ {
			z = z && isZero(v.Field(i))
		}
		return z
	}
	// Compare other types directly:
	z := reflect.Zero(v.Type())
	return v.Interface() == z.Interface()
}

/*
BuildQueryString accepts a generic structure and parses it URL struct. It
converts field names into query names based on "q" tags. So for example, this
type:

	struct {
	   Bar string `q:"x_bar"`
	   Baz int    `q:"lorem_ipsum"`
	}{
	   Bar: "XXX",
	   Baz: "YYY",
	}

will be converted into ?x_bar=XXX&lorem_ipsum=YYYY
*/
func BuildQueryString(opts interface{}) (*url.URL, error) {
	optsValue := reflect.ValueOf(opts)
	if optsValue.Kind() == reflect.Ptr {
		optsValue = optsValue.Elem()
	}

	optsType := reflect.TypeOf(opts)
	if optsType.Kind() == reflect.Ptr {
		optsType = optsType.Elem()
	}

	var optsSlice []string
	if optsValue.Kind() == reflect.Struct {
		for i := 0; i < optsValue.NumField(); i++ {
			v := optsValue.Field(i)
			f := optsType.Field(i)
			qTag := f.Tag.Get("q")

			// if the field has a 'q' tag, it goes in the query string
			if qTag != "" {
				tags := strings.Split(qTag, ",")

				// if the field is set, add it to the slice of query pieces
				if !isZero(v) {
					switch v.Kind() {
					case reflect.String:
						optsSlice = append(optsSlice, tags[0]+"="+v.String())
					case reflect.Int:
						optsSlice = append(optsSlice, tags[0]+"="+strconv.FormatInt(v.Int(), 10))
					case reflect.Bool:
						optsSlice = append(optsSlice, tags[0]+"="+strconv.FormatBool(v.Bool()))
					}
				} else {
					// Otherwise, the field is not set.
					if len(tags) == 2 && tags[1] == "required" {
						// And the field is required. Return an error.
						return nil, fmt.Errorf("Required query parameter [%s] not set.", f.Name)
					}
				}
			}

		}
		// URL encode the string for safety.
		s := strings.Join(optsSlice, "&")
		if s != "" {
			s = "?" + s
		}
		u, err := url.Parse(s)
		if err != nil {
			return nil, err
		}
		return u, nil
	}
	// Return an error if the underlying type of 'opts' isn't a struct.
	return nil, fmt.Errorf("Options type is not a struct.")
}

// BuildHeaders accepts a generic structure and parses it into a string map. It
// converts field names into header names based on "h" tags, and field values
// into header values by a simple one-to-one mapping.
func BuildHeaders(opts interface{}) (map[string]string, error) {
	optsValue := reflect.ValueOf(opts)
	if optsValue.Kind() == reflect.Ptr {
		optsValue = optsValue.Elem()
	}

	optsType := reflect.TypeOf(opts)
	if optsType.Kind() == reflect.Ptr {
		optsType = optsType.Elem()
	}

	optsMap := make(map[string]string)
	if optsValue.Kind() == reflect.Struct {
		for i := 0; i < optsValue.NumField(); i++ {
			v := optsValue.Field(i)
			f := optsType.Field(i)
			hTag := f.Tag.Get("h")

			// if the field has a 'h' tag, it goes in the header
			if hTag != "" {
				tags := strings.Split(hTag, ",")

				// if the field is set, add it to the slice of query pieces
				if !isZero(v) {
					switch v.Kind() {
					case reflect.String:
						optsMap[tags[0]] = v.String()
					case reflect.Int:
						optsMap[tags[0]] = strconv.FormatInt(v.Int(), 10)
					case reflect.Bool:
						optsMap[tags[0]] = strconv.FormatBool(v.Bool())
					}
				} else {
					// Otherwise, the field is not set.
					if len(tags) == 2 && tags[1] == "required" {
						// And the field is required. Return an error.
						return optsMap, fmt.Errorf("Required header not set.")
					}
				}
			}

		}
		return optsMap, nil
	}
	// Return an error if the underlying type of 'opts' isn't a struct.
	return optsMap, fmt.Errorf("Options type is not a struct.")
}
