package gophercloud

import (
	"fmt"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// EnabledState is a convenience type, mostly used in Create and Update
// operations. Because the zero value of a bool is FALSE, we need to use a
// pointer instead to indicate zero-ness.
type EnabledState *bool

// Convenience vars for EnabledState values.
var (
	iTrue  = true
	iFalse = false

	Enabled  EnabledState = &iTrue
	Disabled EnabledState = &iFalse
)

// IntToPointer is a function for converting integers into integer pointers.
// This is useful when passing in options to operations.
func IntToPointer(i int) *int {
	return &i
}

/*
MaybeString is an internal function to be used by request methods in individual
resource packages.

It takes a string that might be a zero value and returns either a pointer to its
address or nil. This is useful for allowing users to conveniently omit values
from an options struct by leaving them zeroed, but still pass nil to the JSON
serializer so they'll be omitted from the request body.
*/
func MaybeString(original string) *string {
	if original != "" {
		return &original
	}
	return nil
}

/*
MaybeInt is an internal function to be used by request methods in individual
resource packages.

Like MaybeString, it accepts an int that may or may not be a zero value, and
returns either a pointer to its address or nil. It's intended to hint that the
JSON serializer should omit its field.
*/
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
BuildQueryString is an internal function to be used by request methods in
individual resource packages.

It accepts a tagged structure and expands it into a URL struct. Field names are
converted into query parameters based on a "q" tag. For example:

	type struct Something {
	   Bar string `q:"x_bar"`
	   Baz int    `q:"lorem_ipsum"`
	}

	instance := Something{
	   Bar: "AAA",
	   Baz: "BBB",
	}

will be converted into "?x_bar=AAA&lorem_ipsum=BBB".

The struct's fields may be strings, integers, or boolean values. Fields left at
their type's zero value will be omitted from the query.
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

	params := url.Values{}

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
						params.Add(tags[0], v.String())
					case reflect.Int:
						params.Add(tags[0], strconv.FormatInt(v.Int(), 10))
					case reflect.Bool:
						params.Add(tags[0], strconv.FormatBool(v.Bool()))
					case reflect.Slice:
						switch v.Type().Elem() {
						case reflect.TypeOf(0):
							for i := 0; i < v.Len(); i++ {
								params.Add(tags[0], strconv.FormatInt(v.Index(i).Int(), 10))
							}
						default:
							for i := 0; i < v.Len(); i++ {
								params.Add(tags[0], v.Index(i).String())
							}
						}
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

		return &url.URL{RawQuery: params.Encode()}, nil
	}
	// Return an error if the underlying type of 'opts' isn't a struct.
	return nil, fmt.Errorf("Options type is not a struct.")
}

/*
BuildHeaders is an internal function to be used by request methods in
individual resource packages.

It accepts an arbitrary tagged structure and produces a string map that's
suitable for use as the HTTP headers of an outgoing request. Field names are
mapped to header names based in "h" tags.

  type struct Something {
    Bar string `h:"x_bar"`
    Baz int    `h:"lorem_ipsum"`
  }

  instance := Something{
    Bar: "AAA",
    Baz: "BBB",
  }

will be converted into:

  map[string]string{
    "x_bar": "AAA",
    "lorem_ipsum": "BBB",
  }

Untagged fields and fields left at their zero values are skipped. Integers,
booleans and string values are supported.
*/
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

// IDSliceToQueryString takes a slice of elements and converts them into a query
// string. For example, if name=foo and slice=[]int{20, 40, 60}, then the
// result would be `?name=20&name=40&name=60'
func IDSliceToQueryString(name string, ids []int) string {
	str := ""
	for k, v := range ids {
		if k == 0 {
			str += "?"
		} else {
			str += "&"
		}
		str += fmt.Sprintf("%s=%s", name, strconv.Itoa(v))
	}
	return str
}

// IntWithinRange returns TRUE if an integer falls within a defined range, and
// FALSE if not.
func IntWithinRange(val, min, max int) bool {
	return val > min && val < max
}
