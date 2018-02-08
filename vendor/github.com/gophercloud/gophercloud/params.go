package gophercloud

import (
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

/*
BuildRequestBody builds a map[string]interface from the given `struct`. If
parent is not an empty string, the final map[string]interface returned will
encapsulate the built one. For example:

  disk := 1
  createOpts := flavors.CreateOpts{
    ID:         "1",
    Name:       "m1.tiny",
    Disk:       &disk,
    RAM:        512,
    VCPUs:      1,
    RxTxFactor: 1.0,
  }

  body, err := gophercloud.BuildRequestBody(createOpts, "flavor")

The above example can be run as-is, however it is recommended to look at how
BuildRequestBody is used within Gophercloud to more fully understand how it
fits within the request process as a whole rather than use it directly as shown
above.
*/
func BuildRequestBody(opts interface{}, parent string) (map[string]interface{}, error) {
	optsValue := reflect.ValueOf(opts)
	if optsValue.Kind() == reflect.Ptr {
		optsValue = optsValue.Elem()
	}

	optsType := reflect.TypeOf(opts)
	if optsType.Kind() == reflect.Ptr {
		optsType = optsType.Elem()
	}

	optsMap := make(map[string]interface{})
	if optsValue.Kind() == reflect.Struct {
		//fmt.Printf("optsValue.Kind() is a reflect.Struct: %+v\n", optsValue.Kind())
		for i := 0; i < optsValue.NumField(); i++ {
			v := optsValue.Field(i)
			f := optsType.Field(i)

			if f.Name != strings.Title(f.Name) {
				//fmt.Printf("Skipping field: %s...\n", f.Name)
				continue
			}

			//fmt.Printf("Starting on field: %s...\n", f.Name)

			zero := isZero(v)
			//fmt.Printf("v is zero?: %v\n", zero)

			// if the field has a required tag that's set to "true"
			if requiredTag := f.Tag.Get("required"); requiredTag == "true" {
				//fmt.Printf("Checking required field [%s]:\n\tv: %+v\n\tisZero:%v\n", f.Name, v.Interface(), zero)
				// if the field's value is zero, return a missing-argument error
				if zero {
					// if the field has a 'required' tag, it can't have a zero-value
					err := ErrMissingInput{}
					err.Argument = f.Name
					return nil, err
				}
			}

			if xorTag := f.Tag.Get("xor"); xorTag != "" {
				//fmt.Printf("Checking `xor` tag for field [%s] with value %+v:\n\txorTag: %s\n", f.Name, v, xorTag)
				xorField := optsValue.FieldByName(xorTag)
				var xorFieldIsZero bool
				if reflect.ValueOf(xorField.Interface()) == reflect.Zero(xorField.Type()) {
					xorFieldIsZero = true
				} else {
					if xorField.Kind() == reflect.Ptr {
						xorField = xorField.Elem()
					}
					xorFieldIsZero = isZero(xorField)
				}
				if !(zero != xorFieldIsZero) {
					err := ErrMissingInput{}
					err.Argument = fmt.Sprintf("%s/%s", f.Name, xorTag)
					err.Info = fmt.Sprintf("Exactly one of %s and %s must be provided", f.Name, xorTag)
					return nil, err
				}
			}

			if orTag := f.Tag.Get("or"); orTag != "" {
				//fmt.Printf("Checking `or` tag for field with:\n\tname: %+v\n\torTag:%s\n", f.Name, orTag)
				//fmt.Printf("field is zero?: %v\n", zero)
				if zero {
					orField := optsValue.FieldByName(orTag)
					var orFieldIsZero bool
					if reflect.ValueOf(orField.Interface()) == reflect.Zero(orField.Type()) {
						orFieldIsZero = true
					} else {
						if orField.Kind() == reflect.Ptr {
							orField = orField.Elem()
						}
						orFieldIsZero = isZero(orField)
					}
					if orFieldIsZero {
						err := ErrMissingInput{}
						err.Argument = fmt.Sprintf("%s/%s", f.Name, orTag)
						err.Info = fmt.Sprintf("At least one of %s and %s must be provided", f.Name, orTag)
						return nil, err
					}
				}
			}

			if v.Kind() == reflect.Struct || (v.Kind() == reflect.Ptr && v.Elem().Kind() == reflect.Struct) {
				if zero {
					//fmt.Printf("value before change: %+v\n", optsValue.Field(i))
					if jsonTag := f.Tag.Get("json"); jsonTag != "" {
						jsonTagPieces := strings.Split(jsonTag, ",")
						if len(jsonTagPieces) > 1 && jsonTagPieces[1] == "omitempty" {
							if v.CanSet() {
								if !v.IsNil() {
									if v.Kind() == reflect.Ptr {
										v.Set(reflect.Zero(v.Type()))
									}
								}
								//fmt.Printf("value after change: %+v\n", optsValue.Field(i))
							}
						}
					}
					continue
				}

				//fmt.Printf("Calling BuildRequestBody with:\n\tv: %+v\n\tf.Name:%s\n", v.Interface(), f.Name)
				_, err := BuildRequestBody(v.Interface(), f.Name)
				if err != nil {
					return nil, err
				}
			}
		}

		//fmt.Printf("opts: %+v \n", opts)

		b, err := json.Marshal(opts)
		if err != nil {
			return nil, err
		}

		//fmt.Printf("string(b): %s\n", string(b))

		err = json.Unmarshal(b, &optsMap)
		if err != nil {
			return nil, err
		}

		//fmt.Printf("optsMap: %+v\n", optsMap)

		if parent != "" {
			optsMap = map[string]interface{}{parent: optsMap}
		}
		//fmt.Printf("optsMap after parent added: %+v\n", optsMap)
		return optsMap, nil
	}
	// Return an error if the underlying type of 'opts' isn't a struct.
	return nil, fmt.Errorf("Options type is not a struct.")
}

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

// IPVersion is a type for the possible IP address versions. Valid instances
// are IPv4 and IPv6
type IPVersion int

const (
	// IPv4 is used for IP version 4 addresses
	IPv4 IPVersion = 4
	// IPv6 is used for IP version 6 addresses
	IPv6 IPVersion = 6
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

/*
func isUnderlyingStructZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Ptr:
		return isUnderlyingStructZero(v.Elem())
	default:
		return isZero(v)
	}
}
*/

var t time.Time

func isZero(v reflect.Value) bool {
	//fmt.Printf("\n\nchecking isZero for value: %+v\n", v)
	switch v.Kind() {
	case reflect.Ptr:
		if v.IsNil() {
			return true
		}
		return false
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
	//fmt.Printf("zero type for value: %+v\n\n\n", z)
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
				loop:
					switch v.Kind() {
					case reflect.Ptr:
						v = v.Elem()
						goto loop
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
					case reflect.Map:
						if v.Type().Key().Kind() == reflect.String && v.Type().Elem().Kind() == reflect.String {
							var s []string
							for _, k := range v.MapKeys() {
								value := v.MapIndex(k).String()
								s = append(s, fmt.Sprintf("'%s':'%s'", k.String(), value))
							}
							params.Add(tags[0], fmt.Sprintf("{%s}", strings.Join(s, ", ")))
						}
					}
				} else {
					// Otherwise, the field is not set.
					if len(tags) == 2 && tags[1] == "required" {
						// And the field is required. Return an error.
						return &url.URL{}, fmt.Errorf("Required query parameter [%s] not set.", f.Name)
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
