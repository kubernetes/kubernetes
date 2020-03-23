package objx

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/url"
	"strings"
)

// MSIConvertable is an interface that defines methods for converting your
// custom types to a map[string]interface{} representation.
type MSIConvertable interface {
	// MSI gets a map[string]interface{} (msi) representing the
	// object.
	MSI() map[string]interface{}
}

// Map provides extended functionality for working with
// untyped data, in particular map[string]interface (msi).
type Map map[string]interface{}

// Value returns the internal value instance
func (m Map) Value() *Value {
	return &Value{data: m}
}

// Nil represents a nil Map.
var Nil = New(nil)

// New creates a new Map containing the map[string]interface{} in the data argument.
// If the data argument is not a map[string]interface, New attempts to call the
// MSI() method on the MSIConvertable interface to create one.
func New(data interface{}) Map {
	if _, ok := data.(map[string]interface{}); !ok {
		if converter, ok := data.(MSIConvertable); ok {
			data = converter.MSI()
		} else {
			return nil
		}
	}
	return Map(data.(map[string]interface{}))
}

// MSI creates a map[string]interface{} and puts it inside a new Map.
//
// The arguments follow a key, value pattern.
//
//
// Returns nil if any key argument is non-string or if there are an odd number of arguments.
//
// Example
//
// To easily create Maps:
//
//     m := objx.MSI("name", "Mat", "age", 29, "subobj", objx.MSI("active", true))
//
//     // creates an Map equivalent to
//     m := objx.Map{"name": "Mat", "age": 29, "subobj": objx.Map{"active": true}}
func MSI(keyAndValuePairs ...interface{}) Map {
	newMap := Map{}
	keyAndValuePairsLen := len(keyAndValuePairs)
	if keyAndValuePairsLen%2 != 0 {
		return nil
	}
	for i := 0; i < keyAndValuePairsLen; i = i + 2 {
		key := keyAndValuePairs[i]
		value := keyAndValuePairs[i+1]

		// make sure the key is a string
		keyString, keyStringOK := key.(string)
		if !keyStringOK {
			return nil
		}
		newMap[keyString] = value
	}
	return newMap
}

// ****** Conversion Constructors

// MustFromJSON creates a new Map containing the data specified in the
// jsonString.
//
// Panics if the JSON is invalid.
func MustFromJSON(jsonString string) Map {
	o, err := FromJSON(jsonString)
	if err != nil {
		panic("objx: MustFromJSON failed with error: " + err.Error())
	}
	return o
}

// FromJSON creates a new Map containing the data specified in the
// jsonString.
//
// Returns an error if the JSON is invalid.
func FromJSON(jsonString string) (Map, error) {
	var m Map
	err := json.Unmarshal([]byte(jsonString), &m)
	if err != nil {
		return Nil, err
	}
	m.tryConvertFloat64()
	return m, nil
}

func (m Map) tryConvertFloat64() {
	for k, v := range m {
		switch v.(type) {
		case float64:
			f := v.(float64)
			if float64(int(f)) == f {
				m[k] = int(f)
			}
		case map[string]interface{}:
			t := New(v)
			t.tryConvertFloat64()
			m[k] = t
		case []interface{}:
			m[k] = tryConvertFloat64InSlice(v.([]interface{}))
		}
	}
}

func tryConvertFloat64InSlice(s []interface{}) []interface{} {
	for k, v := range s {
		switch v.(type) {
		case float64:
			f := v.(float64)
			if float64(int(f)) == f {
				s[k] = int(f)
			}
		case map[string]interface{}:
			t := New(v)
			t.tryConvertFloat64()
			s[k] = t
		case []interface{}:
			s[k] = tryConvertFloat64InSlice(v.([]interface{}))
		}
	}
	return s
}

// FromBase64 creates a new Obj containing the data specified
// in the Base64 string.
//
// The string is an encoded JSON string returned by Base64
func FromBase64(base64String string) (Map, error) {
	decoder := base64.NewDecoder(base64.StdEncoding, strings.NewReader(base64String))
	decoded, err := ioutil.ReadAll(decoder)
	if err != nil {
		return nil, err
	}
	return FromJSON(string(decoded))
}

// MustFromBase64 creates a new Obj containing the data specified
// in the Base64 string and panics if there is an error.
//
// The string is an encoded JSON string returned by Base64
func MustFromBase64(base64String string) Map {
	result, err := FromBase64(base64String)
	if err != nil {
		panic("objx: MustFromBase64 failed with error: " + err.Error())
	}
	return result
}

// FromSignedBase64 creates a new Obj containing the data specified
// in the Base64 string.
//
// The string is an encoded JSON string returned by SignedBase64
func FromSignedBase64(base64String, key string) (Map, error) {
	parts := strings.Split(base64String, SignatureSeparator)
	if len(parts) != 2 {
		return nil, errors.New("objx: Signed base64 string is malformed")
	}

	sig := HashWithKey(parts[0], key)
	if parts[1] != sig {
		return nil, errors.New("objx: Signature for base64 data does not match")
	}
	return FromBase64(parts[0])
}

// MustFromSignedBase64 creates a new Obj containing the data specified
// in the Base64 string and panics if there is an error.
//
// The string is an encoded JSON string returned by Base64
func MustFromSignedBase64(base64String, key string) Map {
	result, err := FromSignedBase64(base64String, key)
	if err != nil {
		panic("objx: MustFromSignedBase64 failed with error: " + err.Error())
	}
	return result
}

// FromURLQuery generates a new Obj by parsing the specified
// query.
//
// For queries with multiple values, the first value is selected.
func FromURLQuery(query string) (Map, error) {
	vals, err := url.ParseQuery(query)
	if err != nil {
		return nil, err
	}
	m := Map{}
	for k, vals := range vals {
		m[k] = vals[0]
	}
	return m, nil
}

// MustFromURLQuery generates a new Obj by parsing the specified
// query.
//
// For queries with multiple values, the first value is selected.
//
// Panics if it encounters an error
func MustFromURLQuery(query string) Map {
	o, err := FromURLQuery(query)
	if err != nil {
		panic("objx: MustFromURLQuery failed with error: " + err.Error())
	}
	return o
}
