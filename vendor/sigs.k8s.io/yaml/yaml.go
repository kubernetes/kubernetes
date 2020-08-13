package yaml

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strconv"

	"gopkg.in/yaml.v2"
)

// Marshal marshals the object into JSON then converts JSON to YAML and returns the
// YAML.
func Marshal(o interface{}) ([]byte, error) {
	j, err := json.Marshal(o)
	if err != nil {
		return nil, fmt.Errorf("error marshaling into JSON: %v", err)
	}

	y, err := JSONToYAML(j)
	if err != nil {
		return nil, fmt.Errorf("error converting JSON to YAML: %v", err)
	}

	return y, nil
}

// JSONOpt is a decoding option for decoding from JSON format.
type JSONOpt func(*json.Decoder) *json.Decoder

// Unmarshal converts YAML to JSON then uses JSON to unmarshal into an object,
// optionally configuring the behavior of the JSON unmarshal.
func Unmarshal(y []byte, o interface{}, opts ...JSONOpt) error {
	return yamlUnmarshal(y, o, false, opts...)
}

// UnmarshalStrict strictly converts YAML to JSON then uses JSON to unmarshal
// into an object, optionally configuring the behavior of the JSON unmarshal.
func UnmarshalStrict(y []byte, o interface{}, opts ...JSONOpt) error {
	return yamlUnmarshal(y, o, true, append(opts, DisallowUnknownFields)...)
}

// yamlUnmarshal unmarshals the given YAML byte stream into the given interface,
// optionally performing the unmarshalling strictly
func yamlUnmarshal(y []byte, o interface{}, strict bool, opts ...JSONOpt) error {
	vo := reflect.ValueOf(o)
	unmarshalFn := yaml.Unmarshal
	if strict {
		unmarshalFn = yaml.UnmarshalStrict
	}
	j, err := yamlToJSON(y, &vo, unmarshalFn)
	if err != nil {
		return fmt.Errorf("error converting YAML to JSON: %v", err)
	}

	err = jsonUnmarshal(bytes.NewReader(j), o, opts...)
	if err != nil {
		return fmt.Errorf("error unmarshaling JSON: %v", err)
	}

	return nil
}

// jsonUnmarshal unmarshals the JSON byte stream from the given reader into the
// object, optionally applying decoder options prior to decoding.  We are not
// using json.Unmarshal directly as we want the chance to pass in non-default
// options.
func jsonUnmarshal(r io.Reader, o interface{}, opts ...JSONOpt) error {
	d := json.NewDecoder(r)
	for _, opt := range opts {
		d = opt(d)
	}
	if err := d.Decode(&o); err != nil {
		return fmt.Errorf("while decoding JSON: %v", err)
	}
	return nil
}

// JSONToYAML Converts JSON to YAML.
func JSONToYAML(j []byte) ([]byte, error) {
	// Convert the JSON to an object.
	var jsonObj interface{}
	// We are using yaml.Unmarshal here (instead of json.Unmarshal) because the
	// Go JSON library doesn't try to pick the right number type (int, float,
	// etc.) when unmarshalling to interface{}, it just picks float64
	// universally. go-yaml does go through the effort of picking the right
	// number type, so we can preserve number type throughout this process.
	err := yaml.Unmarshal(j, &jsonObj)
	if err != nil {
		return nil, err
	}

	// Marshal this object into YAML.
	return yaml.Marshal(jsonObj)
}

// YAMLToJSON converts YAML to JSON. Since JSON is a subset of YAML,
// passing JSON through this method should be a no-op.
//
// Things YAML can do that are not supported by JSON:
// * In YAML you can have binary and null keys in your maps. These are invalid
//   in JSON. (int and float keys are converted to strings.)
// * Binary data in YAML with the !!binary tag is not supported. If you want to
//   use binary data with this library, encode the data as base64 as usual but do
//   not use the !!binary tag in your YAML. This will ensure the original base64
//   encoded data makes it all the way through to the JSON.
//
// For strict decoding of YAML, use YAMLToJSONStrict.
func YAMLToJSON(y []byte) ([]byte, error) {
	return yamlToJSON(y, nil, yaml.Unmarshal)
}

// YAMLToJSONStrict is like YAMLToJSON but enables strict YAML decoding,
// returning an error on any duplicate field names.
func YAMLToJSONStrict(y []byte) ([]byte, error) {
	return yamlToJSON(y, nil, yaml.UnmarshalStrict)
}

func yamlToJSON(y []byte, jsonTarget *reflect.Value, yamlUnmarshal func([]byte, interface{}) error) ([]byte, error) {
	// Convert the YAML to an object.
	var yamlObj interface{}
	err := yamlUnmarshal(y, &yamlObj)
	if err != nil {
		return nil, err
	}

	// YAML objects are not completely compatible with JSON objects (e.g. you
	// can have non-string keys in YAML). So, convert the YAML-compatible object
	// to a JSON-compatible object, failing with an error if irrecoverable
	// incompatibilties happen along the way.
	jsonObj, err := convertToJSONableObject(yamlObj, jsonTarget)
	if err != nil {
		return nil, err
	}

	// Convert this object to JSON and return the data.
	return json.Marshal(jsonObj)
}

func convertToJSONableObject(yamlObj interface{}, jsonTarget *reflect.Value) (interface{}, error) {
	var err error

	// Resolve jsonTarget to a concrete value (i.e. not a pointer or an
	// interface). We pass decodingNull as false because we're not actually
	// decoding into the value, we're just checking if the ultimate target is a
	// string.
	if jsonTarget != nil {
		ju, tu, pv := indirect(*jsonTarget, false)
		// We have a JSON or Text Umarshaler at this level, so we can't be trying
		// to decode into a string.
		if ju != nil || tu != nil {
			jsonTarget = nil
		} else {
			jsonTarget = &pv
		}
	}

	// If yamlObj is a number or a boolean, check if jsonTarget is a string -
	// if so, coerce.  Else return normal.
	// If yamlObj is a map or array, find the field that each key is
	// unmarshaling to, and when you recurse pass the reflect.Value for that
	// field back into this function.
	switch typedYAMLObj := yamlObj.(type) {
	case map[interface{}]interface{}:
		// JSON does not support arbitrary keys in a map, so we must convert
		// these keys to strings.
		//
		// From my reading of go-yaml v2 (specifically the resolve function),
		// keys can only have the types string, int, int64, float64, binary
		// (unsupported), or null (unsupported).
		strMap := make(map[string]interface{})
		for k, v := range typedYAMLObj {
			// Resolve the key to a string first.
			var keyString string
			switch typedKey := k.(type) {
			case string:
				keyString = typedKey
			case int:
				keyString = strconv.Itoa(typedKey)
			case int64:
				// go-yaml will only return an int64 as a key if the system
				// architecture is 32-bit and the key's value is between 32-bit
				// and 64-bit. Otherwise the key type will simply be int.
				keyString = strconv.FormatInt(typedKey, 10)
			case float64:
				// Stolen from go-yaml to use the same conversion to string as
				// the go-yaml library uses to convert float to string when
				// Marshaling.
				s := strconv.FormatFloat(typedKey, 'g', -1, 32)
				switch s {
				case "+Inf":
					s = ".inf"
				case "-Inf":
					s = "-.inf"
				case "NaN":
					s = ".nan"
				}
				keyString = s
			case bool:
				if typedKey {
					keyString = "true"
				} else {
					keyString = "false"
				}
			default:
				return nil, fmt.Errorf("Unsupported map key of type: %s, key: %+#v, value: %+#v",
					reflect.TypeOf(k), k, v)
			}

			// jsonTarget should be a struct or a map. If it's a struct, find
			// the field it's going to map to and pass its reflect.Value. If
			// it's a map, find the element type of the map and pass the
			// reflect.Value created from that type. If it's neither, just pass
			// nil - JSON conversion will error for us if it's a real issue.
			if jsonTarget != nil {
				t := *jsonTarget
				if t.Kind() == reflect.Struct {
					keyBytes := []byte(keyString)
					// Find the field that the JSON library would use.
					var f *field
					fields := cachedTypeFields(t.Type())
					for i := range fields {
						ff := &fields[i]
						if bytes.Equal(ff.nameBytes, keyBytes) {
							f = ff
							break
						}
						// Do case-insensitive comparison.
						if f == nil && ff.equalFold(ff.nameBytes, keyBytes) {
							f = ff
						}
					}
					if f != nil {
						// Find the reflect.Value of the most preferential
						// struct field.
						jtf := t.Field(f.index[0])
						strMap[keyString], err = convertToJSONableObject(v, &jtf)
						if err != nil {
							return nil, err
						}
						continue
					}
				} else if t.Kind() == reflect.Map {
					// Create a zero value of the map's element type to use as
					// the JSON target.
					jtv := reflect.Zero(t.Type().Elem())
					strMap[keyString], err = convertToJSONableObject(v, &jtv)
					if err != nil {
						return nil, err
					}
					continue
				}
			}
			strMap[keyString], err = convertToJSONableObject(v, nil)
			if err != nil {
				return nil, err
			}
		}
		return strMap, nil
	case []interface{}:
		// We need to recurse into arrays in case there are any
		// map[interface{}]interface{}'s inside and to convert any
		// numbers to strings.

		// If jsonTarget is a slice (which it really should be), find the
		// thing it's going to map to. If it's not a slice, just pass nil
		// - JSON conversion will error for us if it's a real issue.
		var jsonSliceElemValue *reflect.Value
		if jsonTarget != nil {
			t := *jsonTarget
			if t.Kind() == reflect.Slice {
				// By default slices point to nil, but we need a reflect.Value
				// pointing to a value of the slice type, so we create one here.
				ev := reflect.Indirect(reflect.New(t.Type().Elem()))
				jsonSliceElemValue = &ev
			}
		}

		// Make and use a new array.
		arr := make([]interface{}, len(typedYAMLObj))
		for i, v := range typedYAMLObj {
			arr[i], err = convertToJSONableObject(v, jsonSliceElemValue)
			if err != nil {
				return nil, err
			}
		}
		return arr, nil
	default:
		// If the target type is a string and the YAML type is a number,
		// convert the YAML type to a string.
		if jsonTarget != nil && (*jsonTarget).Kind() == reflect.String {
			// Based on my reading of go-yaml, it may return int, int64,
			// float64, or uint64.
			var s string
			switch typedVal := typedYAMLObj.(type) {
			case int:
				s = strconv.FormatInt(int64(typedVal), 10)
			case int64:
				s = strconv.FormatInt(typedVal, 10)
			case float64:
				s = strconv.FormatFloat(typedVal, 'g', -1, 32)
			case uint64:
				s = strconv.FormatUint(typedVal, 10)
			case bool:
				if typedVal {
					s = "true"
				} else {
					s = "false"
				}
			}
			if len(s) > 0 {
				yamlObj = interface{}(s)
			}
		}
		return yamlObj, nil
	}
}

// JSONObjectToYAMLObject converts an in-memory JSON object into a YAML in-memory MapSlice,
// without going through a byte representation. A nil or empty map[string]interface{} input is
// converted to an empty map, i.e. yaml.MapSlice(nil).
//
// interface{} slices stay interface{} slices. map[string]interface{} becomes yaml.MapSlice.
//
// int64 and float64 are down casted following the logic of github.com/go-yaml/yaml:
// - float64s are down-casted as far as possible without data-loss to int, int64, uint64.
// - int64s are down-casted to int if possible without data-loss.
//
// Big int/int64/uint64 do not lose precision as in the json-yaml roundtripping case.
//
// string, bool and any other types are unchanged.
func JSONObjectToYAMLObject(j map[string]interface{}) yaml.MapSlice {
	if len(j) == 0 {
		return nil
	}
	ret := make(yaml.MapSlice, 0, len(j))
	for k, v := range j {
		ret = append(ret, yaml.MapItem{Key: k, Value: jsonToYAMLValue(v)})
	}
	return ret
}

func jsonToYAMLValue(j interface{}) interface{} {
	switch j := j.(type) {
	case map[string]interface{}:
		if j == nil {
			return interface{}(nil)
		}
		return JSONObjectToYAMLObject(j)
	case []interface{}:
		if j == nil {
			return interface{}(nil)
		}
		ret := make([]interface{}, len(j))
		for i := range j {
			ret[i] = jsonToYAMLValue(j[i])
		}
		return ret
	case float64:
		// replicate the logic in https://github.com/go-yaml/yaml/blob/51d6538a90f86fe93ac480b35f37b2be17fef232/resolve.go#L151
		if i64 := int64(j); j == float64(i64) {
			if i := int(i64); i64 == int64(i) {
				return i
			}
			return i64
		}
		if ui64 := uint64(j); j == float64(ui64) {
			return ui64
		}
		return j
	case int64:
		if i := int(j); j == int64(i) {
			return i
		}
		return j
	}
	return j
}
