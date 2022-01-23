// Package mapstructure exposes functionality to convert one arbitrary
// Go type into another, typically to convert a map[string]interface{}
// into a native Go structure.
//
// The Go structure can be arbitrarily complex, containing slices,
// other structs, etc. and the decoder will properly decode nested
// maps and so on into the proper structures in the native Go struct.
// See the examples to see what the decoder is capable of.
//
// The simplest function to start with is Decode.
//
// Field Tags
//
// When decoding to a struct, mapstructure will use the field name by
// default to perform the mapping. For example, if a struct has a field
// "Username" then mapstructure will look for a key in the source value
// of "username" (case insensitive).
//
//     type User struct {
//         Username string
//     }
//
// You can change the behavior of mapstructure by using struct tags.
// The default struct tag that mapstructure looks for is "mapstructure"
// but you can customize it using DecoderConfig.
//
// Renaming Fields
//
// To rename the key that mapstructure looks for, use the "mapstructure"
// tag and set a value directly. For example, to change the "username" example
// above to "user":
//
//     type User struct {
//         Username string `mapstructure:"user"`
//     }
//
// Embedded Structs and Squashing
//
// Embedded structs are treated as if they're another field with that name.
// By default, the two structs below are equivalent when decoding with
// mapstructure:
//
//     type Person struct {
//         Name string
//     }
//
//     type Friend struct {
//         Person
//     }
//
//     type Friend struct {
//         Person Person
//     }
//
// This would require an input that looks like below:
//
//     map[string]interface{}{
//         "person": map[string]interface{}{"name": "alice"},
//     }
//
// If your "person" value is NOT nested, then you can append ",squash" to
// your tag value and mapstructure will treat it as if the embedded struct
// were part of the struct directly. Example:
//
//     type Friend struct {
//         Person `mapstructure:",squash"`
//     }
//
// Now the following input would be accepted:
//
//     map[string]interface{}{
//         "name": "alice",
//     }
//
// When decoding from a struct to a map, the squash tag squashes the struct
// fields into a single map. Using the example structs from above:
//
//     Friend{Person: Person{Name: "alice"}}
//
// Will be decoded into a map:
//
//     map[string]interface{}{
//         "name": "alice",
//     }
//
// DecoderConfig has a field that changes the behavior of mapstructure
// to always squash embedded structs.
//
// Remainder Values
//
// If there are any unmapped keys in the source value, mapstructure by
// default will silently ignore them. You can error by setting ErrorUnused
// in DecoderConfig. If you're using Metadata you can also maintain a slice
// of the unused keys.
//
// You can also use the ",remain" suffix on your tag to collect all unused
// values in a map. The field with this tag MUST be a map type and should
// probably be a "map[string]interface{}" or "map[interface{}]interface{}".
// See example below:
//
//     type Friend struct {
//         Name  string
//         Other map[string]interface{} `mapstructure:",remain"`
//     }
//
// Given the input below, Other would be populated with the other
// values that weren't used (everything but "name"):
//
//     map[string]interface{}{
//         "name":    "bob",
//         "address": "123 Maple St.",
//     }
//
// Omit Empty Values
//
// When decoding from a struct to any other value, you may use the
// ",omitempty" suffix on your tag to omit that value if it equates to
// the zero value. The zero value of all types is specified in the Go
// specification.
//
// For example, the zero type of a numeric type is zero ("0"). If the struct
// field value is zero and a numeric type, the field is empty, and it won't
// be encoded into the destination type.
//
//     type Source {
//         Age int `mapstructure:",omitempty"`
//     }
//
// Unexported fields
//
// Since unexported (private) struct fields cannot be set outside the package
// where they are defined, the decoder will simply skip them.
//
// For this output type definition:
//
//     type Exported struct {
//         private string // this unexported field will be skipped
//         Public string
//     }
//
// Using this map as input:
//
//     map[string]interface{}{
//         "private": "I will be ignored",
//         "Public":  "I made it through!",
//     }
//
// The following struct will be decoded:
//
//     type Exported struct {
//         private: "" // field is left with an empty string (zero value)
//         Public: "I made it through!"
//     }
//
// Other Configuration
//
// mapstructure is highly configurable. See the DecoderConfig struct
// for other features and options that are supported.
package mapstructure

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// DecodeHookFunc is the callback function that can be used for
// data transformations. See "DecodeHook" in the DecoderConfig
// struct.
//
// The type must be one of DecodeHookFuncType, DecodeHookFuncKind, or
// DecodeHookFuncValue.
// Values are a superset of Types (Values can return types), and Types are a
// superset of Kinds (Types can return Kinds) and are generally a richer thing
// to use, but Kinds are simpler if you only need those.
//
// The reason DecodeHookFunc is multi-typed is for backwards compatibility:
// we started with Kinds and then realized Types were the better solution,
// but have a promise to not break backwards compat so we now support
// both.
type DecodeHookFunc interface{}

// DecodeHookFuncType is a DecodeHookFunc which has complete information about
// the source and target types.
type DecodeHookFuncType func(reflect.Type, reflect.Type, interface{}) (interface{}, error)

// DecodeHookFuncKind is a DecodeHookFunc which knows only the Kinds of the
// source and target types.
type DecodeHookFuncKind func(reflect.Kind, reflect.Kind, interface{}) (interface{}, error)

// DecodeHookFuncValue is a DecodeHookFunc which has complete access to both the source and target
// values.
type DecodeHookFuncValue func(from reflect.Value, to reflect.Value) (interface{}, error)

// DecoderConfig is the configuration that is used to create a new decoder
// and allows customization of various aspects of decoding.
type DecoderConfig struct {
	// DecodeHook, if set, will be called before any decoding and any
	// type conversion (if WeaklyTypedInput is on). This lets you modify
	// the values before they're set down onto the resulting struct. The
	// DecodeHook is called for every map and value in the input. This means
	// that if a struct has embedded fields with squash tags the decode hook
	// is called only once with all of the input data, not once for each
	// embedded struct.
	//
	// If an error is returned, the entire decode will fail with that error.
	DecodeHook DecodeHookFunc

	// If ErrorUnused is true, then it is an error for there to exist
	// keys in the original map that were unused in the decoding process
	// (extra keys).
	ErrorUnused bool

	// ZeroFields, if set to true, will zero fields before writing them.
	// For example, a map will be emptied before decoded values are put in
	// it. If this is false, a map will be merged.
	ZeroFields bool

	// If WeaklyTypedInput is true, the decoder will make the following
	// "weak" conversions:
	//
	//   - bools to string (true = "1", false = "0")
	//   - numbers to string (base 10)
	//   - bools to int/uint (true = 1, false = 0)
	//   - strings to int/uint (base implied by prefix)
	//   - int to bool (true if value != 0)
	//   - string to bool (accepts: 1, t, T, TRUE, true, True, 0, f, F,
	//     FALSE, false, False. Anything else is an error)
	//   - empty array = empty map and vice versa
	//   - negative numbers to overflowed uint values (base 10)
	//   - slice of maps to a merged map
	//   - single values are converted to slices if required. Each
	//     element is weakly decoded. For example: "4" can become []int{4}
	//     if the target type is an int slice.
	//
	WeaklyTypedInput bool

	// Squash will squash embedded structs.  A squash tag may also be
	// added to an individual struct field using a tag.  For example:
	//
	//  type Parent struct {
	//      Child `mapstructure:",squash"`
	//  }
	Squash bool

	// Metadata is the struct that will contain extra metadata about
	// the decoding. If this is nil, then no metadata will be tracked.
	Metadata *Metadata

	// Result is a pointer to the struct that will contain the decoded
	// value.
	Result interface{}

	// The tag name that mapstructure reads for field names. This
	// defaults to "mapstructure"
	TagName string

	// MatchName is the function used to match the map key to the struct
	// field name or tag. Defaults to `strings.EqualFold`. This can be used
	// to implement case-sensitive tag values, support snake casing, etc.
	MatchName func(mapKey, fieldName string) bool
}

// A Decoder takes a raw interface value and turns it into structured
// data, keeping track of rich error information along the way in case
// anything goes wrong. Unlike the basic top-level Decode method, you can
// more finely control how the Decoder behaves using the DecoderConfig
// structure. The top-level Decode method is just a convenience that sets
// up the most basic Decoder.
type Decoder struct {
	config *DecoderConfig
}

// Metadata contains information about decoding a structure that
// is tedious or difficult to get otherwise.
type Metadata struct {
	// Keys are the keys of the structure which were successfully decoded
	Keys []string

	// Unused is a slice of keys that were found in the raw value but
	// weren't decoded since there was no matching field in the result interface
	Unused []string
}

// Decode takes an input structure and uses reflection to translate it to
// the output structure. output must be a pointer to a map or struct.
func Decode(input interface{}, output interface{}) error {
	config := &DecoderConfig{
		Metadata: nil,
		Result:   output,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		return err
	}

	return decoder.Decode(input)
}

// WeakDecode is the same as Decode but is shorthand to enable
// WeaklyTypedInput. See DecoderConfig for more info.
func WeakDecode(input, output interface{}) error {
	config := &DecoderConfig{
		Metadata:         nil,
		Result:           output,
		WeaklyTypedInput: true,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		return err
	}

	return decoder.Decode(input)
}

// DecodeMetadata is the same as Decode, but is shorthand to
// enable metadata collection. See DecoderConfig for more info.
func DecodeMetadata(input interface{}, output interface{}, metadata *Metadata) error {
	config := &DecoderConfig{
		Metadata: metadata,
		Result:   output,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		return err
	}

	return decoder.Decode(input)
}

// WeakDecodeMetadata is the same as Decode, but is shorthand to
// enable both WeaklyTypedInput and metadata collection. See
// DecoderConfig for more info.
func WeakDecodeMetadata(input interface{}, output interface{}, metadata *Metadata) error {
	config := &DecoderConfig{
		Metadata:         metadata,
		Result:           output,
		WeaklyTypedInput: true,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		return err
	}

	return decoder.Decode(input)
}

// NewDecoder returns a new decoder for the given configuration. Once
// a decoder has been returned, the same configuration must not be used
// again.
func NewDecoder(config *DecoderConfig) (*Decoder, error) {
	val := reflect.ValueOf(config.Result)
	if val.Kind() != reflect.Ptr {
		return nil, errors.New("result must be a pointer")
	}

	val = val.Elem()
	if !val.CanAddr() {
		return nil, errors.New("result must be addressable (a pointer)")
	}

	if config.Metadata != nil {
		if config.Metadata.Keys == nil {
			config.Metadata.Keys = make([]string, 0)
		}

		if config.Metadata.Unused == nil {
			config.Metadata.Unused = make([]string, 0)
		}
	}

	if config.TagName == "" {
		config.TagName = "mapstructure"
	}

	if config.MatchName == nil {
		config.MatchName = strings.EqualFold
	}

	result := &Decoder{
		config: config,
	}

	return result, nil
}

// Decode decodes the given raw interface to the target pointer specified
// by the configuration.
func (d *Decoder) Decode(input interface{}) error {
	return d.decode("", input, reflect.ValueOf(d.config.Result).Elem())
}

// Decodes an unknown data type into a specific reflection value.
func (d *Decoder) decode(name string, input interface{}, outVal reflect.Value) error {
	var inputVal reflect.Value
	if input != nil {
		inputVal = reflect.ValueOf(input)

		// We need to check here if input is a typed nil. Typed nils won't
		// match the "input == nil" below so we check that here.
		if inputVal.Kind() == reflect.Ptr && inputVal.IsNil() {
			input = nil
		}
	}

	if input == nil {
		// If the data is nil, then we don't set anything, unless ZeroFields is set
		// to true.
		if d.config.ZeroFields {
			outVal.Set(reflect.Zero(outVal.Type()))

			if d.config.Metadata != nil && name != "" {
				d.config.Metadata.Keys = append(d.config.Metadata.Keys, name)
			}
		}
		return nil
	}

	if !inputVal.IsValid() {
		// If the input value is invalid, then we just set the value
		// to be the zero value.
		outVal.Set(reflect.Zero(outVal.Type()))
		if d.config.Metadata != nil && name != "" {
			d.config.Metadata.Keys = append(d.config.Metadata.Keys, name)
		}
		return nil
	}

	if d.config.DecodeHook != nil {
		// We have a DecodeHook, so let's pre-process the input.
		var err error
		input, err = DecodeHookExec(d.config.DecodeHook, inputVal, outVal)
		if err != nil {
			return fmt.Errorf("error decoding '%s': %s", name, err)
		}
	}

	var err error
	outputKind := getKind(outVal)
	addMetaKey := true
	switch outputKind {
	case reflect.Bool:
		err = d.decodeBool(name, input, outVal)
	case reflect.Interface:
		err = d.decodeBasic(name, input, outVal)
	case reflect.String:
		err = d.decodeString(name, input, outVal)
	case reflect.Int:
		err = d.decodeInt(name, input, outVal)
	case reflect.Uint:
		err = d.decodeUint(name, input, outVal)
	case reflect.Float32:
		err = d.decodeFloat(name, input, outVal)
	case reflect.Struct:
		err = d.decodeStruct(name, input, outVal)
	case reflect.Map:
		err = d.decodeMap(name, input, outVal)
	case reflect.Ptr:
		addMetaKey, err = d.decodePtr(name, input, outVal)
	case reflect.Slice:
		err = d.decodeSlice(name, input, outVal)
	case reflect.Array:
		err = d.decodeArray(name, input, outVal)
	case reflect.Func:
		err = d.decodeFunc(name, input, outVal)
	default:
		// If we reached this point then we weren't able to decode it
		return fmt.Errorf("%s: unsupported type: %s", name, outputKind)
	}

	// If we reached here, then we successfully decoded SOMETHING, so
	// mark the key as used if we're tracking metainput.
	if addMetaKey && d.config.Metadata != nil && name != "" {
		d.config.Metadata.Keys = append(d.config.Metadata.Keys, name)
	}

	return err
}

// This decodes a basic type (bool, int, string, etc.) and sets the
// value to "data" of that type.
func (d *Decoder) decodeBasic(name string, data interface{}, val reflect.Value) error {
	if val.IsValid() && val.Elem().IsValid() {
		elem := val.Elem()

		// If we can't address this element, then its not writable. Instead,
		// we make a copy of the value (which is a pointer and therefore
		// writable), decode into that, and replace the whole value.
		copied := false
		if !elem.CanAddr() {
			copied = true

			// Make *T
			copy := reflect.New(elem.Type())

			// *T = elem
			copy.Elem().Set(elem)

			// Set elem so we decode into it
			elem = copy
		}

		// Decode. If we have an error then return. We also return right
		// away if we're not a copy because that means we decoded directly.
		if err := d.decode(name, data, elem); err != nil || !copied {
			return err
		}

		// If we're a copy, we need to set te final result
		val.Set(elem.Elem())
		return nil
	}

	dataVal := reflect.ValueOf(data)

	// If the input data is a pointer, and the assigned type is the dereference
	// of that exact pointer, then indirect it so that we can assign it.
	// Example: *string to string
	if dataVal.Kind() == reflect.Ptr && dataVal.Type().Elem() == val.Type() {
		dataVal = reflect.Indirect(dataVal)
	}

	if !dataVal.IsValid() {
		dataVal = reflect.Zero(val.Type())
	}

	dataValType := dataVal.Type()
	if !dataValType.AssignableTo(val.Type()) {
		return fmt.Errorf(
			"'%s' expected type '%s', got '%s'",
			name, val.Type(), dataValType)
	}

	val.Set(dataVal)
	return nil
}

func (d *Decoder) decodeString(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataKind := getKind(dataVal)

	converted := true
	switch {
	case dataKind == reflect.String:
		val.SetString(dataVal.String())
	case dataKind == reflect.Bool && d.config.WeaklyTypedInput:
		if dataVal.Bool() {
			val.SetString("1")
		} else {
			val.SetString("0")
		}
	case dataKind == reflect.Int && d.config.WeaklyTypedInput:
		val.SetString(strconv.FormatInt(dataVal.Int(), 10))
	case dataKind == reflect.Uint && d.config.WeaklyTypedInput:
		val.SetString(strconv.FormatUint(dataVal.Uint(), 10))
	case dataKind == reflect.Float32 && d.config.WeaklyTypedInput:
		val.SetString(strconv.FormatFloat(dataVal.Float(), 'f', -1, 64))
	case dataKind == reflect.Slice && d.config.WeaklyTypedInput,
		dataKind == reflect.Array && d.config.WeaklyTypedInput:
		dataType := dataVal.Type()
		elemKind := dataType.Elem().Kind()
		switch elemKind {
		case reflect.Uint8:
			var uints []uint8
			if dataKind == reflect.Array {
				uints = make([]uint8, dataVal.Len(), dataVal.Len())
				for i := range uints {
					uints[i] = dataVal.Index(i).Interface().(uint8)
				}
			} else {
				uints = dataVal.Interface().([]uint8)
			}
			val.SetString(string(uints))
		default:
			converted = false
		}
	default:
		converted = false
	}

	if !converted {
		return fmt.Errorf(
			"'%s' expected type '%s', got unconvertible type '%s', value: '%v'",
			name, val.Type(), dataVal.Type(), data)
	}

	return nil
}

func (d *Decoder) decodeInt(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataKind := getKind(dataVal)
	dataType := dataVal.Type()

	switch {
	case dataKind == reflect.Int:
		val.SetInt(dataVal.Int())
	case dataKind == reflect.Uint:
		val.SetInt(int64(dataVal.Uint()))
	case dataKind == reflect.Float32:
		val.SetInt(int64(dataVal.Float()))
	case dataKind == reflect.Bool && d.config.WeaklyTypedInput:
		if dataVal.Bool() {
			val.SetInt(1)
		} else {
			val.SetInt(0)
		}
	case dataKind == reflect.String && d.config.WeaklyTypedInput:
		str := dataVal.String()
		if str == "" {
			str = "0"
		}

		i, err := strconv.ParseInt(str, 0, val.Type().Bits())
		if err == nil {
			val.SetInt(i)
		} else {
			return fmt.Errorf("cannot parse '%s' as int: %s", name, err)
		}
	case dataType.PkgPath() == "encoding/json" && dataType.Name() == "Number":
		jn := data.(json.Number)
		i, err := jn.Int64()
		if err != nil {
			return fmt.Errorf(
				"error decoding json.Number into %s: %s", name, err)
		}
		val.SetInt(i)
	default:
		return fmt.Errorf(
			"'%s' expected type '%s', got unconvertible type '%s', value: '%v'",
			name, val.Type(), dataVal.Type(), data)
	}

	return nil
}

func (d *Decoder) decodeUint(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataKind := getKind(dataVal)
	dataType := dataVal.Type()

	switch {
	case dataKind == reflect.Int:
		i := dataVal.Int()
		if i < 0 && !d.config.WeaklyTypedInput {
			return fmt.Errorf("cannot parse '%s', %d overflows uint",
				name, i)
		}
		val.SetUint(uint64(i))
	case dataKind == reflect.Uint:
		val.SetUint(dataVal.Uint())
	case dataKind == reflect.Float32:
		f := dataVal.Float()
		if f < 0 && !d.config.WeaklyTypedInput {
			return fmt.Errorf("cannot parse '%s', %f overflows uint",
				name, f)
		}
		val.SetUint(uint64(f))
	case dataKind == reflect.Bool && d.config.WeaklyTypedInput:
		if dataVal.Bool() {
			val.SetUint(1)
		} else {
			val.SetUint(0)
		}
	case dataKind == reflect.String && d.config.WeaklyTypedInput:
		str := dataVal.String()
		if str == "" {
			str = "0"
		}

		i, err := strconv.ParseUint(str, 0, val.Type().Bits())
		if err == nil {
			val.SetUint(i)
		} else {
			return fmt.Errorf("cannot parse '%s' as uint: %s", name, err)
		}
	case dataType.PkgPath() == "encoding/json" && dataType.Name() == "Number":
		jn := data.(json.Number)
		i, err := strconv.ParseUint(string(jn), 0, 64)
		if err != nil {
			return fmt.Errorf(
				"error decoding json.Number into %s: %s", name, err)
		}
		val.SetUint(i)
	default:
		return fmt.Errorf(
			"'%s' expected type '%s', got unconvertible type '%s', value: '%v'",
			name, val.Type(), dataVal.Type(), data)
	}

	return nil
}

func (d *Decoder) decodeBool(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataKind := getKind(dataVal)

	switch {
	case dataKind == reflect.Bool:
		val.SetBool(dataVal.Bool())
	case dataKind == reflect.Int && d.config.WeaklyTypedInput:
		val.SetBool(dataVal.Int() != 0)
	case dataKind == reflect.Uint && d.config.WeaklyTypedInput:
		val.SetBool(dataVal.Uint() != 0)
	case dataKind == reflect.Float32 && d.config.WeaklyTypedInput:
		val.SetBool(dataVal.Float() != 0)
	case dataKind == reflect.String && d.config.WeaklyTypedInput:
		b, err := strconv.ParseBool(dataVal.String())
		if err == nil {
			val.SetBool(b)
		} else if dataVal.String() == "" {
			val.SetBool(false)
		} else {
			return fmt.Errorf("cannot parse '%s' as bool: %s", name, err)
		}
	default:
		return fmt.Errorf(
			"'%s' expected type '%s', got unconvertible type '%s', value: '%v'",
			name, val.Type(), dataVal.Type(), data)
	}

	return nil
}

func (d *Decoder) decodeFloat(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataKind := getKind(dataVal)
	dataType := dataVal.Type()

	switch {
	case dataKind == reflect.Int:
		val.SetFloat(float64(dataVal.Int()))
	case dataKind == reflect.Uint:
		val.SetFloat(float64(dataVal.Uint()))
	case dataKind == reflect.Float32:
		val.SetFloat(dataVal.Float())
	case dataKind == reflect.Bool && d.config.WeaklyTypedInput:
		if dataVal.Bool() {
			val.SetFloat(1)
		} else {
			val.SetFloat(0)
		}
	case dataKind == reflect.String && d.config.WeaklyTypedInput:
		str := dataVal.String()
		if str == "" {
			str = "0"
		}

		f, err := strconv.ParseFloat(str, val.Type().Bits())
		if err == nil {
			val.SetFloat(f)
		} else {
			return fmt.Errorf("cannot parse '%s' as float: %s", name, err)
		}
	case dataType.PkgPath() == "encoding/json" && dataType.Name() == "Number":
		jn := data.(json.Number)
		i, err := jn.Float64()
		if err != nil {
			return fmt.Errorf(
				"error decoding json.Number into %s: %s", name, err)
		}
		val.SetFloat(i)
	default:
		return fmt.Errorf(
			"'%s' expected type '%s', got unconvertible type '%s', value: '%v'",
			name, val.Type(), dataVal.Type(), data)
	}

	return nil
}

func (d *Decoder) decodeMap(name string, data interface{}, val reflect.Value) error {
	valType := val.Type()
	valKeyType := valType.Key()
	valElemType := valType.Elem()

	// By default we overwrite keys in the current map
	valMap := val

	// If the map is nil or we're purposely zeroing fields, make a new map
	if valMap.IsNil() || d.config.ZeroFields {
		// Make a new map to hold our result
		mapType := reflect.MapOf(valKeyType, valElemType)
		valMap = reflect.MakeMap(mapType)
	}

	// Check input type and based on the input type jump to the proper func
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	switch dataVal.Kind() {
	case reflect.Map:
		return d.decodeMapFromMap(name, dataVal, val, valMap)

	case reflect.Struct:
		return d.decodeMapFromStruct(name, dataVal, val, valMap)

	case reflect.Array, reflect.Slice:
		if d.config.WeaklyTypedInput {
			return d.decodeMapFromSlice(name, dataVal, val, valMap)
		}

		fallthrough

	default:
		return fmt.Errorf("'%s' expected a map, got '%s'", name, dataVal.Kind())
	}
}

func (d *Decoder) decodeMapFromSlice(name string, dataVal reflect.Value, val reflect.Value, valMap reflect.Value) error {
	// Special case for BC reasons (covered by tests)
	if dataVal.Len() == 0 {
		val.Set(valMap)
		return nil
	}

	for i := 0; i < dataVal.Len(); i++ {
		err := d.decode(
			name+"["+strconv.Itoa(i)+"]",
			dataVal.Index(i).Interface(), val)
		if err != nil {
			return err
		}
	}

	return nil
}

func (d *Decoder) decodeMapFromMap(name string, dataVal reflect.Value, val reflect.Value, valMap reflect.Value) error {
	valType := val.Type()
	valKeyType := valType.Key()
	valElemType := valType.Elem()

	// Accumulate errors
	errors := make([]string, 0)

	// If the input data is empty, then we just match what the input data is.
	if dataVal.Len() == 0 {
		if dataVal.IsNil() {
			if !val.IsNil() {
				val.Set(dataVal)
			}
		} else {
			// Set to empty allocated value
			val.Set(valMap)
		}

		return nil
	}

	for _, k := range dataVal.MapKeys() {
		fieldName := name + "[" + k.String() + "]"

		// First decode the key into the proper type
		currentKey := reflect.Indirect(reflect.New(valKeyType))
		if err := d.decode(fieldName, k.Interface(), currentKey); err != nil {
			errors = appendErrors(errors, err)
			continue
		}

		// Next decode the data into the proper type
		v := dataVal.MapIndex(k).Interface()
		currentVal := reflect.Indirect(reflect.New(valElemType))
		if err := d.decode(fieldName, v, currentVal); err != nil {
			errors = appendErrors(errors, err)
			continue
		}

		valMap.SetMapIndex(currentKey, currentVal)
	}

	// Set the built up map to the value
	val.Set(valMap)

	// If we had errors, return those
	if len(errors) > 0 {
		return &Error{errors}
	}

	return nil
}

func (d *Decoder) decodeMapFromStruct(name string, dataVal reflect.Value, val reflect.Value, valMap reflect.Value) error {
	typ := dataVal.Type()
	for i := 0; i < typ.NumField(); i++ {
		// Get the StructField first since this is a cheap operation. If the
		// field is unexported, then ignore it.
		f := typ.Field(i)
		if f.PkgPath != "" {
			continue
		}

		// Next get the actual value of this field and verify it is assignable
		// to the map value.
		v := dataVal.Field(i)
		if !v.Type().AssignableTo(valMap.Type().Elem()) {
			return fmt.Errorf("cannot assign type '%s' to map value field of type '%s'", v.Type(), valMap.Type().Elem())
		}

		tagValue := f.Tag.Get(d.config.TagName)
		keyName := f.Name

		// If Squash is set in the config, we squash the field down.
		squash := d.config.Squash && v.Kind() == reflect.Struct && f.Anonymous

		// Determine the name of the key in the map
		if index := strings.Index(tagValue, ","); index != -1 {
			if tagValue[:index] == "-" {
				continue
			}
			// If "omitempty" is specified in the tag, it ignores empty values.
			if strings.Index(tagValue[index+1:], "omitempty") != -1 && isEmptyValue(v) {
				continue
			}

			// If "squash" is specified in the tag, we squash the field down.
			squash = !squash && strings.Index(tagValue[index+1:], "squash") != -1
			if squash {
				// When squashing, the embedded type can be a pointer to a struct.
				if v.Kind() == reflect.Ptr && v.Elem().Kind() == reflect.Struct {
					v = v.Elem()
				}

				// The final type must be a struct
				if v.Kind() != reflect.Struct {
					return fmt.Errorf("cannot squash non-struct type '%s'", v.Type())
				}
			}
			keyName = tagValue[:index]
		} else if len(tagValue) > 0 {
			if tagValue == "-" {
				continue
			}
			keyName = tagValue
		}

		switch v.Kind() {
		// this is an embedded struct, so handle it differently
		case reflect.Struct:
			x := reflect.New(v.Type())
			x.Elem().Set(v)

			vType := valMap.Type()
			vKeyType := vType.Key()
			vElemType := vType.Elem()
			mType := reflect.MapOf(vKeyType, vElemType)
			vMap := reflect.MakeMap(mType)

			// Creating a pointer to a map so that other methods can completely
			// overwrite the map if need be (looking at you decodeMapFromMap). The
			// indirection allows the underlying map to be settable (CanSet() == true)
			// where as reflect.MakeMap returns an unsettable map.
			addrVal := reflect.New(vMap.Type())
			reflect.Indirect(addrVal).Set(vMap)

			err := d.decode(keyName, x.Interface(), reflect.Indirect(addrVal))
			if err != nil {
				return err
			}

			// the underlying map may have been completely overwritten so pull
			// it indirectly out of the enclosing value.
			vMap = reflect.Indirect(addrVal)

			if squash {
				for _, k := range vMap.MapKeys() {
					valMap.SetMapIndex(k, vMap.MapIndex(k))
				}
			} else {
				valMap.SetMapIndex(reflect.ValueOf(keyName), vMap)
			}

		default:
			valMap.SetMapIndex(reflect.ValueOf(keyName), v)
		}
	}

	if val.CanAddr() {
		val.Set(valMap)
	}

	return nil
}

func (d *Decoder) decodePtr(name string, data interface{}, val reflect.Value) (bool, error) {
	// If the input data is nil, then we want to just set the output
	// pointer to be nil as well.
	isNil := data == nil
	if !isNil {
		switch v := reflect.Indirect(reflect.ValueOf(data)); v.Kind() {
		case reflect.Chan,
			reflect.Func,
			reflect.Interface,
			reflect.Map,
			reflect.Ptr,
			reflect.Slice:
			isNil = v.IsNil()
		}
	}
	if isNil {
		if !val.IsNil() && val.CanSet() {
			nilValue := reflect.New(val.Type()).Elem()
			val.Set(nilValue)
		}

		return true, nil
	}

	// Create an element of the concrete (non pointer) type and decode
	// into that. Then set the value of the pointer to this type.
	valType := val.Type()
	valElemType := valType.Elem()
	if val.CanSet() {
		realVal := val
		if realVal.IsNil() || d.config.ZeroFields {
			realVal = reflect.New(valElemType)
		}

		if err := d.decode(name, data, reflect.Indirect(realVal)); err != nil {
			return false, err
		}

		val.Set(realVal)
	} else {
		if err := d.decode(name, data, reflect.Indirect(val)); err != nil {
			return false, err
		}
	}
	return false, nil
}

func (d *Decoder) decodeFunc(name string, data interface{}, val reflect.Value) error {
	// Create an element of the concrete (non pointer) type and decode
	// into that. Then set the value of the pointer to this type.
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	if val.Type() != dataVal.Type() {
		return fmt.Errorf(
			"'%s' expected type '%s', got unconvertible type '%s', value: '%v'",
			name, val.Type(), dataVal.Type(), data)
	}
	val.Set(dataVal)
	return nil
}

func (d *Decoder) decodeSlice(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataValKind := dataVal.Kind()
	valType := val.Type()
	valElemType := valType.Elem()
	sliceType := reflect.SliceOf(valElemType)

	// If we have a non array/slice type then we first attempt to convert.
	if dataValKind != reflect.Array && dataValKind != reflect.Slice {
		if d.config.WeaklyTypedInput {
			switch {
			// Slice and array we use the normal logic
			case dataValKind == reflect.Slice, dataValKind == reflect.Array:
				break

			// Empty maps turn into empty slices
			case dataValKind == reflect.Map:
				if dataVal.Len() == 0 {
					val.Set(reflect.MakeSlice(sliceType, 0, 0))
					return nil
				}
				// Create slice of maps of other sizes
				return d.decodeSlice(name, []interface{}{data}, val)

			case dataValKind == reflect.String && valElemType.Kind() == reflect.Uint8:
				return d.decodeSlice(name, []byte(dataVal.String()), val)

			// All other types we try to convert to the slice type
			// and "lift" it into it. i.e. a string becomes a string slice.
			default:
				// Just re-try this function with data as a slice.
				return d.decodeSlice(name, []interface{}{data}, val)
			}
		}

		return fmt.Errorf(
			"'%s': source data must be an array or slice, got %s", name, dataValKind)
	}

	// If the input value is nil, then don't allocate since empty != nil
	if dataVal.IsNil() {
		return nil
	}

	valSlice := val
	if valSlice.IsNil() || d.config.ZeroFields {
		// Make a new slice to hold our result, same size as the original data.
		valSlice = reflect.MakeSlice(sliceType, dataVal.Len(), dataVal.Len())
	}

	// Accumulate any errors
	errors := make([]string, 0)

	for i := 0; i < dataVal.Len(); i++ {
		currentData := dataVal.Index(i).Interface()
		for valSlice.Len() <= i {
			valSlice = reflect.Append(valSlice, reflect.Zero(valElemType))
		}
		currentField := valSlice.Index(i)

		fieldName := name + "[" + strconv.Itoa(i) + "]"
		if err := d.decode(fieldName, currentData, currentField); err != nil {
			errors = appendErrors(errors, err)
		}
	}

	// Finally, set the value to the slice we built up
	val.Set(valSlice)

	// If there were errors, we return those
	if len(errors) > 0 {
		return &Error{errors}
	}

	return nil
}

func (d *Decoder) decodeArray(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))
	dataValKind := dataVal.Kind()
	valType := val.Type()
	valElemType := valType.Elem()
	arrayType := reflect.ArrayOf(valType.Len(), valElemType)

	valArray := val

	if valArray.Interface() == reflect.Zero(valArray.Type()).Interface() || d.config.ZeroFields {
		// Check input type
		if dataValKind != reflect.Array && dataValKind != reflect.Slice {
			if d.config.WeaklyTypedInput {
				switch {
				// Empty maps turn into empty arrays
				case dataValKind == reflect.Map:
					if dataVal.Len() == 0 {
						val.Set(reflect.Zero(arrayType))
						return nil
					}

				// All other types we try to convert to the array type
				// and "lift" it into it. i.e. a string becomes a string array.
				default:
					// Just re-try this function with data as a slice.
					return d.decodeArray(name, []interface{}{data}, val)
				}
			}

			return fmt.Errorf(
				"'%s': source data must be an array or slice, got %s", name, dataValKind)

		}
		if dataVal.Len() > arrayType.Len() {
			return fmt.Errorf(
				"'%s': expected source data to have length less or equal to %d, got %d", name, arrayType.Len(), dataVal.Len())

		}

		// Make a new array to hold our result, same size as the original data.
		valArray = reflect.New(arrayType).Elem()
	}

	// Accumulate any errors
	errors := make([]string, 0)

	for i := 0; i < dataVal.Len(); i++ {
		currentData := dataVal.Index(i).Interface()
		currentField := valArray.Index(i)

		fieldName := name + "[" + strconv.Itoa(i) + "]"
		if err := d.decode(fieldName, currentData, currentField); err != nil {
			errors = appendErrors(errors, err)
		}
	}

	// Finally, set the value to the array we built up
	val.Set(valArray)

	// If there were errors, we return those
	if len(errors) > 0 {
		return &Error{errors}
	}

	return nil
}

func (d *Decoder) decodeStruct(name string, data interface{}, val reflect.Value) error {
	dataVal := reflect.Indirect(reflect.ValueOf(data))

	// If the type of the value to write to and the data match directly,
	// then we just set it directly instead of recursing into the structure.
	if dataVal.Type() == val.Type() {
		val.Set(dataVal)
		return nil
	}

	dataValKind := dataVal.Kind()
	switch dataValKind {
	case reflect.Map:
		return d.decodeStructFromMap(name, dataVal, val)

	case reflect.Struct:
		// Not the most efficient way to do this but we can optimize later if
		// we want to. To convert from struct to struct we go to map first
		// as an intermediary.

		// Make a new map to hold our result
		mapType := reflect.TypeOf((map[string]interface{})(nil))
		mval := reflect.MakeMap(mapType)

		// Creating a pointer to a map so that other methods can completely
		// overwrite the map if need be (looking at you decodeMapFromMap). The
		// indirection allows the underlying map to be settable (CanSet() == true)
		// where as reflect.MakeMap returns an unsettable map.
		addrVal := reflect.New(mval.Type())

		reflect.Indirect(addrVal).Set(mval)
		if err := d.decodeMapFromStruct(name, dataVal, reflect.Indirect(addrVal), mval); err != nil {
			return err
		}

		result := d.decodeStructFromMap(name, reflect.Indirect(addrVal), val)
		return result

	default:
		return fmt.Errorf("'%s' expected a map, got '%s'", name, dataVal.Kind())
	}
}

func (d *Decoder) decodeStructFromMap(name string, dataVal, val reflect.Value) error {
	dataValType := dataVal.Type()
	if kind := dataValType.Key().Kind(); kind != reflect.String && kind != reflect.Interface {
		return fmt.Errorf(
			"'%s' needs a map with string keys, has '%s' keys",
			name, dataValType.Key().Kind())
	}

	dataValKeys := make(map[reflect.Value]struct{})
	dataValKeysUnused := make(map[interface{}]struct{})
	for _, dataValKey := range dataVal.MapKeys() {
		dataValKeys[dataValKey] = struct{}{}
		dataValKeysUnused[dataValKey.Interface()] = struct{}{}
	}

	errors := make([]string, 0)

	// This slice will keep track of all the structs we'll be decoding.
	// There can be more than one struct if there are embedded structs
	// that are squashed.
	structs := make([]reflect.Value, 1, 5)
	structs[0] = val

	// Compile the list of all the fields that we're going to be decoding
	// from all the structs.
	type field struct {
		field reflect.StructField
		val   reflect.Value
	}

	// remainField is set to a valid field set with the "remain" tag if
	// we are keeping track of remaining values.
	var remainField *field

	fields := []field{}
	for len(structs) > 0 {
		structVal := structs[0]
		structs = structs[1:]

		structType := structVal.Type()

		for i := 0; i < structType.NumField(); i++ {
			fieldType := structType.Field(i)
			fieldVal := structVal.Field(i)
			if fieldVal.Kind() == reflect.Ptr && fieldVal.Elem().Kind() == reflect.Struct {
				// Handle embedded struct pointers as embedded structs.
				fieldVal = fieldVal.Elem()
			}

			// If "squash" is specified in the tag, we squash the field down.
			squash := d.config.Squash && fieldVal.Kind() == reflect.Struct && fieldType.Anonymous
			remain := false

			// We always parse the tags cause we're looking for other tags too
			tagParts := strings.Split(fieldType.Tag.Get(d.config.TagName), ",")
			for _, tag := range tagParts[1:] {
				if tag == "squash" {
					squash = true
					break
				}

				if tag == "remain" {
					remain = true
					break
				}
			}

			if squash {
				if fieldVal.Kind() != reflect.Struct {
					errors = appendErrors(errors,
						fmt.Errorf("%s: unsupported type for squash: %s", fieldType.Name, fieldVal.Kind()))
				} else {
					structs = append(structs, fieldVal)
				}
				continue
			}

			// Build our field
			if remain {
				remainField = &field{fieldType, fieldVal}
			} else {
				// Normal struct field, store it away
				fields = append(fields, field{fieldType, fieldVal})
			}
		}
	}

	// for fieldType, field := range fields {
	for _, f := range fields {
		field, fieldValue := f.field, f.val
		fieldName := field.Name

		tagValue := field.Tag.Get(d.config.TagName)
		tagValue = strings.SplitN(tagValue, ",", 2)[0]
		if tagValue != "" {
			fieldName = tagValue
		}

		rawMapKey := reflect.ValueOf(fieldName)
		rawMapVal := dataVal.MapIndex(rawMapKey)
		if !rawMapVal.IsValid() {
			// Do a slower search by iterating over each key and
			// doing case-insensitive search.
			for dataValKey := range dataValKeys {
				mK, ok := dataValKey.Interface().(string)
				if !ok {
					// Not a string key
					continue
				}

				if d.config.MatchName(mK, fieldName) {
					rawMapKey = dataValKey
					rawMapVal = dataVal.MapIndex(dataValKey)
					break
				}
			}

			if !rawMapVal.IsValid() {
				// There was no matching key in the map for the value in
				// the struct. Just ignore.
				continue
			}
		}

		if !fieldValue.IsValid() {
			// This should never happen
			panic("field is not valid")
		}

		// If we can't set the field, then it is unexported or something,
		// and we just continue onwards.
		if !fieldValue.CanSet() {
			continue
		}

		// Delete the key we're using from the unused map so we stop tracking
		delete(dataValKeysUnused, rawMapKey.Interface())

		// If the name is empty string, then we're at the root, and we
		// don't dot-join the fields.
		if name != "" {
			fieldName = name + "." + fieldName
		}

		if err := d.decode(fieldName, rawMapVal.Interface(), fieldValue); err != nil {
			errors = appendErrors(errors, err)
		}
	}

	// If we have a "remain"-tagged field and we have unused keys then
	// we put the unused keys directly into the remain field.
	if remainField != nil && len(dataValKeysUnused) > 0 {
		// Build a map of only the unused values
		remain := map[interface{}]interface{}{}
		for key := range dataValKeysUnused {
			remain[key] = dataVal.MapIndex(reflect.ValueOf(key)).Interface()
		}

		// Decode it as-if we were just decoding this map onto our map.
		if err := d.decodeMap(name, remain, remainField.val); err != nil {
			errors = appendErrors(errors, err)
		}

		// Set the map to nil so we have none so that the next check will
		// not error (ErrorUnused)
		dataValKeysUnused = nil
	}

	if d.config.ErrorUnused && len(dataValKeysUnused) > 0 {
		keys := make([]string, 0, len(dataValKeysUnused))
		for rawKey := range dataValKeysUnused {
			keys = append(keys, rawKey.(string))
		}
		sort.Strings(keys)

		err := fmt.Errorf("'%s' has invalid keys: %s", name, strings.Join(keys, ", "))
		errors = appendErrors(errors, err)
	}

	if len(errors) > 0 {
		return &Error{errors}
	}

	// Add the unused keys to the list of unused keys if we're tracking metadata
	if d.config.Metadata != nil {
		for rawKey := range dataValKeysUnused {
			key := rawKey.(string)
			if name != "" {
				key = name + "." + key
			}

			d.config.Metadata.Unused = append(d.config.Metadata.Unused, key)
		}
	}

	return nil
}

func isEmptyValue(v reflect.Value) bool {
	switch getKind(v) {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func getKind(val reflect.Value) reflect.Kind {
	kind := val.Kind()

	switch {
	case kind >= reflect.Int && kind <= reflect.Int64:
		return reflect.Int
	case kind >= reflect.Uint && kind <= reflect.Uint64:
		return reflect.Uint
	case kind >= reflect.Float32 && kind <= reflect.Float64:
		return reflect.Float32
	default:
		return kind
	}
}
