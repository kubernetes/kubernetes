package toml

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
)

type tomlOpts struct {
	name      string
	include   bool
	omitempty bool
}

var timeType = reflect.TypeOf(time.Time{})
var marshalerType = reflect.TypeOf(new(Marshaler)).Elem()

// Check if the given marshall type maps to a Tree primitive
func isPrimitive(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isPrimitive(mtype.Elem())
	case reflect.Bool:
		return true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return true
	case reflect.Float32, reflect.Float64:
		return true
	case reflect.String:
		return true
	case reflect.Struct:
		return mtype == timeType || isCustomMarshaler(mtype)
	default:
		return false
	}
}

// Check if the given marshall type maps to a Tree slice
func isTreeSlice(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Slice:
		return !isOtherSlice(mtype)
	default:
		return false
	}
}

// Check if the given marshall type maps to a non-Tree slice
func isOtherSlice(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isOtherSlice(mtype.Elem())
	case reflect.Slice:
		return isPrimitive(mtype.Elem()) || isOtherSlice(mtype.Elem())
	default:
		return false
	}
}

// Check if the given marshall type maps to a Tree
func isTree(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Map:
		return true
	case reflect.Struct:
		return !isPrimitive(mtype)
	default:
		return false
	}
}

func isCustomMarshaler(mtype reflect.Type) bool {
	return mtype.Implements(marshalerType)
}

func callCustomMarshaler(mval reflect.Value) ([]byte, error) {
	return mval.Interface().(Marshaler).MarshalTOML()
}

// Marshaler is the interface implemented by types that
// can marshal themselves into valid TOML.
type Marshaler interface {
	MarshalTOML() ([]byte, error)
}

/*
Marshal returns the TOML encoding of v.  Behavior is similar to the Go json
encoder, except that there is no concept of a Marshaler interface or MarshalTOML
function for sub-structs, and currently only definite types can be marshaled
(i.e. no `interface{}`).

Note that pointers are automatically assigned the "omitempty" option, as TOML
explicity does not handle null values (saying instead the label should be
dropped).

Tree structural types and corresponding marshal types:

  *Tree                            (*)struct, (*)map[string]interface{}
  []*Tree                          (*)[](*)struct, (*)[](*)map[string]interface{}
  []interface{} (as interface{})   (*)[]primitive, (*)[]([]interface{})
  interface{}                      (*)primitive

Tree primitive types and corresponding marshal types:

  uint64     uint, uint8-uint64, pointers to same
  int64      int, int8-uint64, pointers to same
  float64    float32, float64, pointers to same
  string     string, pointers to same
  bool       bool, pointers to same
  time.Time  time.Time{}, pointers to same
*/
func Marshal(v interface{}) ([]byte, error) {
	mtype := reflect.TypeOf(v)
	if mtype.Kind() != reflect.Struct {
		return []byte{}, errors.New("Only a struct can be marshaled to TOML")
	}
	sval := reflect.ValueOf(v)
	if isCustomMarshaler(mtype) {
		return callCustomMarshaler(sval)
	}
	t, err := valueToTree(mtype, sval)
	if err != nil {
		return []byte{}, err
	}
	s, err := t.ToTomlString()
	return []byte(s), err
}

// Convert given marshal struct or map value to toml tree
func valueToTree(mtype reflect.Type, mval reflect.Value) (*Tree, error) {
	if mtype.Kind() == reflect.Ptr {
		return valueToTree(mtype.Elem(), mval.Elem())
	}
	tval := newTree()
	switch mtype.Kind() {
	case reflect.Struct:
		for i := 0; i < mtype.NumField(); i++ {
			mtypef, mvalf := mtype.Field(i), mval.Field(i)
			opts := tomlOptions(mtypef)
			if opts.include && (!opts.omitempty || !isZero(mvalf)) {
				val, err := valueToToml(mtypef.Type, mvalf)
				if err != nil {
					return nil, err
				}
				tval.Set(opts.name, val)
			}
		}
	case reflect.Map:
		for _, key := range mval.MapKeys() {
			mvalf := mval.MapIndex(key)
			val, err := valueToToml(mtype.Elem(), mvalf)
			if err != nil {
				return nil, err
			}
			tval.Set(key.String(), val)
		}
	}
	return tval, nil
}

// Convert given marshal slice to slice of Toml trees
func valueToTreeSlice(mtype reflect.Type, mval reflect.Value) ([]*Tree, error) {
	tval := make([]*Tree, mval.Len(), mval.Len())
	for i := 0; i < mval.Len(); i++ {
		val, err := valueToTree(mtype.Elem(), mval.Index(i))
		if err != nil {
			return nil, err
		}
		tval[i] = val
	}
	return tval, nil
}

// Convert given marshal slice to slice of toml values
func valueToOtherSlice(mtype reflect.Type, mval reflect.Value) (interface{}, error) {
	tval := make([]interface{}, mval.Len(), mval.Len())
	for i := 0; i < mval.Len(); i++ {
		val, err := valueToToml(mtype.Elem(), mval.Index(i))
		if err != nil {
			return nil, err
		}
		tval[i] = val
	}
	return tval, nil
}

// Convert given marshal value to toml value
func valueToToml(mtype reflect.Type, mval reflect.Value) (interface{}, error) {
	if mtype.Kind() == reflect.Ptr {
		return valueToToml(mtype.Elem(), mval.Elem())
	}
	switch {
	case isCustomMarshaler(mtype):
		return callCustomMarshaler(mval)
	case isTree(mtype):
		return valueToTree(mtype, mval)
	case isTreeSlice(mtype):
		return valueToTreeSlice(mtype, mval)
	case isOtherSlice(mtype):
		return valueToOtherSlice(mtype, mval)
	default:
		switch mtype.Kind() {
		case reflect.Bool:
			return mval.Bool(), nil
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return mval.Int(), nil
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return mval.Uint(), nil
		case reflect.Float32, reflect.Float64:
			return mval.Float(), nil
		case reflect.String:
			return mval.String(), nil
		case reflect.Struct:
			return mval.Interface().(time.Time), nil
		default:
			return nil, fmt.Errorf("Marshal can't handle %v(%v)", mtype, mtype.Kind())
		}
	}
}

// Unmarshal attempts to unmarshal the Tree into a Go struct pointed by v.
// Neither Unmarshaler interfaces nor UnmarshalTOML functions are supported for
// sub-structs, and only definite types can be unmarshaled.
func (t *Tree) Unmarshal(v interface{}) error {
	mtype := reflect.TypeOf(v)
	if mtype.Kind() != reflect.Ptr || mtype.Elem().Kind() != reflect.Struct {
		return errors.New("Only a pointer to struct can be unmarshaled from TOML")
	}

	sval, err := valueFromTree(mtype.Elem(), t)
	if err != nil {
		return err
	}
	reflect.ValueOf(v).Elem().Set(sval)
	return nil
}

// Unmarshal parses the TOML-encoded data and stores the result in the value
// pointed to by v. Behavior is similar to the Go json encoder, except that there
// is no concept of an Unmarshaler interface or UnmarshalTOML function for
// sub-structs, and currently only definite types can be unmarshaled to (i.e. no
// `interface{}`).
//
// See Marshal() documentation for types mapping table.
func Unmarshal(data []byte, v interface{}) error {
	t, err := LoadReader(bytes.NewReader(data))
	if err != nil {
		return err
	}
	return t.Unmarshal(v)
}

// Convert toml tree to marshal struct or map, using marshal type
func valueFromTree(mtype reflect.Type, tval *Tree) (reflect.Value, error) {
	if mtype.Kind() == reflect.Ptr {
		return unwrapPointer(mtype, tval)
	}
	var mval reflect.Value
	switch mtype.Kind() {
	case reflect.Struct:
		mval = reflect.New(mtype).Elem()
		for i := 0; i < mtype.NumField(); i++ {
			mtypef := mtype.Field(i)
			opts := tomlOptions(mtypef)
			if opts.include {
				baseKey := opts.name
				keysToTry := []string{baseKey, strings.ToLower(baseKey), strings.ToTitle(baseKey)}
				for _, key := range keysToTry {
					exists := tval.Has(key)
					if !exists {
						continue
					}
					val := tval.Get(key)
					mvalf, err := valueFromToml(mtypef.Type, val)
					if err != nil {
						return mval, formatError(err, tval.GetPosition(key))
					}
					mval.Field(i).Set(mvalf)
					break
				}
			}
		}
	case reflect.Map:
		mval = reflect.MakeMap(mtype)
		for _, key := range tval.Keys() {
			val := tval.Get(key)
			mvalf, err := valueFromToml(mtype.Elem(), val)
			if err != nil {
				return mval, formatError(err, tval.GetPosition(key))
			}
			mval.SetMapIndex(reflect.ValueOf(key), mvalf)
		}
	}
	return mval, nil
}

// Convert toml value to marshal struct/map slice, using marshal type
func valueFromTreeSlice(mtype reflect.Type, tval []*Tree) (reflect.Value, error) {
	mval := reflect.MakeSlice(mtype, len(tval), len(tval))
	for i := 0; i < len(tval); i++ {
		val, err := valueFromTree(mtype.Elem(), tval[i])
		if err != nil {
			return mval, err
		}
		mval.Index(i).Set(val)
	}
	return mval, nil
}

// Convert toml value to marshal primitive slice, using marshal type
func valueFromOtherSlice(mtype reflect.Type, tval []interface{}) (reflect.Value, error) {
	mval := reflect.MakeSlice(mtype, len(tval), len(tval))
	for i := 0; i < len(tval); i++ {
		val, err := valueFromToml(mtype.Elem(), tval[i])
		if err != nil {
			return mval, err
		}
		mval.Index(i).Set(val)
	}
	return mval, nil
}

// Convert toml value to marshal value, using marshal type
func valueFromToml(mtype reflect.Type, tval interface{}) (reflect.Value, error) {
	if mtype.Kind() == reflect.Ptr {
		return unwrapPointer(mtype, tval)
	}
	switch {
	case isTree(mtype):
		return valueFromTree(mtype, tval.(*Tree))
	case isTreeSlice(mtype):
		return valueFromTreeSlice(mtype, tval.([]*Tree))
	case isOtherSlice(mtype):
		return valueFromOtherSlice(mtype, tval.([]interface{}))
	default:
		switch mtype.Kind() {
		case reflect.Bool:
			val, ok := tval.(bool)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to bool", tval, tval)
			}
			return reflect.ValueOf(val), nil
		case reflect.Int:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to int", tval, tval)
			}
			return reflect.ValueOf(int(val)), nil
		case reflect.Int8:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to int", tval, tval)
			}
			return reflect.ValueOf(int8(val)), nil
		case reflect.Int16:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to int", tval, tval)
			}
			return reflect.ValueOf(int16(val)), nil
		case reflect.Int32:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to int", tval, tval)
			}
			return reflect.ValueOf(int32(val)), nil
		case reflect.Int64:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to int", tval, tval)
			}
			return reflect.ValueOf(val), nil
		case reflect.Uint:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to uint", tval, tval)
			}
			return reflect.ValueOf(uint(val)), nil
		case reflect.Uint8:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to uint", tval, tval)
			}
			return reflect.ValueOf(uint8(val)), nil
		case reflect.Uint16:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to uint", tval, tval)
			}
			return reflect.ValueOf(uint16(val)), nil
		case reflect.Uint32:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to uint", tval, tval)
			}
			return reflect.ValueOf(uint32(val)), nil
		case reflect.Uint64:
			val, ok := tval.(int64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to uint", tval, tval)
			}
			return reflect.ValueOf(uint64(val)), nil
		case reflect.Float32:
			val, ok := tval.(float64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to float", tval, tval)
			}
			return reflect.ValueOf(float32(val)), nil
		case reflect.Float64:
			val, ok := tval.(float64)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to float", tval, tval)
			}
			return reflect.ValueOf(val), nil
		case reflect.String:
			val, ok := tval.(string)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to string", tval, tval)
			}
			return reflect.ValueOf(val), nil
		case reflect.Struct:
			val, ok := tval.(time.Time)
			if !ok {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to time", tval, tval)
			}
			return reflect.ValueOf(val), nil
		default:
			return reflect.ValueOf(nil), fmt.Errorf("Unmarshal can't handle %v(%v)", mtype, mtype.Kind())
		}
	}
}

func unwrapPointer(mtype reflect.Type, tval interface{}) (reflect.Value, error) {
	val, err := valueFromToml(mtype.Elem(), tval)
	if err != nil {
		return reflect.ValueOf(nil), err
	}
	mval := reflect.New(mtype.Elem())
	mval.Elem().Set(val)
	return mval, nil
}

func tomlOptions(vf reflect.StructField) tomlOpts {
	tag := vf.Tag.Get("toml")
	parse := strings.Split(tag, ",")
	result := tomlOpts{vf.Name, true, false}
	if parse[0] != "" {
		if parse[0] == "-" && len(parse) == 1 {
			result.include = false
		} else {
			result.name = strings.Trim(parse[0], " ")
		}
	}
	if vf.PkgPath != "" {
		result.include = false
	}
	if len(parse) > 1 && strings.Trim(parse[1], " ") == "omitempty" {
		result.omitempty = true
	}
	if vf.Type.Kind() == reflect.Ptr {
		result.omitempty = true
	}
	return result
}

func isZero(val reflect.Value) bool {
	switch val.Type().Kind() {
	case reflect.Map:
		fallthrough
	case reflect.Array:
		fallthrough
	case reflect.Slice:
		return val.Len() == 0
	default:
		return reflect.DeepEqual(val.Interface(), reflect.Zero(val.Type()).Interface())
	}
}

func formatError(err error, pos Position) error {
	if err.Error()[0] == '(' { // Error already contains position information
		return err
	}
	return fmt.Errorf("%s: %s", pos, err)
}
