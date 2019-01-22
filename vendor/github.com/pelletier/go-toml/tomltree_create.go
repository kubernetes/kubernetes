package toml

import (
	"fmt"
	"reflect"
	"time"
)

var kindToType = [reflect.String + 1]reflect.Type{
	reflect.Bool:    reflect.TypeOf(true),
	reflect.String:  reflect.TypeOf(""),
	reflect.Float32: reflect.TypeOf(float64(1)),
	reflect.Float64: reflect.TypeOf(float64(1)),
	reflect.Int:     reflect.TypeOf(int64(1)),
	reflect.Int8:    reflect.TypeOf(int64(1)),
	reflect.Int16:   reflect.TypeOf(int64(1)),
	reflect.Int32:   reflect.TypeOf(int64(1)),
	reflect.Int64:   reflect.TypeOf(int64(1)),
	reflect.Uint:    reflect.TypeOf(uint64(1)),
	reflect.Uint8:   reflect.TypeOf(uint64(1)),
	reflect.Uint16:  reflect.TypeOf(uint64(1)),
	reflect.Uint32:  reflect.TypeOf(uint64(1)),
	reflect.Uint64:  reflect.TypeOf(uint64(1)),
}

// typeFor returns a reflect.Type for a reflect.Kind, or nil if none is found.
// supported values:
// string, bool, int64, uint64, float64, time.Time, int, int8, int16, int32, uint, uint8, uint16, uint32, float32
func typeFor(k reflect.Kind) reflect.Type {
	if k > 0 && int(k) < len(kindToType) {
		return kindToType[k]
	}
	return nil
}

func simpleValueCoercion(object interface{}) (interface{}, error) {
	switch original := object.(type) {
	case string, bool, int64, uint64, float64, time.Time:
		return original, nil
	case int:
		return int64(original), nil
	case int8:
		return int64(original), nil
	case int16:
		return int64(original), nil
	case int32:
		return int64(original), nil
	case uint:
		return uint64(original), nil
	case uint8:
		return uint64(original), nil
	case uint16:
		return uint64(original), nil
	case uint32:
		return uint64(original), nil
	case float32:
		return float64(original), nil
	case fmt.Stringer:
		return original.String(), nil
	default:
		return nil, fmt.Errorf("cannot convert type %T to Tree", object)
	}
}

func sliceToTree(object interface{}) (interface{}, error) {
	// arrays are a bit tricky, since they can represent either a
	// collection of simple values, which is represented by one
	// *tomlValue, or an array of tables, which is represented by an
	// array of *Tree.

	// holding the assumption that this function is called from toTree only when value.Kind() is Array or Slice
	value := reflect.ValueOf(object)
	insideType := value.Type().Elem()
	length := value.Len()
	if length > 0 {
		insideType = reflect.ValueOf(value.Index(0).Interface()).Type()
	}
	if insideType.Kind() == reflect.Map {
		// this is considered as an array of tables
		tablesArray := make([]*Tree, 0, length)
		for i := 0; i < length; i++ {
			table := value.Index(i)
			tree, err := toTree(table.Interface())
			if err != nil {
				return nil, err
			}
			tablesArray = append(tablesArray, tree.(*Tree))
		}
		return tablesArray, nil
	}

	sliceType := typeFor(insideType.Kind())
	if sliceType == nil {
		sliceType = insideType
	}

	arrayValue := reflect.MakeSlice(reflect.SliceOf(sliceType), 0, length)

	for i := 0; i < length; i++ {
		val := value.Index(i).Interface()
		simpleValue, err := simpleValueCoercion(val)
		if err != nil {
			return nil, err
		}
		arrayValue = reflect.Append(arrayValue, reflect.ValueOf(simpleValue))
	}
	return &tomlValue{value: arrayValue.Interface(), position: Position{}}, nil
}

func toTree(object interface{}) (interface{}, error) {
	value := reflect.ValueOf(object)

	if value.Kind() == reflect.Map {
		values := map[string]interface{}{}
		keys := value.MapKeys()
		for _, key := range keys {
			if key.Kind() != reflect.String {
				if _, ok := key.Interface().(string); !ok {
					return nil, fmt.Errorf("map key needs to be a string, not %T (%v)", key.Interface(), key.Kind())
				}
			}

			v := value.MapIndex(key)
			newValue, err := toTree(v.Interface())
			if err != nil {
				return nil, err
			}
			values[key.String()] = newValue
		}
		return &Tree{values: values, position: Position{}}, nil
	}

	if value.Kind() == reflect.Array || value.Kind() == reflect.Slice {
		return sliceToTree(object)
	}

	simpleValue, err := simpleValueCoercion(object)
	if err != nil {
		return nil, err
	}
	return &tomlValue{value: simpleValue, position: Position{}}, nil
}
