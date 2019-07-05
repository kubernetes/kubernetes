package runtime

import (
	"fmt"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/grpc-ecosystem/grpc-gateway/utilities"
	"google.golang.org/grpc/grpclog"
)

// PopulateQueryParameters populates "values" into "msg".
// A value is ignored if its key starts with one of the elements in "filter".
func PopulateQueryParameters(msg proto.Message, values url.Values, filter *utilities.DoubleArray) error {
	for key, values := range values {
		fieldPath := strings.Split(key, ".")
		if filter.HasCommonPrefix(fieldPath) {
			continue
		}
		if err := populateFieldValueFromPath(msg, fieldPath, values); err != nil {
			return err
		}
	}
	return nil
}

// PopulateFieldFromPath sets a value in a nested Protobuf structure.
// It instantiates missing protobuf fields as it goes.
func PopulateFieldFromPath(msg proto.Message, fieldPathString string, value string) error {
	fieldPath := strings.Split(fieldPathString, ".")
	return populateFieldValueFromPath(msg, fieldPath, []string{value})
}

func populateFieldValueFromPath(msg proto.Message, fieldPath []string, values []string) error {
	m := reflect.ValueOf(msg)
	if m.Kind() != reflect.Ptr {
		return fmt.Errorf("unexpected type %T: %v", msg, msg)
	}
	var props *proto.Properties
	m = m.Elem()
	for i, fieldName := range fieldPath {
		isLast := i == len(fieldPath)-1
		if !isLast && m.Kind() != reflect.Struct {
			return fmt.Errorf("non-aggregate type in the mid of path: %s", strings.Join(fieldPath, "."))
		}
		var f reflect.Value
		var err error
		f, props, err = fieldByProtoName(m, fieldName)
		if err != nil {
			return err
		} else if !f.IsValid() {
			grpclog.Printf("field not found in %T: %s", msg, strings.Join(fieldPath, "."))
			return nil
		}

		switch f.Kind() {
		case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64, reflect.String, reflect.Uint32, reflect.Uint64:
			if !isLast {
				return fmt.Errorf("unexpected nested field %s in %s", fieldPath[i+1], strings.Join(fieldPath[:i+1], "."))
			}
			m = f
		case reflect.Slice:
			// TODO(yugui) Support []byte
			if !isLast {
				return fmt.Errorf("unexpected repeated field in %s", strings.Join(fieldPath, "."))
			}
			return populateRepeatedField(f, values, props)
		case reflect.Ptr:
			if f.IsNil() {
				m = reflect.New(f.Type().Elem())
				f.Set(m.Convert(f.Type()))
			}
			m = f.Elem()
			continue
		case reflect.Struct:
			m = f
			continue
		default:
			return fmt.Errorf("unexpected type %s in %T", f.Type(), msg)
		}
	}
	switch len(values) {
	case 0:
		return fmt.Errorf("no value of field: %s", strings.Join(fieldPath, "."))
	case 1:
	default:
		grpclog.Printf("too many field values: %s", strings.Join(fieldPath, "."))
	}
	return populateField(m, values[0], props)
}

// fieldByProtoName looks up a field whose corresponding protobuf field name is "name".
// "m" must be a struct value. It returns zero reflect.Value if no such field found.
func fieldByProtoName(m reflect.Value, name string) (reflect.Value, *proto.Properties, error) {
	props := proto.GetProperties(m.Type())

	// look up field name in oneof map
	if op, ok := props.OneofTypes[name]; ok {
		v := reflect.New(op.Type.Elem())
		field := m.Field(op.Field)
		if !field.IsNil() {
			return reflect.Value{}, nil, fmt.Errorf("field already set for %s oneof", props.Prop[op.Field].OrigName)
		}
		field.Set(v)
		return v.Elem().Field(0), op.Prop, nil
	}

	for _, p := range props.Prop {
		if p.OrigName == name {
			return m.FieldByName(p.Name), p, nil
		}
		if p.JSONName == name {
			return m.FieldByName(p.Name), p, nil
		}
	}
	return reflect.Value{}, nil, nil
}

func populateRepeatedField(f reflect.Value, values []string, props *proto.Properties) error {
	elemType := f.Type().Elem()

	// is the destination field a slice of an enumeration type?
	if enumValMap := proto.EnumValueMap(props.Enum); enumValMap != nil {
		return populateFieldEnumRepeated(f, values, enumValMap)
	}

	conv, ok := convFromType[elemType.Kind()]
	if !ok {
		return fmt.Errorf("unsupported field type %s", elemType)
	}
	f.Set(reflect.MakeSlice(f.Type(), len(values), len(values)).Convert(f.Type()))
	for i, v := range values {
		result := conv.Call([]reflect.Value{reflect.ValueOf(v)})
		if err := result[1].Interface(); err != nil {
			return err.(error)
		}
		f.Index(i).Set(result[0].Convert(f.Index(i).Type()))
	}
	return nil
}

func populateField(f reflect.Value, value string, props *proto.Properties) error {
	// Handle well known type
	type wkt interface {
		XXX_WellKnownType() string
	}
	if wkt, ok := f.Addr().Interface().(wkt); ok {
		switch wkt.XXX_WellKnownType() {
		case "Timestamp":
			if value == "null" {
				f.Field(0).SetInt(0)
				f.Field(1).SetInt(0)
				return nil
			}

			t, err := time.Parse(time.RFC3339Nano, value)
			if err != nil {
				return fmt.Errorf("bad Timestamp: %v", err)
			}
			f.Field(0).SetInt(int64(t.Unix()))
			f.Field(1).SetInt(int64(t.Nanosecond()))
			return nil
		case "DoubleValue":
			fallthrough
		case "FloatValue":
			float64Val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return fmt.Errorf("bad DoubleValue: %s", value)
			}
			f.Field(0).SetFloat(float64Val)
			return nil
		case "Int64Value":
			fallthrough
		case "Int32Value":
			int64Val, err := strconv.ParseInt(value, 10, 64)
			if err != nil {
				return fmt.Errorf("bad DoubleValue: %s", value)
			}
			f.Field(0).SetInt(int64Val)
			return nil
		case "UInt64Value":
			fallthrough
		case "UInt32Value":
			uint64Val, err := strconv.ParseUint(value, 10, 64)
			if err != nil {
				return fmt.Errorf("bad DoubleValue: %s", value)
			}
			f.Field(0).SetUint(uint64Val)
			return nil
		case "BoolValue":
			if value == "true" {
				f.Field(0).SetBool(true)
			} else if value == "false" {
				f.Field(0).SetBool(false)
			} else {
				return fmt.Errorf("bad BoolValue: %s", value)
			}
			return nil
		case "StringValue":
			f.Field(0).SetString(value)
			return nil
		}
	}

	// is the destination field an enumeration type?
	if enumValMap := proto.EnumValueMap(props.Enum); enumValMap != nil {
		return populateFieldEnum(f, value, enumValMap)
	}

	conv, ok := convFromType[f.Kind()]
	if !ok {
		return fmt.Errorf("unsupported field type %T", f)
	}
	result := conv.Call([]reflect.Value{reflect.ValueOf(value)})
	if err := result[1].Interface(); err != nil {
		return err.(error)
	}
	f.Set(result[0].Convert(f.Type()))
	return nil
}

func convertEnum(value string, t reflect.Type, enumValMap map[string]int32) (reflect.Value, error) {
	// see if it's an enumeration string
	if enumVal, ok := enumValMap[value]; ok {
		return reflect.ValueOf(enumVal).Convert(t), nil
	}

	// check for an integer that matches an enumeration value
	eVal, err := strconv.Atoi(value)
	if err != nil {
		return reflect.Value{}, fmt.Errorf("%s is not a valid %s", value, t)
	}
	for _, v := range enumValMap {
		if v == int32(eVal) {
			return reflect.ValueOf(eVal).Convert(t), nil
		}
	}
	return reflect.Value{}, fmt.Errorf("%s is not a valid %s", value, t)
}

func populateFieldEnum(f reflect.Value, value string, enumValMap map[string]int32) error {
	cval, err := convertEnum(value, f.Type(), enumValMap)
	if err != nil {
		return err
	}
	f.Set(cval)
	return nil
}

func populateFieldEnumRepeated(f reflect.Value, values []string, enumValMap map[string]int32) error {
	elemType := f.Type().Elem()
	f.Set(reflect.MakeSlice(f.Type(), len(values), len(values)).Convert(f.Type()))
	for i, v := range values {
		result, err := convertEnum(v, elemType, enumValMap)
		if err != nil {
			return err
		}
		f.Index(i).Set(result)
	}
	return nil
}

var (
	convFromType = map[reflect.Kind]reflect.Value{
		reflect.String:  reflect.ValueOf(String),
		reflect.Bool:    reflect.ValueOf(Bool),
		reflect.Float64: reflect.ValueOf(Float64),
		reflect.Float32: reflect.ValueOf(Float32),
		reflect.Int64:   reflect.ValueOf(Int64),
		reflect.Int32:   reflect.ValueOf(Int32),
		reflect.Uint64:  reflect.ValueOf(Uint64),
		reflect.Uint32:  reflect.ValueOf(Uint32),
		// TODO(yugui) Support []byte
	}
)
