package runtime

import (
	"encoding/base64"
	"fmt"
	"net/url"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/grpc-ecosystem/grpc-gateway/utilities"
	"google.golang.org/grpc/grpclog"
)

var valuesKeyRegexp = regexp.MustCompile("^(.*)\\[(.*)\\]$")

var currentQueryParser QueryParameterParser = &defaultQueryParser{}

// QueryParameterParser defines interface for all query parameter parsers
type QueryParameterParser interface {
	Parse(msg proto.Message, values url.Values, filter *utilities.DoubleArray) error
}

// PopulateQueryParameters parses query parameters
// into "msg" using current query parser
func PopulateQueryParameters(msg proto.Message, values url.Values, filter *utilities.DoubleArray) error {
	return currentQueryParser.Parse(msg, values, filter)
}

type defaultQueryParser struct{}

// Parse populates "values" into "msg".
// A value is ignored if its key starts with one of the elements in "filter".
func (*defaultQueryParser) Parse(msg proto.Message, values url.Values, filter *utilities.DoubleArray) error {
	for key, values := range values {
		match := valuesKeyRegexp.FindStringSubmatch(key)
		if len(match) == 3 {
			key = match[1]
			values = append([]string{match[2]}, values...)
		}
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
			grpclog.Infof("field not found in %T: %s", msg, strings.Join(fieldPath, "."))
			return nil
		}

		switch f.Kind() {
		case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64, reflect.String, reflect.Uint32, reflect.Uint64:
			if !isLast {
				return fmt.Errorf("unexpected nested field %s in %s", fieldPath[i+1], strings.Join(fieldPath[:i+1], "."))
			}
			m = f
		case reflect.Slice:
			if !isLast {
				return fmt.Errorf("unexpected repeated field in %s", strings.Join(fieldPath, "."))
			}
			// Handle []byte
			if f.Type().Elem().Kind() == reflect.Uint8 {
				m = f
				break
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
		case reflect.Map:
			if !isLast {
				return fmt.Errorf("unexpected nested field %s in %s", fieldPath[i+1], strings.Join(fieldPath[:i+1], "."))
			}
			return populateMapField(f, values, props)
		default:
			return fmt.Errorf("unexpected type %s in %T", f.Type(), msg)
		}
	}
	switch len(values) {
	case 0:
		return fmt.Errorf("no value of field: %s", strings.Join(fieldPath, "."))
	case 1:
	default:
		grpclog.Infof("too many field values: %s", strings.Join(fieldPath, "."))
	}
	return populateField(m, values[0], props)
}

// fieldByProtoName looks up a field whose corresponding protobuf field name is "name".
// "m" must be a struct value. It returns zero reflect.Value if no such field found.
func fieldByProtoName(m reflect.Value, name string) (reflect.Value, *proto.Properties, error) {
	props := proto.GetProperties(m.Type())

	// look up field name in oneof map
	for _, op := range props.OneofTypes {
		if name == op.Prop.OrigName || name == op.Prop.JSONName {
			v := reflect.New(op.Type.Elem())
			field := m.Field(op.Field)
			if !field.IsNil() {
				return reflect.Value{}, nil, fmt.Errorf("field already set for %s oneof", props.Prop[op.Field].OrigName)
			}
			field.Set(v)
			return v.Elem().Field(0), op.Prop, nil
		}
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

func populateMapField(f reflect.Value, values []string, props *proto.Properties) error {
	if len(values) != 2 {
		return fmt.Errorf("more than one value provided for key %s in map %s", values[0], props.Name)
	}

	key, value := values[0], values[1]
	keyType := f.Type().Key()
	valueType := f.Type().Elem()
	if f.IsNil() {
		f.Set(reflect.MakeMap(f.Type()))
	}

	keyConv, ok := convFromType[keyType.Kind()]
	if !ok {
		return fmt.Errorf("unsupported key type %s in map %s", keyType, props.Name)
	}
	valueConv, ok := convFromType[valueType.Kind()]
	if !ok {
		return fmt.Errorf("unsupported value type %s in map %s", valueType, props.Name)
	}

	keyV := keyConv.Call([]reflect.Value{reflect.ValueOf(key)})
	if err := keyV[1].Interface(); err != nil {
		return err.(error)
	}
	valueV := valueConv.Call([]reflect.Value{reflect.ValueOf(value)})
	if err := valueV[1].Interface(); err != nil {
		return err.(error)
	}

	f.SetMapIndex(keyV[0].Convert(keyType), valueV[0].Convert(valueType))

	return nil
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
	i := f.Addr().Interface()

	// Handle protobuf well known types
	var name string
	switch m := i.(type) {
	case interface{ XXX_WellKnownType() string }:
		name = m.XXX_WellKnownType()
	case proto.Message:
		const wktPrefix = "google.protobuf."
		if fullName := proto.MessageName(m); strings.HasPrefix(fullName, wktPrefix) {
			name = fullName[len(wktPrefix):]
		}
	}
	switch name {
	case "Timestamp":
		if value == "null" {
			f.FieldByName("Seconds").SetInt(0)
			f.FieldByName("Nanos").SetInt(0)
			return nil
		}

		t, err := time.Parse(time.RFC3339Nano, value)
		if err != nil {
			return fmt.Errorf("bad Timestamp: %v", err)
		}
		f.FieldByName("Seconds").SetInt(int64(t.Unix()))
		f.FieldByName("Nanos").SetInt(int64(t.Nanosecond()))
		return nil
	case "Duration":
		if value == "null" {
			f.FieldByName("Seconds").SetInt(0)
			f.FieldByName("Nanos").SetInt(0)
			return nil
		}
		d, err := time.ParseDuration(value)
		if err != nil {
			return fmt.Errorf("bad Duration: %v", err)
		}

		ns := d.Nanoseconds()
		s := ns / 1e9
		ns %= 1e9
		f.FieldByName("Seconds").SetInt(s)
		f.FieldByName("Nanos").SetInt(ns)
		return nil
	case "DoubleValue":
		fallthrough
	case "FloatValue":
		float64Val, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf("bad DoubleValue: %s", value)
		}
		f.FieldByName("Value").SetFloat(float64Val)
		return nil
	case "Int64Value":
		fallthrough
	case "Int32Value":
		int64Val, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return fmt.Errorf("bad DoubleValue: %s", value)
		}
		f.FieldByName("Value").SetInt(int64Val)
		return nil
	case "UInt64Value":
		fallthrough
	case "UInt32Value":
		uint64Val, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return fmt.Errorf("bad DoubleValue: %s", value)
		}
		f.FieldByName("Value").SetUint(uint64Val)
		return nil
	case "BoolValue":
		if value == "true" {
			f.FieldByName("Value").SetBool(true)
		} else if value == "false" {
			f.FieldByName("Value").SetBool(false)
		} else {
			return fmt.Errorf("bad BoolValue: %s", value)
		}
		return nil
	case "StringValue":
		f.FieldByName("Value").SetString(value)
		return nil
	case "BytesValue":
		bytesVal, err := base64.StdEncoding.DecodeString(value)
		if err != nil {
			return fmt.Errorf("bad BytesValue: %s", value)
		}
		f.FieldByName("Value").SetBytes(bytesVal)
		return nil
	case "FieldMask":
		p := f.FieldByName("Paths")
		for _, v := range strings.Split(value, ",") {
			if v != "" {
				p.Set(reflect.Append(p, reflect.ValueOf(v)))
			}
		}
		return nil
	}

	// Handle Time and Duration stdlib types
	switch t := i.(type) {
	case *time.Time:
		pt, err := time.Parse(time.RFC3339Nano, value)
		if err != nil {
			return fmt.Errorf("bad Timestamp: %v", err)
		}
		*t = pt
		return nil
	case *time.Duration:
		d, err := time.ParseDuration(value)
		if err != nil {
			return fmt.Errorf("bad Duration: %v", err)
		}
		*t = d
		return nil
	}

	// is the destination field an enumeration type?
	if enumValMap := proto.EnumValueMap(props.Enum); enumValMap != nil {
		return populateFieldEnum(f, value, enumValMap)
	}

	conv, ok := convFromType[f.Kind()]
	if !ok {
		return fmt.Errorf("field type %T is not supported in query parameters", i)
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
		reflect.Slice:   reflect.ValueOf(Bytes),
	}
)
