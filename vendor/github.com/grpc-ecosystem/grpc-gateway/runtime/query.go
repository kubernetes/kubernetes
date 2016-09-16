package runtime

import (
	"fmt"
	"net/url"
	"reflect"
	"strings"

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
	m = m.Elem()
	for i, fieldName := range fieldPath {
		isLast := i == len(fieldPath)-1
		if !isLast && m.Kind() != reflect.Struct {
			return fmt.Errorf("non-aggregate type in the mid of path: %s", strings.Join(fieldPath, "."))
		}
		f := fieldByProtoName(m, fieldName)
		if !f.IsValid() {
			grpclog.Printf("field not found in %T: %s", msg, strings.Join(fieldPath, "."))
			return nil
		}

		switch f.Kind() {
		case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64, reflect.String, reflect.Uint32, reflect.Uint64:
			m = f
		case reflect.Slice:
			// TODO(yugui) Support []byte
			if !isLast {
				return fmt.Errorf("unexpected repeated field in %s", strings.Join(fieldPath, "."))
			}
			return populateRepeatedField(f, values)
		case reflect.Ptr:
			if f.IsNil() {
				m = reflect.New(f.Type().Elem())
				f.Set(m)
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
	return populateField(m, values[0])
}

// fieldByProtoName looks up a field whose corresponding protobuf field name is "name".
// "m" must be a struct value. It returns zero reflect.Value if no such field found.
func fieldByProtoName(m reflect.Value, name string) reflect.Value {
	props := proto.GetProperties(m.Type())
	for _, p := range props.Prop {
		if p.OrigName == name {
			return m.FieldByName(p.Name)
		}
	}
	return reflect.Value{}
}

func populateRepeatedField(f reflect.Value, values []string) error {
	elemType := f.Type().Elem()
	conv, ok := convFromType[elemType.Kind()]
	if !ok {
		return fmt.Errorf("unsupported field type %s", elemType)
	}
	f.Set(reflect.MakeSlice(f.Type(), len(values), len(values)))
	for i, v := range values {
		result := conv.Call([]reflect.Value{reflect.ValueOf(v)})
		if err := result[1].Interface(); err != nil {
			return err.(error)
		}
		f.Index(i).Set(result[0])
	}
	return nil
}

func populateField(f reflect.Value, value string) error {
	conv, ok := convFromType[f.Kind()]
	if !ok {
		return fmt.Errorf("unsupported field type %T", f)
	}
	result := conv.Call([]reflect.Value{reflect.ValueOf(value)})
	if err := result[1].Interface(); err != nil {
		return err.(error)
	}
	f.Set(result[0])
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
