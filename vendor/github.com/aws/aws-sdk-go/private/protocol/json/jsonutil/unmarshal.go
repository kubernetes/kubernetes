package jsonutil

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/private/protocol"
)

// UnmarshalJSON reads a stream and unmarshals the results in object v.
func UnmarshalJSON(v interface{}, stream io.Reader) error {
	var out interface{}

	b, err := ioutil.ReadAll(stream)
	if err != nil {
		return err
	}

	if len(b) == 0 {
		return nil
	}

	if err := json.Unmarshal(b, &out); err != nil {
		return err
	}

	return unmarshalAny(reflect.ValueOf(v), out, "")
}

func unmarshalAny(value reflect.Value, data interface{}, tag reflect.StructTag) error {
	vtype := value.Type()
	if vtype.Kind() == reflect.Ptr {
		vtype = vtype.Elem() // check kind of actual element type
	}

	t := tag.Get("type")
	if t == "" {
		switch vtype.Kind() {
		case reflect.Struct:
			// also it can't be a time object
			if _, ok := value.Interface().(*time.Time); !ok {
				t = "structure"
			}
		case reflect.Slice:
			// also it can't be a byte slice
			if _, ok := value.Interface().([]byte); !ok {
				t = "list"
			}
		case reflect.Map:
			// cannot be a JSONValue map
			if _, ok := value.Interface().(aws.JSONValue); !ok {
				t = "map"
			}
		}
	}

	switch t {
	case "structure":
		if field, ok := vtype.FieldByName("_"); ok {
			tag = field.Tag
		}
		return unmarshalStruct(value, data, tag)
	case "list":
		return unmarshalList(value, data, tag)
	case "map":
		return unmarshalMap(value, data, tag)
	default:
		return unmarshalScalar(value, data, tag)
	}
}

func unmarshalStruct(value reflect.Value, data interface{}, tag reflect.StructTag) error {
	if data == nil {
		return nil
	}
	mapData, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Errorf("JSON value is not a structure (%#v)", data)
	}

	t := value.Type()
	if value.Kind() == reflect.Ptr {
		if value.IsNil() { // create the structure if it's nil
			s := reflect.New(value.Type().Elem())
			value.Set(s)
			value = s
		}

		value = value.Elem()
		t = t.Elem()
	}

	// unwrap any payloads
	if payload := tag.Get("payload"); payload != "" {
		field, _ := t.FieldByName(payload)
		return unmarshalAny(value.FieldByName(payload), data, field.Tag)
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" {
			continue // ignore unexported fields
		}

		// figure out what this field is called
		name := field.Name
		if locName := field.Tag.Get("locationName"); locName != "" {
			name = locName
		}

		member := value.FieldByIndex(field.Index)
		err := unmarshalAny(member, mapData[name], field.Tag)
		if err != nil {
			return err
		}
	}
	return nil
}

func unmarshalList(value reflect.Value, data interface{}, tag reflect.StructTag) error {
	if data == nil {
		return nil
	}
	listData, ok := data.([]interface{})
	if !ok {
		return fmt.Errorf("JSON value is not a list (%#v)", data)
	}

	if value.IsNil() {
		l := len(listData)
		value.Set(reflect.MakeSlice(value.Type(), l, l))
	}

	for i, c := range listData {
		err := unmarshalAny(value.Index(i), c, "")
		if err != nil {
			return err
		}
	}

	return nil
}

func unmarshalMap(value reflect.Value, data interface{}, tag reflect.StructTag) error {
	if data == nil {
		return nil
	}
	mapData, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Errorf("JSON value is not a map (%#v)", data)
	}

	if value.IsNil() {
		value.Set(reflect.MakeMap(value.Type()))
	}

	for k, v := range mapData {
		kvalue := reflect.ValueOf(k)
		vvalue := reflect.New(value.Type().Elem()).Elem()

		unmarshalAny(vvalue, v, "")
		value.SetMapIndex(kvalue, vvalue)
	}

	return nil
}

func unmarshalScalar(value reflect.Value, data interface{}, tag reflect.StructTag) error {
	errf := func() error {
		return fmt.Errorf("unsupported value: %v (%s)", value.Interface(), value.Type())
	}

	switch d := data.(type) {
	case nil:
		return nil // nothing to do here
	case string:
		switch value.Interface().(type) {
		case *string:
			value.Set(reflect.ValueOf(&d))
		case []byte:
			b, err := base64.StdEncoding.DecodeString(d)
			if err != nil {
				return err
			}
			value.Set(reflect.ValueOf(b))
		case aws.JSONValue:
			// No need to use escaping as the value is a non-quoted string.
			v, err := protocol.DecodeJSONValue(d, protocol.NoEscape)
			if err != nil {
				return err
			}
			value.Set(reflect.ValueOf(v))
		default:
			return errf()
		}
	case float64:
		switch value.Interface().(type) {
		case *int64:
			di := int64(d)
			value.Set(reflect.ValueOf(&di))
		case *float64:
			value.Set(reflect.ValueOf(&d))
		case *time.Time:
			t := time.Unix(int64(d), 0).UTC()
			value.Set(reflect.ValueOf(&t))
		default:
			return errf()
		}
	case bool:
		switch value.Interface().(type) {
		case *bool:
			value.Set(reflect.ValueOf(&d))
		default:
			return errf()
		}
	default:
		return fmt.Errorf("unsupported JSON value (%v)", data)
	}
	return nil
}
