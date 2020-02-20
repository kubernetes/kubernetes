package jsonutil

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/private/protocol"
)

// UnmarshalJSONError unmarshal's the reader's JSON document into the passed in
// type. The value to unmarshal the json document into must be a pointer to the
// type.
func UnmarshalJSONError(v interface{}, stream io.Reader) error {
	var errBuf bytes.Buffer
	body := io.TeeReader(stream, &errBuf)

	err := json.NewDecoder(body).Decode(v)
	if err != nil {
		msg := "failed decoding error message"
		if err == io.EOF {
			msg = "error message missing"
			err = nil
		}
		return awserr.NewUnmarshalError(err, msg, errBuf.Bytes())
	}

	return nil
}

// UnmarshalJSON reads a stream and unmarshals the results in object v.
func UnmarshalJSON(v interface{}, stream io.Reader) error {
	var out interface{}

	err := json.NewDecoder(stream).Decode(&out)
	if err == io.EOF {
		return nil
	} else if err != nil {
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
		case *time.Time:
			format := tag.Get("timestampFormat")
			if len(format) == 0 {
				format = protocol.ISO8601TimeFormatName
			}

			t, err := protocol.ParseTime(format, d)
			if err != nil {
				return err
			}
			value.Set(reflect.ValueOf(&t))
		case aws.JSONValue:
			// No need to use escaping as the value is a non-quoted string.
			v, err := protocol.DecodeJSONValue(d, protocol.NoEscape)
			if err != nil {
				return err
			}
			value.Set(reflect.ValueOf(v))
		default:
			return fmt.Errorf("unsupported value: %v (%s)", value.Interface(), value.Type())
		}
	case float64:
		switch value.Interface().(type) {
		case *int64:
			di := int64(d)
			value.Set(reflect.ValueOf(&di))
		case *float64:
			value.Set(reflect.ValueOf(&d))
		case *time.Time:
			// Time unmarshaled from a float64 can only be epoch seconds
			t := time.Unix(int64(d), 0).UTC()
			value.Set(reflect.ValueOf(&t))
		default:
			return fmt.Errorf("unsupported value: %v (%s)", value.Interface(), value.Type())
		}
	case bool:
		switch value.Interface().(type) {
		case *bool:
			value.Set(reflect.ValueOf(&d))
		default:
			return fmt.Errorf("unsupported value: %v (%s)", value.Interface(), value.Type())
		}
	default:
		return fmt.Errorf("unsupported JSON value (%v)", data)
	}
	return nil
}
