package jsonutil

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"reflect"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/private/protocol"
)

var millisecondsFloat = new(big.Float).SetInt64(1e3)

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

	decoder := json.NewDecoder(stream)
	decoder.UseNumber()
	err := decoder.Decode(&out)
	if err == io.EOF {
		return nil
	} else if err != nil {
		return err
	}

	return unmarshaler{}.unmarshalAny(reflect.ValueOf(v), out, "")
}

// UnmarshalJSONCaseInsensitive reads a stream and unmarshals the result into the
// object v. Ignores casing for structure members.
func UnmarshalJSONCaseInsensitive(v interface{}, stream io.Reader) error {
	var out interface{}

	decoder := json.NewDecoder(stream)
	decoder.UseNumber()
	err := decoder.Decode(&out)
	if err == io.EOF {
		return nil
	} else if err != nil {
		return err
	}

	return unmarshaler{
		caseInsensitive: true,
	}.unmarshalAny(reflect.ValueOf(v), out, "")
}

type unmarshaler struct {
	caseInsensitive bool
}

func (u unmarshaler) unmarshalAny(value reflect.Value, data interface{}, tag reflect.StructTag) error {
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
		return u.unmarshalStruct(value, data, tag)
	case "list":
		return u.unmarshalList(value, data, tag)
	case "map":
		return u.unmarshalMap(value, data, tag)
	default:
		return u.unmarshalScalar(value, data, tag)
	}
}

func (u unmarshaler) unmarshalStruct(value reflect.Value, data interface{}, tag reflect.StructTag) error {
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
		return u.unmarshalAny(value.FieldByName(payload), data, field.Tag)
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
		if u.caseInsensitive {
			if _, ok := mapData[name]; !ok {
				// Fallback to uncased name search if the exact name didn't match.
				for kn, v := range mapData {
					if strings.EqualFold(kn, name) {
						mapData[name] = v
					}
				}
			}
		}

		member := value.FieldByIndex(field.Index)
		err := u.unmarshalAny(member, mapData[name], field.Tag)
		if err != nil {
			return err
		}
	}
	return nil
}

func (u unmarshaler) unmarshalList(value reflect.Value, data interface{}, tag reflect.StructTag) error {
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
		err := u.unmarshalAny(value.Index(i), c, "")
		if err != nil {
			return err
		}
	}

	return nil
}

func (u unmarshaler) unmarshalMap(value reflect.Value, data interface{}, tag reflect.StructTag) error {
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

		u.unmarshalAny(vvalue, v, "")
		value.SetMapIndex(kvalue, vvalue)
	}

	return nil
}

func (u unmarshaler) unmarshalScalar(value reflect.Value, data interface{}, tag reflect.StructTag) error {

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
		case *float64:
			// These are regular strings when parsed by encoding/json's unmarshaler.
			switch {
			case strings.EqualFold(d, floatNaN):
				value.Set(reflect.ValueOf(aws.Float64(math.NaN())))
			case strings.EqualFold(d, floatInf):
				value.Set(reflect.ValueOf(aws.Float64(math.Inf(1))))
			case strings.EqualFold(d, floatNegInf):
				value.Set(reflect.ValueOf(aws.Float64(math.Inf(-1))))
			default:
				return fmt.Errorf("unknown JSON number value: %s", d)
			}
		default:
			return fmt.Errorf("unsupported value: %v (%s)", value.Interface(), value.Type())
		}
	case json.Number:
		switch value.Interface().(type) {
		case *int64:
			// Retain the old behavior where we would just truncate the float64
			// calling d.Int64() here could cause an invalid syntax error due to the usage of strconv.ParseInt
			f, err := d.Float64()
			if err != nil {
				return err
			}
			di := int64(f)
			value.Set(reflect.ValueOf(&di))
		case *float64:
			f, err := d.Float64()
			if err != nil {
				return err
			}
			value.Set(reflect.ValueOf(&f))
		case *time.Time:
			float, ok := new(big.Float).SetString(d.String())
			if !ok {
				return fmt.Errorf("unsupported float time representation: %v", d.String())
			}
			float = float.Mul(float, millisecondsFloat)
			ms, _ := float.Int64()
			t := time.Unix(0, ms*1e6).UTC()
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
