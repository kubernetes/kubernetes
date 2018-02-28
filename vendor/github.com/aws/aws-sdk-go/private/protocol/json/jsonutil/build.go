// Package jsonutil provides JSON serialization of AWS requests and responses.
package jsonutil

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/private/protocol"
)

var timeType = reflect.ValueOf(time.Time{}).Type()
var byteSliceType = reflect.ValueOf([]byte{}).Type()

// BuildJSON builds a JSON string for a given object v.
func BuildJSON(v interface{}) ([]byte, error) {
	var buf bytes.Buffer

	err := buildAny(reflect.ValueOf(v), &buf, "")
	return buf.Bytes(), err
}

func buildAny(value reflect.Value, buf *bytes.Buffer, tag reflect.StructTag) error {
	origVal := value
	value = reflect.Indirect(value)
	if !value.IsValid() {
		return nil
	}

	vtype := value.Type()

	t := tag.Get("type")
	if t == "" {
		switch vtype.Kind() {
		case reflect.Struct:
			// also it can't be a time object
			if value.Type() != timeType {
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
		return buildStruct(value, buf, tag)
	case "list":
		return buildList(value, buf, tag)
	case "map":
		return buildMap(value, buf, tag)
	default:
		return buildScalar(origVal, buf, tag)
	}
}

func buildStruct(value reflect.Value, buf *bytes.Buffer, tag reflect.StructTag) error {
	if !value.IsValid() {
		return nil
	}

	// unwrap payloads
	if payload := tag.Get("payload"); payload != "" {
		field, _ := value.Type().FieldByName(payload)
		tag = field.Tag
		value = elemOf(value.FieldByName(payload))

		if !value.IsValid() {
			return nil
		}
	}

	buf.WriteByte('{')

	t := value.Type()
	first := true
	for i := 0; i < t.NumField(); i++ {
		member := value.Field(i)

		// This allocates the most memory.
		// Additionally, we cannot skip nil fields due to
		// idempotency auto filling.
		field := t.Field(i)

		if field.PkgPath != "" {
			continue // ignore unexported fields
		}
		if field.Tag.Get("json") == "-" {
			continue
		}
		if field.Tag.Get("location") != "" {
			continue // ignore non-body elements
		}
		if field.Tag.Get("ignore") != "" {
			continue
		}

		if protocol.CanSetIdempotencyToken(member, field) {
			token := protocol.GetIdempotencyToken()
			member = reflect.ValueOf(&token)
		}

		if (member.Kind() == reflect.Ptr || member.Kind() == reflect.Slice || member.Kind() == reflect.Map) && member.IsNil() {
			continue // ignore unset fields
		}

		if first {
			first = false
		} else {
			buf.WriteByte(',')
		}

		// figure out what this field is called
		name := field.Name
		if locName := field.Tag.Get("locationName"); locName != "" {
			name = locName
		}

		writeString(name, buf)
		buf.WriteString(`:`)

		err := buildAny(member, buf, field.Tag)
		if err != nil {
			return err
		}

	}

	buf.WriteString("}")

	return nil
}

func buildList(value reflect.Value, buf *bytes.Buffer, tag reflect.StructTag) error {
	buf.WriteString("[")

	for i := 0; i < value.Len(); i++ {
		buildAny(value.Index(i), buf, "")

		if i < value.Len()-1 {
			buf.WriteString(",")
		}
	}

	buf.WriteString("]")

	return nil
}

type sortedValues []reflect.Value

func (sv sortedValues) Len() int           { return len(sv) }
func (sv sortedValues) Swap(i, j int)      { sv[i], sv[j] = sv[j], sv[i] }
func (sv sortedValues) Less(i, j int) bool { return sv[i].String() < sv[j].String() }

func buildMap(value reflect.Value, buf *bytes.Buffer, tag reflect.StructTag) error {
	buf.WriteString("{")

	sv := sortedValues(value.MapKeys())
	sort.Sort(sv)

	for i, k := range sv {
		if i > 0 {
			buf.WriteByte(',')
		}

		writeString(k.String(), buf)
		buf.WriteString(`:`)

		buildAny(value.MapIndex(k), buf, "")
	}

	buf.WriteString("}")

	return nil
}

func buildScalar(v reflect.Value, buf *bytes.Buffer, tag reflect.StructTag) error {
	// prevents allocation on the heap.
	scratch := [64]byte{}
	switch value := reflect.Indirect(v); value.Kind() {
	case reflect.String:
		writeString(value.String(), buf)
	case reflect.Bool:
		if value.Bool() {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case reflect.Int64:
		buf.Write(strconv.AppendInt(scratch[:0], value.Int(), 10))
	case reflect.Float64:
		f := value.Float()
		if math.IsInf(f, 0) || math.IsNaN(f) {
			return &json.UnsupportedValueError{Value: v, Str: strconv.FormatFloat(f, 'f', -1, 64)}
		}
		buf.Write(strconv.AppendFloat(scratch[:0], f, 'f', -1, 64))
	default:
		switch converted := value.Interface().(type) {
		case time.Time:
			buf.Write(strconv.AppendInt(scratch[:0], converted.UTC().Unix(), 10))
		case []byte:
			if !value.IsNil() {
				buf.WriteByte('"')
				if len(converted) < 1024 {
					// for small buffers, using Encode directly is much faster.
					dst := make([]byte, base64.StdEncoding.EncodedLen(len(converted)))
					base64.StdEncoding.Encode(dst, converted)
					buf.Write(dst)
				} else {
					// for large buffers, avoid unnecessary extra temporary
					// buffer space.
					enc := base64.NewEncoder(base64.StdEncoding, buf)
					enc.Write(converted)
					enc.Close()
				}
				buf.WriteByte('"')
			}
		case aws.JSONValue:
			str, err := protocol.EncodeJSONValue(converted, protocol.QuotedEscape)
			if err != nil {
				return fmt.Errorf("unable to encode JSONValue, %v", err)
			}
			buf.WriteString(str)
		default:
			return fmt.Errorf("unsupported JSON value %v (%s)", value.Interface(), value.Type())
		}
	}
	return nil
}

var hex = "0123456789abcdef"

func writeString(s string, buf *bytes.Buffer) {
	buf.WriteByte('"')
	for i := 0; i < len(s); i++ {
		if s[i] == '"' {
			buf.WriteString(`\"`)
		} else if s[i] == '\\' {
			buf.WriteString(`\\`)
		} else if s[i] == '\b' {
			buf.WriteString(`\b`)
		} else if s[i] == '\f' {
			buf.WriteString(`\f`)
		} else if s[i] == '\r' {
			buf.WriteString(`\r`)
		} else if s[i] == '\t' {
			buf.WriteString(`\t`)
		} else if s[i] == '\n' {
			buf.WriteString(`\n`)
		} else if s[i] < 32 {
			buf.WriteString("\\u00")
			buf.WriteByte(hex[s[i]>>4])
			buf.WriteByte(hex[s[i]&0xF])
		} else {
			buf.WriteByte(s[i])
		}
	}
	buf.WriteByte('"')
}

// Returns the reflection element of a value, if it is a pointer.
func elemOf(value reflect.Value) reflect.Value {
	for value.Kind() == reflect.Ptr {
		value = value.Elem()
	}
	return value
}
