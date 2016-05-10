// Package jsonutil provides JSON serialisation of AWS requests and responses.
package jsonutil

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"time"

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
			t = "map"
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
		return buildScalar(value, buf, tag)
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

	var sv sortedValues = value.MapKeys()
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

func buildScalar(value reflect.Value, buf *bytes.Buffer, tag reflect.StructTag) error {
	switch value.Kind() {
	case reflect.String:
		writeString(value.String(), buf)
	case reflect.Bool:
		buf.WriteString(strconv.FormatBool(value.Bool()))
	case reflect.Int64:
		buf.WriteString(strconv.FormatInt(value.Int(), 10))
	case reflect.Float64:
		buf.WriteString(strconv.FormatFloat(value.Float(), 'f', -1, 64))
	default:
		switch value.Type() {
		case timeType:
			converted := value.Interface().(time.Time)
			buf.WriteString(strconv.FormatInt(converted.UTC().Unix(), 10))
		case byteSliceType:
			if !value.IsNil() {
				converted := value.Interface().([]byte)
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
		default:
			return fmt.Errorf("unsupported JSON value %v (%s)", value.Interface(), value.Type())
		}
	}
	return nil
}

func writeString(s string, buf *bytes.Buffer) {
	buf.WriteByte('"')
	for _, r := range s {
		if r == '"' {
			buf.WriteString(`\"`)
		} else if r == '\\' {
			buf.WriteString(`\\`)
		} else if r == '\b' {
			buf.WriteString(`\b`)
		} else if r == '\f' {
			buf.WriteString(`\f`)
		} else if r == '\r' {
			buf.WriteString(`\r`)
		} else if r == '\t' {
			buf.WriteString(`\t`)
		} else if r == '\n' {
			buf.WriteString(`\n`)
		} else if r < 32 {
			fmt.Fprintf(buf, "\\u%0.4x", r)
		} else {
			buf.WriteRune(r)
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
