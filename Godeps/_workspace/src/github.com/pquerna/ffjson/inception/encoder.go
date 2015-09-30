/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ffjsoninception

import (
	"fmt"
	"github.com/pquerna/ffjson/shared"
	"reflect"
)

func typeInInception(ic *Inception, typ reflect.Type, f shared.Feature) bool {
	for _, v := range ic.objs {
		if v.Typ == typ {
			return v.Options.HasFeature(f)
		}
		if typ.Kind() == reflect.Ptr {
			if v.Typ == typ.Elem() {
				return v.Options.HasFeature(f)
			}
		}
	}

	return false
}

func getOmitEmpty(ic *Inception, sf *StructField) string {
	ptname := "mj." + sf.Name
	if sf.Pointer {
		ptname = "*" + ptname
		return "if true {\n"
	}
	switch sf.Typ.Kind() {

	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return "if len(" + ptname + ") != 0 {" + "\n"

	case reflect.Int,
		reflect.Int8,
		reflect.Int16,
		reflect.Int32,
		reflect.Int64,
		reflect.Uint,
		reflect.Uint8,
		reflect.Uint16,
		reflect.Uint32,
		reflect.Uint64,
		reflect.Uintptr,
		reflect.Float32,
		reflect.Float64:
		return "if " + ptname + " != 0 {" + "\n"

	case reflect.Bool:
		return "if " + ptname + " != false {" + "\n"

	case reflect.Interface, reflect.Ptr:
		return "if " + ptname + " != nil {" + "\n"

	default:
		// TODO(pquerna): fix types
		return "if true {" + "\n"
	}
}

func getMapValue(ic *Inception, name string, typ reflect.Type, ptr bool, forceString bool) string {
	var out = ""

	if typ.Key().Kind() != reflect.String {
		out += fmt.Sprintf("/* Falling back. type=%v kind=%v */\n", typ, typ.Kind())
		out += ic.q.Flush()
		out += "err = buf.Encode(" + name + ")" + "\n"
		out += "if err != nil {" + "\n"
		out += "  return err" + "\n"
		out += "}" + "\n"
		return out
	}

	var elemKind reflect.Kind
	elemKind = typ.Elem().Kind()

	switch elemKind {
	case reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32,
		reflect.Float64,
		reflect.Bool:

		ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true

		out += "if " + name + " == nil  {" + "\n"
		ic.q.Write("null")
		out += ic.q.GetQueued()
		ic.q.DeleteLast()
		out += "} else {" + "\n"
		out += ic.q.WriteFlush("{ ")
		out += "  for key, value := range " + name + " {" + "\n"
		out += "    fflib.WriteJsonString(buf, key)" + "\n"
		out += "    buf.WriteString(`:`)" + "\n"
		out += getGetInnerValue(ic, "value", typ.Elem(), false, forceString)
		out += "    buf.WriteByte(',')" + "\n"
		out += "  }" + "\n"
		out += "buf.Rewind(1)" + "\n"
		out += ic.q.WriteFlush("}")
		out += "}" + "\n"

	default:
		out += ic.q.Flush()
		out += fmt.Sprintf("/* Falling back. type=%v kind=%v */\n", typ, typ.Kind())
		out += "err = buf.Encode(" + name + ")" + "\n"
		out += "if err != nil {" + "\n"
		out += "  return err" + "\n"
		out += "}" + "\n"
	}
	return out
}

func getGetInnerValue(ic *Inception, name string, typ reflect.Type, ptr bool, forceString bool) string {
	var out = ""

	// Flush if not bool or maps
	if typ.Kind() != reflect.Bool && typ.Kind() != reflect.Map && typ.Kind() != reflect.Struct {
		out += ic.q.Flush()
	}

	if typ.Implements(marshalerFasterType) ||
		reflect.PtrTo(typ).Implements(marshalerFasterType) ||
		typeInInception(ic, typ, shared.MustEncoder) ||
		typ.Implements(marshalerType) ||
		reflect.PtrTo(typ).Implements(marshalerType) {

		out += ic.q.Flush()
		out += tplStr(encodeTpl["handleMarshaler"], handleMarshaler{
			IC:             ic,
			Name:           name,
			MarshalJSONBuf: typ.Implements(marshalerFasterType) || reflect.PtrTo(typ).Implements(marshalerFasterType) || typeInInception(ic, typ, shared.MustEncoder),
			Marshaler:      typ.Implements(marshalerType) || reflect.PtrTo(typ).Implements(marshalerType),
		})
		return out
	}

	ptname := name
	if ptr {
		ptname = "*" + name
	}

	switch typ.Kind() {
	case reflect.Int,
		reflect.Int8,
		reflect.Int16,
		reflect.Int32,
		reflect.Int64:
		ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true
		out += "fflib.FormatBits2(buf, uint64(" + ptname + "), 10, " + ptname + " < 0)" + "\n"
	case reflect.Uint,
		reflect.Uint8,
		reflect.Uint16,
		reflect.Uint32,
		reflect.Uint64,
		reflect.Uintptr:
		ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true
		out += "fflib.FormatBits2(buf, uint64(" + ptname + "), 10, false)" + "\n"
	case reflect.Float32:
		ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true
		out += "fflib.AppendFloat(buf, float64(" + ptname + "), 'g', -1, 32)" + "\n"
	case reflect.Float64:
		ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true
		out += "fflib.AppendFloat(buf, float64(" + ptname + "), 'g', -1, 64)" + "\n"
	case reflect.Array,
		reflect.Slice:

		// Arrays cannot be nil
		if typ.Kind() != reflect.Array {
			out += "if " + name + "!= nil {" + "\n"
		}
		// Array and slice values encode as JSON arrays, except that
		// []byte encodes as a base64-encoded string, and a nil slice
		// encodes as the null JSON object.
		if typ.Kind() == reflect.Slice && typ.Elem().Kind() == reflect.Uint8 {
			ic.OutputImports[`"encoding/base64"`] = true

			out += "buf.WriteString(`\"`)" + "\n"
			out += `{` + "\n"
			out += `enc := base64.NewEncoder(base64.StdEncoding, buf)` + "\n"
			if typ.Elem().Name() != "byte" {
				ic.OutputImports[`"reflect"`] = true
				out += `enc.Write(reflect.Indirect(reflect.ValueOf(` + ptname + `)).Bytes())` + "\n"

			} else {
				out += `enc.Write(` + ptname + `)` + "\n"
			}
			out += `enc.Close()` + "\n"
			out += `}` + "\n"
			out += "buf.WriteString(`\"`)" + "\n"
		} else {
			out += "buf.WriteString(`[`)" + "\n"
			out += "for i, v := range " + name + "{" + "\n"
			out += "if i != 0 {" + "\n"
			out += "buf.WriteString(`,`)" + "\n"
			out += "}" + "\n"
			out += getGetInnerValue(ic, "v", typ.Elem(), false, false)
			out += "}" + "\n"
			out += "buf.WriteString(`]`)" + "\n"
		}
		if typ.Kind() != reflect.Array {
			out += "} else {" + "\n"
			out += "buf.WriteString(`null`)" + "\n"
			out += "}" + "\n"
		}
	case reflect.String:
		ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true
		if forceString {
			// Forcestring on strings does double-escaping of the entire value.
			// We create a temporary buffer, encode to that an re-encode it.
			out += "tmpbuf := fflib.Buffer{}" + "\n"
			out += "tmpbuf.Grow(len(" + ptname + ") + 16)" + "\n"
			out += "fflib.WriteJsonString(&tmpbuf, string(" + ptname + "))" + "\n"
			out += "fflib.WriteJsonString(buf, string( tmpbuf.Bytes() " + `))` + "\n"
		} else {
			out += "fflib.WriteJsonString(buf, string(" + ptname + "))" + "\n"
		}
	case reflect.Ptr:
		out += "if " + name + "!= nil {" + "\n"
		switch typ.Elem().Kind() {
		case reflect.Struct:
			out += getGetInnerValue(ic, name, typ.Elem(), false, false)
		default:
			out += getGetInnerValue(ic, "*"+name, typ.Elem(), false, false)
		}
		out += "} else {" + "\n"
		out += "buf.WriteString(`null`)" + "\n"
		out += "}" + "\n"
	case reflect.Bool:
		out += "if " + ptname + " {" + "\n"
		ic.q.Write("true")
		out += ic.q.GetQueued()
		out += "} else {" + "\n"
		// Delete 'true'
		ic.q.DeleteLast()
		out += ic.q.WriteFlush("false")
		out += "}" + "\n"
	case reflect.Interface:
		out += fmt.Sprintf("/* Interface types must use runtime reflection. type=%v kind=%v */\n", typ, typ.Kind())
		out += "err = buf.Encode(" + name + ")" + "\n"
		out += "if err != nil {" + "\n"
		out += "  return err" + "\n"
		out += "}" + "\n"
	case reflect.Map:
		out += getMapValue(ic, ptname, typ, ptr, forceString)
	case reflect.Struct:
		if typ.Name() == "" {
			ic.q.Write("{")
			ic.q.Write(" ")
			out += fmt.Sprintf("/* Inline struct. type=%v kind=%v */\n", typ, typ.Kind())
			newV := reflect.Indirect(reflect.New(typ)).Interface()
			fields := extractFields(newV)

			// Output all fields
			for _, field := range fields {
				// Adjust field name
				field.Name = name + "." + field.Name
				out += getField(ic, field, "")
			}

			if lastConditional(fields) {
				out += ic.q.Flush()
				out += `buf.Rewind(1)` + "\n"
			} else {
				ic.q.DeleteLast()
			}
			out += ic.q.WriteFlush("}")
		} else {
			out += fmt.Sprintf("/* Struct fall back. type=%v kind=%v */\n", typ, typ.Kind())
			out += ic.q.Flush()
			if ptr {
				out += "err = buf.Encode(" + name + ")" + "\n"
			} else {
				// We send pointer to avoid copying entire struct
				out += "err = buf.Encode(&" + name + ")" + "\n"
			}
			out += "if err != nil {" + "\n"
			out += "  return err" + "\n"
			out += "}" + "\n"
		}
	default:
		out += fmt.Sprintf("/* Falling back. type=%v kind=%v */\n", typ, typ.Kind())
		out += "err = buf.Encode(" + name + ")" + "\n"
		out += "if err != nil {" + "\n"
		out += "  return err" + "\n"
		out += "}" + "\n"
	}

	return out
}

func getValue(ic *Inception, sf *StructField, prefix string) string {
	closequote := false
	if sf.ForceString {
		switch sf.Typ.Kind() {
		case reflect.Int,
			reflect.Int8,
			reflect.Int16,
			reflect.Int32,
			reflect.Int64,
			reflect.Uint,
			reflect.Uint8,
			reflect.Uint16,
			reflect.Uint32,
			reflect.Uint64,
			reflect.Uintptr,
			reflect.Float32,
			reflect.Float64,
			reflect.Bool:
			ic.q.Write(`"`)
			closequote = true
		}
	}
	out := getGetInnerValue(ic, prefix+sf.Name, sf.Typ, sf.Pointer, sf.ForceString)
	if closequote {
		if sf.Pointer {
			out += ic.q.WriteFlush(`"`)
		} else {
			ic.q.Write(`"`)
		}
	}

	return out
}

func p2(v uint32) uint32 {
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v++
	return v
}

func getTypeSize(t reflect.Type) uint32 {
	switch t.Kind() {
	case reflect.String:
		// TODO: consider runtime analysis.
		return 32
	case reflect.Array, reflect.Map, reflect.Slice:
		// TODO: consider runtime analysis.
		return 4 * getTypeSize(t.Elem())
	case reflect.Int,
		reflect.Int8,
		reflect.Int16,
		reflect.Int32,
		reflect.Uint,
		reflect.Uint8,
		reflect.Uint16,
		reflect.Uint32:
		return 8
	case reflect.Int64,
		reflect.Uint64,
		reflect.Uintptr:
		return 16
	case reflect.Float32,
		reflect.Float64:
		return 16
	case reflect.Bool:
		return 4
	case reflect.Ptr:
		return getTypeSize(t.Elem())
	default:
		return 16
	}
}

func getTotalSize(si *StructInfo) uint32 {
	rv := uint32(si.Typ.Size())
	for _, f := range si.Fields {
		rv += getTypeSize(f.Typ)
	}
	return rv
}

func getBufGrowSize(si *StructInfo) uint32 {

	// TOOD(pquerna): automatically calc a better grow size based on history
	// of a struct.
	return p2(getTotalSize(si))
}

func isIntish(t reflect.Type) bool {
	if t.Kind() >= reflect.Int && t.Kind() <= reflect.Uintptr {
		return true
	}
	if t.Kind() == reflect.Array || t.Kind() == reflect.Slice || t.Kind() == reflect.Ptr {
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			// base64 special case.
			return false
		} else {
			return isIntish(t.Elem())
		}
	}
	return false
}

func getField(ic *Inception, f *StructField, prefix string) string {
	out := ""
	if f.OmitEmpty {
		out += ic.q.Flush()
		if f.Pointer {
			out += "if " + prefix + f.Name + " != nil {" + "\n"
		}
		out += getOmitEmpty(ic, f)
	}

	if f.Pointer && !f.OmitEmpty {
		// Pointer values encode as the value pointed to. A nil pointer encodes as the null JSON object.
		out += "if " + prefix + f.Name + " != nil {" + "\n"
	}

	// JsonName is already escaped and quoted.
	// getInnervalue should flush
	ic.q.Write(f.JsonName + ":")
	// We save a copy in case we need it
	t := ic.q

	out += getValue(ic, f, prefix)
	ic.q.Write(",")

	if f.Pointer && !f.OmitEmpty {
		out += "} else {" + "\n"
		out += t.WriteFlush("null")
		out += "}" + "\n"
	}

	if f.OmitEmpty {
		out += ic.q.Flush()
		if f.Pointer {
			out += "}" + "\n"
		}
		out += "}" + "\n"
	}
	return out
}

// We check if the last field is conditional.
func lastConditional(fields []*StructField) bool {
	if len(fields) > 0 {
		f := fields[len(fields)-1]
		return f.OmitEmpty
	}
	return false
}

func CreateMarshalJSON(ic *Inception, si *StructInfo) error {
	conditionalWrites := lastConditional(si.Fields)
	out := ""

	out += `func (mj *` + si.Name + `) MarshalJSON() ([]byte, error) {` + "\n"
	out += `var buf fflib.Buffer` + "\n"

	out += `err := mj.MarshalJSONBuf(&buf)` + "\n"
	out += `if err != nil {` + "\n"
	out += "  return nil, err" + "\n"
	out += `}` + "\n"
	out += `return buf.Bytes(), nil` + "\n"
	out += `}` + "\n"

	out += `func (mj *` + si.Name + `) MarshalJSONBuf(buf fflib.EncodingBuffer) (error) {` + "\n"
	out += `var err error` + "\n"
	out += `var obj []byte` + "\n"
	out += `_ = obj` + "\n"
	out += `_ = err` + "\n"

	ic.q.Write("{")

	// The extra space is inserted here.
	// If nothing is written to the field this will be deleted
	// instead of the last comma.
	if conditionalWrites || len(si.Fields) == 0 {
		ic.q.Write(" ")
	}

	for _, f := range si.Fields {
		out += getField(ic, f, "mj.")
	}

	// Handling the last comma is tricky.
	// If the last field has omitempty, conditionalWrites is set.
	// If something has been written, we delete the last comma,
	// by backing up the buffer, otherwise it will delete a space.
	if conditionalWrites {
		out += ic.q.Flush()
		out += `buf.Rewind(1)` + "\n"
	} else {
		ic.q.DeleteLast()
	}

	out += ic.q.WriteFlush("}")
	out += `return nil` + "\n"
	out += `}` + "\n"
	ic.OutputFuncs = append(ic.OutputFuncs, out)
	return nil
}
