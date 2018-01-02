package gen

import (
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/mailru/easyjson"
)

func (g *Generator) getEncoderName(t reflect.Type) string {
	return g.functionName("encode", t)
}

var primitiveEncoders = map[reflect.Kind]string{
	reflect.String:  "out.String(string(%v))",
	reflect.Bool:    "out.Bool(bool(%v))",
	reflect.Int:     "out.Int(int(%v))",
	reflect.Int8:    "out.Int8(int8(%v))",
	reflect.Int16:   "out.Int16(int16(%v))",
	reflect.Int32:   "out.Int32(int32(%v))",
	reflect.Int64:   "out.Int64(int64(%v))",
	reflect.Uint:    "out.Uint(uint(%v))",
	reflect.Uint8:   "out.Uint8(uint8(%v))",
	reflect.Uint16:  "out.Uint16(uint16(%v))",
	reflect.Uint32:  "out.Uint32(uint32(%v))",
	reflect.Uint64:  "out.Uint64(uint64(%v))",
	reflect.Float32: "out.Float32(float32(%v))",
	reflect.Float64: "out.Float64(float64(%v))",
}

var primitiveStringEncoders = map[reflect.Kind]string{
	reflect.Int:    "out.IntStr(int(%v))",
	reflect.Int8:   "out.Int8Str(int8(%v))",
	reflect.Int16:  "out.Int16Str(int16(%v))",
	reflect.Int32:  "out.Int32Str(int32(%v))",
	reflect.Int64:  "out.Int64Str(int64(%v))",
	reflect.Uint:   "out.UintStr(uint(%v))",
	reflect.Uint8:  "out.Uint8Str(uint8(%v))",
	reflect.Uint16: "out.Uint16Str(uint16(%v))",
	reflect.Uint32: "out.Uint32Str(uint32(%v))",
	reflect.Uint64: "out.Uint64Str(uint64(%v))",
}

// fieldTags contains parsed version of json struct field tags.
type fieldTags struct {
	name string

	omit        bool
	omitEmpty   bool
	noOmitEmpty bool
	asString    bool
	required    bool
}

// parseFieldTags parses the json field tag into a structure.
func parseFieldTags(f reflect.StructField) fieldTags {
	var ret fieldTags

	for i, s := range strings.Split(f.Tag.Get("json"), ",") {
		switch {
		case i == 0 && s == "-":
			ret.omit = true
		case i == 0:
			ret.name = s
		case s == "omitempty":
			ret.omitEmpty = true
		case s == "!omitempty":
			ret.noOmitEmpty = true
		case s == "string":
			ret.asString = true
		case s == "required":
			ret.required = true
		}
	}

	return ret
}

// genTypeEncoder generates code that encodes in of type t into the writer, but uses marshaler interface if implemented by t.
func (g *Generator) genTypeEncoder(t reflect.Type, in string, tags fieldTags, indent int) error {
	ws := strings.Repeat("  ", indent)

	marshalerIface := reflect.TypeOf((*easyjson.Marshaler)(nil)).Elem()
	if reflect.PtrTo(t).Implements(marshalerIface) {
		fmt.Fprintln(g.out, ws+"("+in+").MarshalEasyJSON(out)")
		return nil
	}

	marshalerIface = reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	if reflect.PtrTo(t).Implements(marshalerIface) {
		fmt.Fprintln(g.out, ws+"out.Raw( ("+in+").MarshalJSON() )")
		return nil
	}

	marshalerIface = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
	if reflect.PtrTo(t).Implements(marshalerIface) {
		fmt.Fprintln(g.out, ws+"out.RawText( ("+in+").MarshalText() )")
		return nil
	}

	err := g.genTypeEncoderNoCheck(t, in, tags, indent)
	return err
}

// genTypeEncoderNoCheck generates code that encodes in of type t into the writer.
func (g *Generator) genTypeEncoderNoCheck(t reflect.Type, in string, tags fieldTags, indent int) error {
	ws := strings.Repeat("  ", indent)

	// Check whether type is primitive, needs to be done after interface check.
	if enc := primitiveStringEncoders[t.Kind()]; enc != "" && tags.asString {
		fmt.Fprintf(g.out, ws+enc+"\n", in)
		return nil
	} else if enc := primitiveEncoders[t.Kind()]; enc != "" {
		fmt.Fprintf(g.out, ws+enc+"\n", in)
		return nil
	}

	switch t.Kind() {
	case reflect.Slice:
		elem := t.Elem()
		iVar := g.uniqueVarName()
		vVar := g.uniqueVarName()

		if t.Elem().Kind() == reflect.Uint8 {
			fmt.Fprintln(g.out, ws+"out.Base64Bytes("+in+")")
		} else {
			fmt.Fprintln(g.out, ws+"if "+in+" == nil && (out.Flags & jwriter.NilSliceAsEmpty) == 0 {")
			fmt.Fprintln(g.out, ws+`  out.RawString("null")`)
			fmt.Fprintln(g.out, ws+"} else {")
			fmt.Fprintln(g.out, ws+"  out.RawByte('[')")
			fmt.Fprintln(g.out, ws+"  for "+iVar+", "+vVar+" := range "+in+" {")
			fmt.Fprintln(g.out, ws+"    if "+iVar+" > 0 {")
			fmt.Fprintln(g.out, ws+"      out.RawByte(',')")
			fmt.Fprintln(g.out, ws+"    }")

			g.genTypeEncoder(elem, vVar, tags, indent+2)

			fmt.Fprintln(g.out, ws+"  }")
			fmt.Fprintln(g.out, ws+"  out.RawByte(']')")
			fmt.Fprintln(g.out, ws+"}")
		}

	case reflect.Array:
		elem := t.Elem()
		iVar := g.uniqueVarName()

		if t.Elem().Kind() == reflect.Uint8 {
			fmt.Fprintln(g.out, ws+"out.Base64Bytes("+in+"[:])")
		} else {
			fmt.Fprintln(g.out, ws+"out.RawByte('[')")
			fmt.Fprintln(g.out, ws+"for "+iVar+" := range "+in+" {")
			fmt.Fprintln(g.out, ws+"  if "+iVar+" > 0 {")
			fmt.Fprintln(g.out, ws+"    out.RawByte(',')")
			fmt.Fprintln(g.out, ws+"  }")

			g.genTypeEncoder(elem, in+"["+iVar+"]", tags, indent+1)

			fmt.Fprintln(g.out, ws+"}")
			fmt.Fprintln(g.out, ws+"out.RawByte(']')")
		}

	case reflect.Struct:
		enc := g.getEncoderName(t)
		g.addType(t)

		fmt.Fprintln(g.out, ws+enc+"(out, "+in+")")

	case reflect.Ptr:
		fmt.Fprintln(g.out, ws+"if "+in+" == nil {")
		fmt.Fprintln(g.out, ws+`  out.RawString("null")`)
		fmt.Fprintln(g.out, ws+"} else {")

		g.genTypeEncoder(t.Elem(), "*"+in, tags, indent+1)

		fmt.Fprintln(g.out, ws+"}")

	case reflect.Map:
		key := t.Key()
		if key.Kind() != reflect.String {
			return fmt.Errorf("map type %v not supported: only string keys are allowed", key)
		}
		tmpVar := g.uniqueVarName()

		fmt.Fprintln(g.out, ws+"if "+in+" == nil && (out.Flags & jwriter.NilMapAsEmpty) == 0 {")
		fmt.Fprintln(g.out, ws+"  out.RawString(`null`)")
		fmt.Fprintln(g.out, ws+"} else {")
		fmt.Fprintln(g.out, ws+"  out.RawByte('{')")
		fmt.Fprintln(g.out, ws+"  "+tmpVar+"First := true")
		fmt.Fprintln(g.out, ws+"  for "+tmpVar+"Name, "+tmpVar+"Value := range "+in+" {")
		fmt.Fprintln(g.out, ws+"    if !"+tmpVar+"First { out.RawByte(',') }")
		fmt.Fprintln(g.out, ws+"    "+tmpVar+"First = false")
		fmt.Fprintln(g.out, ws+"    out.String(string("+tmpVar+"Name))")
		fmt.Fprintln(g.out, ws+"    out.RawByte(':')")

		g.genTypeEncoder(t.Elem(), tmpVar+"Value", tags, indent+2)

		fmt.Fprintln(g.out, ws+"  }")
		fmt.Fprintln(g.out, ws+"  out.RawByte('}')")
		fmt.Fprintln(g.out, ws+"}")

	case reflect.Interface:
		if t.NumMethod() != 0 {
			return fmt.Errorf("interface type %v not supported: only interface{} is allowed", t)
		}
		fmt.Fprintln(g.out, ws+"if m, ok := "+in+".(easyjson.Marshaler); ok {")
		fmt.Fprintln(g.out, ws+"  m.MarshalEasyJSON(out)")
		fmt.Fprintln(g.out, ws+"} else if m, ok := "+in+".(json.Marshaler); ok {")
		fmt.Fprintln(g.out, ws+"  out.Raw(m.MarshalJSON())")
		fmt.Fprintln(g.out, ws+"} else {")
		fmt.Fprintln(g.out, ws+"  out.Raw(json.Marshal("+in+"))")
		fmt.Fprintln(g.out, ws+"}")

	default:
		return fmt.Errorf("don't know how to encode %v", t)
	}
	return nil
}

func (g *Generator) notEmptyCheck(t reflect.Type, v string) string {
	optionalIface := reflect.TypeOf((*easyjson.Optional)(nil)).Elem()
	if reflect.PtrTo(t).Implements(optionalIface) {
		return "(" + v + ").IsDefined()"
	}

	switch t.Kind() {
	case reflect.Slice, reflect.Map:
		return "len(" + v + ") != 0"
	case reflect.Interface, reflect.Ptr:
		return v + " != nil"
	case reflect.Bool:
		return v
	case reflect.String:
		return v + ` != ""`
	case reflect.Float32, reflect.Float64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:

		return v + " != 0"

	default:
		// note: Array types don't have a useful empty value
		return "true"
	}
}

func (g *Generator) genStructFieldEncoder(t reflect.Type, f reflect.StructField) error {
	jsonName := g.fieldNamer.GetJSONFieldName(t, f)
	tags := parseFieldTags(f)

	if tags.omit {
		return nil
	}
	if !tags.omitEmpty && !g.omitEmpty || tags.noOmitEmpty {
		fmt.Fprintln(g.out, "  if !first { out.RawByte(',') }")
		fmt.Fprintln(g.out, "  first = false")
		fmt.Fprintf(g.out, "  out.RawString(%q)\n", strconv.Quote(jsonName)+":")
		return g.genTypeEncoder(f.Type, "in."+f.Name, tags, 1)
	}

	fmt.Fprintln(g.out, "  if", g.notEmptyCheck(f.Type, "in."+f.Name), "{")
	fmt.Fprintln(g.out, "    if !first { out.RawByte(',') }")
	fmt.Fprintln(g.out, "    first = false")

	fmt.Fprintf(g.out, "    out.RawString(%q)\n", strconv.Quote(jsonName)+":")
	if err := g.genTypeEncoder(f.Type, "in."+f.Name, tags, 2); err != nil {
		return err
	}
	fmt.Fprintln(g.out, "  }")
	return nil
}

func (g *Generator) genEncoder(t reflect.Type) error {
	switch t.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		return g.genSliceArrayMapEncoder(t)
	default:
		return g.genStructEncoder(t)
	}
}

func (g *Generator) genSliceArrayMapEncoder(t reflect.Type) error {
	switch t.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
	default:
		return fmt.Errorf("cannot generate encoder/decoder for %v, not a slice/array/map type", t)
	}

	fname := g.getEncoderName(t)
	typ := g.getType(t)

	fmt.Fprintln(g.out, "func "+fname+"(out *jwriter.Writer, in "+typ+") {")
	err := g.genTypeEncoderNoCheck(t, "in", fieldTags{}, 1)
	if err != nil {
		return err
	}
	fmt.Fprintln(g.out, "}")
	return nil
}

func (g *Generator) genStructEncoder(t reflect.Type) error {
	if t.Kind() != reflect.Struct {
		return fmt.Errorf("cannot generate encoder/decoder for %v, not a struct type", t)
	}

	fname := g.getEncoderName(t)
	typ := g.getType(t)

	fmt.Fprintln(g.out, "func "+fname+"(out *jwriter.Writer, in "+typ+") {")
	fmt.Fprintln(g.out, "  out.RawByte('{')")
	fmt.Fprintln(g.out, "  first := true")
	fmt.Fprintln(g.out, "  _ = first")

	fs, err := getStructFields(t)
	if err != nil {
		return fmt.Errorf("cannot generate encoder for %v: %v", t, err)
	}
	for _, f := range fs {
		if err := g.genStructFieldEncoder(t, f); err != nil {
			return err
		}
	}

	fmt.Fprintln(g.out, "  out.RawByte('}')")
	fmt.Fprintln(g.out, "}")

	return nil
}

func (g *Generator) genStructMarshaler(t reflect.Type) error {
	switch t.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map, reflect.Struct:
	default:
		return fmt.Errorf("cannot generate encoder/decoder for %v, not a struct/slice/array/map type", t)
	}

	fname := g.getEncoderName(t)
	typ := g.getType(t)

	if !g.noStdMarshalers {
		fmt.Fprintln(g.out, "// MarshalJSON supports json.Marshaler interface")
		fmt.Fprintln(g.out, "func (v "+typ+") MarshalJSON() ([]byte, error) {")
		fmt.Fprintln(g.out, "  w := jwriter.Writer{}")
		fmt.Fprintln(g.out, "  "+fname+"(&w, v)")
		fmt.Fprintln(g.out, "  return w.Buffer.BuildBytes(), w.Error")
		fmt.Fprintln(g.out, "}")
	}

	fmt.Fprintln(g.out, "// MarshalEasyJSON supports easyjson.Marshaler interface")
	fmt.Fprintln(g.out, "func (v "+typ+") MarshalEasyJSON(w *jwriter.Writer) {")
	fmt.Fprintln(g.out, "  "+fname+"(w, v)")
	fmt.Fprintln(g.out, "}")

	return nil
}
