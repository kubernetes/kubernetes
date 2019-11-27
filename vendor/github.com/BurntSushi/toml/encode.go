package toml

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"
)

type tomlEncodeError struct{ error }

var (
	errArrayMixedElementTypes = errors.New(
		"toml: cannot encode array with mixed element types")
	errArrayNilElement = errors.New(
		"toml: cannot encode array with nil element")
	errNonString = errors.New(
		"toml: cannot encode a map with non-string key type")
	errAnonNonStruct = errors.New(
		"toml: cannot encode an anonymous field that is not a struct")
	errArrayNoTable = errors.New(
		"toml: TOML array element cannot contain a table")
	errNoKey = errors.New(
		"toml: top-level values must be Go maps or structs")
	errAnything = errors.New("") // used in testing
)

var quotedReplacer = strings.NewReplacer(
	"\t", "\\t",
	"\n", "\\n",
	"\r", "\\r",
	"\"", "\\\"",
	"\\", "\\\\",
)

// Encoder controls the encoding of Go values to a TOML document to some
// io.Writer.
//
// The indentation level can be controlled with the Indent field.
type Encoder struct {
	// A single indentation level. By default it is two spaces.
	Indent string

	// hasWritten is whether we have written any output to w yet.
	hasWritten bool
	w          *bufio.Writer
}

// NewEncoder returns a TOML encoder that encodes Go values to the io.Writer
// given. By default, a single indentation level is 2 spaces.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		w:      bufio.NewWriter(w),
		Indent: "  ",
	}
}

// Encode writes a TOML representation of the Go value to the underlying
// io.Writer. If the value given cannot be encoded to a valid TOML document,
// then an error is returned.
//
// The mapping between Go values and TOML values should be precisely the same
// as for the Decode* functions. Similarly, the TextMarshaler interface is
// supported by encoding the resulting bytes as strings. (If you want to write
// arbitrary binary data then you will need to use something like base64 since
// TOML does not have any binary types.)
//
// When encoding TOML hashes (i.e., Go maps or structs), keys without any
// sub-hashes are encoded first.
//
// If a Go map is encoded, then its keys are sorted alphabetically for
// deterministic output. More control over this behavior may be provided if
// there is demand for it.
//
// Encoding Go values without a corresponding TOML representation---like map
// types with non-string keys---will cause an error to be returned. Similarly
// for mixed arrays/slices, arrays/slices with nil elements, embedded
// non-struct types and nested slices containing maps or structs.
// (e.g., [][]map[string]string is not allowed but []map[string]string is OK
// and so is []map[string][]string.)
func (enc *Encoder) Encode(v interface{}) error {
	rv := eindirect(reflect.ValueOf(v))
	if err := enc.safeEncode(Key([]string{}), rv); err != nil {
		return err
	}
	return enc.w.Flush()
}

func (enc *Encoder) safeEncode(key Key, rv reflect.Value) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if terr, ok := r.(tomlEncodeError); ok {
				err = terr.error
				return
			}
			panic(r)
		}
	}()
	enc.encode(key, rv)
	return nil
}

func (enc *Encoder) encode(key Key, rv reflect.Value) {
	// Special case. Time needs to be in ISO8601 format.
	// Special case. If we can marshal the type to text, then we used that.
	// Basically, this prevents the encoder for handling these types as
	// generic structs (or whatever the underlying type of a TextMarshaler is).
	switch rv.Interface().(type) {
	case time.Time, TextMarshaler:
		enc.keyEqElement(key, rv)
		return
	}

	k := rv.Kind()
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32,
		reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32,
		reflect.Uint64,
		reflect.Float32, reflect.Float64, reflect.String, reflect.Bool:
		enc.keyEqElement(key, rv)
	case reflect.Array, reflect.Slice:
		if typeEqual(tomlArrayHash, tomlTypeOfGo(rv)) {
			enc.eArrayOfTables(key, rv)
		} else {
			enc.keyEqElement(key, rv)
		}
	case reflect.Interface:
		if rv.IsNil() {
			return
		}
		enc.encode(key, rv.Elem())
	case reflect.Map:
		if rv.IsNil() {
			return
		}
		enc.eTable(key, rv)
	case reflect.Ptr:
		if rv.IsNil() {
			return
		}
		enc.encode(key, rv.Elem())
	case reflect.Struct:
		enc.eTable(key, rv)
	default:
		panic(e("unsupported type for key '%s': %s", key, k))
	}
}

// eElement encodes any value that can be an array element (primitives and
// arrays).
func (enc *Encoder) eElement(rv reflect.Value) {
	switch v := rv.Interface().(type) {
	case time.Time:
		// Special case time.Time as a primitive. Has to come before
		// TextMarshaler below because time.Time implements
		// encoding.TextMarshaler, but we need to always use UTC.
		enc.wf(v.UTC().Format("2006-01-02T15:04:05Z"))
		return
	case TextMarshaler:
		// Special case. Use text marshaler if it's available for this value.
		if s, err := v.MarshalText(); err != nil {
			encPanic(err)
		} else {
			enc.writeQuoted(string(s))
		}
		return
	}
	switch rv.Kind() {
	case reflect.Bool:
		enc.wf(strconv.FormatBool(rv.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32,
		reflect.Int64:
		enc.wf(strconv.FormatInt(rv.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16,
		reflect.Uint32, reflect.Uint64:
		enc.wf(strconv.FormatUint(rv.Uint(), 10))
	case reflect.Float32:
		enc.wf(floatAddDecimal(strconv.FormatFloat(rv.Float(), 'f', -1, 32)))
	case reflect.Float64:
		enc.wf(floatAddDecimal(strconv.FormatFloat(rv.Float(), 'f', -1, 64)))
	case reflect.Array, reflect.Slice:
		enc.eArrayOrSliceElement(rv)
	case reflect.Interface:
		enc.eElement(rv.Elem())
	case reflect.String:
		enc.writeQuoted(rv.String())
	default:
		panic(e("unexpected primitive type: %s", rv.Kind()))
	}
}

// By the TOML spec, all floats must have a decimal with at least one
// number on either side.
func floatAddDecimal(fstr string) string {
	if !strings.Contains(fstr, ".") {
		return fstr + ".0"
	}
	return fstr
}

func (enc *Encoder) writeQuoted(s string) {
	enc.wf("\"%s\"", quotedReplacer.Replace(s))
}

func (enc *Encoder) eArrayOrSliceElement(rv reflect.Value) {
	length := rv.Len()
	enc.wf("[")
	for i := 0; i < length; i++ {
		elem := rv.Index(i)
		enc.eElement(elem)
		if i != length-1 {
			enc.wf(", ")
		}
	}
	enc.wf("]")
}

func (enc *Encoder) eArrayOfTables(key Key, rv reflect.Value) {
	if len(key) == 0 {
		encPanic(errNoKey)
	}
	for i := 0; i < rv.Len(); i++ {
		trv := rv.Index(i)
		if isNil(trv) {
			continue
		}
		panicIfInvalidKey(key)
		enc.newline()
		enc.wf("%s[[%s]]", enc.indentStr(key), key.maybeQuotedAll())
		enc.newline()
		enc.eMapOrStruct(key, trv)
	}
}

func (enc *Encoder) eTable(key Key, rv reflect.Value) {
	panicIfInvalidKey(key)
	if len(key) == 1 {
		// Output an extra newline between top-level tables.
		// (The newline isn't written if nothing else has been written though.)
		enc.newline()
	}
	if len(key) > 0 {
		enc.wf("%s[%s]", enc.indentStr(key), key.maybeQuotedAll())
		enc.newline()
	}
	enc.eMapOrStruct(key, rv)
}

func (enc *Encoder) eMapOrStruct(key Key, rv reflect.Value) {
	switch rv := eindirect(rv); rv.Kind() {
	case reflect.Map:
		enc.eMap(key, rv)
	case reflect.Struct:
		enc.eStruct(key, rv)
	default:
		panic("eTable: unhandled reflect.Value Kind: " + rv.Kind().String())
	}
}

func (enc *Encoder) eMap(key Key, rv reflect.Value) {
	rt := rv.Type()
	if rt.Key().Kind() != reflect.String {
		encPanic(errNonString)
	}

	// Sort keys so that we have deterministic output. And write keys directly
	// underneath this key first, before writing sub-structs or sub-maps.
	var mapKeysDirect, mapKeysSub []string
	for _, mapKey := range rv.MapKeys() {
		k := mapKey.String()
		if typeIsHash(tomlTypeOfGo(rv.MapIndex(mapKey))) {
			mapKeysSub = append(mapKeysSub, k)
		} else {
			mapKeysDirect = append(mapKeysDirect, k)
		}
	}

	var writeMapKeys = func(mapKeys []string) {
		sort.Strings(mapKeys)
		for _, mapKey := range mapKeys {
			mrv := rv.MapIndex(reflect.ValueOf(mapKey))
			if isNil(mrv) {
				// Don't write anything for nil fields.
				continue
			}
			enc.encode(key.add(mapKey), mrv)
		}
	}
	writeMapKeys(mapKeysDirect)
	writeMapKeys(mapKeysSub)
}

func (enc *Encoder) eStruct(key Key, rv reflect.Value) {
	// Write keys for fields directly under this key first, because if we write
	// a field that creates a new table, then all keys under it will be in that
	// table (not the one we're writing here).
	rt := rv.Type()
	var fieldsDirect, fieldsSub [][]int
	var addFields func(rt reflect.Type, rv reflect.Value, start []int)
	addFields = func(rt reflect.Type, rv reflect.Value, start []int) {
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			// skip unexported fields
			if f.PkgPath != "" && !f.Anonymous {
				continue
			}
			frv := rv.Field(i)
			if f.Anonymous {
				t := f.Type
				switch t.Kind() {
				case reflect.Struct:
					// Treat anonymous struct fields with
					// tag names as though they are not
					// anonymous, like encoding/json does.
					if getOptions(f.Tag).name == "" {
						addFields(t, frv, f.Index)
						continue
					}
				case reflect.Ptr:
					if t.Elem().Kind() == reflect.Struct &&
						getOptions(f.Tag).name == "" {
						if !frv.IsNil() {
							addFields(t.Elem(), frv.Elem(), f.Index)
						}
						continue
					}
					// Fall through to the normal field encoding logic below
					// for non-struct anonymous fields.
				}
			}

			if typeIsHash(tomlTypeOfGo(frv)) {
				fieldsSub = append(fieldsSub, append(start, f.Index...))
			} else {
				fieldsDirect = append(fieldsDirect, append(start, f.Index...))
			}
		}
	}
	addFields(rt, rv, nil)

	var writeFields = func(fields [][]int) {
		for _, fieldIndex := range fields {
			sft := rt.FieldByIndex(fieldIndex)
			sf := rv.FieldByIndex(fieldIndex)
			if isNil(sf) {
				// Don't write anything for nil fields.
				continue
			}

			opts := getOptions(sft.Tag)
			if opts.skip {
				continue
			}
			keyName := sft.Name
			if opts.name != "" {
				keyName = opts.name
			}
			if opts.omitempty && isEmpty(sf) {
				continue
			}
			if opts.omitzero && isZero(sf) {
				continue
			}

			enc.encode(key.add(keyName), sf)
		}
	}
	writeFields(fieldsDirect)
	writeFields(fieldsSub)
}

// tomlTypeName returns the TOML type name of the Go value's type. It is
// used to determine whether the types of array elements are mixed (which is
// forbidden). If the Go value is nil, then it is illegal for it to be an array
// element, and valueIsNil is returned as true.

// Returns the TOML type of a Go value. The type may be `nil`, which means
// no concrete TOML type could be found.
func tomlTypeOfGo(rv reflect.Value) tomlType {
	if isNil(rv) || !rv.IsValid() {
		return nil
	}
	switch rv.Kind() {
	case reflect.Bool:
		return tomlBool
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32,
		reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32,
		reflect.Uint64:
		return tomlInteger
	case reflect.Float32, reflect.Float64:
		return tomlFloat
	case reflect.Array, reflect.Slice:
		if typeEqual(tomlHash, tomlArrayType(rv)) {
			return tomlArrayHash
		}
		return tomlArray
	case reflect.Ptr, reflect.Interface:
		return tomlTypeOfGo(rv.Elem())
	case reflect.String:
		return tomlString
	case reflect.Map:
		return tomlHash
	case reflect.Struct:
		switch rv.Interface().(type) {
		case time.Time:
			return tomlDatetime
		case TextMarshaler:
			return tomlString
		default:
			return tomlHash
		}
	default:
		panic("unexpected reflect.Kind: " + rv.Kind().String())
	}
}

// tomlArrayType returns the element type of a TOML array. The type returned
// may be nil if it cannot be determined (e.g., a nil slice or a zero length
// slize). This function may also panic if it finds a type that cannot be
// expressed in TOML (such as nil elements, heterogeneous arrays or directly
// nested arrays of tables).
func tomlArrayType(rv reflect.Value) tomlType {
	if isNil(rv) || !rv.IsValid() || rv.Len() == 0 {
		return nil
	}
	firstType := tomlTypeOfGo(rv.Index(0))
	if firstType == nil {
		encPanic(errArrayNilElement)
	}

	rvlen := rv.Len()
	for i := 1; i < rvlen; i++ {
		elem := rv.Index(i)
		switch elemType := tomlTypeOfGo(elem); {
		case elemType == nil:
			encPanic(errArrayNilElement)
		case !typeEqual(firstType, elemType):
			encPanic(errArrayMixedElementTypes)
		}
	}
	// If we have a nested array, then we must make sure that the nested
	// array contains ONLY primitives.
	// This checks arbitrarily nested arrays.
	if typeEqual(firstType, tomlArray) || typeEqual(firstType, tomlArrayHash) {
		nest := tomlArrayType(eindirect(rv.Index(0)))
		if typeEqual(nest, tomlHash) || typeEqual(nest, tomlArrayHash) {
			encPanic(errArrayNoTable)
		}
	}
	return firstType
}

type tagOptions struct {
	skip      bool // "-"
	name      string
	omitempty bool
	omitzero  bool
}

func getOptions(tag reflect.StructTag) tagOptions {
	t := tag.Get("toml")
	if t == "-" {
		return tagOptions{skip: true}
	}
	var opts tagOptions
	parts := strings.Split(t, ",")
	opts.name = parts[0]
	for _, s := range parts[1:] {
		switch s {
		case "omitempty":
			opts.omitempty = true
		case "omitzero":
			opts.omitzero = true
		}
	}
	return opts
}

func isZero(rv reflect.Value) bool {
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return rv.Float() == 0.0
	}
	return false
}

func isEmpty(rv reflect.Value) bool {
	switch rv.Kind() {
	case reflect.Array, reflect.Slice, reflect.Map, reflect.String:
		return rv.Len() == 0
	case reflect.Bool:
		return !rv.Bool()
	}
	return false
}

func (enc *Encoder) newline() {
	if enc.hasWritten {
		enc.wf("\n")
	}
}

func (enc *Encoder) keyEqElement(key Key, val reflect.Value) {
	if len(key) == 0 {
		encPanic(errNoKey)
	}
	panicIfInvalidKey(key)
	enc.wf("%s%s = ", enc.indentStr(key), key.maybeQuoted(len(key)-1))
	enc.eElement(val)
	enc.newline()
}

func (enc *Encoder) wf(format string, v ...interface{}) {
	if _, err := fmt.Fprintf(enc.w, format, v...); err != nil {
		encPanic(err)
	}
	enc.hasWritten = true
}

func (enc *Encoder) indentStr(key Key) string {
	return strings.Repeat(enc.Indent, len(key)-1)
}

func encPanic(err error) {
	panic(tomlEncodeError{err})
}

func eindirect(v reflect.Value) reflect.Value {
	switch v.Kind() {
	case reflect.Ptr, reflect.Interface:
		return eindirect(v.Elem())
	default:
		return v
	}
}

func isNil(rv reflect.Value) bool {
	switch rv.Kind() {
	case reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return rv.IsNil()
	default:
		return false
	}
}

func panicIfInvalidKey(key Key) {
	for _, k := range key {
		if len(k) == 0 {
			encPanic(e("Key '%s' is not a valid table name. Key names "+
				"cannot be empty.", key.maybeQuotedAll()))
		}
	}
}

func isValidKeyName(s string) bool {
	return len(s) != 0
}
