package toml

import (
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	tagFieldName    = "toml"
	tagFieldComment = "comment"
	tagCommented    = "commented"
	tagMultiline    = "multiline"
	tagLiteral      = "literal"
	tagDefault      = "default"
)

type tomlOpts struct {
	name         string
	nameFromTag  bool
	comment      string
	commented    bool
	multiline    bool
	literal      bool
	include      bool
	omitempty    bool
	defaultValue string
}

type encOpts struct {
	quoteMapKeys            bool
	arraysOneElementPerLine bool
}

var encOptsDefaults = encOpts{
	quoteMapKeys: false,
}

type annotation struct {
	tag          string
	comment      string
	commented    string
	multiline    string
	literal      string
	defaultValue string
}

var annotationDefault = annotation{
	tag:          tagFieldName,
	comment:      tagFieldComment,
	commented:    tagCommented,
	multiline:    tagMultiline,
	literal:      tagLiteral,
	defaultValue: tagDefault,
}

type MarshalOrder int

// Orders the Encoder can write the fields to the output stream.
const (
	// Sort fields alphabetically.
	OrderAlphabetical MarshalOrder = iota + 1
	// Preserve the order the fields are encountered. For example, the order of fields in
	// a struct.
	OrderPreserve
)

var timeType = reflect.TypeOf(time.Time{})
var marshalerType = reflect.TypeOf(new(Marshaler)).Elem()
var unmarshalerType = reflect.TypeOf(new(Unmarshaler)).Elem()
var textMarshalerType = reflect.TypeOf(new(encoding.TextMarshaler)).Elem()
var textUnmarshalerType = reflect.TypeOf(new(encoding.TextUnmarshaler)).Elem()
var localDateType = reflect.TypeOf(LocalDate{})
var localTimeType = reflect.TypeOf(LocalTime{})
var localDateTimeType = reflect.TypeOf(LocalDateTime{})
var mapStringInterfaceType = reflect.TypeOf(map[string]interface{}{})

// Check if the given marshal type maps to a Tree primitive
func isPrimitive(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isPrimitive(mtype.Elem())
	case reflect.Bool:
		return true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return true
	case reflect.Float32, reflect.Float64:
		return true
	case reflect.String:
		return true
	case reflect.Struct:
		return isTimeType(mtype)
	default:
		return false
	}
}

func isTimeType(mtype reflect.Type) bool {
	return mtype == timeType || mtype == localDateType || mtype == localDateTimeType || mtype == localTimeType
}

// Check if the given marshal type maps to a Tree slice or array
func isTreeSequence(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isTreeSequence(mtype.Elem())
	case reflect.Slice, reflect.Array:
		return isTree(mtype.Elem())
	default:
		return false
	}
}

// Check if the given marshal type maps to a slice or array of a custom marshaler type
func isCustomMarshalerSequence(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isCustomMarshalerSequence(mtype.Elem())
	case reflect.Slice, reflect.Array:
		return isCustomMarshaler(mtype.Elem()) || isCustomMarshaler(reflect.New(mtype.Elem()).Type())
	default:
		return false
	}
}

// Check if the given marshal type maps to a slice or array of a text marshaler type
func isTextMarshalerSequence(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isTextMarshalerSequence(mtype.Elem())
	case reflect.Slice, reflect.Array:
		return isTextMarshaler(mtype.Elem()) || isTextMarshaler(reflect.New(mtype.Elem()).Type())
	default:
		return false
	}
}

// Check if the given marshal type maps to a non-Tree slice or array
func isOtherSequence(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isOtherSequence(mtype.Elem())
	case reflect.Slice, reflect.Array:
		return !isTreeSequence(mtype)
	default:
		return false
	}
}

// Check if the given marshal type maps to a Tree
func isTree(mtype reflect.Type) bool {
	switch mtype.Kind() {
	case reflect.Ptr:
		return isTree(mtype.Elem())
	case reflect.Map:
		return true
	case reflect.Struct:
		return !isPrimitive(mtype)
	default:
		return false
	}
}

func isCustomMarshaler(mtype reflect.Type) bool {
	return mtype.Implements(marshalerType)
}

func callCustomMarshaler(mval reflect.Value) ([]byte, error) {
	return mval.Interface().(Marshaler).MarshalTOML()
}

func isTextMarshaler(mtype reflect.Type) bool {
	return mtype.Implements(textMarshalerType) && !isTimeType(mtype)
}

func callTextMarshaler(mval reflect.Value) ([]byte, error) {
	return mval.Interface().(encoding.TextMarshaler).MarshalText()
}

func isCustomUnmarshaler(mtype reflect.Type) bool {
	return mtype.Implements(unmarshalerType)
}

func callCustomUnmarshaler(mval reflect.Value, tval interface{}) error {
	return mval.Interface().(Unmarshaler).UnmarshalTOML(tval)
}

func isTextUnmarshaler(mtype reflect.Type) bool {
	return mtype.Implements(textUnmarshalerType)
}

func callTextUnmarshaler(mval reflect.Value, text []byte) error {
	return mval.Interface().(encoding.TextUnmarshaler).UnmarshalText(text)
}

// Marshaler is the interface implemented by types that
// can marshal themselves into valid TOML.
type Marshaler interface {
	MarshalTOML() ([]byte, error)
}

// Unmarshaler is the interface implemented by types that
// can unmarshal a TOML description of themselves.
type Unmarshaler interface {
	UnmarshalTOML(interface{}) error
}

/*
Marshal returns the TOML encoding of v.  Behavior is similar to the Go json
encoder, except that there is no concept of a Marshaler interface or MarshalTOML
function for sub-structs, and currently only definite types can be marshaled
(i.e. no `interface{}`).

The following struct annotations are supported:

  toml:"Field"      Overrides the field's name to output.
  omitempty         When set, empty values and groups are not emitted.
  comment:"comment" Emits a # comment on the same line. This supports new lines.
  commented:"true"  Emits the value as commented.

Note that pointers are automatically assigned the "omitempty" option, as TOML
explicitly does not handle null values (saying instead the label should be
dropped).

Tree structural types and corresponding marshal types:

  *Tree                            (*)struct, (*)map[string]interface{}
  []*Tree                          (*)[](*)struct, (*)[](*)map[string]interface{}
  []interface{} (as interface{})   (*)[]primitive, (*)[]([]interface{})
  interface{}                      (*)primitive

Tree primitive types and corresponding marshal types:

  uint64     uint, uint8-uint64, pointers to same
  int64      int, int8-uint64, pointers to same
  float64    float32, float64, pointers to same
  string     string, pointers to same
  bool       bool, pointers to same
  time.LocalTime  time.LocalTime{}, pointers to same

For additional flexibility, use the Encoder API.
*/
func Marshal(v interface{}) ([]byte, error) {
	return NewEncoder(nil).marshal(v)
}

// Encoder writes TOML values to an output stream.
type Encoder struct {
	w io.Writer
	encOpts
	annotation
	line            int
	col             int
	order           MarshalOrder
	promoteAnon     bool
	compactComments bool
	indentation     string
}

// NewEncoder returns a new encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		w:           w,
		encOpts:     encOptsDefaults,
		annotation:  annotationDefault,
		line:        0,
		col:         1,
		order:       OrderAlphabetical,
		indentation: "  ",
	}
}

// Encode writes the TOML encoding of v to the stream.
//
// See the documentation for Marshal for details.
func (e *Encoder) Encode(v interface{}) error {
	b, err := e.marshal(v)
	if err != nil {
		return err
	}
	if _, err := e.w.Write(b); err != nil {
		return err
	}
	return nil
}

// QuoteMapKeys sets up the encoder to encode
// maps with string type keys with quoted TOML keys.
//
// This relieves the character limitations on map keys.
func (e *Encoder) QuoteMapKeys(v bool) *Encoder {
	e.quoteMapKeys = v
	return e
}

// ArraysWithOneElementPerLine sets up the encoder to encode arrays
// with more than one element on multiple lines instead of one.
//
// For example:
//
//   A = [1,2,3]
//
// Becomes
//
//   A = [
//     1,
//     2,
//     3,
//   ]
func (e *Encoder) ArraysWithOneElementPerLine(v bool) *Encoder {
	e.arraysOneElementPerLine = v
	return e
}

// Order allows to change in which order fields will be written to the output stream.
func (e *Encoder) Order(ord MarshalOrder) *Encoder {
	e.order = ord
	return e
}

// Indentation allows to change indentation when marshalling.
func (e *Encoder) Indentation(indent string) *Encoder {
	e.indentation = indent
	return e
}

// SetTagName allows changing default tag "toml"
func (e *Encoder) SetTagName(v string) *Encoder {
	e.tag = v
	return e
}

// SetTagComment allows changing default tag "comment"
func (e *Encoder) SetTagComment(v string) *Encoder {
	e.comment = v
	return e
}

// SetTagCommented allows changing default tag "commented"
func (e *Encoder) SetTagCommented(v string) *Encoder {
	e.commented = v
	return e
}

// SetTagMultiline allows changing default tag "multiline"
func (e *Encoder) SetTagMultiline(v string) *Encoder {
	e.multiline = v
	return e
}

// PromoteAnonymous allows to change how anonymous struct fields are marshaled.
// Usually, they are marshaled as if the inner exported fields were fields in
// the outer struct. However, if an anonymous struct field is given a name in
// its TOML tag, it is treated like a regular struct field with that name.
// rather than being anonymous.
//
// In case anonymous promotion is enabled, all anonymous structs are promoted
// and treated like regular struct fields.
func (e *Encoder) PromoteAnonymous(promote bool) *Encoder {
	e.promoteAnon = promote
	return e
}

// CompactComments removes the new line before each comment in the tree.
func (e *Encoder) CompactComments(cc bool) *Encoder {
	e.compactComments = cc
	return e
}

func (e *Encoder) marshal(v interface{}) ([]byte, error) {
	// Check if indentation is valid
	for _, char := range e.indentation {
		if !isSpace(char) {
			return []byte{}, fmt.Errorf("invalid indentation: must only contains space or tab characters")
		}
	}

	mtype := reflect.TypeOf(v)
	if mtype == nil {
		return []byte{}, errors.New("nil cannot be marshaled to TOML")
	}

	switch mtype.Kind() {
	case reflect.Struct, reflect.Map:
	case reflect.Ptr:
		if mtype.Elem().Kind() != reflect.Struct {
			return []byte{}, errors.New("Only pointer to struct can be marshaled to TOML")
		}
		if reflect.ValueOf(v).IsNil() {
			return []byte{}, errors.New("nil pointer cannot be marshaled to TOML")
		}
	default:
		return []byte{}, errors.New("Only a struct or map can be marshaled to TOML")
	}

	sval := reflect.ValueOf(v)
	if isCustomMarshaler(mtype) {
		return callCustomMarshaler(sval)
	}
	if isTextMarshaler(mtype) {
		return callTextMarshaler(sval)
	}
	t, err := e.valueToTree(mtype, sval)
	if err != nil {
		return []byte{}, err
	}

	var buf bytes.Buffer
	_, err = t.writeToOrdered(&buf, "", "", 0, e.arraysOneElementPerLine, e.order, e.indentation, e.compactComments, false)

	return buf.Bytes(), err
}

// Create next tree with a position based on Encoder.line
func (e *Encoder) nextTree() *Tree {
	return newTreeWithPosition(Position{Line: e.line, Col: 1})
}

// Convert given marshal struct or map value to toml tree
func (e *Encoder) valueToTree(mtype reflect.Type, mval reflect.Value) (*Tree, error) {
	if mtype.Kind() == reflect.Ptr {
		return e.valueToTree(mtype.Elem(), mval.Elem())
	}
	tval := e.nextTree()
	switch mtype.Kind() {
	case reflect.Struct:
		switch mval.Interface().(type) {
		case Tree:
			reflect.ValueOf(tval).Elem().Set(mval)
		default:
			for i := 0; i < mtype.NumField(); i++ {
				mtypef, mvalf := mtype.Field(i), mval.Field(i)
				opts := tomlOptions(mtypef, e.annotation)
				if opts.include && ((mtypef.Type.Kind() != reflect.Interface && !opts.omitempty) || !isZero(mvalf)) {
					val, err := e.valueToToml(mtypef.Type, mvalf)
					if err != nil {
						return nil, err
					}
					if tree, ok := val.(*Tree); ok && mtypef.Anonymous && !opts.nameFromTag && !e.promoteAnon {
						e.appendTree(tval, tree)
					} else {
						val = e.wrapTomlValue(val, tval)
						tval.SetPathWithOptions([]string{opts.name}, SetOptions{
							Comment:   opts.comment,
							Commented: opts.commented,
							Multiline: opts.multiline,
							Literal:   opts.literal,
						}, val)
					}
				}
			}
		}
	case reflect.Map:
		keys := mval.MapKeys()
		if e.order == OrderPreserve && len(keys) > 0 {
			// Sorting []reflect.Value is not straight forward.
			//
			// OrderPreserve will support deterministic results when string is used
			// as the key to maps.
			typ := keys[0].Type()
			kind := keys[0].Kind()
			if kind == reflect.String {
				ikeys := make([]string, len(keys))
				for i := range keys {
					ikeys[i] = keys[i].Interface().(string)
				}
				sort.Strings(ikeys)
				for i := range ikeys {
					keys[i] = reflect.ValueOf(ikeys[i]).Convert(typ)
				}
			}
		}
		for _, key := range keys {
			mvalf := mval.MapIndex(key)
			if (mtype.Elem().Kind() == reflect.Ptr || mtype.Elem().Kind() == reflect.Interface) && mvalf.IsNil() {
				continue
			}
			val, err := e.valueToToml(mtype.Elem(), mvalf)
			if err != nil {
				return nil, err
			}
			val = e.wrapTomlValue(val, tval)
			if e.quoteMapKeys {
				keyStr, err := tomlValueStringRepresentation(key.String(), "", "", e.order, e.arraysOneElementPerLine)
				if err != nil {
					return nil, err
				}
				tval.SetPath([]string{keyStr}, val)
			} else {
				tval.SetPath([]string{key.String()}, val)
			}
		}
	}
	return tval, nil
}

// Convert given marshal slice to slice of Toml trees
func (e *Encoder) valueToTreeSlice(mtype reflect.Type, mval reflect.Value) ([]*Tree, error) {
	tval := make([]*Tree, mval.Len(), mval.Len())
	for i := 0; i < mval.Len(); i++ {
		val, err := e.valueToTree(mtype.Elem(), mval.Index(i))
		if err != nil {
			return nil, err
		}
		tval[i] = val
	}
	return tval, nil
}

// Convert given marshal slice to slice of toml values
func (e *Encoder) valueToOtherSlice(mtype reflect.Type, mval reflect.Value) (interface{}, error) {
	tval := make([]interface{}, mval.Len(), mval.Len())
	for i := 0; i < mval.Len(); i++ {
		val, err := e.valueToToml(mtype.Elem(), mval.Index(i))
		if err != nil {
			return nil, err
		}
		tval[i] = val
	}
	return tval, nil
}

// Convert given marshal value to toml value
func (e *Encoder) valueToToml(mtype reflect.Type, mval reflect.Value) (interface{}, error) {
	if mtype.Kind() == reflect.Ptr {
		switch {
		case isCustomMarshaler(mtype):
			return callCustomMarshaler(mval)
		case isTextMarshaler(mtype):
			b, err := callTextMarshaler(mval)
			return string(b), err
		default:
			return e.valueToToml(mtype.Elem(), mval.Elem())
		}
	}
	if mtype.Kind() == reflect.Interface {
		return e.valueToToml(mval.Elem().Type(), mval.Elem())
	}
	switch {
	case isCustomMarshaler(mtype):
		return callCustomMarshaler(mval)
	case isTextMarshaler(mtype):
		b, err := callTextMarshaler(mval)
		return string(b), err
	case isTree(mtype):
		return e.valueToTree(mtype, mval)
	case isOtherSequence(mtype), isCustomMarshalerSequence(mtype), isTextMarshalerSequence(mtype):
		return e.valueToOtherSlice(mtype, mval)
	case isTreeSequence(mtype):
		return e.valueToTreeSlice(mtype, mval)
	default:
		switch mtype.Kind() {
		case reflect.Bool:
			return mval.Bool(), nil
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			if mtype.Kind() == reflect.Int64 && mtype == reflect.TypeOf(time.Duration(1)) {
				return fmt.Sprint(mval), nil
			}
			return mval.Int(), nil
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return mval.Uint(), nil
		case reflect.Float32, reflect.Float64:
			return mval.Float(), nil
		case reflect.String:
			return mval.String(), nil
		case reflect.Struct:
			return mval.Interface(), nil
		default:
			return nil, fmt.Errorf("Marshal can't handle %v(%v)", mtype, mtype.Kind())
		}
	}
}

func (e *Encoder) appendTree(t, o *Tree) error {
	for key, value := range o.values {
		if _, ok := t.values[key]; ok {
			continue
		}
		if tomlValue, ok := value.(*tomlValue); ok {
			tomlValue.position.Col = t.position.Col
		}
		t.values[key] = value
	}
	return nil
}

// Create a toml value with the current line number as the position line
func (e *Encoder) wrapTomlValue(val interface{}, parent *Tree) interface{} {
	_, isTree := val.(*Tree)
	_, isTreeS := val.([]*Tree)
	if isTree || isTreeS {
		e.line++
		return val
	}

	ret := &tomlValue{
		value: val,
		position: Position{
			e.line,
			parent.position.Col,
		},
	}
	e.line++
	return ret
}

// Unmarshal attempts to unmarshal the Tree into a Go struct pointed by v.
// Neither Unmarshaler interfaces nor UnmarshalTOML functions are supported for
// sub-structs, and only definite types can be unmarshaled.
func (t *Tree) Unmarshal(v interface{}) error {
	d := Decoder{tval: t, tagName: tagFieldName}
	return d.unmarshal(v)
}

// Marshal returns the TOML encoding of Tree.
// See Marshal() documentation for types mapping table.
func (t *Tree) Marshal() ([]byte, error) {
	var buf bytes.Buffer
	_, err := t.WriteTo(&buf)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Unmarshal parses the TOML-encoded data and stores the result in the value
// pointed to by v. Behavior is similar to the Go json encoder, except that there
// is no concept of an Unmarshaler interface or UnmarshalTOML function for
// sub-structs, and currently only definite types can be unmarshaled to (i.e. no
// `interface{}`).
//
// The following struct annotations are supported:
//
//   toml:"Field" Overrides the field's name to map to.
//   default:"foo" Provides a default value.
//
// For default values, only fields of the following types are supported:
//   * string
//   * bool
//   * int
//   * int64
//   * float64
//
// See Marshal() documentation for types mapping table.
func Unmarshal(data []byte, v interface{}) error {
	t, err := LoadReader(bytes.NewReader(data))
	if err != nil {
		return err
	}
	return t.Unmarshal(v)
}

// Decoder reads and decodes TOML values from an input stream.
type Decoder struct {
	r    io.Reader
	tval *Tree
	encOpts
	tagName string
	strict  bool
	visitor visitorState
}

// NewDecoder returns a new decoder that reads from r.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{
		r:       r,
		encOpts: encOptsDefaults,
		tagName: tagFieldName,
	}
}

// Decode reads a TOML-encoded value from it's input
// and unmarshals it in the value pointed at by v.
//
// See the documentation for Marshal for details.
func (d *Decoder) Decode(v interface{}) error {
	var err error
	d.tval, err = LoadReader(d.r)
	if err != nil {
		return err
	}
	return d.unmarshal(v)
}

// SetTagName allows changing default tag "toml"
func (d *Decoder) SetTagName(v string) *Decoder {
	d.tagName = v
	return d
}

// Strict allows changing to strict decoding. Any fields that are found in the
// input data and do not have a corresponding struct member cause an error.
func (d *Decoder) Strict(strict bool) *Decoder {
	d.strict = strict
	return d
}

func (d *Decoder) unmarshal(v interface{}) error {
	mtype := reflect.TypeOf(v)
	if mtype == nil {
		return errors.New("nil cannot be unmarshaled from TOML")
	}
	if mtype.Kind() != reflect.Ptr {
		return errors.New("only a pointer to struct or map can be unmarshaled from TOML")
	}

	elem := mtype.Elem()

	switch elem.Kind() {
	case reflect.Struct, reflect.Map:
	case reflect.Interface:
		elem = mapStringInterfaceType
	default:
		return errors.New("only a pointer to struct or map can be unmarshaled from TOML")
	}

	if reflect.ValueOf(v).IsNil() {
		return errors.New("nil pointer cannot be unmarshaled from TOML")
	}

	vv := reflect.ValueOf(v).Elem()

	if d.strict {
		d.visitor = newVisitorState(d.tval)
	}

	sval, err := d.valueFromTree(elem, d.tval, &vv)
	if err != nil {
		return err
	}
	if err := d.visitor.validate(); err != nil {
		return err
	}
	reflect.ValueOf(v).Elem().Set(sval)
	return nil
}

// Convert toml tree to marshal struct or map, using marshal type. When mval1
// is non-nil, merge fields into the given value instead of allocating a new one.
func (d *Decoder) valueFromTree(mtype reflect.Type, tval *Tree, mval1 *reflect.Value) (reflect.Value, error) {
	if mtype.Kind() == reflect.Ptr {
		return d.unwrapPointer(mtype, tval, mval1)
	}

	// Check if pointer to value implements the Unmarshaler interface.
	if mvalPtr := reflect.New(mtype); isCustomUnmarshaler(mvalPtr.Type()) {
		d.visitor.visitAll()

		if tval == nil {
			return mvalPtr.Elem(), nil
		}

		if err := callCustomUnmarshaler(mvalPtr, tval.ToMap()); err != nil {
			return reflect.ValueOf(nil), fmt.Errorf("unmarshal toml: %v", err)
		}
		return mvalPtr.Elem(), nil
	}

	var mval reflect.Value
	switch mtype.Kind() {
	case reflect.Struct:
		if mval1 != nil {
			mval = *mval1
		} else {
			mval = reflect.New(mtype).Elem()
		}

		switch mval.Interface().(type) {
		case Tree:
			mval.Set(reflect.ValueOf(tval).Elem())
		default:
			for i := 0; i < mtype.NumField(); i++ {
				mtypef := mtype.Field(i)
				an := annotation{tag: d.tagName}
				opts := tomlOptions(mtypef, an)
				if !opts.include {
					continue
				}
				baseKey := opts.name
				keysToTry := []string{
					baseKey,
					strings.ToLower(baseKey),
					strings.ToTitle(baseKey),
					strings.ToLower(string(baseKey[0])) + baseKey[1:],
				}

				found := false
				if tval != nil {
					for _, key := range keysToTry {
						exists := tval.HasPath([]string{key})
						if !exists {
							continue
						}

						d.visitor.push(key)
						val := tval.GetPath([]string{key})
						fval := mval.Field(i)
						mvalf, err := d.valueFromToml(mtypef.Type, val, &fval)
						if err != nil {
							return mval, formatError(err, tval.GetPositionPath([]string{key}))
						}
						mval.Field(i).Set(mvalf)
						found = true
						d.visitor.pop()
						break
					}
				}

				if !found && opts.defaultValue != "" {
					mvalf := mval.Field(i)
					var val interface{}
					var err error
					switch mvalf.Kind() {
					case reflect.String:
						val = opts.defaultValue
					case reflect.Bool:
						val, err = strconv.ParseBool(opts.defaultValue)
					case reflect.Uint:
						val, err = strconv.ParseUint(opts.defaultValue, 10, 0)
					case reflect.Uint8:
						val, err = strconv.ParseUint(opts.defaultValue, 10, 8)
					case reflect.Uint16:
						val, err = strconv.ParseUint(opts.defaultValue, 10, 16)
					case reflect.Uint32:
						val, err = strconv.ParseUint(opts.defaultValue, 10, 32)
					case reflect.Uint64:
						val, err = strconv.ParseUint(opts.defaultValue, 10, 64)
					case reflect.Int:
						val, err = strconv.ParseInt(opts.defaultValue, 10, 0)
					case reflect.Int8:
						val, err = strconv.ParseInt(opts.defaultValue, 10, 8)
					case reflect.Int16:
						val, err = strconv.ParseInt(opts.defaultValue, 10, 16)
					case reflect.Int32:
						val, err = strconv.ParseInt(opts.defaultValue, 10, 32)
					case reflect.Int64:
						// Check if the provided number has a non-numeric extension.
						var hasExtension bool
						if len(opts.defaultValue) > 0 {
							lastChar := opts.defaultValue[len(opts.defaultValue)-1]
							if lastChar < '0' || lastChar > '9' {
								hasExtension = true
							}
						}
						// If the value is a time.Duration with extension, parse as duration.
						// If the value is an int64 or a time.Duration without extension, parse as number.
						if hasExtension && mvalf.Type().String() == "time.Duration" {
							val, err = time.ParseDuration(opts.defaultValue)
						} else {
							val, err = strconv.ParseInt(opts.defaultValue, 10, 64)
						}
					case reflect.Float32:
						val, err = strconv.ParseFloat(opts.defaultValue, 32)
					case reflect.Float64:
						val, err = strconv.ParseFloat(opts.defaultValue, 64)
					default:
						return mvalf, fmt.Errorf("unsupported field type for default option")
					}

					if err != nil {
						return mvalf, err
					}
					mvalf.Set(reflect.ValueOf(val).Convert(mvalf.Type()))
				}

				// save the old behavior above and try to check structs
				if !found && opts.defaultValue == "" && mtypef.Type.Kind() == reflect.Struct {
					tmpTval := tval
					if !mtypef.Anonymous {
						tmpTval = nil
					}
					fval := mval.Field(i)
					v, err := d.valueFromTree(mtypef.Type, tmpTval, &fval)
					if err != nil {
						return v, err
					}
					mval.Field(i).Set(v)
				}
			}
		}
	case reflect.Map:
		mval = reflect.MakeMap(mtype)
		for _, key := range tval.Keys() {
			d.visitor.push(key)
			// TODO: path splits key
			val := tval.GetPath([]string{key})
			mvalf, err := d.valueFromToml(mtype.Elem(), val, nil)
			if err != nil {
				return mval, formatError(err, tval.GetPositionPath([]string{key}))
			}
			mval.SetMapIndex(reflect.ValueOf(key).Convert(mtype.Key()), mvalf)
			d.visitor.pop()
		}
	}
	return mval, nil
}

// Convert toml value to marshal struct/map slice, using marshal type
func (d *Decoder) valueFromTreeSlice(mtype reflect.Type, tval []*Tree) (reflect.Value, error) {
	mval, err := makeSliceOrArray(mtype, len(tval))
	if err != nil {
		return mval, err
	}

	for i := 0; i < len(tval); i++ {
		d.visitor.push(strconv.Itoa(i))
		val, err := d.valueFromTree(mtype.Elem(), tval[i], nil)
		if err != nil {
			return mval, err
		}
		mval.Index(i).Set(val)
		d.visitor.pop()
	}
	return mval, nil
}

// Convert toml value to marshal primitive slice, using marshal type
func (d *Decoder) valueFromOtherSlice(mtype reflect.Type, tval []interface{}) (reflect.Value, error) {
	mval, err := makeSliceOrArray(mtype, len(tval))
	if err != nil {
		return mval, err
	}

	for i := 0; i < len(tval); i++ {
		val, err := d.valueFromToml(mtype.Elem(), tval[i], nil)
		if err != nil {
			return mval, err
		}
		mval.Index(i).Set(val)
	}
	return mval, nil
}

// Convert toml value to marshal primitive slice, using marshal type
func (d *Decoder) valueFromOtherSliceI(mtype reflect.Type, tval interface{}) (reflect.Value, error) {
	val := reflect.ValueOf(tval)
	length := val.Len()

	mval, err := makeSliceOrArray(mtype, length)
	if err != nil {
		return mval, err
	}

	for i := 0; i < length; i++ {
		val, err := d.valueFromToml(mtype.Elem(), val.Index(i).Interface(), nil)
		if err != nil {
			return mval, err
		}
		mval.Index(i).Set(val)
	}
	return mval, nil
}

// Create a new slice or a new array with specified length
func makeSliceOrArray(mtype reflect.Type, tLength int) (reflect.Value, error) {
	var mval reflect.Value
	switch mtype.Kind() {
	case reflect.Slice:
		mval = reflect.MakeSlice(mtype, tLength, tLength)
	case reflect.Array:
		mval = reflect.New(reflect.ArrayOf(mtype.Len(), mtype.Elem())).Elem()
		if tLength > mtype.Len() {
			return mval, fmt.Errorf("unmarshal: TOML array length (%v) exceeds destination array length (%v)", tLength, mtype.Len())
		}
	}
	return mval, nil
}

// Convert toml value to marshal value, using marshal type. When mval1 is non-nil
// and the given type is a struct value, merge fields into it.
func (d *Decoder) valueFromToml(mtype reflect.Type, tval interface{}, mval1 *reflect.Value) (reflect.Value, error) {
	if mtype.Kind() == reflect.Ptr {
		return d.unwrapPointer(mtype, tval, mval1)
	}

	switch t := tval.(type) {
	case *Tree:
		var mval11 *reflect.Value
		if mtype.Kind() == reflect.Struct {
			mval11 = mval1
		}

		if isTree(mtype) {
			return d.valueFromTree(mtype, t, mval11)
		}

		if mtype.Kind() == reflect.Interface {
			if mval1 == nil || mval1.IsNil() {
				return d.valueFromTree(reflect.TypeOf(map[string]interface{}{}), t, nil)
			} else {
				return d.valueFromToml(mval1.Elem().Type(), t, nil)
			}
		}

		return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to a tree", tval, tval)
	case []*Tree:
		if isTreeSequence(mtype) {
			return d.valueFromTreeSlice(mtype, t)
		}
		if mtype.Kind() == reflect.Interface {
			if mval1 == nil || mval1.IsNil() {
				return d.valueFromTreeSlice(reflect.TypeOf([]map[string]interface{}{}), t)
			} else {
				ival := mval1.Elem()
				return d.valueFromToml(mval1.Elem().Type(), t, &ival)
			}
		}
		return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to trees", tval, tval)
	case []interface{}:
		d.visitor.visit()
		if isOtherSequence(mtype) {
			return d.valueFromOtherSlice(mtype, t)
		}
		if mtype.Kind() == reflect.Interface {
			if mval1 == nil || mval1.IsNil() {
				return d.valueFromOtherSlice(reflect.TypeOf([]interface{}{}), t)
			} else {
				ival := mval1.Elem()
				return d.valueFromToml(mval1.Elem().Type(), t, &ival)
			}
		}
		return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to a slice", tval, tval)
	default:
		d.visitor.visit()
		mvalPtr := reflect.New(mtype)

		// Check if pointer to value implements the Unmarshaler interface.
		if isCustomUnmarshaler(mvalPtr.Type()) {
			if err := callCustomUnmarshaler(mvalPtr, tval); err != nil {
				return reflect.ValueOf(nil), fmt.Errorf("unmarshal toml: %v", err)
			}
			return mvalPtr.Elem(), nil
		}

		// Check if pointer to value implements the encoding.TextUnmarshaler.
		if isTextUnmarshaler(mvalPtr.Type()) && !isTimeType(mtype) {
			if err := d.unmarshalText(tval, mvalPtr); err != nil {
				return reflect.ValueOf(nil), fmt.Errorf("unmarshal text: %v", err)
			}
			return mvalPtr.Elem(), nil
		}

		switch mtype.Kind() {
		case reflect.Bool, reflect.Struct:
			val := reflect.ValueOf(tval)

			switch val.Type() {
			case localDateType:
				localDate := val.Interface().(LocalDate)
				switch mtype {
				case timeType:
					return reflect.ValueOf(time.Date(localDate.Year, localDate.Month, localDate.Day, 0, 0, 0, 0, time.Local)), nil
				}
			case localDateTimeType:
				localDateTime := val.Interface().(LocalDateTime)
				switch mtype {
				case timeType:
					return reflect.ValueOf(time.Date(
						localDateTime.Date.Year,
						localDateTime.Date.Month,
						localDateTime.Date.Day,
						localDateTime.Time.Hour,
						localDateTime.Time.Minute,
						localDateTime.Time.Second,
						localDateTime.Time.Nanosecond,
						time.Local)), nil
				}
			}

			// if this passes for when mtype is reflect.Struct, tval is a time.LocalTime
			if !val.Type().ConvertibleTo(mtype) {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v", tval, tval, mtype.String())
			}

			return val.Convert(mtype), nil
		case reflect.String:
			val := reflect.ValueOf(tval)
			// stupidly, int64 is convertible to string. So special case this.
			if !val.Type().ConvertibleTo(mtype) || val.Kind() == reflect.Int64 {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v", tval, tval, mtype.String())
			}

			return val.Convert(mtype), nil
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			val := reflect.ValueOf(tval)
			if mtype.Kind() == reflect.Int64 && mtype == reflect.TypeOf(time.Duration(1)) && val.Kind() == reflect.String {
				d, err := time.ParseDuration(val.String())
				if err != nil {
					return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v. %s", tval, tval, mtype.String(), err)
				}
				return reflect.ValueOf(d), nil
			}
			if !val.Type().ConvertibleTo(mtype) || val.Kind() == reflect.Float64 {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v", tval, tval, mtype.String())
			}
			if reflect.Indirect(reflect.New(mtype)).OverflowInt(val.Convert(reflect.TypeOf(int64(0))).Int()) {
				return reflect.ValueOf(nil), fmt.Errorf("%v(%T) would overflow %v", tval, tval, mtype.String())
			}

			return val.Convert(mtype), nil
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			val := reflect.ValueOf(tval)
			if !val.Type().ConvertibleTo(mtype) || val.Kind() == reflect.Float64 {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v", tval, tval, mtype.String())
			}

			if val.Convert(reflect.TypeOf(int(1))).Int() < 0 {
				return reflect.ValueOf(nil), fmt.Errorf("%v(%T) is negative so does not fit in %v", tval, tval, mtype.String())
			}
			if reflect.Indirect(reflect.New(mtype)).OverflowUint(val.Convert(reflect.TypeOf(uint64(0))).Uint()) {
				return reflect.ValueOf(nil), fmt.Errorf("%v(%T) would overflow %v", tval, tval, mtype.String())
			}

			return val.Convert(mtype), nil
		case reflect.Float32, reflect.Float64:
			val := reflect.ValueOf(tval)
			if !val.Type().ConvertibleTo(mtype) || val.Kind() == reflect.Int64 {
				return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v", tval, tval, mtype.String())
			}
			if reflect.Indirect(reflect.New(mtype)).OverflowFloat(val.Convert(reflect.TypeOf(float64(0))).Float()) {
				return reflect.ValueOf(nil), fmt.Errorf("%v(%T) would overflow %v", tval, tval, mtype.String())
			}

			return val.Convert(mtype), nil
		case reflect.Interface:
			if mval1 == nil || mval1.IsNil() {
				return reflect.ValueOf(tval), nil
			} else {
				ival := mval1.Elem()
				return d.valueFromToml(mval1.Elem().Type(), t, &ival)
			}
		case reflect.Slice, reflect.Array:
			if isOtherSequence(mtype) && isOtherSequence(reflect.TypeOf(t)) {
				return d.valueFromOtherSliceI(mtype, t)
			}
			return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v(%v)", tval, tval, mtype, mtype.Kind())
		default:
			return reflect.ValueOf(nil), fmt.Errorf("Can't convert %v(%T) to %v(%v)", tval, tval, mtype, mtype.Kind())
		}
	}
}

func (d *Decoder) unwrapPointer(mtype reflect.Type, tval interface{}, mval1 *reflect.Value) (reflect.Value, error) {
	var melem *reflect.Value

	if mval1 != nil && !mval1.IsNil() && (mtype.Elem().Kind() == reflect.Struct || mtype.Elem().Kind() == reflect.Interface) {
		elem := mval1.Elem()
		melem = &elem
	}

	val, err := d.valueFromToml(mtype.Elem(), tval, melem)
	if err != nil {
		return reflect.ValueOf(nil), err
	}
	mval := reflect.New(mtype.Elem())
	mval.Elem().Set(val)
	return mval, nil
}

func (d *Decoder) unmarshalText(tval interface{}, mval reflect.Value) error {
	var buf bytes.Buffer
	fmt.Fprint(&buf, tval)
	return callTextUnmarshaler(mval, buf.Bytes())
}

func tomlOptions(vf reflect.StructField, an annotation) tomlOpts {
	tag := vf.Tag.Get(an.tag)
	parse := strings.Split(tag, ",")
	var comment string
	if c := vf.Tag.Get(an.comment); c != "" {
		comment = c
	}
	commented, _ := strconv.ParseBool(vf.Tag.Get(an.commented))
	multiline, _ := strconv.ParseBool(vf.Tag.Get(an.multiline))
	literal, _ := strconv.ParseBool(vf.Tag.Get(an.literal))
	defaultValue := vf.Tag.Get(tagDefault)
	result := tomlOpts{
		name:         vf.Name,
		nameFromTag:  false,
		comment:      comment,
		commented:    commented,
		multiline:    multiline,
		literal:      literal,
		include:      true,
		omitempty:    false,
		defaultValue: defaultValue,
	}
	if parse[0] != "" {
		if parse[0] == "-" && len(parse) == 1 {
			result.include = false
		} else {
			result.name = strings.Trim(parse[0], " ")
			result.nameFromTag = true
		}
	}
	if vf.PkgPath != "" {
		result.include = false
	}
	if len(parse) > 1 && strings.Trim(parse[1], " ") == "omitempty" {
		result.omitempty = true
	}
	if vf.Type.Kind() == reflect.Ptr {
		result.omitempty = true
	}
	return result
}

func isZero(val reflect.Value) bool {
	switch val.Type().Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		return val.Len() == 0
	default:
		return reflect.DeepEqual(val.Interface(), reflect.Zero(val.Type()).Interface())
	}
}

func formatError(err error, pos Position) error {
	if err.Error()[0] == '(' { // Error already contains position information
		return err
	}
	return fmt.Errorf("%s: %s", pos, err)
}

// visitorState keeps track of which keys were unmarshaled.
type visitorState struct {
	tree   *Tree
	path   []string
	keys   map[string]struct{}
	active bool
}

func newVisitorState(tree *Tree) visitorState {
	path, result := []string{}, map[string]struct{}{}
	insertKeys(path, result, tree)
	return visitorState{
		tree:   tree,
		path:   path[:0],
		keys:   result,
		active: true,
	}
}

func (s *visitorState) push(key string) {
	if s.active {
		s.path = append(s.path, key)
	}
}

func (s *visitorState) pop() {
	if s.active {
		s.path = s.path[:len(s.path)-1]
	}
}

func (s *visitorState) visit() {
	if s.active {
		delete(s.keys, strings.Join(s.path, "."))
	}
}

func (s *visitorState) visitAll() {
	if s.active {
		for k := range s.keys {
			if strings.HasPrefix(k, strings.Join(s.path, ".")) {
				delete(s.keys, k)
			}
		}
	}
}

func (s *visitorState) validate() error {
	if !s.active {
		return nil
	}
	undecoded := make([]string, 0, len(s.keys))
	for key := range s.keys {
		undecoded = append(undecoded, key)
	}
	sort.Strings(undecoded)
	if len(undecoded) > 0 {
		return fmt.Errorf("undecoded keys: %q", undecoded)
	}
	return nil
}

func insertKeys(path []string, m map[string]struct{}, tree *Tree) {
	for k, v := range tree.values {
		switch node := v.(type) {
		case []*Tree:
			for i, item := range node {
				insertKeys(append(path, k, strconv.Itoa(i)), m, item)
			}
		case *Tree:
			insertKeys(append(path, k), m, node)
		case *tomlValue:
			m[strings.Join(append(path, k), ".")] = struct{}{}
		}
	}
}
