//
// Copyright (c) 2011-2019 Canonical Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package yaml implements YAML support for the Go language.
//
// Source code and other details for the project are available at GitHub:
//
//   https://github.com/go-yaml/yaml
//
package yaml

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"unicode/utf8"
)

// The Unmarshaler interface may be implemented by types to customize their
// behavior when being unmarshaled from a YAML document.
type Unmarshaler interface {
	UnmarshalYAML(value *Node) error
}

type obsoleteUnmarshaler interface {
	UnmarshalYAML(unmarshal func(interface{}) error) error
}

// The Marshaler interface may be implemented by types to customize their
// behavior when being marshaled into a YAML document. The returned value
// is marshaled in place of the original value implementing Marshaler.
//
// If an error is returned by MarshalYAML, the marshaling procedure stops
// and returns with the provided error.
type Marshaler interface {
	MarshalYAML() (interface{}, error)
}

// Unmarshal decodes the first document found within the in byte slice
// and assigns decoded values into the out value.
//
// Maps and pointers (to a struct, string, int, etc) are accepted as out
// values. If an internal pointer within a struct is not initialized,
// the yaml package will initialize it if necessary for unmarshalling
// the provided data. The out parameter must not be nil.
//
// The type of the decoded values should be compatible with the respective
// values in out. If one or more values cannot be decoded due to a type
// mismatches, decoding continues partially until the end of the YAML
// content, and a *yaml.TypeError is returned with details for all
// missed values.
//
// Struct fields are only unmarshalled if they are exported (have an
// upper case first letter), and are unmarshalled using the field name
// lowercased as the default key. Custom keys may be defined via the
// "yaml" name in the field tag: the content preceding the first comma
// is used as the key, and the following comma-separated options are
// used to tweak the marshalling process (see Marshal).
// Conflicting names result in a runtime error.
//
// For example:
//
//     type T struct {
//         F int `yaml:"a,omitempty"`
//         B int
//     }
//     var t T
//     yaml.Unmarshal([]byte("a: 1\nb: 2"), &t)
//
// See the documentation of Marshal for the format of tags and a list of
// supported tag options.
//
func Unmarshal(in []byte, out interface{}) (err error) {
	return unmarshal(in, out, false)
}

// A Decoder reads and decodes YAML values from an input stream.
type Decoder struct {
	parser      *parser
	knownFields bool
}

// NewDecoder returns a new decoder that reads from r.
//
// The decoder introduces its own buffering and may read
// data from r beyond the YAML values requested.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{
		parser: newParserFromReader(r),
	}
}

// KnownFields ensures that the keys in decoded mappings to
// exist as fields in the struct being decoded into.
func (dec *Decoder) KnownFields(enable bool) {
	dec.knownFields = enable
}

// Decode reads the next YAML-encoded value from its input
// and stores it in the value pointed to by v.
//
// See the documentation for Unmarshal for details about the
// conversion of YAML into a Go value.
func (dec *Decoder) Decode(v interface{}) (err error) {
	d := newDecoder()
	d.knownFields = dec.knownFields
	defer handleErr(&err)
	node := dec.parser.parse()
	if node == nil {
		return io.EOF
	}
	out := reflect.ValueOf(v)
	if out.Kind() == reflect.Ptr && !out.IsNil() {
		out = out.Elem()
	}
	d.unmarshal(node, out)
	if len(d.terrors) > 0 {
		return &TypeError{d.terrors}
	}
	return nil
}

// Decode decodes the node and stores its data into the value pointed to by v.
//
// See the documentation for Unmarshal for details about the
// conversion of YAML into a Go value.
func (n *Node) Decode(v interface{}) (err error) {
	d := newDecoder()
	defer handleErr(&err)
	out := reflect.ValueOf(v)
	if out.Kind() == reflect.Ptr && !out.IsNil() {
		out = out.Elem()
	}
	d.unmarshal(n, out)
	if len(d.terrors) > 0 {
		return &TypeError{d.terrors}
	}
	return nil
}

func unmarshal(in []byte, out interface{}, strict bool) (err error) {
	defer handleErr(&err)
	d := newDecoder()
	p := newParser(in)
	defer p.destroy()
	node := p.parse()
	if node != nil {
		v := reflect.ValueOf(out)
		if v.Kind() == reflect.Ptr && !v.IsNil() {
			v = v.Elem()
		}
		d.unmarshal(node, v)
	}
	if len(d.terrors) > 0 {
		return &TypeError{d.terrors}
	}
	return nil
}

// Marshal serializes the value provided into a YAML document. The structure
// of the generated document will reflect the structure of the value itself.
// Maps and pointers (to struct, string, int, etc) are accepted as the in value.
//
// Struct fields are only marshalled if they are exported (have an upper case
// first letter), and are marshalled using the field name lowercased as the
// default key. Custom keys may be defined via the "yaml" name in the field
// tag: the content preceding the first comma is used as the key, and the
// following comma-separated options are used to tweak the marshalling process.
// Conflicting names result in a runtime error.
//
// The field tag format accepted is:
//
//     `(...) yaml:"[<key>][,<flag1>[,<flag2>]]" (...)`
//
// The following flags are currently supported:
//
//     omitempty    Only include the field if it's not set to the zero
//                  value for the type or to empty slices or maps.
//                  Zero valued structs will be omitted if all their public
//                  fields are zero, unless they implement an IsZero
//                  method (see the IsZeroer interface type), in which
//                  case the field will be excluded if IsZero returns true.
//
//     flow         Marshal using a flow style (useful for structs,
//                  sequences and maps).
//
//     inline       Inline the field, which must be a struct or a map,
//                  causing all of its fields or keys to be processed as if
//                  they were part of the outer struct. For maps, keys must
//                  not conflict with the yaml keys of other struct fields.
//
// In addition, if the key is "-", the field is ignored.
//
// For example:
//
//     type T struct {
//         F int `yaml:"a,omitempty"`
//         B int
//     }
//     yaml.Marshal(&T{B: 2}) // Returns "b: 2\n"
//     yaml.Marshal(&T{F: 1}} // Returns "a: 1\nb: 0\n"
//
func Marshal(in interface{}) (out []byte, err error) {
	defer handleErr(&err)
	e := newEncoder()
	defer e.destroy()
	e.marshalDoc("", reflect.ValueOf(in))
	e.finish()
	out = e.out
	return
}

// An Encoder writes YAML values to an output stream.
type Encoder struct {
	encoder *encoder
}

// NewEncoder returns a new encoder that writes to w.
// The Encoder should be closed after use to flush all data
// to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		encoder: newEncoderWithWriter(w),
	}
}

// Encode writes the YAML encoding of v to the stream.
// If multiple items are encoded to the stream, the
// second and subsequent document will be preceded
// with a "---" document separator, but the first will not.
//
// See the documentation for Marshal for details about the conversion of Go
// values to YAML.
func (e *Encoder) Encode(v interface{}) (err error) {
	defer handleErr(&err)
	e.encoder.marshalDoc("", reflect.ValueOf(v))
	return nil
}

// Encode encodes value v and stores its representation in n.
//
// See the documentation for Marshal for details about the
// conversion of Go values into YAML.
func (n *Node) Encode(v interface{}) (err error) {
	defer handleErr(&err)
	e := newEncoder()
	defer e.destroy()
	e.marshalDoc("", reflect.ValueOf(v))
	e.finish()
	p := newParser(e.out)
	p.textless = true
	defer p.destroy()
	doc := p.parse()
	*n = *doc.Content[0]
	return nil
}

// SetIndent changes the used indentation used when encoding.
func (e *Encoder) SetIndent(spaces int) {
	if spaces < 0 {
		panic("yaml: cannot indent to a negative number of spaces")
	}
	e.encoder.indent = spaces
}

// Close closes the encoder by writing any remaining data.
// It does not write a stream terminating string "...".
func (e *Encoder) Close() (err error) {
	defer handleErr(&err)
	e.encoder.finish()
	return nil
}

func handleErr(err *error) {
	if v := recover(); v != nil {
		if e, ok := v.(yamlError); ok {
			*err = e.err
		} else {
			panic(v)
		}
	}
}

type yamlError struct {
	err error
}

func fail(err error) {
	panic(yamlError{err})
}

func failf(format string, args ...interface{}) {
	panic(yamlError{fmt.Errorf("yaml: "+format, args...)})
}

// A TypeError is returned by Unmarshal when one or more fields in
// the YAML document cannot be properly decoded into the requested
// types. When this error is returned, the value is still
// unmarshaled partially.
type TypeError struct {
	Errors []string
}

func (e *TypeError) Error() string {
	return fmt.Sprintf("yaml: unmarshal errors:\n  %s", strings.Join(e.Errors, "\n  "))
}

type Kind uint32

const (
	DocumentNode Kind = 1 << iota
	SequenceNode
	MappingNode
	ScalarNode
	AliasNode
)

type Style uint32

const (
	TaggedStyle Style = 1 << iota
	DoubleQuotedStyle
	SingleQuotedStyle
	LiteralStyle
	FoldedStyle
	FlowStyle
)

// Node represents an element in the YAML document hierarchy. While documents
// are typically encoded and decoded into higher level types, such as structs
// and maps, Node is an intermediate representation that allows detailed
// control over the content being decoded or encoded.
//
// It's worth noting that although Node offers access into details such as
// line numbers, colums, and comments, the content when re-encoded will not
// have its original textual representation preserved. An effort is made to
// render the data plesantly, and to preserve comments near the data they
// describe, though.
//
// Values that make use of the Node type interact with the yaml package in the
// same way any other type would do, by encoding and decoding yaml data
// directly or indirectly into them.
//
// For example:
//
//     var person struct {
//             Name    string
//             Address yaml.Node
//     }
//     err := yaml.Unmarshal(data, &person)
// 
// Or by itself:
//
//     var person Node
//     err := yaml.Unmarshal(data, &person)
//
type Node struct {
	// Kind defines whether the node is a document, a mapping, a sequence,
	// a scalar value, or an alias to another node. The specific data type of
	// scalar nodes may be obtained via the ShortTag and LongTag methods.
	Kind  Kind

	// Style allows customizing the apperance of the node in the tree.
	Style Style

	// Tag holds the YAML tag defining the data type for the value.
	// When decoding, this field will always be set to the resolved tag,
	// even when it wasn't explicitly provided in the YAML content.
	// When encoding, if this field is unset the value type will be
	// implied from the node properties, and if it is set, it will only
	// be serialized into the representation if TaggedStyle is used or
	// the implicit tag diverges from the provided one.
	Tag string

	// Value holds the unescaped and unquoted represenation of the value.
	Value string

	// Anchor holds the anchor name for this node, which allows aliases to point to it.
	Anchor string

	// Alias holds the node that this alias points to. Only valid when Kind is AliasNode.
	Alias *Node

	// Content holds contained nodes for documents, mappings, and sequences.
	Content []*Node

	// HeadComment holds any comments in the lines preceding the node and
	// not separated by an empty line.
	HeadComment string

	// LineComment holds any comments at the end of the line where the node is in.
	LineComment string

	// FootComment holds any comments following the node and before empty lines.
	FootComment string

	// Line and Column hold the node position in the decoded YAML text.
	// These fields are not respected when encoding the node.
	Line   int
	Column int
}

// IsZero returns whether the node has all of its fields unset.
func (n *Node) IsZero() bool {
	return n.Kind == 0 && n.Style == 0 && n.Tag == "" && n.Value == "" && n.Anchor == "" && n.Alias == nil && n.Content == nil &&
		n.HeadComment == "" && n.LineComment == "" && n.FootComment == "" && n.Line == 0 && n.Column == 0
}


// LongTag returns the long form of the tag that indicates the data type for
// the node. If the Tag field isn't explicitly defined, one will be computed
// based on the node properties.
func (n *Node) LongTag() string {
	return longTag(n.ShortTag())
}

// ShortTag returns the short form of the YAML tag that indicates data type for
// the node. If the Tag field isn't explicitly defined, one will be computed
// based on the node properties.
func (n *Node) ShortTag() string {
	if n.indicatedString() {
		return strTag
	}
	if n.Tag == "" || n.Tag == "!" {
		switch n.Kind {
		case MappingNode:
			return mapTag
		case SequenceNode:
			return seqTag
		case AliasNode:
			if n.Alias != nil {
				return n.Alias.ShortTag()
			}
		case ScalarNode:
			tag, _ := resolve("", n.Value)
			return tag
		case 0:
			// Special case to make the zero value convenient.
			if n.IsZero() {
				return nullTag
			}
		}
		return ""
	}
	return shortTag(n.Tag)
}

func (n *Node) indicatedString() bool {
	return n.Kind == ScalarNode &&
		(shortTag(n.Tag) == strTag ||
			(n.Tag == "" || n.Tag == "!") && n.Style&(SingleQuotedStyle|DoubleQuotedStyle|LiteralStyle|FoldedStyle) != 0)
}

// SetString is a convenience function that sets the node to a string value
// and defines its style in a pleasant way depending on its content.
func (n *Node) SetString(s string) {
	n.Kind = ScalarNode
	if utf8.ValidString(s) {
		n.Value = s
		n.Tag = strTag
	} else {
		n.Value = encodeBase64(s)
		n.Tag = binaryTag
	}
	if strings.Contains(n.Value, "\n") {
		n.Style = LiteralStyle
	}
}

// --------------------------------------------------------------------------
// Maintain a mapping of keys to structure field indexes

// The code in this section was copied from mgo/bson.

// structInfo holds details for the serialization of fields of
// a given struct.
type structInfo struct {
	FieldsMap  map[string]fieldInfo
	FieldsList []fieldInfo

	// InlineMap is the number of the field in the struct that
	// contains an ,inline map, or -1 if there's none.
	InlineMap int

	// InlineUnmarshalers holds indexes to inlined fields that
	// contain unmarshaler values.
	InlineUnmarshalers [][]int
}

type fieldInfo struct {
	Key       string
	Num       int
	OmitEmpty bool
	Flow      bool
	// Id holds the unique field identifier, so we can cheaply
	// check for field duplicates without maintaining an extra map.
	Id int

	// Inline holds the field index if the field is part of an inlined struct.
	Inline []int
}

var structMap = make(map[reflect.Type]*structInfo)
var fieldMapMutex sync.RWMutex
var unmarshalerType reflect.Type

func init() {
	var v Unmarshaler
	unmarshalerType = reflect.ValueOf(&v).Elem().Type()
}

func getStructInfo(st reflect.Type) (*structInfo, error) {
	fieldMapMutex.RLock()
	sinfo, found := structMap[st]
	fieldMapMutex.RUnlock()
	if found {
		return sinfo, nil
	}

	n := st.NumField()
	fieldsMap := make(map[string]fieldInfo)
	fieldsList := make([]fieldInfo, 0, n)
	inlineMap := -1
	inlineUnmarshalers := [][]int(nil)
	for i := 0; i != n; i++ {
		field := st.Field(i)
		if field.PkgPath != "" && !field.Anonymous {
			continue // Private field
		}

		info := fieldInfo{Num: i}

		tag := field.Tag.Get("yaml")
		if tag == "" && strings.Index(string(field.Tag), ":") < 0 {
			tag = string(field.Tag)
		}
		if tag == "-" {
			continue
		}

		inline := false
		fields := strings.Split(tag, ",")
		if len(fields) > 1 {
			for _, flag := range fields[1:] {
				switch flag {
				case "omitempty":
					info.OmitEmpty = true
				case "flow":
					info.Flow = true
				case "inline":
					inline = true
				default:
					return nil, errors.New(fmt.Sprintf("unsupported flag %q in tag %q of type %s", flag, tag, st))
				}
			}
			tag = fields[0]
		}

		if inline {
			switch field.Type.Kind() {
			case reflect.Map:
				if inlineMap >= 0 {
					return nil, errors.New("multiple ,inline maps in struct " + st.String())
				}
				if field.Type.Key() != reflect.TypeOf("") {
					return nil, errors.New("option ,inline needs a map with string keys in struct " + st.String())
				}
				inlineMap = info.Num
			case reflect.Struct, reflect.Ptr:
				ftype := field.Type
				for ftype.Kind() == reflect.Ptr {
					ftype = ftype.Elem()
				}
				if ftype.Kind() != reflect.Struct {
					return nil, errors.New("option ,inline may only be used on a struct or map field")
				}
				if reflect.PtrTo(ftype).Implements(unmarshalerType) {
					inlineUnmarshalers = append(inlineUnmarshalers, []int{i})
				} else {
					sinfo, err := getStructInfo(ftype)
					if err != nil {
						return nil, err
					}
					for _, index := range sinfo.InlineUnmarshalers {
						inlineUnmarshalers = append(inlineUnmarshalers, append([]int{i}, index...))
					}
					for _, finfo := range sinfo.FieldsList {
						if _, found := fieldsMap[finfo.Key]; found {
							msg := "duplicated key '" + finfo.Key + "' in struct " + st.String()
							return nil, errors.New(msg)
						}
						if finfo.Inline == nil {
							finfo.Inline = []int{i, finfo.Num}
						} else {
							finfo.Inline = append([]int{i}, finfo.Inline...)
						}
						finfo.Id = len(fieldsList)
						fieldsMap[finfo.Key] = finfo
						fieldsList = append(fieldsList, finfo)
					}
				}
			default:
				return nil, errors.New("option ,inline may only be used on a struct or map field")
			}
			continue
		}

		if tag != "" {
			info.Key = tag
		} else {
			info.Key = strings.ToLower(field.Name)
		}

		if _, found = fieldsMap[info.Key]; found {
			msg := "duplicated key '" + info.Key + "' in struct " + st.String()
			return nil, errors.New(msg)
		}

		info.Id = len(fieldsList)
		fieldsList = append(fieldsList, info)
		fieldsMap[info.Key] = info
	}

	sinfo = &structInfo{
		FieldsMap:          fieldsMap,
		FieldsList:         fieldsList,
		InlineMap:          inlineMap,
		InlineUnmarshalers: inlineUnmarshalers,
	}

	fieldMapMutex.Lock()
	structMap[st] = sinfo
	fieldMapMutex.Unlock()
	return sinfo, nil
}

// IsZeroer is used to check whether an object is zero to
// determine whether it should be omitted when marshaling
// with the omitempty flag. One notable implementation
// is time.Time.
type IsZeroer interface {
	IsZero() bool
}

func isZero(v reflect.Value) bool {
	kind := v.Kind()
	if z, ok := v.Interface().(IsZeroer); ok {
		if (kind == reflect.Ptr || kind == reflect.Interface) && v.IsNil() {
			return true
		}
		return z.IsZero()
	}
	switch kind {
	case reflect.String:
		return len(v.String()) == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	case reflect.Slice:
		return v.Len() == 0
	case reflect.Map:
		return v.Len() == 0
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Struct:
		vt := v.Type()
		for i := v.NumField() - 1; i >= 0; i-- {
			if vt.Field(i).PkgPath != "" {
				continue // Private field
			}
			if !isZero(v.Field(i)) {
				return false
			}
		}
		return true
	}
	return false
}
