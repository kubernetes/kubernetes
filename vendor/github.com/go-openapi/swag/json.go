// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"sync"

	"github.com/mailru/easyjson/jlexer"
	"github.com/mailru/easyjson/jwriter"
)

// DefaultJSONNameProvider the default cache for types
var DefaultJSONNameProvider = NewNameProvider()

const comma = byte(',')

var closers = map[byte]byte{
	'{': '}',
	'[': ']',
}

type ejMarshaler interface {
	MarshalEasyJSON(w *jwriter.Writer)
}

type ejUnmarshaler interface {
	UnmarshalEasyJSON(w *jlexer.Lexer)
}

// WriteJSON writes json data, prefers finding an appropriate interface to short-circuit the marshaller
// so it takes the fastest option available.
func WriteJSON(data interface{}) ([]byte, error) {
	if d, ok := data.(ejMarshaler); ok {
		jw := new(jwriter.Writer)
		d.MarshalEasyJSON(jw)
		return jw.BuildBytes()
	}
	if d, ok := data.(json.Marshaler); ok {
		return d.MarshalJSON()
	}
	return json.Marshal(data)
}

// ReadJSON reads json data, prefers finding an appropriate interface to short-circuit the unmarshaller
// so it takes the fastes option available
func ReadJSON(data []byte, value interface{}) error {
	if d, ok := value.(ejUnmarshaler); ok {
		jl := &jlexer.Lexer{Data: data}
		d.UnmarshalEasyJSON(jl)
		return jl.Error()
	}
	if d, ok := value.(json.Unmarshaler); ok {
		return d.UnmarshalJSON(data)
	}
	return json.Unmarshal(data, value)
}

// DynamicJSONToStruct converts an untyped json structure into a struct
func DynamicJSONToStruct(data interface{}, target interface{}) error {
	// TODO: convert straight to a json typed map  (mergo + iterate?)
	b, err := WriteJSON(data)
	if err != nil {
		return err
	}
	if err := ReadJSON(b, target); err != nil {
		return err
	}
	return nil
}

// ConcatJSON concatenates multiple json objects efficiently
func ConcatJSON(blobs ...[]byte) []byte {
	if len(blobs) == 0 {
		return nil
	}
	if len(blobs) == 1 {
		return blobs[0]
	}

	last := len(blobs) - 1
	var opening, closing byte
	a := 0
	idx := 0
	buf := bytes.NewBuffer(nil)

	for i, b := range blobs {
		if len(b) > 0 && opening == 0 { // is this an array or an object?
			opening, closing = b[0], closers[b[0]]
		}

		if opening != '{' && opening != '[' {
			continue // don't know how to concatenate non container objects
		}

		if len(b) < 3 { // yep empty but also the last one, so closing this thing
			if i == last && a > 0 {
				buf.WriteByte(closing)
			}
			continue
		}

		idx = 0
		if a > 0 { // we need to join with a comma for everything beyond the first non-empty item
			buf.WriteByte(comma)
			idx = 1 // this is not the first or the last so we want to drop the leading bracket
		}

		if i != last { // not the last one, strip brackets
			buf.Write(b[idx : len(b)-1])
		} else { // last one, strip only the leading bracket
			buf.Write(b[idx:])
		}
		a++
	}
	// somehow it ended up being empty, so provide a default value
	if buf.Len() == 0 {
		buf.WriteByte(opening)
		buf.WriteByte(closing)
	}
	return buf.Bytes()
}

// ToDynamicJSON turns an object into a properly JSON typed structure
func ToDynamicJSON(data interface{}) interface{} {
	// TODO: convert straight to a json typed map (mergo + iterate?)
	b, _ := json.Marshal(data)
	var res interface{}
	json.Unmarshal(b, &res)
	return res
}

// FromDynamicJSON turns an object into a properly JSON typed structure
func FromDynamicJSON(data, target interface{}) error {
	b, _ := json.Marshal(data)
	return json.Unmarshal(b, target)
}

// NameProvider represents an object capabale of translating from go property names
// to json property names
// This type is thread-safe.
type NameProvider struct {
	lock  *sync.Mutex
	index map[reflect.Type]nameIndex
}

type nameIndex struct {
	jsonNames map[string]string
	goNames   map[string]string
}

// NewNameProvider creates a new name provider
func NewNameProvider() *NameProvider {
	return &NameProvider{
		lock:  &sync.Mutex{},
		index: make(map[reflect.Type]nameIndex),
	}
}

func buildnameIndex(tpe reflect.Type, idx, reverseIdx map[string]string) {
	for i := 0; i < tpe.NumField(); i++ {
		targetDes := tpe.Field(i)

		if targetDes.PkgPath != "" { // unexported
			continue
		}

		if targetDes.Anonymous { // walk embedded structures tree down first
			buildnameIndex(targetDes.Type, idx, reverseIdx)
			continue
		}

		if tag := targetDes.Tag.Get("json"); tag != "" {

			parts := strings.Split(tag, ",")
			if len(parts) == 0 {
				continue
			}

			nm := parts[0]
			if nm == "-" {
				continue
			}
			if nm == "" { // empty string means we want to use the Go name
				nm = targetDes.Name
			}

			idx[nm] = targetDes.Name
			reverseIdx[targetDes.Name] = nm
		}
	}
}

func newNameIndex(tpe reflect.Type) nameIndex {
	var idx = make(map[string]string, tpe.NumField())
	var reverseIdx = make(map[string]string, tpe.NumField())

	buildnameIndex(tpe, idx, reverseIdx)
	return nameIndex{jsonNames: idx, goNames: reverseIdx}
}

// GetJSONNames gets all the json property names for a type
func (n *NameProvider) GetJSONNames(subject interface{}) []string {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
	names, ok := n.index[tpe]
	if !ok {
		names = n.makeNameIndex(tpe)
	}

	var res []string
	for k := range names.jsonNames {
		res = append(res, k)
	}
	return res
}

// GetJSONName gets the json name for a go property name
func (n *NameProvider) GetJSONName(subject interface{}, name string) (string, bool) {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
	return n.GetJSONNameForType(tpe, name)
}

// GetJSONNameForType gets the json name for a go property name on a given type
func (n *NameProvider) GetJSONNameForType(tpe reflect.Type, name string) (string, bool) {
	names, ok := n.index[tpe]
	if !ok {
		names = n.makeNameIndex(tpe)
	}
	nme, ok := names.goNames[name]
	return nme, ok
}

func (n *NameProvider) makeNameIndex(tpe reflect.Type) nameIndex {
	n.lock.Lock()
	defer n.lock.Unlock()
	names := newNameIndex(tpe)
	n.index[tpe] = names
	return names
}

// GetGoName gets the go name for a json property name
func (n *NameProvider) GetGoName(subject interface{}, name string) (string, bool) {
	tpe := reflect.Indirect(reflect.ValueOf(subject)).Type()
	return n.GetGoNameForType(tpe, name)
}

// GetGoNameForType gets the go name for a given type for a json property name
func (n *NameProvider) GetGoNameForType(tpe reflect.Type, name string) (string, bool) {
	names, ok := n.index[tpe]
	if !ok {
		names = n.makeNameIndex(tpe)
	}
	nme, ok := names.jsonNames[name]
	return nme, ok
}
