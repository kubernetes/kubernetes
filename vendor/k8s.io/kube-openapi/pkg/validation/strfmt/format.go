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

package strfmt

import (
	"encoding"
	"encoding/json"
	"reflect"
	"strings"
	"sync"

	"k8s.io/kube-openapi/pkg/validation/errors"
)

// Default is the default formats registry
var Default = NewSeededFormats(nil, nil)

// Validator represents a validator for a string format.
type Validator func(string) bool

// Format represents a string format.
//
// All implementations of Format provide a string representation and text
// marshaling/unmarshaling interface to be used by encoders (e.g. encoding/json).
type Format interface {
	String() string
	encoding.TextMarshaler
	encoding.TextUnmarshaler
}

// Registry is a registry of string formats, with a validation method.
type Registry interface {
	Add(string, Format, Validator) bool
	DelByName(string) bool
	GetType(string) (reflect.Type, bool)
	ContainsName(string) bool
	Validates(string, string) bool
	Parse(string, string) (interface{}, error)
}

type knownFormat struct {
	Name      string
	OrigName  string
	Type      reflect.Type
	Validator Validator
}

// NameNormalizer is a function that normalizes a format name.
type NameNormalizer func(string) string

// DefaultNameNormalizer removes all dashes
func DefaultNameNormalizer(name string) string {
	return strings.Replace(name, "-", "", -1)
}

type defaultFormats struct {
	sync.Mutex
	data          []knownFormat
	normalizeName NameNormalizer
}

// NewFormats creates a new formats registry seeded with the values from the default
func NewFormats() Registry {
	return NewSeededFormats(Default.(*defaultFormats).data, nil)
}

// NewSeededFormats creates a new formats registry
func NewSeededFormats(seeds []knownFormat, normalizer NameNormalizer) Registry {
	if normalizer == nil {
		normalizer = DefaultNameNormalizer
	}
	// copy here, don't modify original
	d := append([]knownFormat(nil), seeds...)
	return &defaultFormats{
		data:          d,
		normalizeName: normalizer,
	}
}

// Add adds a new format, return true if this was a new item instead of a replacement
func (f *defaultFormats) Add(name string, strfmt Format, validator Validator) bool {
	f.Lock()
	defer f.Unlock()

	nme := f.normalizeName(name)

	tpe := reflect.TypeOf(strfmt)
	if tpe.Kind() == reflect.Ptr {
		tpe = tpe.Elem()
	}

	for i := range f.data {
		v := &f.data[i]
		if v.Name == nme {
			v.Type = tpe
			v.Validator = validator
			return false
		}
	}

	// turns out it's new after all
	f.data = append(f.data, knownFormat{Name: nme, OrigName: name, Type: tpe, Validator: validator})
	return true
}

// GetType gets the type for the specified name
func (f *defaultFormats) GetType(name string) (reflect.Type, bool) {
	f.Lock()
	defer f.Unlock()
	nme := f.normalizeName(name)
	for _, v := range f.data {
		if v.Name == nme {
			return v.Type, true
		}
	}
	return nil, false
}

// DelByName removes the format by the specified name, returns true when an item was actually removed
func (f *defaultFormats) DelByName(name string) bool {
	f.Lock()
	defer f.Unlock()

	nme := f.normalizeName(name)

	for i, v := range f.data {
		if v.Name == nme {
			f.data[i] = knownFormat{} // release
			f.data = append(f.data[:i], f.data[i+1:]...)
			return true
		}
	}
	return false
}

// DelByFormat removes the specified format, returns true when an item was actually removed
func (f *defaultFormats) DelByFormat(strfmt Format) bool {
	f.Lock()
	defer f.Unlock()

	tpe := reflect.TypeOf(strfmt)
	if tpe.Kind() == reflect.Ptr {
		tpe = tpe.Elem()
	}

	for i, v := range f.data {
		if v.Type == tpe {
			f.data[i] = knownFormat{} // release
			f.data = append(f.data[:i], f.data[i+1:]...)
			return true
		}
	}
	return false
}

// ContainsName returns true if this registry contains the specified name
func (f *defaultFormats) ContainsName(name string) bool {
	f.Lock()
	defer f.Unlock()
	nme := f.normalizeName(name)
	for _, v := range f.data {
		if v.Name == nme {
			return true
		}
	}
	return false
}

// ContainsFormat returns true if this registry contains the specified format
func (f *defaultFormats) ContainsFormat(strfmt Format) bool {
	f.Lock()
	defer f.Unlock()
	tpe := reflect.TypeOf(strfmt)
	if tpe.Kind() == reflect.Ptr {
		tpe = tpe.Elem()
	}

	for _, v := range f.data {
		if v.Type == tpe {
			return true
		}
	}
	return false
}

// Validates passed data against format.
//
// Note that the format name is automatically normalized, e.g. one may
// use "date-time" to use the "datetime" format validator.
func (f *defaultFormats) Validates(name, data string) bool {
	f.Lock()
	defer f.Unlock()
	nme := f.normalizeName(name)
	for _, v := range f.data {
		if v.Name == nme {
			return v.Validator(data)
		}
	}
	return false
}

// Parse a string into the appropriate format representation type.
//
// E.g. parsing a string a "date" will return a Date type.
func (f *defaultFormats) Parse(name, data string) (interface{}, error) {
	f.Lock()
	defer f.Unlock()
	nme := f.normalizeName(name)
	for _, v := range f.data {
		if v.Name == nme {
			nw := reflect.New(v.Type).Interface()
			if dec, ok := nw.(encoding.TextUnmarshaler); ok {
				if err := dec.UnmarshalText([]byte(data)); err != nil {
					return nil, err
				}
				return nw, nil
			}
			return nil, errors.InvalidTypeName(name)
		}
	}
	return nil, errors.InvalidTypeName(name)
}

// unmarshalJSON provides a generic implementation of json.Unmarshaler interface's UnmarshalJSON function for basic string formats.
func unmarshalJSON[T ~string](r *T, data []byte) error {
	if string(data) == jsonNull {
		return nil
	}
	var ustr string
	if err := json.Unmarshal(data, &ustr); err != nil {
		return err
	}
	*r = T(ustr)
	return nil
}

// deepCopy provides a generic implementation of DeepCopy for basic string formats.
func deepCopy[T ~string](r *T) *T {
	if r == nil {
		return nil
	}
	out := new(T)
	*out = *r
	return out
}
