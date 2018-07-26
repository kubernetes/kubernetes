/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package json

import (
	"io"
	"strconv"
	"unsafe"

	"github.com/json-iterator/go"
	"github.com/modern-go/reflect2"
)

var (
	// Private copy of jsoniter to try to shield against possible mutations
	// from outside. Still does not protect from package level jsoniter.Register*() functions - someone calling them
	// in some other library will mess with every usage of the jsoniter library in the whole program.
	// See https://github.com/json-iterator/go/issues/265
	caseSensitiveJSONIterator = CaseSensitiveJSONIterator()
)

type customNumberExtension struct {
	jsoniter.DummyExtension
}

func (cne *customNumberExtension) CreateDecoder(typ reflect2.Type) jsoniter.ValDecoder {
	if typ.String() == "interface {}" {
		return customNumberDecoder{}
	}
	return nil
}

type customNumberDecoder struct {
}

func (customNumberDecoder) Decode(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
	switch iter.WhatIsNext() {
	case jsoniter.NumberValue:
		var number jsoniter.Number
		iter.ReadVal(&number)
		i64, err := strconv.ParseInt(string(number), 10, 64)
		if err == nil {
			*(*interface{})(ptr) = i64
			return
		}
		f64, err := strconv.ParseFloat(string(number), 64)
		if err == nil {
			*(*interface{})(ptr) = f64
			return
		}
		iter.ReportError("DecodeNumber", err.Error())
	default:
		*(*interface{})(ptr) = iter.Read()
	}
}

// CaseSensitiveJSONIterator returns a jsoniterator API that's configured to be
// case-sensitive when unmarshalling, and otherwise compatible with
// the encoding/json standard library.
func CaseSensitiveJSONIterator() jsoniter.API {
	config := jsoniter.Config{
		EscapeHTML:             true,
		SortMapKeys:            true,
		ValidateJsonRawMessage: true,
		CaseSensitive:          true,
	}.Froze()
	// Force jsoniter to decode number to interface{} via int64/float64, if possible.
	config.RegisterExtension(&customNumberExtension{})
	return config
}

// NewEncoder mirrors json.NewEncoder().
// It is here so this package can be a drop-in for common encoding/json uses.
func NewEncoder(w io.Writer) *jsoniter.Encoder {
	return caseSensitiveJSONIterator.NewEncoder(w)
}

// Marshal mirrors json.Marshal().
// It is here so this package can be a drop-in for common encoding/json uses.
func Marshal(v interface{}) ([]byte, error) {
	return caseSensitiveJSONIterator.Marshal(v)
}

// MarshalIndent mirrors json.MarshalIndent().
// It is here so this package can be a drop-in for common encoding/json uses.
func MarshalIndent(v interface{}, prefix, indent string) ([]byte, error) {
	return caseSensitiveJSONIterator.MarshalIndent(v, prefix, indent)
}

// Unmarshal mirrors json.Unmarshal().
// Numbers are converted to int64 or float64.
func Unmarshal(data []byte, v interface{}) error {
	return caseSensitiveJSONIterator.Unmarshal(data, v)
}
