// Copyright 2013 sigu-399 ( https://github.com/sigu-399 )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author       sigu-399
// author-github  https://github.com/sigu-399
// author-mail    sigu.399@gmail.com
//
// repository-name  jsonpointer
// repository-desc  An implementation of JSON Pointer - Go language
//
// description    Main and unique file.
//
// created        25-02-2013

package jsonpointer

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/go-openapi/swag"
)

const (
	emptyPointer     = ``
	pointerSeparator = `/`

	invalidStart = `JSON pointer must be empty or start with a "` + pointerSeparator
	notFound     = `Can't find the pointer in the document`
)

var jsonPointableType = reflect.TypeOf(new(JSONPointable)).Elem()
var jsonSetableType = reflect.TypeOf(new(JSONSetable)).Elem()

// JSONPointable is an interface for structs to implement when they need to customize the
// json pointer process
type JSONPointable interface {
	JSONLookup(string) (any, error)
}

// JSONSetable is an interface for structs to implement when they need to customize the
// json pointer process
type JSONSetable interface {
	JSONSet(string, any) error
}

// New creates a new json pointer for the given string
func New(jsonPointerString string) (Pointer, error) {

	var p Pointer
	err := p.parse(jsonPointerString)
	return p, err

}

// Pointer the json pointer reprsentation
type Pointer struct {
	referenceTokens []string
}

// "Constructor", parses the given string JSON pointer
func (p *Pointer) parse(jsonPointerString string) error {

	var err error

	if jsonPointerString != emptyPointer {
		if !strings.HasPrefix(jsonPointerString, pointerSeparator) {
			err = errors.New(invalidStart)
		} else {
			referenceTokens := strings.Split(jsonPointerString, pointerSeparator)
			p.referenceTokens = append(p.referenceTokens, referenceTokens[1:]...)
		}
	}

	return err
}

// Get uses the pointer to retrieve a value from a JSON document
func (p *Pointer) Get(document any) (any, reflect.Kind, error) {
	return p.get(document, swag.DefaultJSONNameProvider)
}

// Set uses the pointer to set a value from a JSON document
func (p *Pointer) Set(document any, value any) (any, error) {
	return document, p.set(document, value, swag.DefaultJSONNameProvider)
}

// GetForToken gets a value for a json pointer token 1 level deep
func GetForToken(document any, decodedToken string) (any, reflect.Kind, error) {
	return getSingleImpl(document, decodedToken, swag.DefaultJSONNameProvider)
}

// SetForToken gets a value for a json pointer token 1 level deep
func SetForToken(document any, decodedToken string, value any) (any, error) {
	return document, setSingleImpl(document, value, decodedToken, swag.DefaultJSONNameProvider)
}

func isNil(input any) bool {
	if input == nil {
		return true
	}

	kind := reflect.TypeOf(input).Kind()
	switch kind { //nolint:exhaustive
	case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Chan:
		return reflect.ValueOf(input).IsNil()
	default:
		return false
	}
}

func getSingleImpl(node any, decodedToken string, nameProvider *swag.NameProvider) (any, reflect.Kind, error) {
	rValue := reflect.Indirect(reflect.ValueOf(node))
	kind := rValue.Kind()
	if isNil(node) {
		return nil, kind, fmt.Errorf("nil value has not field %q", decodedToken)
	}

	switch typed := node.(type) {
	case JSONPointable:
		r, err := typed.JSONLookup(decodedToken)
		if err != nil {
			return nil, kind, err
		}
		return r, kind, nil
	case *any: // case of a pointer to interface, that is not resolved by reflect.Indirect
		return getSingleImpl(*typed, decodedToken, nameProvider)
	}

	switch kind { //nolint:exhaustive
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return nil, kind, fmt.Errorf("object has no field %q", decodedToken)
		}
		fld := rValue.FieldByName(nm)
		return fld.Interface(), kind, nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		mv := rValue.MapIndex(kv)

		if mv.IsValid() {
			return mv.Interface(), kind, nil
		}
		return nil, kind, fmt.Errorf("object has no key %q", decodedToken)

	case reflect.Slice:
		tokenIndex, err := strconv.Atoi(decodedToken)
		if err != nil {
			return nil, kind, err
		}
		sLength := rValue.Len()
		if tokenIndex < 0 || tokenIndex >= sLength {
			return nil, kind, fmt.Errorf("index out of bounds array[0,%d] index '%d'", sLength-1, tokenIndex)
		}

		elem := rValue.Index(tokenIndex)
		return elem.Interface(), kind, nil

	default:
		return nil, kind, fmt.Errorf("invalid token reference %q", decodedToken)
	}

}

func setSingleImpl(node, data any, decodedToken string, nameProvider *swag.NameProvider) error {
	rValue := reflect.Indirect(reflect.ValueOf(node))

	if ns, ok := node.(JSONSetable); ok { // pointer impl
		return ns.JSONSet(decodedToken, data)
	}

	if rValue.Type().Implements(jsonSetableType) {
		return node.(JSONSetable).JSONSet(decodedToken, data)
	}

	switch rValue.Kind() { //nolint:exhaustive
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return fmt.Errorf("object has no field %q", decodedToken)
		}
		fld := rValue.FieldByName(nm)
		if fld.IsValid() {
			fld.Set(reflect.ValueOf(data))
		}
		return nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		rValue.SetMapIndex(kv, reflect.ValueOf(data))
		return nil

	case reflect.Slice:
		tokenIndex, err := strconv.Atoi(decodedToken)
		if err != nil {
			return err
		}
		sLength := rValue.Len()
		if tokenIndex < 0 || tokenIndex >= sLength {
			return fmt.Errorf("index out of bounds array[0,%d] index '%d'", sLength, tokenIndex)
		}

		elem := rValue.Index(tokenIndex)
		if !elem.CanSet() {
			return fmt.Errorf("can't set slice index %s to %v", decodedToken, data)
		}
		elem.Set(reflect.ValueOf(data))
		return nil

	default:
		return fmt.Errorf("invalid token reference %q", decodedToken)
	}

}

func (p *Pointer) get(node any, nameProvider *swag.NameProvider) (any, reflect.Kind, error) {

	if nameProvider == nil {
		nameProvider = swag.DefaultJSONNameProvider
	}

	kind := reflect.Invalid

	// Full document when empty
	if len(p.referenceTokens) == 0 {
		return node, kind, nil
	}

	for _, token := range p.referenceTokens {

		decodedToken := Unescape(token)

		r, knd, err := getSingleImpl(node, decodedToken, nameProvider)
		if err != nil {
			return nil, knd, err
		}
		node = r
	}

	rValue := reflect.ValueOf(node)
	kind = rValue.Kind()

	return node, kind, nil
}

func (p *Pointer) set(node, data any, nameProvider *swag.NameProvider) error {
	knd := reflect.ValueOf(node).Kind()

	if knd != reflect.Ptr && knd != reflect.Struct && knd != reflect.Map && knd != reflect.Slice && knd != reflect.Array {
		return errors.New("only structs, pointers, maps and slices are supported for setting values")
	}

	if nameProvider == nil {
		nameProvider = swag.DefaultJSONNameProvider
	}

	// Full document when empty
	if len(p.referenceTokens) == 0 {
		return nil
	}

	lastI := len(p.referenceTokens) - 1
	for i, token := range p.referenceTokens {
		isLastToken := i == lastI
		decodedToken := Unescape(token)

		if isLastToken {

			return setSingleImpl(node, data, decodedToken, nameProvider)
		}

		rValue := reflect.Indirect(reflect.ValueOf(node))
		kind := rValue.Kind()

		if rValue.Type().Implements(jsonPointableType) {
			r, err := node.(JSONPointable).JSONLookup(decodedToken)
			if err != nil {
				return err
			}
			fld := reflect.ValueOf(r)
			if fld.CanAddr() && fld.Kind() != reflect.Interface && fld.Kind() != reflect.Map && fld.Kind() != reflect.Slice && fld.Kind() != reflect.Ptr {
				node = fld.Addr().Interface()
				continue
			}
			node = r
			continue
		}

		switch kind { //nolint:exhaustive
		case reflect.Struct:
			nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
			if !ok {
				return fmt.Errorf("object has no field %q", decodedToken)
			}
			fld := rValue.FieldByName(nm)
			if fld.CanAddr() && fld.Kind() != reflect.Interface && fld.Kind() != reflect.Map && fld.Kind() != reflect.Slice && fld.Kind() != reflect.Ptr {
				node = fld.Addr().Interface()
				continue
			}
			node = fld.Interface()

		case reflect.Map:
			kv := reflect.ValueOf(decodedToken)
			mv := rValue.MapIndex(kv)

			if !mv.IsValid() {
				return fmt.Errorf("object has no key %q", decodedToken)
			}
			if mv.CanAddr() && mv.Kind() != reflect.Interface && mv.Kind() != reflect.Map && mv.Kind() != reflect.Slice && mv.Kind() != reflect.Ptr {
				node = mv.Addr().Interface()
				continue
			}
			node = mv.Interface()

		case reflect.Slice:
			tokenIndex, err := strconv.Atoi(decodedToken)
			if err != nil {
				return err
			}
			sLength := rValue.Len()
			if tokenIndex < 0 || tokenIndex >= sLength {
				return fmt.Errorf("index out of bounds array[0,%d] index '%d'", sLength, tokenIndex)
			}

			elem := rValue.Index(tokenIndex)
			if elem.CanAddr() && elem.Kind() != reflect.Interface && elem.Kind() != reflect.Map && elem.Kind() != reflect.Slice && elem.Kind() != reflect.Ptr {
				node = elem.Addr().Interface()
				continue
			}
			node = elem.Interface()

		default:
			return fmt.Errorf("invalid token reference %q", decodedToken)
		}

	}

	return nil
}

// DecodedTokens returns the decoded tokens
func (p *Pointer) DecodedTokens() []string {
	result := make([]string, 0, len(p.referenceTokens))
	for _, t := range p.referenceTokens {
		result = append(result, Unescape(t))
	}
	return result
}

// IsEmpty returns true if this is an empty json pointer
// this indicates that it points to the root document
func (p *Pointer) IsEmpty() bool {
	return len(p.referenceTokens) == 0
}

// Pointer to string representation function
func (p *Pointer) String() string {

	if len(p.referenceTokens) == 0 {
		return emptyPointer
	}

	pointerString := pointerSeparator + strings.Join(p.referenceTokens, pointerSeparator)

	return pointerString
}

func (p *Pointer) Offset(document string) (int64, error) {
	dec := json.NewDecoder(strings.NewReader(document))
	var offset int64
	for _, ttk := range p.DecodedTokens() {
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}
		switch tk := tk.(type) {
		case json.Delim:
			switch tk {
			case '{':
				offset, err = offsetSingleObject(dec, ttk)
				if err != nil {
					return 0, err
				}
			case '[':
				offset, err = offsetSingleArray(dec, ttk)
				if err != nil {
					return 0, err
				}
			default:
				return 0, fmt.Errorf("invalid token %#v", tk)
			}
		default:
			return 0, fmt.Errorf("invalid token %#v", tk)
		}
	}
	return offset, nil
}

func offsetSingleObject(dec *json.Decoder, decodedToken string) (int64, error) {
	for dec.More() {
		offset := dec.InputOffset()
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}
		switch tk := tk.(type) {
		case json.Delim:
			switch tk {
			case '{':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			case '[':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			}
		case string:
			if tk == decodedToken {
				return offset, nil
			}
		default:
			return 0, fmt.Errorf("invalid token %#v", tk)
		}
	}
	return 0, fmt.Errorf("token reference %q not found", decodedToken)
}

func offsetSingleArray(dec *json.Decoder, decodedToken string) (int64, error) {
	idx, err := strconv.Atoi(decodedToken)
	if err != nil {
		return 0, fmt.Errorf("token reference %q is not a number: %v", decodedToken, err)
	}
	var i int
	for i = 0; i < idx && dec.More(); i++ {
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}

		if delim, isDelim := tk.(json.Delim); isDelim {
			switch delim {
			case '{':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			case '[':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			}
		}
	}

	if !dec.More() {
		return 0, fmt.Errorf("token reference %q not found", decodedToken)
	}
	return dec.InputOffset(), nil
}

// drainSingle drains a single level of object or array.
// The decoder has to guarantee the beginning delim (i.e. '{' or '[') has been consumed.
func drainSingle(dec *json.Decoder) error {
	for dec.More() {
		tk, err := dec.Token()
		if err != nil {
			return err
		}
		if delim, isDelim := tk.(json.Delim); isDelim {
			switch delim {
			case '{':
				if err = drainSingle(dec); err != nil {
					return err
				}
			case '[':
				if err = drainSingle(dec); err != nil {
					return err
				}
			}
		}
	}

	// Consumes the ending delim
	if _, err := dec.Token(); err != nil {
		return err
	}
	return nil
}

// Specific JSON pointer encoding here
// ~0 => ~
// ~1 => /
// ... and vice versa

const (
	encRefTok0 = `~0`
	encRefTok1 = `~1`
	decRefTok0 = `~`
	decRefTok1 = `/`
)

// Unescape unescapes a json pointer reference token string to the original representation
func Unescape(token string) string {
	step1 := strings.ReplaceAll(token, encRefTok1, decRefTok1)
	step2 := strings.ReplaceAll(step1, encRefTok0, decRefTok0)
	return step2
}

// Escape escapes a pointer reference token string
func Escape(token string) string {
	step1 := strings.ReplaceAll(token, decRefTok0, encRefTok0)
	step2 := strings.ReplaceAll(step1, decRefTok1, encRefTok1)
	return step2
}
