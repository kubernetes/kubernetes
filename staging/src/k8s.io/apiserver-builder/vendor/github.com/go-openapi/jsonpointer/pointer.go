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
)

var jsonPointableType = reflect.TypeOf(new(JSONPointable)).Elem()

// JSONPointable is an interface for structs to implement when they need to customize the
// json pointer process
type JSONPointable interface {
	JSONLookup(string) (interface{}, error)
}

type implStruct struct {
	mode string // "SET" or "GET"

	inDocument interface{}

	setInValue interface{}

	getOutNode interface{}
	getOutKind reflect.Kind
	outError   error
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
			for _, referenceToken := range referenceTokens[1:] {
				p.referenceTokens = append(p.referenceTokens, referenceToken)
			}
		}
	}

	return err
}

// Get uses the pointer to retrieve a value from a JSON document
func (p *Pointer) Get(document interface{}) (interface{}, reflect.Kind, error) {
	return p.get(document, swag.DefaultJSONNameProvider)
}

// GetForToken gets a value for a json pointer token 1 level deep
func GetForToken(document interface{}, decodedToken string) (interface{}, reflect.Kind, error) {
	return getSingleImpl(document, decodedToken, swag.DefaultJSONNameProvider)
}

func getSingleImpl(node interface{}, decodedToken string, nameProvider *swag.NameProvider) (interface{}, reflect.Kind, error) {
	kind := reflect.Invalid
	rValue := reflect.Indirect(reflect.ValueOf(node))
	kind = rValue.Kind()
	switch kind {

	case reflect.Struct:
		if rValue.Type().Implements(jsonPointableType) {
			r, err := node.(JSONPointable).JSONLookup(decodedToken)
			if err != nil {
				return nil, kind, err
			}
			return r, kind, nil
		}
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return nil, kind, fmt.Errorf("object has no field %q", decodedToken)
		}
		fld := rValue.FieldByName(nm)
		return fld.Interface(), kind, nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		mv := rValue.MapIndex(kv)
		if mv.IsValid() && !swag.IsZero(mv) {
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
			return nil, kind, fmt.Errorf("index out of bounds array[0,%d] index '%d'", sLength, tokenIndex)
		}

		elem := rValue.Index(tokenIndex)
		return elem.Interface(), kind, nil

	default:
		return nil, kind, fmt.Errorf("invalid token reference %q", decodedToken)
	}

}

func (p *Pointer) get(node interface{}, nameProvider *swag.NameProvider) (interface{}, reflect.Kind, error) {

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
		node, kind = r, knd

	}

	rValue := reflect.ValueOf(node)
	kind = rValue.Kind()

	return node, kind, nil
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
	step1 := strings.Replace(token, encRefTok1, decRefTok1, -1)
	step2 := strings.Replace(step1, encRefTok0, decRefTok0, -1)
	return step2
}

// Escape escapes a pointer reference token string
func Escape(token string) string {
	step1 := strings.Replace(token, decRefTok0, encRefTok0, -1)
	step2 := strings.Replace(step1, decRefTok1, encRefTok1, -1)
	return step2
}
