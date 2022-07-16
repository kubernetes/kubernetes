// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
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

// author  			xeipuuv
// author-github 	https://github.com/xeipuuv
// author-mail		xeipuuv@gmail.com
//
// repository-name	gojsonpointer
// repository-desc	An implementation of JSON Pointer - Go language
//
// description		Main and unique file.
//
// created      	25-02-2013

package gojsonpointer

import (
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

const (
	const_empty_pointer     = ``
	const_pointer_separator = `/`

	const_invalid_start = `JSON pointer must be empty or start with a "` + const_pointer_separator + `"`
)

type implStruct struct {
	mode string // "SET" or "GET"

	inDocument interface{}

	setInValue interface{}

	getOutNode interface{}
	getOutKind reflect.Kind
	outError   error
}

type JsonPointer struct {
	referenceTokens []string
}

// NewJsonPointer parses the given string JSON pointer and returns an object
func NewJsonPointer(jsonPointerString string) (p JsonPointer, err error) {

	// Pointer to the root of the document
	if len(jsonPointerString) == 0 {
		// Keep referenceTokens nil
		return
	}
	if jsonPointerString[0] != '/' {
		return p, errors.New(const_invalid_start)
	}

	p.referenceTokens = strings.Split(jsonPointerString[1:], const_pointer_separator)
	return
}

// Uses the pointer to retrieve a value from a JSON document
func (p *JsonPointer) Get(document interface{}) (interface{}, reflect.Kind, error) {

	is := &implStruct{mode: "GET", inDocument: document}
	p.implementation(is)
	return is.getOutNode, is.getOutKind, is.outError

}

// Uses the pointer to update a value from a JSON document
func (p *JsonPointer) Set(document interface{}, value interface{}) (interface{}, error) {

	is := &implStruct{mode: "SET", inDocument: document, setInValue: value}
	p.implementation(is)
	return document, is.outError

}

// Uses the pointer to delete a value from a JSON document
func (p *JsonPointer) Delete(document interface{}) (interface{}, error) {
	is := &implStruct{mode: "DEL", inDocument: document}
	p.implementation(is)
	return document, is.outError
}

// Both Get and Set functions use the same implementation to avoid code duplication
func (p *JsonPointer) implementation(i *implStruct) {

	kind := reflect.Invalid

	// Full document when empty
	if len(p.referenceTokens) == 0 {
		i.getOutNode = i.inDocument
		i.outError = nil
		i.getOutKind = kind
		i.outError = nil
		return
	}

	node := i.inDocument

	previousNodes := make([]interface{}, len(p.referenceTokens))
	previousTokens := make([]string, len(p.referenceTokens))

	for ti, token := range p.referenceTokens {

		isLastToken := ti == len(p.referenceTokens)-1
		previousNodes[ti] = node
		previousTokens[ti] = token

		switch v := node.(type) {

		case map[string]interface{}:
			decodedToken := decodeReferenceToken(token)
			if _, ok := v[decodedToken]; ok {
				node = v[decodedToken]
				if isLastToken && i.mode == "SET" {
					v[decodedToken] = i.setInValue
				} else if isLastToken && i.mode =="DEL" {
					delete(v,decodedToken)
				}
			} else if (isLastToken && i.mode == "SET") {
				v[decodedToken] = i.setInValue
			} else {
				i.outError = fmt.Errorf("Object has no key '%s'", decodedToken)
				i.getOutKind = reflect.Map
				i.getOutNode = nil
				return
			}

		case []interface{}:
			tokenIndex, err := strconv.Atoi(token)
			if err != nil {
				i.outError = fmt.Errorf("Invalid array index '%s'", token)
				i.getOutKind = reflect.Slice
				i.getOutNode = nil
				return
			}
			if tokenIndex < 0 || tokenIndex >= len(v) {
				i.outError = fmt.Errorf("Out of bound array[0,%d] index '%d'", len(v), tokenIndex)
				i.getOutKind = reflect.Slice
				i.getOutNode = nil
				return
			}

			node = v[tokenIndex]
			if isLastToken && i.mode == "SET" {
				v[tokenIndex] = i.setInValue
			}  else if isLastToken && i.mode =="DEL" {
				v[tokenIndex] = v[len(v)-1]
				v[len(v)-1] = nil
				v = v[:len(v)-1]
				previousNodes[ti-1].(map[string]interface{})[previousTokens[ti-1]] = v
			}

		default:
			i.outError = fmt.Errorf("Invalid token reference '%s'", token)
			i.getOutKind = reflect.ValueOf(node).Kind()
			i.getOutNode = nil
			return
		}

	}

	i.getOutNode = node
	i.getOutKind = reflect.ValueOf(node).Kind()
	i.outError = nil
}

// Pointer to string representation function
func (p *JsonPointer) String() string {

	if len(p.referenceTokens) == 0 {
		return const_empty_pointer
	}

	pointerString := const_pointer_separator + strings.Join(p.referenceTokens, const_pointer_separator)

	return pointerString
}

// Specific JSON pointer encoding here
// ~0 => ~
// ~1 => /
// ... and vice versa

func decodeReferenceToken(token string) string {
	step1 := strings.Replace(token, `~1`, `/`, -1)
	step2 := strings.Replace(step1, `~0`, `~`, -1)
	return step2
}

func encodeReferenceToken(token string) string {
	step1 := strings.Replace(token, `~`, `~0`, -1)
	step2 := strings.Replace(step1, `/`, `~1`, -1)
	return step2
}
