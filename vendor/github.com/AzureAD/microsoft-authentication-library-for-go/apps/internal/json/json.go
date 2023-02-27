// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Package json provide functions for marshalling an unmarshalling types to JSON. These functions are meant to
// be utilized inside of structs that implement json.Unmarshaler and json.Marshaler interfaces.
// This package provides the additional functionality of writing fields that are not in the struct when marshalling
// to a field called AdditionalFields if that field exists and is a map[string]interface{}.
// When marshalling, if the struct has all the same prerequisites, it will uses the keys in AdditionalFields as
// extra fields. This package uses encoding/json underneath.
package json

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

const addField = "AdditionalFields"
const (
	marshalJSON   = "MarshalJSON"
	unmarshalJSON = "UnmarshalJSON"
)

var (
	leftBrace  = []byte("{")[0]
	rightBrace = []byte("}")[0]
	comma      = []byte(",")[0]
	leftParen  = []byte("[")[0]
	rightParen = []byte("]")[0]
)

var mapStrInterType = reflect.TypeOf(map[string]interface{}{})

// stateFn defines a state machine function. This will be used in all state
// machines in this package.
type stateFn func() (stateFn, error)

// Marshal is used to marshal a type into its JSON representation. It
// wraps the stdlib calls in order to marshal a struct or *struct so
// that a field called "AdditionalFields" of type map[string]interface{}
// with "-" used inside struct tag `json:"-"` can be marshalled as if
// they were fields within the struct.
func Marshal(i interface{}) ([]byte, error) {
	buff := bytes.Buffer{}
	enc := json.NewEncoder(&buff)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "")

	v := reflect.ValueOf(i)
	if v.Kind() != reflect.Ptr && v.CanAddr() {
		v = v.Addr()
	}
	err := marshalStruct(v, &buff, enc)
	if err != nil {
		return nil, err
	}
	return buff.Bytes(), nil
}

// Unmarshal unmarshals a []byte representing JSON into i, which must be a *struct. In addition, if the struct has
// a field called AdditionalFields of type map[string]interface{}, JSON data representing fields not in the struct
// will be written as key/value pairs to AdditionalFields.
func Unmarshal(b []byte, i interface{}) error {
	if len(b) == 0 {
		return nil
	}

	jdec := json.NewDecoder(bytes.NewBuffer(b))
	jdec.UseNumber()
	return unmarshalStruct(jdec, i)
}

// MarshalRaw marshals i into a json.RawMessage. If I cannot be marshalled,
// this will panic. This is exposed to help test AdditionalField values
// which are stored as json.RawMessage.
func MarshalRaw(i interface{}) json.RawMessage {
	b, err := json.Marshal(i)
	if err != nil {
		panic(err)
	}
	return json.RawMessage(b)
}

// isDelim simply tests to see if a json.Token is a delimeter.
func isDelim(got json.Token) bool {
	switch got.(type) {
	case json.Delim:
		return true
	}
	return false
}

// delimIs tests got to see if it is want.
func delimIs(got json.Token, want rune) bool {
	switch v := got.(type) {
	case json.Delim:
		if v == json.Delim(want) {
			return true
		}
	}
	return false
}

// hasMarshalJSON will determine if the value or a pointer to this value has
// the MarshalJSON method.
func hasMarshalJSON(v reflect.Value) bool {
	if method := v.MethodByName(marshalJSON); method.Kind() != reflect.Invalid {
		_, ok := v.Interface().(json.Marshaler)
		return ok
	}

	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	} else {
		if !v.CanAddr() {
			return false
		}
		v = v.Addr()
	}

	if method := v.MethodByName(marshalJSON); method.Kind() != reflect.Invalid {
		_, ok := v.Interface().(json.Marshaler)
		return ok
	}
	return false
}

// callMarshalJSON will call MarshalJSON() method on the value or a pointer to this value.
// This will panic if the method is not defined.
func callMarshalJSON(v reflect.Value) ([]byte, error) {
	if method := v.MethodByName(marshalJSON); method.Kind() != reflect.Invalid {
		marsh := v.Interface().(json.Marshaler)
		return marsh.MarshalJSON()
	}

	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	} else {
		if v.CanAddr() {
			v = v.Addr()
		}
	}

	if method := v.MethodByName(unmarshalJSON); method.Kind() != reflect.Invalid {
		marsh := v.Interface().(json.Marshaler)
		return marsh.MarshalJSON()
	}

	panic(fmt.Sprintf("callMarshalJSON called on type %T that does not have MarshalJSON defined", v.Interface()))
}

// hasUnmarshalJSON will determine if the value or a pointer to this value has
// the UnmarshalJSON method.
func hasUnmarshalJSON(v reflect.Value) bool {
	// You can't unmarshal on a non-pointer type.
	if v.Kind() != reflect.Ptr {
		if !v.CanAddr() {
			return false
		}
		v = v.Addr()
	}

	if method := v.MethodByName(unmarshalJSON); method.Kind() != reflect.Invalid {
		_, ok := v.Interface().(json.Unmarshaler)
		return ok
	}

	return false
}

// hasOmitEmpty indicates if the field has instructed us to not output
// the field if omitempty is set on the tag. tag is the string
// returned by reflect.StructField.Tag().Get().
func hasOmitEmpty(tag string) bool {
	sl := strings.Split(tag, ",")
	for _, str := range sl {
		if str == "omitempty" {
			return true
		}
	}
	return false
}
