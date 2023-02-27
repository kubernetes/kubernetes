// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package json

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"unicode"
)

// marshalStruct takes in i, which must be a *struct or struct and marshals its content
// as JSON into buff (sometimes with writes to buff directly, sometimes via enc).
// This call is recursive for all fields of *struct or struct type.
func marshalStruct(v reflect.Value, buff *bytes.Buffer, enc *json.Encoder) error {
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	// We only care about custom Marshalling a struct.
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("bug: marshal() received a non *struct or struct, received type %T", v.Interface())
	}

	if hasMarshalJSON(v) {
		b, err := callMarshalJSON(v)
		if err != nil {
			return err
		}
		buff.Write(b)
		return nil
	}

	t := v.Type()

	// If it has an AdditionalFields field make sure its the right type.
	f := v.FieldByName(addField)
	if f.Kind() != reflect.Invalid {
		if f.Kind() != reflect.Map {
			return fmt.Errorf("type %T has field 'AdditionalFields' that is not a map[string]interface{}", v.Interface())
		}
		if !f.Type().AssignableTo(mapStrInterType) {
			return fmt.Errorf("type %T has field 'AdditionalFields' that is not a map[string]interface{}", v.Interface())
		}
	}

	translator, err := findFields(v)
	if err != nil {
		return err
	}

	buff.WriteByte(leftBrace)
	for x := 0; x < v.NumField(); x++ {
		field := v.Field(x)

		// We don't access private fields.
		if unicode.IsLower(rune(t.Field(x).Name[0])) {
			continue
		}

		if t.Field(x).Name == addField {
			if v.Field(x).Len() > 0 {
				if err := writeAddFields(field.Interface(), buff, enc); err != nil {
					return err
				}
				buff.WriteByte(comma)
			}
			continue
		}

		// If they have omitempty set, we don't write out the field if
		// it is the zero value.
		if hasOmitEmpty(t.Field(x).Tag.Get("json")) {
			if v.Field(x).IsZero() {
				continue
			}
		}

		// Write out the field name part.
		jsonName := translator.jsonName(t.Field(x).Name)
		buff.WriteString(fmt.Sprintf("%q:", jsonName))

		if field.Kind() == reflect.Ptr {
			field = field.Elem()
		}

		if err := marshalStructField(field, buff, enc); err != nil {
			return err
		}
	}

	buff.Truncate(buff.Len() - 1) // Remove final comma
	buff.WriteByte(rightBrace)

	return nil
}

func marshalStructField(field reflect.Value, buff *bytes.Buffer, enc *json.Encoder) error {
	// Determine if we need a trailing comma.
	defer buff.WriteByte(comma)

	switch field.Kind() {
	// If it was a *struct or struct, we need to recursively all marshal().
	case reflect.Struct:
		if field.CanAddr() {
			field = field.Addr()
		}
		return marshalStruct(field, buff, enc)
	case reflect.Map:
		return marshalMap(field, buff, enc)
	case reflect.Slice:
		return marshalSlice(field, buff, enc)
	}

	// It is just a basic type, so encode it.
	if err := enc.Encode(field.Interface()); err != nil {
		return err
	}
	buff.Truncate(buff.Len() - 1) // Remove Encode() added \n

	return nil
}

func marshalMap(v reflect.Value, buff *bytes.Buffer, enc *json.Encoder) error {
	if v.Kind() != reflect.Map {
		return fmt.Errorf("bug: marshalMap() called on %T", v.Interface())
	}
	if v.Len() == 0 {
		buff.WriteByte(leftBrace)
		buff.WriteByte(rightBrace)
		return nil
	}
	encoder := mapEncode{m: v, buff: buff, enc: enc}
	return encoder.run()
}

type mapEncode struct {
	m    reflect.Value
	buff *bytes.Buffer
	enc  *json.Encoder

	valueBaseType reflect.Type
}

// run runs our encoder state machine.
func (m *mapEncode) run() error {
	var state = m.start
	var err error
	for {
		state, err = state()
		if err != nil {
			return err
		}
		if state == nil {
			return nil
		}
	}
}

func (m *mapEncode) start() (stateFn, error) {
	if hasMarshalJSON(m.m) {
		b, err := callMarshalJSON(m.m)
		if err != nil {
			return nil, err
		}
		m.buff.Write(b)
		return nil, nil
	}

	valueBaseType := m.m.Type().Elem()
	if valueBaseType.Kind() == reflect.Ptr {
		valueBaseType = valueBaseType.Elem()
	}
	m.valueBaseType = valueBaseType

	switch valueBaseType.Kind() {
	case reflect.Ptr:
		return nil, fmt.Errorf("Marshal does not support **<type> or *<reference>")
	case reflect.Struct, reflect.Map, reflect.Slice:
		return m.encode, nil
	}

	// If the map value doesn't have a struct/map/slice, just Encode() it.
	if err := m.enc.Encode(m.m.Interface()); err != nil {
		return nil, err
	}
	m.buff.Truncate(m.buff.Len() - 1) // Remove Encode() added \n
	return nil, nil
}

func (m *mapEncode) encode() (stateFn, error) {
	m.buff.WriteByte(leftBrace)

	iter := m.m.MapRange()
	for iter.Next() {
		// Write the key.
		k := iter.Key()
		m.buff.WriteString(fmt.Sprintf("%q:", k.String()))

		v := iter.Value()
		switch m.valueBaseType.Kind() {
		case reflect.Struct:
			if v.CanAddr() {
				v = v.Addr()
			}
			if err := marshalStruct(v, m.buff, m.enc); err != nil {
				return nil, err
			}
		case reflect.Map:
			if err := marshalMap(v, m.buff, m.enc); err != nil {
				return nil, err
			}
		case reflect.Slice:
			if err := marshalSlice(v, m.buff, m.enc); err != nil {
				return nil, err
			}
		default:
			panic(fmt.Sprintf("critical bug: mapEncode.encode() called with value base type: %v", m.valueBaseType.Kind()))
		}
		m.buff.WriteByte(comma)
	}
	m.buff.Truncate(m.buff.Len() - 1) // Remove final comma
	m.buff.WriteByte(rightBrace)

	return nil, nil
}

func marshalSlice(v reflect.Value, buff *bytes.Buffer, enc *json.Encoder) error {
	if v.Kind() != reflect.Slice {
		return fmt.Errorf("bug: marshalSlice() called on %T", v.Interface())
	}
	if v.Len() == 0 {
		buff.WriteByte(leftParen)
		buff.WriteByte(rightParen)
		return nil
	}
	encoder := sliceEncode{s: v, buff: buff, enc: enc}
	return encoder.run()
}

type sliceEncode struct {
	s    reflect.Value
	buff *bytes.Buffer
	enc  *json.Encoder

	valueBaseType reflect.Type
}

// run runs our encoder state machine.
func (s *sliceEncode) run() error {
	var state = s.start
	var err error
	for {
		state, err = state()
		if err != nil {
			return err
		}
		if state == nil {
			return nil
		}
	}
}

func (s *sliceEncode) start() (stateFn, error) {
	if hasMarshalJSON(s.s) {
		b, err := callMarshalJSON(s.s)
		if err != nil {
			return nil, err
		}
		s.buff.Write(b)
		return nil, nil
	}

	valueBaseType := s.s.Type().Elem()
	if valueBaseType.Kind() == reflect.Ptr {
		valueBaseType = valueBaseType.Elem()
	}
	s.valueBaseType = valueBaseType

	switch valueBaseType.Kind() {
	case reflect.Ptr:
		return nil, fmt.Errorf("Marshal does not support **<type> or *<reference>")
	case reflect.Struct, reflect.Map, reflect.Slice:
		return s.encode, nil
	}

	// If the map value doesn't have a struct/map/slice, just Encode() it.
	if err := s.enc.Encode(s.s.Interface()); err != nil {
		return nil, err
	}
	s.buff.Truncate(s.buff.Len() - 1) // Remove Encode added \n

	return nil, nil
}

func (s *sliceEncode) encode() (stateFn, error) {
	s.buff.WriteByte(leftParen)
	for i := 0; i < s.s.Len(); i++ {
		v := s.s.Index(i)
		switch s.valueBaseType.Kind() {
		case reflect.Struct:
			if v.CanAddr() {
				v = v.Addr()
			}
			if err := marshalStruct(v, s.buff, s.enc); err != nil {
				return nil, err
			}
		case reflect.Map:
			if err := marshalMap(v, s.buff, s.enc); err != nil {
				return nil, err
			}
		case reflect.Slice:
			if err := marshalSlice(v, s.buff, s.enc); err != nil {
				return nil, err
			}
		default:
			panic(fmt.Sprintf("critical bug: mapEncode.encode() called with value base type: %v", s.valueBaseType.Kind()))
		}
		s.buff.WriteByte(comma)
	}
	s.buff.Truncate(s.buff.Len() - 1) // Remove final comma
	s.buff.WriteByte(rightParen)
	return nil, nil
}

// writeAddFields writes the AdditionalFields struct field out to JSON as field
// values. i must be a map[string]interface{} or this will panic.
func writeAddFields(i interface{}, buff *bytes.Buffer, enc *json.Encoder) error {
	m := i.(map[string]interface{})

	x := 0
	for k, v := range m {
		buff.WriteString(fmt.Sprintf("%q:", k))
		if err := enc.Encode(v); err != nil {
			return err
		}
		buff.Truncate(buff.Len() - 1) // Remove Encode() added \n

		if x+1 != len(m) {
			buff.WriteByte(comma)
		}
		x++
	}
	return nil
}
