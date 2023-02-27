// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package json

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// unmarshalMap unmarshal's a map.
func unmarshalMap(dec *json.Decoder, m reflect.Value) error {
	if m.Kind() != reflect.Ptr || m.Elem().Kind() != reflect.Map {
		panic("unmarshalMap called on non-*map value")
	}
	mapValueType := m.Elem().Type().Elem()
	walk := mapWalk{dec: dec, m: m, valueType: mapValueType}
	if err := walk.run(); err != nil {
		return err
	}
	return nil
}

type mapWalk struct {
	dec       *json.Decoder
	key       string
	m         reflect.Value
	valueType reflect.Type
}

// run runs our decoder state machine.
func (m *mapWalk) run() error {
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

func (m *mapWalk) start() (stateFn, error) {
	// maps can have custom unmarshaler's.
	if hasUnmarshalJSON(m.m) {
		err := m.dec.Decode(m.m.Interface())
		if err != nil {
			return nil, err
		}
		return nil, nil
	}

	// We only want to use this if the map value is:
	// *struct/struct/map/slice
	// otherwise use standard decode
	t, _ := m.valueBaseType()
	switch t.Kind() {
	case reflect.Struct, reflect.Map, reflect.Slice:
		delim, err := m.dec.Token()
		if err != nil {
			return nil, err
		}
		// This indicates the value was set to JSON null.
		if delim == nil {
			return nil, nil
		}
		if !delimIs(delim, '{') {
			return nil, fmt.Errorf("Unmarshal expected opening {, received %v", delim)
		}
		return m.next, nil
	case reflect.Ptr:
		return nil, fmt.Errorf("do not support maps with values of '**type' or '*reference")
	}

	// This is a basic map type, so just use Decode().
	if err := m.dec.Decode(m.m.Interface()); err != nil {
		return nil, err
	}

	return nil, nil
}

func (m *mapWalk) next() (stateFn, error) {
	if m.dec.More() {
		key, err := m.dec.Token()
		if err != nil {
			return nil, err
		}
		m.key = key.(string)
		return m.storeValue, nil
	}
	// No more entries, so remove final }.
	_, err := m.dec.Token()
	if err != nil {
		return nil, err
	}
	return nil, nil
}

func (m *mapWalk) storeValue() (stateFn, error) {
	v := m.valueType
	for {
		switch v.Kind() {
		case reflect.Ptr:
			v = v.Elem()
			continue
		case reflect.Struct:
			return m.storeStruct, nil
		case reflect.Map:
			return m.storeMap, nil
		case reflect.Slice:
			return m.storeSlice, nil
		}
		return nil, fmt.Errorf("bug: mapWalk.storeValue() called on unsupported type: %v", v.Kind())
	}
}

func (m *mapWalk) storeStruct() (stateFn, error) {
	v := newValue(m.valueType)
	if err := unmarshalStruct(m.dec, v.Interface()); err != nil {
		return nil, err
	}

	if m.valueType.Kind() == reflect.Ptr {
		m.m.Elem().SetMapIndex(reflect.ValueOf(m.key), v)
		return m.next, nil
	}
	m.m.Elem().SetMapIndex(reflect.ValueOf(m.key), v.Elem())

	return m.next, nil
}

func (m *mapWalk) storeMap() (stateFn, error) {
	v := reflect.MakeMap(m.valueType)
	ptr := newValue(v.Type())
	ptr.Elem().Set(v)
	if err := unmarshalMap(m.dec, ptr); err != nil {
		return nil, err
	}

	m.m.Elem().SetMapIndex(reflect.ValueOf(m.key), v)

	return m.next, nil
}

func (m *mapWalk) storeSlice() (stateFn, error) {
	v := newValue(m.valueType)
	if err := unmarshalSlice(m.dec, v); err != nil {
		return nil, err
	}

	m.m.Elem().SetMapIndex(reflect.ValueOf(m.key), v.Elem())

	return m.next, nil
}

// valueType returns the underlying Type. So a *struct would yield
// struct, etc...
func (m *mapWalk) valueBaseType() (reflect.Type, bool) {
	ptr := false
	v := m.valueType
	if v.Kind() == reflect.Ptr {
		ptr = true
		v = v.Elem()
	}
	return v, ptr
}

// unmarshalSlice unmarshal's the next value, which must be a slice, into
// ptrSlice, which must be a pointer to a slice. newValue() can be use to
// create the slice.
func unmarshalSlice(dec *json.Decoder, ptrSlice reflect.Value) error {
	if ptrSlice.Kind() != reflect.Ptr || ptrSlice.Elem().Kind() != reflect.Slice {
		panic("unmarshalSlice called on non-*[]slice value")
	}
	sliceValueType := ptrSlice.Elem().Type().Elem()
	walk := sliceWalk{
		dec:       dec,
		s:         ptrSlice,
		valueType: sliceValueType,
	}
	if err := walk.run(); err != nil {
		return err
	}

	return nil
}

type sliceWalk struct {
	dec       *json.Decoder
	s         reflect.Value // *[]slice
	valueType reflect.Type
}

// run runs our decoder state machine.
func (s *sliceWalk) run() error {
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

func (s *sliceWalk) start() (stateFn, error) {
	// slices can have custom unmarshaler's.
	if hasUnmarshalJSON(s.s) {
		err := s.dec.Decode(s.s.Interface())
		if err != nil {
			return nil, err
		}
		return nil, nil
	}

	// We only want to use this if the slice value is:
	// []*struct/[]struct/[]map/[]slice
	// otherwise use standard decode
	t := s.valueBaseType()

	switch t.Kind() {
	case reflect.Ptr:
		return nil, fmt.Errorf("cannot unmarshal into a **<type> or *<reference>")
	case reflect.Struct, reflect.Map, reflect.Slice:
		delim, err := s.dec.Token()
		if err != nil {
			return nil, err
		}
		// This indicates the value was set to nil.
		if delim == nil {
			return nil, nil
		}
		if !delimIs(delim, '[') {
			return nil, fmt.Errorf("Unmarshal expected opening [, received %v", delim)
		}
		return s.next, nil
	}

	if err := s.dec.Decode(s.s.Interface()); err != nil {
		return nil, err
	}
	return nil, nil
}

func (s *sliceWalk) next() (stateFn, error) {
	if s.dec.More() {
		return s.storeValue, nil
	}
	// Nothing left in the slice, remove closing ]
	_, err := s.dec.Token()
	return nil, err
}

func (s *sliceWalk) storeValue() (stateFn, error) {
	t := s.valueBaseType()
	switch t.Kind() {
	case reflect.Ptr:
		return nil, fmt.Errorf("do not support 'pointer to pointer' or 'pointer to reference' types")
	case reflect.Struct:
		return s.storeStruct, nil
	case reflect.Map:
		return s.storeMap, nil
	case reflect.Slice:
		return s.storeSlice, nil
	}
	return nil, fmt.Errorf("bug: sliceWalk.storeValue() called on unsupported type: %v", t.Kind())
}

func (s *sliceWalk) storeStruct() (stateFn, error) {
	v := newValue(s.valueType)
	if err := unmarshalStruct(s.dec, v.Interface()); err != nil {
		return nil, err
	}

	if s.valueType.Kind() == reflect.Ptr {
		s.s.Elem().Set(reflect.Append(s.s.Elem(), v))
		return s.next, nil
	}

	s.s.Elem().Set(reflect.Append(s.s.Elem(), v.Elem()))
	return s.next, nil
}

func (s *sliceWalk) storeMap() (stateFn, error) {
	v := reflect.MakeMap(s.valueType)
	ptr := newValue(v.Type())
	ptr.Elem().Set(v)

	if err := unmarshalMap(s.dec, ptr); err != nil {
		return nil, err
	}

	s.s.Elem().Set(reflect.Append(s.s.Elem(), v))

	return s.next, nil
}

func (s *sliceWalk) storeSlice() (stateFn, error) {
	v := newValue(s.valueType)
	if err := unmarshalSlice(s.dec, v); err != nil {
		return nil, err
	}

	s.s.Elem().Set(reflect.Append(s.s.Elem(), v.Elem()))

	return s.next, nil
}

// valueType returns the underlying Type. So a *struct would yield
// struct, etc...
func (s *sliceWalk) valueBaseType() reflect.Type {
	v := s.valueType
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	return v
}

// newValue() returns a new *type that represents type passed.
func newValue(valueType reflect.Type) reflect.Value {
	if valueType.Kind() == reflect.Ptr {
		return reflect.New(valueType.Elem())
	}
	return reflect.New(valueType)
}
