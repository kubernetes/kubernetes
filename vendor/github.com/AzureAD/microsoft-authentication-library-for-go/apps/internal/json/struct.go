// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package json

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

func unmarshalStruct(jdec *json.Decoder, i interface{}) error {
	v := reflect.ValueOf(i)
	if v.Kind() != reflect.Ptr {
		return fmt.Errorf("Unmarshal() received type %T, which is not a *struct", i)
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("Unmarshal() received type %T, which is not a *struct", i)
	}

	if hasUnmarshalJSON(v) {
		// Indicates that this type has a custom Unmarshaler.
		return jdec.Decode(v.Addr().Interface())
	}

	f := v.FieldByName(addField)
	if f.Kind() == reflect.Invalid {
		return fmt.Errorf("Unmarshal(%T) only supports structs that have the field AdditionalFields or implements json.Unmarshaler", i)
	}

	if f.Kind() != reflect.Map || !f.Type().AssignableTo(mapStrInterType) {
		return fmt.Errorf("type %T has field 'AdditionalFields' that is not a map[string]interface{}", i)
	}

	dec := newDecoder(jdec, v)
	return dec.run()
}

type decoder struct {
	dec        *json.Decoder
	value      reflect.Value // This will be a reflect.Struct
	translator translateFields
	key        string
}

func newDecoder(dec *json.Decoder, value reflect.Value) *decoder {
	return &decoder{value: value, dec: dec}
}

// run runs our decoder state machine.
func (d *decoder) run() error {
	var state = d.start
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

// start looks for our opening delimeter '{' and then transitions to looping through our fields.
func (d *decoder) start() (stateFn, error) {
	var err error
	d.translator, err = findFields(d.value)
	if err != nil {
		return nil, err
	}

	delim, err := d.dec.Token()
	if err != nil {
		return nil, err
	}
	if !delimIs(delim, '{') {
		return nil, fmt.Errorf("Unmarshal expected opening {, received %v", delim)
	}

	return d.next, nil
}

// next gets the next struct field name from the raw json or stops the machine if we get our closing }.
func (d *decoder) next() (stateFn, error) {
	if !d.dec.More() {
		// Remove the closing }.
		if _, err := d.dec.Token(); err != nil {
			return nil, err
		}
		return nil, nil
	}

	key, err := d.dec.Token()
	if err != nil {
		return nil, err
	}

	d.key = key.(string)
	return d.storeValue, nil
}

// storeValue takes the next value and stores it our struct. If the field can't be found
// in the struct, it pushes the operation to storeAdditional().
func (d *decoder) storeValue() (stateFn, error) {
	goName := d.translator.goName(d.key)
	if goName == "" {
		goName = d.key
	}

	// We don't have the field in the struct, so it goes in AdditionalFields.
	f := d.value.FieldByName(goName)
	if f.Kind() == reflect.Invalid {
		return d.storeAdditional, nil
	}

	// Indicates that this type has a custom Unmarshaler.
	if hasUnmarshalJSON(f) {
		err := d.dec.Decode(f.Addr().Interface())
		if err != nil {
			return nil, err
		}
		return d.next, nil
	}

	t, isPtr, err := fieldBaseType(d.value, goName)
	if err != nil {
		return nil, fmt.Errorf("type(%s) had field(%s) %w", d.value.Type().Name(), goName, err)
	}

	switch t.Kind() {
	// We need to recursively call ourselves on any *struct or struct.
	case reflect.Struct:
		if isPtr {
			if f.IsNil() {
				f.Set(reflect.New(t))
			}
		} else {
			f = f.Addr()
		}
		if err := unmarshalStruct(d.dec, f.Interface()); err != nil {
			return nil, err
		}
		return d.next, nil
	case reflect.Map:
		v := reflect.MakeMap(f.Type())
		ptr := newValue(f.Type())
		ptr.Elem().Set(v)
		if err := unmarshalMap(d.dec, ptr); err != nil {
			return nil, err
		}
		f.Set(ptr.Elem())
		return d.next, nil
	case reflect.Slice:
		v := reflect.MakeSlice(f.Type(), 0, 0)
		ptr := newValue(f.Type())
		ptr.Elem().Set(v)
		if err := unmarshalSlice(d.dec, ptr); err != nil {
			return nil, err
		}
		f.Set(ptr.Elem())
		return d.next, nil
	}

	if !isPtr {
		f = f.Addr()
	}

	// For values that are pointers, we need them to be non-nil in order
	// to decode into them.
	if f.IsNil() {
		f.Set(reflect.New(t))
	}

	if err := d.dec.Decode(f.Interface()); err != nil {
		return nil, err
	}

	return d.next, nil
}

// storeAdditional pushes the key/value into our .AdditionalFields map.
func (d *decoder) storeAdditional() (stateFn, error) {
	rw := json.RawMessage{}
	if err := d.dec.Decode(&rw); err != nil {
		return nil, err
	}
	field := d.value.FieldByName(addField)
	if field.IsNil() {
		field.Set(reflect.MakeMap(field.Type()))
	}
	field.SetMapIndex(reflect.ValueOf(d.key), reflect.ValueOf(rw))
	return d.next, nil
}

func fieldBaseType(v reflect.Value, fieldName string) (t reflect.Type, isPtr bool, err error) {
	sf, ok := v.Type().FieldByName(fieldName)
	if !ok {
		return nil, false, fmt.Errorf("bug: fieldBaseType() lookup of field(%s) on type(%s): do not have field", fieldName, v.Type().Name())
	}
	t = sf.Type
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
		isPtr = true
	}
	if t.Kind() == reflect.Ptr {
		return nil, isPtr, fmt.Errorf("received pointer to pointer type, not supported")
	}
	return t, isPtr, nil
}

type translateField struct {
	jsonName string
	goName   string
}

// translateFields is a list of translateFields with a handy lookup method.
type translateFields []translateField

// goName loops through a list of fields looking for one contaning the jsonName and
// returning the goName. If not found, returns the empty string.
// Note: not a map because at this size slices are faster even in tight loops.
func (t translateFields) goName(jsonName string) string {
	for _, entry := range t {
		if entry.jsonName == jsonName {
			return entry.goName
		}
	}
	return ""
}

// jsonName loops through a list of fields looking for one contaning the goName and
// returning the jsonName. If not found, returns the empty string.
// Note: not a map because at this size slices are faster even in tight loops.
func (t translateFields) jsonName(goName string) string {
	for _, entry := range t {
		if entry.goName == goName {
			return entry.jsonName
		}
	}
	return ""
}

var umarshalerType = reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()

// findFields parses a struct and writes the field tags for lookup. It will return an error
// if any field has a type of *struct or struct that does not implement json.Marshaler.
func findFields(v reflect.Value) (translateFields, error) {
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("findFields received a %s type, expected *struct or struct", v.Type().Name())
	}
	tfs := make([]translateField, 0, v.NumField())
	for i := 0; i < v.NumField(); i++ {
		tf := translateField{
			goName:   v.Type().Field(i).Name,
			jsonName: parseTag(v.Type().Field(i).Tag.Get("json")),
		}
		switch tf.jsonName {
		case "", "-":
			tf.jsonName = tf.goName
		}
		tfs = append(tfs, tf)

		f := v.Field(i)
		if f.Kind() == reflect.Ptr {
			f = f.Elem()
		}
		if f.Kind() == reflect.Struct {
			if f.Type().Implements(umarshalerType) {
				return nil, fmt.Errorf("struct type %q which has field %q which "+
					"doesn't implement json.Unmarshaler", v.Type().Name(), v.Type().Field(i).Name)
			}
		}
	}
	return tfs, nil
}

// parseTag just returns the first entry in the tag. tag is the string
// returned by reflect.StructField.Tag().Get().
func parseTag(tag string) string {
	if idx := strings.Index(tag, ","); idx != -1 {
		return tag[:idx]
	}
	return tag
}
