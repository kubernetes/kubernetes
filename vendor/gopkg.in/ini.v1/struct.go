// Copyright 2014 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package ini

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
	"unicode"
)

// NameMapper represents a ini tag name mapper.
type NameMapper func(string) string

// Built-in name getters.
var (
	// SnackCase converts to format SNACK_CASE.
	SnackCase NameMapper = func(raw string) string {
		newstr := make([]rune, 0, len(raw))
		for i, chr := range raw {
			if isUpper := 'A' <= chr && chr <= 'Z'; isUpper {
				if i > 0 {
					newstr = append(newstr, '_')
				}
			}
			newstr = append(newstr, unicode.ToUpper(chr))
		}
		return string(newstr)
	}
	// TitleUnderscore converts to format title_underscore.
	TitleUnderscore NameMapper = func(raw string) string {
		newstr := make([]rune, 0, len(raw))
		for i, chr := range raw {
			if isUpper := 'A' <= chr && chr <= 'Z'; isUpper {
				if i > 0 {
					newstr = append(newstr, '_')
				}
				chr -= 'A' - 'a'
			}
			newstr = append(newstr, chr)
		}
		return string(newstr)
	}
)

func (s *Section) parseFieldName(raw, actual string) string {
	if len(actual) > 0 {
		return actual
	}
	if s.f.NameMapper != nil {
		return s.f.NameMapper(raw)
	}
	return raw
}

func parseDelim(actual string) string {
	if len(actual) > 0 {
		return actual
	}
	return ","
}

var reflectTime = reflect.TypeOf(time.Now()).Kind()

// setSliceWithProperType sets proper values to slice based on its type.
func setSliceWithProperType(key *Key, field reflect.Value, delim string, allowShadow, isStrict bool) error {
	var strs []string
	if allowShadow {
		strs = key.StringsWithShadows(delim)
	} else {
		strs = key.Strings(delim)
	}

	numVals := len(strs)
	if numVals == 0 {
		return nil
	}

	var vals interface{}
	var err error

	sliceOf := field.Type().Elem().Kind()
	switch sliceOf {
	case reflect.String:
		vals = strs
	case reflect.Int:
		vals, err = key.parseInts(strs, true, false)
	case reflect.Int64:
		vals, err = key.parseInt64s(strs, true, false)
	case reflect.Uint:
		vals, err = key.parseUints(strs, true, false)
	case reflect.Uint64:
		vals, err = key.parseUint64s(strs, true, false)
	case reflect.Float64:
		vals, err = key.parseFloat64s(strs, true, false)
	case reflect.Bool:
		vals, err = key.parseBools(strs, true, false)
	case reflectTime:
		vals, err = key.parseTimesFormat(time.RFC3339, strs, true, false)
	default:
		return fmt.Errorf("unsupported type '[]%s'", sliceOf)
	}
	if err != nil && isStrict {
		return err
	}

	slice := reflect.MakeSlice(field.Type(), numVals, numVals)
	for i := 0; i < numVals; i++ {
		switch sliceOf {
		case reflect.String:
			slice.Index(i).Set(reflect.ValueOf(vals.([]string)[i]))
		case reflect.Int:
			slice.Index(i).Set(reflect.ValueOf(vals.([]int)[i]))
		case reflect.Int64:
			slice.Index(i).Set(reflect.ValueOf(vals.([]int64)[i]))
		case reflect.Uint:
			slice.Index(i).Set(reflect.ValueOf(vals.([]uint)[i]))
		case reflect.Uint64:
			slice.Index(i).Set(reflect.ValueOf(vals.([]uint64)[i]))
		case reflect.Float64:
			slice.Index(i).Set(reflect.ValueOf(vals.([]float64)[i]))
		case reflect.Bool:
			slice.Index(i).Set(reflect.ValueOf(vals.([]bool)[i]))
		case reflectTime:
			slice.Index(i).Set(reflect.ValueOf(vals.([]time.Time)[i]))
		}
	}
	field.Set(slice)
	return nil
}

func wrapStrictError(err error, isStrict bool) error {
	if isStrict {
		return err
	}
	return nil
}

// setWithProperType sets proper value to field based on its type,
// but it does not return error for failing parsing,
// because we want to use default value that is already assigned to struct.
func setWithProperType(t reflect.Type, key *Key, field reflect.Value, delim string, allowShadow, isStrict bool) error {
	vt := t
	isPtr := t.Kind() == reflect.Ptr
	if isPtr {
		vt = t.Elem()
	}
	switch vt.Kind() {
	case reflect.String:
		stringVal := key.String()
		if isPtr {
			field.Set(reflect.ValueOf(&stringVal))
		} else if len(stringVal) > 0 {
			field.SetString(key.String())
		}
	case reflect.Bool:
		boolVal, err := key.Bool()
		if err != nil {
			return wrapStrictError(err, isStrict)
		}
		if isPtr {
			field.Set(reflect.ValueOf(&boolVal))
		} else {
			field.SetBool(boolVal)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		// ParseDuration will not return err for `0`, so check the type name
		if vt.Name() == "Duration" {
			durationVal, err := key.Duration()
			if err != nil {
				return wrapStrictError(err, isStrict)
			}
			if isPtr {
				field.Set(reflect.ValueOf(&durationVal))
			} else if int64(durationVal) > 0 {
				field.Set(reflect.ValueOf(durationVal))
			}
			return nil
		}

		intVal, err := key.Int64()
		if err != nil {
			return wrapStrictError(err, isStrict)
		}
		if isPtr {
			pv := reflect.New(t.Elem())
			pv.Elem().SetInt(intVal)
			field.Set(pv)
		} else {
			field.SetInt(intVal)
		}
	//	byte is an alias for uint8, so supporting uint8 breaks support for byte
	case reflect.Uint, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		durationVal, err := key.Duration()
		// Skip zero value
		if err == nil && uint64(durationVal) > 0 {
			if isPtr {
				field.Set(reflect.ValueOf(&durationVal))
			} else {
				field.Set(reflect.ValueOf(durationVal))
			}
			return nil
		}

		uintVal, err := key.Uint64()
		if err != nil {
			return wrapStrictError(err, isStrict)
		}
		if isPtr {
			pv := reflect.New(t.Elem())
			pv.Elem().SetUint(uintVal)
			field.Set(pv)
		} else {
			field.SetUint(uintVal)
		}

	case reflect.Float32, reflect.Float64:
		floatVal, err := key.Float64()
		if err != nil {
			return wrapStrictError(err, isStrict)
		}
		if isPtr {
			pv := reflect.New(t.Elem())
			pv.Elem().SetFloat(floatVal)
			field.Set(pv)
		} else {
			field.SetFloat(floatVal)
		}
	case reflectTime:
		timeVal, err := key.Time()
		if err != nil {
			return wrapStrictError(err, isStrict)
		}
		if isPtr {
			field.Set(reflect.ValueOf(&timeVal))
		} else {
			field.Set(reflect.ValueOf(timeVal))
		}
	case reflect.Slice:
		return setSliceWithProperType(key, field, delim, allowShadow, isStrict)
	default:
		return fmt.Errorf("unsupported type '%s'", t)
	}
	return nil
}

func parseTagOptions(tag string) (rawName string, omitEmpty bool, allowShadow bool) {
	opts := strings.SplitN(tag, ",", 3)
	rawName = opts[0]
	if len(opts) > 1 {
		omitEmpty = opts[1] == "omitempty"
	}
	if len(opts) > 2 {
		allowShadow = opts[2] == "allowshadow"
	}
	return rawName, omitEmpty, allowShadow
}

func (s *Section) mapTo(val reflect.Value, isStrict bool) error {
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}
	typ := val.Type()

	for i := 0; i < typ.NumField(); i++ {
		field := val.Field(i)
		tpField := typ.Field(i)

		tag := tpField.Tag.Get("ini")
		if tag == "-" {
			continue
		}

		rawName, _, allowShadow := parseTagOptions(tag)
		fieldName := s.parseFieldName(tpField.Name, rawName)
		if len(fieldName) == 0 || !field.CanSet() {
			continue
		}

		isStruct := tpField.Type.Kind() == reflect.Struct
		isStructPtr := tpField.Type.Kind() == reflect.Ptr && tpField.Type.Elem().Kind() == reflect.Struct
		isAnonymous := tpField.Type.Kind() == reflect.Ptr && tpField.Anonymous
		if isAnonymous {
			field.Set(reflect.New(tpField.Type.Elem()))
		}

		if isAnonymous || isStruct || isStructPtr {
			if sec, err := s.f.GetSection(fieldName); err == nil {
				// Only set the field to non-nil struct value if we have
				// a section for it. Otherwise, we end up with a non-nil
				// struct ptr even though there is no data.
				if isStructPtr && field.IsNil() {
					field.Set(reflect.New(tpField.Type.Elem()))
				}
				if err = sec.mapTo(field, isStrict); err != nil {
					return fmt.Errorf("error mapping field(%s): %v", fieldName, err)
				}
				continue
			}
		}
		if key, err := s.GetKey(fieldName); err == nil {
			delim := parseDelim(tpField.Tag.Get("delim"))
			if err = setWithProperType(tpField.Type, key, field, delim, allowShadow, isStrict); err != nil {
				return fmt.Errorf("error mapping field(%s): %v", fieldName, err)
			}
		}
	}
	return nil
}

// MapTo maps section to given struct.
func (s *Section) MapTo(v interface{}) error {
	typ := reflect.TypeOf(v)
	val := reflect.ValueOf(v)
	if typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
		val = val.Elem()
	} else {
		return errors.New("cannot map to non-pointer struct")
	}

	return s.mapTo(val, false)
}

// StrictMapTo maps section to given struct in strict mode,
// which returns all possible error including value parsing error.
func (s *Section) StrictMapTo(v interface{}) error {
	typ := reflect.TypeOf(v)
	val := reflect.ValueOf(v)
	if typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
		val = val.Elem()
	} else {
		return errors.New("cannot map to non-pointer struct")
	}

	return s.mapTo(val, true)
}

// MapTo maps file to given struct.
func (f *File) MapTo(v interface{}) error {
	return f.Section("").MapTo(v)
}

// StrictMapTo maps file to given struct in strict mode,
// which returns all possible error including value parsing error.
func (f *File) StrictMapTo(v interface{}) error {
	return f.Section("").StrictMapTo(v)
}

// MapToWithMapper maps data sources to given struct with name mapper.
func MapToWithMapper(v interface{}, mapper NameMapper, source interface{}, others ...interface{}) error {
	cfg, err := Load(source, others...)
	if err != nil {
		return err
	}
	cfg.NameMapper = mapper
	return cfg.MapTo(v)
}

// StrictMapToWithMapper maps data sources to given struct with name mapper in strict mode,
// which returns all possible error including value parsing error.
func StrictMapToWithMapper(v interface{}, mapper NameMapper, source interface{}, others ...interface{}) error {
	cfg, err := Load(source, others...)
	if err != nil {
		return err
	}
	cfg.NameMapper = mapper
	return cfg.StrictMapTo(v)
}

// MapTo maps data sources to given struct.
func MapTo(v, source interface{}, others ...interface{}) error {
	return MapToWithMapper(v, nil, source, others...)
}

// StrictMapTo maps data sources to given struct in strict mode,
// which returns all possible error including value parsing error.
func StrictMapTo(v, source interface{}, others ...interface{}) error {
	return StrictMapToWithMapper(v, nil, source, others...)
}

// reflectSliceWithProperType does the opposite thing as setSliceWithProperType.
func reflectSliceWithProperType(key *Key, field reflect.Value, delim string, allowShadow bool) error {
	slice := field.Slice(0, field.Len())
	if field.Len() == 0 {
		return nil
	}
	sliceOf := field.Type().Elem().Kind()

	if allowShadow {
		var keyWithShadows *Key
		for i := 0; i < field.Len(); i++ {
			var val string
			switch sliceOf {
			case reflect.String:
				val = slice.Index(i).String()
			case reflect.Int, reflect.Int64:
				val = fmt.Sprint(slice.Index(i).Int())
			case reflect.Uint, reflect.Uint64:
				val = fmt.Sprint(slice.Index(i).Uint())
			case reflect.Float64:
				val = fmt.Sprint(slice.Index(i).Float())
			case reflect.Bool:
				val = fmt.Sprint(slice.Index(i).Bool())
			case reflectTime:
				val = slice.Index(i).Interface().(time.Time).Format(time.RFC3339)
			default:
				return fmt.Errorf("unsupported type '[]%s'", sliceOf)
			}

			if i == 0 {
				keyWithShadows = newKey(key.s, key.name, val)
			} else {
				keyWithShadows.AddShadow(val)
			}
		}
		key = keyWithShadows
		return nil
	}

	var buf bytes.Buffer
	for i := 0; i < field.Len(); i++ {
		switch sliceOf {
		case reflect.String:
			buf.WriteString(slice.Index(i).String())
		case reflect.Int, reflect.Int64:
			buf.WriteString(fmt.Sprint(slice.Index(i).Int()))
		case reflect.Uint, reflect.Uint64:
			buf.WriteString(fmt.Sprint(slice.Index(i).Uint()))
		case reflect.Float64:
			buf.WriteString(fmt.Sprint(slice.Index(i).Float()))
		case reflect.Bool:
			buf.WriteString(fmt.Sprint(slice.Index(i).Bool()))
		case reflectTime:
			buf.WriteString(slice.Index(i).Interface().(time.Time).Format(time.RFC3339))
		default:
			return fmt.Errorf("unsupported type '[]%s'", sliceOf)
		}
		buf.WriteString(delim)
	}
	key.SetValue(buf.String()[:buf.Len()-len(delim)])
	return nil
}

// reflectWithProperType does the opposite thing as setWithProperType.
func reflectWithProperType(t reflect.Type, key *Key, field reflect.Value, delim string, allowShadow bool) error {
	switch t.Kind() {
	case reflect.String:
		key.SetValue(field.String())
	case reflect.Bool:
		key.SetValue(fmt.Sprint(field.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		key.SetValue(fmt.Sprint(field.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		key.SetValue(fmt.Sprint(field.Uint()))
	case reflect.Float32, reflect.Float64:
		key.SetValue(fmt.Sprint(field.Float()))
	case reflectTime:
		key.SetValue(fmt.Sprint(field.Interface().(time.Time).Format(time.RFC3339)))
	case reflect.Slice:
		return reflectSliceWithProperType(key, field, delim, allowShadow)
	case reflect.Ptr:
		if !field.IsNil() {
			return reflectWithProperType(t.Elem(), key, field.Elem(), delim, allowShadow)
		}
	default:
		return fmt.Errorf("unsupported type '%s'", t)
	}
	return nil
}

// CR: copied from encoding/json/encode.go with modifications of time.Time support.
// TODO: add more test coverage.
func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	case reflectTime:
		t, ok := v.Interface().(time.Time)
		return ok && t.IsZero()
	}
	return false
}

func (s *Section) reflectFrom(val reflect.Value) error {
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}
	typ := val.Type()

	for i := 0; i < typ.NumField(); i++ {
		field := val.Field(i)
		tpField := typ.Field(i)

		tag := tpField.Tag.Get("ini")
		if tag == "-" {
			continue
		}

		rawName, omitEmpty, allowShadow := parseTagOptions(tag)
		if omitEmpty && isEmptyValue(field) {
			continue
		}

		fieldName := s.parseFieldName(tpField.Name, rawName)
		if len(fieldName) == 0 || !field.CanSet() {
			continue
		}

		if (tpField.Type.Kind() == reflect.Ptr && tpField.Anonymous) ||
			(tpField.Type.Kind() == reflect.Struct && tpField.Type.Name() != "Time") {
			// Note: The only error here is section doesn't exist.
			sec, err := s.f.GetSection(fieldName)
			if err != nil {
				// Note: fieldName can never be empty here, ignore error.
				sec, _ = s.f.NewSection(fieldName)
			}

			// Add comment from comment tag
			if len(sec.Comment) == 0 {
				sec.Comment = tpField.Tag.Get("comment")
			}

			if err = sec.reflectFrom(field); err != nil {
				return fmt.Errorf("error reflecting field (%s): %v", fieldName, err)
			}
			continue
		}

		// Note: Same reason as secion.
		key, err := s.GetKey(fieldName)
		if err != nil {
			key, _ = s.NewKey(fieldName, "")
		}

		// Add comment from comment tag
		if len(key.Comment) == 0 {
			key.Comment = tpField.Tag.Get("comment")
		}

		if err = reflectWithProperType(tpField.Type, key, field, parseDelim(tpField.Tag.Get("delim")), allowShadow); err != nil {
			return fmt.Errorf("error reflecting field (%s): %v", fieldName, err)
		}

	}
	return nil
}

// ReflectFrom reflects secion from given struct.
func (s *Section) ReflectFrom(v interface{}) error {
	typ := reflect.TypeOf(v)
	val := reflect.ValueOf(v)
	if typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
		val = val.Elem()
	} else {
		return errors.New("cannot reflect from non-pointer struct")
	}

	return s.reflectFrom(val)
}

// ReflectFrom reflects file from given struct.
func (f *File) ReflectFrom(v interface{}) error {
	return f.Section("").ReflectFrom(v)
}

// ReflectFromWithMapper reflects data sources from given struct with name mapper.
func ReflectFromWithMapper(cfg *File, v interface{}, mapper NameMapper) error {
	cfg.NameMapper = mapper
	return cfg.ReflectFrom(v)
}

// ReflectFrom reflects data sources from given struct.
func ReflectFrom(cfg *File, v interface{}) error {
	return ReflectFromWithMapper(cfg, v, nil)
}
