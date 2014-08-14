/*
Copyright 2014 Google Inc. All rights reserved.

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

package conversion

import (
	"fmt"
	"reflect"
)

type typePair struct {
	source reflect.Type
	dest   reflect.Type
}

// DebugLogger allows you to get debugging messages if necessary.
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

// Converter knows how to convert one type to another.
type Converter struct {
	// Map from the conversion pair to a function which can
	// do the conversion.
	funcs map[typePair]reflect.Value

	// If true, print helpful debugging info. Quite verbose.
	Debug DebugLogger
}

// NewConverter makes a new Converter object.
func NewConverter() *Converter {
	return &Converter{
		funcs: map[typePair]reflect.Value{},
	}
}

// Register registers a conversion func with the Converter. conversionFunc must take
// two parameters, the input and output type. It must take a pointer to each. It must
// return an error.
//
// Example:
// c.Register(func(in *Pod, out *v1beta1.Pod) error { ... return nil })
func (c *Converter) Register(conversionFunc interface{}) error {
	fv := reflect.ValueOf(conversionFunc)
	ft := fv.Type()
	if ft.Kind() != reflect.Func {
		return fmt.Errorf("expected func, got: %v", ft)
	}
	if ft.NumIn() != 2 {
		return fmt.Errorf("expected two in params, got: %v", ft)
	}
	if ft.NumOut() != 1 {
		return fmt.Errorf("expected one out param, got: %v", ft)
	}
	if ft.In(0).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for in param 0, got: %v", ft)
	}
	if ft.In(1).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for in param 1, got: %v", ft)
	}
	var forErrorType error
	// This convolution is necessary, otherwise TypeOf picks up on the fact
	// that forErrorType is nil.
	errorType := reflect.TypeOf(&forErrorType).Elem()
	if ft.Out(0) != errorType {
		return fmt.Errorf("expected error return, got: %v", ft)
	}
	c.funcs[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	return nil
}

// FieldMatchingType contains a list of ways in which struct fields could be
// copied. These constants may be | combined.
type FieldMatchingFlags int

const (
	// Loop through destination fields, search for matching source
	// field to copy it from. Source fields with no corresponding
	// destination field will be ignored. If SourceToDest is
	// specified, this flag is ignored. If niether is specified,
	// or no flags are passed, this flag is the default.
	DestFromSource FieldMatchingFlags = 0
	// Loop through source fields, search for matching dest field
	// to copy it into. Destination fields with no corresponding
	// source field will be ignored.
	SourceToDest FieldMatchingFlags = 1 << iota
	// Don't treat it as an error if the corresponding source or
	// dest field can't be found.
	IgnoreMissingFields
	// Don't require type names to match.
	AllowDifferentFieldTypeNames
)

// Returns true if the given flag or combination of flags is set.
func (f FieldMatchingFlags) IsSet(flag FieldMatchingFlags) bool {
	return f&flag == flag
}

// Convert will translate src to dest if it knows how. Both must be pointers.
// If no conversion func is registered and the default copying mechanism
// doesn't work on this type pair, an error will be returned.
// Not safe for objects with cyclic references!
func (c *Converter) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	dv, sv := reflect.ValueOf(dest), reflect.ValueOf(src)
	if dv.Kind() != reflect.Ptr {
		return fmt.Errorf("Need pointer, but got %#v", dest)
	}
	if sv.Kind() != reflect.Ptr {
		return fmt.Errorf("Need pointer, but got %#v", src)
	}
	dv = dv.Elem()
	sv = sv.Elem()
	if !dv.CanAddr() {
		return fmt.Errorf("Can't write to dest")
	}
	return c.convert(sv, dv, flags)
}

// convert recursively copies sv into dv, calling an appropriate conversion function if
// one is registered.
func (c *Converter) convert(sv, dv reflect.Value, flags FieldMatchingFlags) error {
	dt, st := dv.Type(), sv.Type()
	if fv, ok := c.funcs[typePair{st, dt}]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Calling custom conversion of '%v' to '%v'", st, dt)
		}
		ret := fv.Call([]reflect.Value{sv.Addr(), dv.Addr()})[0].Interface()
		// This convolution is necssary because nil interfaces won't convert
		// to errors.
		if ret == nil {
			return nil
		}
		return ret.(error)
	}

	if !flags.IsSet(AllowDifferentFieldTypeNames) && dt.Name() != st.Name() {
		return fmt.Errorf("Can't convert %v to %v because type names don't match.", st, dt)
	}

	// This should handle all simple types.
	if st.AssignableTo(dt) {
		dv.Set(sv)
		return nil
	}
	if st.ConvertibleTo(dt) {
		dv.Set(sv.Convert(dt))
		return nil
	}

	if c.Debug != nil {
		c.Debug.Logf("Trying to convert '%v' to '%v'", st, dt)
	}

	switch dv.Kind() {
	case reflect.Struct:
		listType := dt
		if flags.IsSet(SourceToDest) {
			listType = st
		}
		for i := 0; i < listType.NumField(); i++ {
			f := listType.Field(i)
			df := dv.FieldByName(f.Name)
			sf := sv.FieldByName(f.Name)
			if !df.IsValid() || !sf.IsValid() {
				switch {
				case flags.IsSet(IgnoreMissingFields):
					// No error.
				case flags.IsSet(SourceToDest):
					return fmt.Errorf("%v not present in dest (%v to %v)", f.Name, st, dt)
				default:
					return fmt.Errorf("%v not present in src (%v to %v)", f.Name, st, dt)
				}
				continue
			}
			if err := c.convert(sf, df, flags); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if sv.IsNil() {
			// Don't make a zero-length slice.
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
		for i := 0; i < sv.Len(); i++ {
			if err := c.convert(sv.Index(i), dv.Index(i), flags); err != nil {
				return err
			}
		}
	case reflect.Ptr:
		if sv.IsNil() {
			// Don't copy a nil ptr!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.New(dt.Elem()))
		return c.convert(sv.Elem(), dv.Elem(), flags)
	case reflect.Map:
		if sv.IsNil() {
			// Don't copy a nil ptr!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeMap(dt))
		for _, sk := range sv.MapKeys() {
			dk := reflect.New(dt.Key()).Elem()
			if err := c.convert(sk, dk, flags); err != nil {
				return err
			}
			dkv := reflect.New(dt.Elem()).Elem()
			if err := c.convert(sv.MapIndex(sk), dkv, flags); err != nil {
				return err
			}
			dv.SetMapIndex(dk, dkv)
		}
	default:
		return fmt.Errorf("Couldn't copy '%v' into '%v'", st, dt)
	}
	return nil
}
