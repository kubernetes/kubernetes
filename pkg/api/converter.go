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

package api

import (
	"fmt"
	"reflect"
)

type typePair struct {
	source reflect.Type
	dest   reflect.Type
}

type debugLogger interface {
	Logf(format string, args ...interface{})
}

// Converter knows how to convert one type to another.
type Converter struct {
	// Map from the conversion pair to a function which can
	// do the conversion.
	funcs map[typePair]reflect.Value

	// If true, print helpful debugging info. Quite verbose.
	debug debugLogger
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

// Convert will translate src to dest if it knows how. Both must be pointers.
// If no conversion func is registered and the default copying mechanism
// doesn't work on this type pair, an error will be returned.
// Not safe for objects with cyclic references!
func (c *Converter) Convert(src, dest interface{}) error {
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
	return c.convert(sv, dv)
}

// convert recursively copies sv into dv, calling an appropriate conversion function if
// one is registered.
func (c *Converter) convert(sv, dv reflect.Value) error {
	dt, st := dv.Type(), sv.Type()
	if fv, ok := c.funcs[typePair{st, dt}]; ok {
		if c.debug != nil {
			c.debug.Logf("Calling custom conversion of '%v' to '%v'", st, dt)
		}
		ret := fv.Call([]reflect.Value{sv.Addr(), dv.Addr()})[0].Interface()
		// This convolution is necssary because nil interfaces won't convert
		// to errors.
		if ret == nil {
			return nil
		}
		return ret.(error)
	}

	if dt.Name() != st.Name() {
		return fmt.Errorf("Type names don't match: %v, %v", dt.Name(), st.Name())
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

	if c.debug != nil {
		c.debug.Logf("Trying to convert '%v' to '%v'", st, dt)
	}

	switch dv.Kind() {
	case reflect.Struct:
		for i := 0; i < dt.NumField(); i++ {
			f := dv.Type().Field(i)
			sf := sv.FieldByName(f.Name)
			if !sf.IsValid() {
				return fmt.Errorf("%v not present in source %v for dest %v", f.Name, sv.Type(), dv.Type())
			}
			df := dv.Field(i)
			if err := c.convert(sf, df); err != nil {
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
			if err := c.convert(sv.Index(i), dv.Index(i)); err != nil {
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
		return c.convert(sv.Elem(), dv.Elem())
	case reflect.Map:
		if sv.IsNil() {
			// Don't copy a nil ptr!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeMap(dt))
		for _, sk := range sv.MapKeys() {
			dk := reflect.New(dt.Key()).Elem()
			if err := c.convert(sk, dk); err != nil {
				return err
			}
			dkv := reflect.New(dt.Elem()).Elem()
			if err := c.convert(sv.MapIndex(sk), dkv); err != nil {
				return err
			}
			dv.SetMapIndex(dk, dkv)
		}
	default:
		return fmt.Errorf("Couldn't copy '%v' into '%v'", st, dt)
	}
	return nil
}
