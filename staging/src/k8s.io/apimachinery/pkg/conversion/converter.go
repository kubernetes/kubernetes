/*
Copyright 2014 The Kubernetes Authors.

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

type NameFunc func(t reflect.Type) string

var DefaultNameFunc = func(t reflect.Type) string { return t.Name() }

// ConversionFunc converts the object a into the object b, reusing arrays or objects
// or pointers if necessary. It should return an error if the object cannot be converted
// or if some data is invalid. If you do not wish a and b to share fields or nested
// objects, you must copy a before calling this function.
type ConversionFunc func(a, b interface{}, scope Scope) error

// Converter knows how to convert one type to another.
type Converter struct {
	// Map from the conversion pair to a function which can
	// do the conversion.
	conversionFuncs          ConversionFuncs
	generatedConversionFuncs ConversionFuncs

	// Set of conversions that should be treated as a no-op
	ignoredConversions        map[typePair]struct{}
	ignoredUntypedConversions map[typePair]struct{}

	// Map from an input type to a function which can apply a key name mapping
	inputFieldMappingFuncs map[reflect.Type]FieldMappingFunc

	// Map from an input type to a set of default conversion flags.
	inputDefaultFlags map[reflect.Type]FieldMatchingFlags

	// nameFunc is called to retrieve the name of a type; this name is used for the
	// purpose of deciding whether two types match or not (i.e., will we attempt to
	// do a conversion). The default returns the go type name.
	nameFunc func(t reflect.Type) string
}

// NewConverter creates a new Converter object.
func NewConverter(nameFn NameFunc) *Converter {
	c := &Converter{
		conversionFuncs:           NewConversionFuncs(),
		generatedConversionFuncs:  NewConversionFuncs(),
		ignoredConversions:        make(map[typePair]struct{}),
		ignoredUntypedConversions: make(map[typePair]struct{}),
		nameFunc:                  nameFn,

		inputFieldMappingFuncs: make(map[reflect.Type]FieldMappingFunc),
		inputDefaultFlags:      make(map[reflect.Type]FieldMatchingFlags),
	}
	c.RegisterUntypedConversionFunc(
		(*[]byte)(nil), (*[]byte)(nil),
		func(a, b interface{}, s Scope) error {
			return Convert_Slice_byte_To_Slice_byte(a.(*[]byte), b.(*[]byte), s)
		},
	)
	return c
}

// WithConversions returns a Converter that is a copy of c but with the additional
// fns merged on top.
func (c *Converter) WithConversions(fns ConversionFuncs) *Converter {
	copied := *c
	copied.conversionFuncs = c.conversionFuncs.Merge(fns)
	return &copied
}

// DefaultMeta returns the conversion FieldMappingFunc and meta for a given type.
func (c *Converter) DefaultMeta(t reflect.Type) (FieldMatchingFlags, *Meta) {
	return c.inputDefaultFlags[t], &Meta{
		KeyNameMapping: c.inputFieldMappingFuncs[t],
	}
}

// Convert_Slice_byte_To_Slice_byte prevents recursing into every byte
func Convert_Slice_byte_To_Slice_byte(in *[]byte, out *[]byte, s Scope) error {
	if *in == nil {
		*out = nil
		return nil
	}
	*out = make([]byte, len(*in))
	copy(*out, *in)
	return nil
}

// Scope is passed to conversion funcs to allow them to continue an ongoing conversion.
// If multiple converters exist in the system, Scope will allow you to use the correct one
// from a conversion function--that is, the one your conversion function was called by.
type Scope interface {
	// Call Convert to convert sub-objects. Note that if you call it with your own exact
	// parameters, you'll run out of stack space before anything useful happens.
	Convert(src, dest interface{}, flags FieldMatchingFlags) error

	// Flags returns the flags with which the conversion was started.
	Flags() FieldMatchingFlags

	// Meta returns any information originally passed to Convert.
	Meta() *Meta
}

// FieldMappingFunc can convert an input field value into different values, depending on
// the value of the source or destination struct tags.
type FieldMappingFunc func(key string, sourceTag, destTag reflect.StructTag) (source string, dest string)

func NewConversionFuncs() ConversionFuncs {
	return ConversionFuncs{
		untyped: make(map[typePair]ConversionFunc),
	}
}

type ConversionFuncs struct {
	untyped map[typePair]ConversionFunc
}

// AddUntyped adds the provided conversion function to the lookup table for the types that are
// supplied as a and b. a and b must be pointers or an error is returned. This method overwrites
// previously defined functions.
func (c ConversionFuncs) AddUntyped(a, b interface{}, fn ConversionFunc) error {
	tA, tB := reflect.TypeOf(a), reflect.TypeOf(b)
	if tA.Kind() != reflect.Ptr {
		return fmt.Errorf("the type %T must be a pointer to register as an untyped conversion", a)
	}
	if tB.Kind() != reflect.Ptr {
		return fmt.Errorf("the type %T must be a pointer to register as an untyped conversion", b)
	}
	c.untyped[typePair{tA, tB}] = fn
	return nil
}

// Merge returns a new ConversionFuncs that contains all conversions from
// both other and c, with other conversions taking precedence.
func (c ConversionFuncs) Merge(other ConversionFuncs) ConversionFuncs {
	merged := NewConversionFuncs()
	for k, v := range c.untyped {
		merged.untyped[k] = v
	}
	for k, v := range other.untyped {
		merged.untyped[k] = v
	}
	return merged
}

// Meta is supplied by Scheme, when it calls Convert.
type Meta struct {
	// KeyNameMapping is an optional function which may map the listed key (field name)
	// into a source and destination value.
	KeyNameMapping FieldMappingFunc
	// Context is an optional field that callers may use to pass info to conversion functions.
	Context interface{}
}

// scope contains information about an ongoing conversion.
type scope struct {
	converter *Converter
	meta      *Meta
	flags     FieldMatchingFlags
}

// Convert continues a conversion.
func (s *scope) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.Convert(src, dest, flags, s.meta)
}

// Flags returns the flags with which the current conversion was started.
func (s *scope) Flags() FieldMatchingFlags {
	return s.flags
}

// Meta returns the meta object that was originally passed to Convert.
func (s *scope) Meta() *Meta {
	return s.meta
}

// RegisterUntypedConversionFunc registers a function that converts between a and b by passing objects of those
// types to the provided function. The function *must* accept objects of a and b - this machinery will not enforce
// any other guarantee.
func (c *Converter) RegisterUntypedConversionFunc(a, b interface{}, fn ConversionFunc) error {
	return c.conversionFuncs.AddUntyped(a, b, fn)
}

// RegisterGeneratedUntypedConversionFunc registers a function that converts between a and b by passing objects of those
// types to the provided function. The function *must* accept objects of a and b - this machinery will not enforce
// any other guarantee.
func (c *Converter) RegisterGeneratedUntypedConversionFunc(a, b interface{}, fn ConversionFunc) error {
	return c.generatedConversionFuncs.AddUntyped(a, b, fn)
}

// RegisterIgnoredConversion registers a "no-op" for conversion, where any requested
// conversion between from and to is ignored.
func (c *Converter) RegisterIgnoredConversion(from, to interface{}) error {
	typeFrom := reflect.TypeOf(from)
	typeTo := reflect.TypeOf(to)
	if reflect.TypeOf(from).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'from' param 0, got: %v", typeFrom)
	}
	if typeTo.Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'to' param 1, got: %v", typeTo)
	}
	c.ignoredConversions[typePair{typeFrom.Elem(), typeTo.Elem()}] = struct{}{}
	c.ignoredUntypedConversions[typePair{typeFrom, typeTo}] = struct{}{}
	return nil
}

// RegisterInputDefaults registers a field name mapping function, used when converting
// from maps to structs. Inputs to the conversion methods are checked for this type and a mapping
// applied automatically if the input matches in. A set of default flags for the input conversion
// may also be provided, which will be used when no explicit flags are requested.
func (c *Converter) RegisterInputDefaults(in interface{}, fn FieldMappingFunc, defaultFlags FieldMatchingFlags) error {
	fv := reflect.ValueOf(in)
	ft := fv.Type()
	if ft.Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer 'in' argument, got: %v", ft)
	}
	c.inputFieldMappingFuncs[ft] = fn
	c.inputDefaultFlags[ft] = defaultFlags
	return nil
}

// FieldMatchingFlags contains a list of ways in which struct fields could be
// copied. These constants may be | combined.
type FieldMatchingFlags int

const (
	// Loop through destination fields, search for matching source
	// field to copy it from. Source fields with no corresponding
	// destination field will be ignored. If SourceToDest is
	// specified, this flag is ignored. If neither is specified,
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

// IsSet returns true if the given flag or combination of flags is set.
func (f FieldMatchingFlags) IsSet(flag FieldMatchingFlags) bool {
	if flag == DestFromSource {
		// The bit logic doesn't work on the default value.
		return f&SourceToDest != SourceToDest
	}
	return f&flag == flag
}

// Convert will translate src to dest if it knows how. Both must be pointers.
// If no conversion func is registered and the default copying mechanism
// doesn't work on this type pair, an error will be returned.
// Read the comments on the various FieldMatchingFlags constants to understand
// what the 'flags' parameter does.
// 'meta' is given to allow you to pass information to conversion functions,
// it is not used by Convert() other than storing it in the scope.
// Not safe for objects with cyclic references!
func (c *Converter) Convert(src, dest interface{}, flags FieldMatchingFlags, meta *Meta) error {
	pair := typePair{reflect.TypeOf(src), reflect.TypeOf(dest)}
	scope := &scope{
		converter: c,
		flags:     flags,
		meta:      meta,
	}

	// ignore conversions of this type
	if _, ok := c.ignoredUntypedConversions[pair]; ok {
		return nil
	}
	if fn, ok := c.conversionFuncs.untyped[pair]; ok {
		return fn(src, dest, scope)
	}
	if fn, ok := c.generatedConversionFuncs.untyped[pair]; ok {
		return fn(src, dest, scope)
	}

	dv, err := EnforcePtr(dest)
	if err != nil {
		return err
	}
	sv, err := EnforcePtr(src)
	if err != nil {
		return err
	}
	return fmt.Errorf("converting (%s) to (%s): unknown conversion", sv.Type(), dv.Type())
}
