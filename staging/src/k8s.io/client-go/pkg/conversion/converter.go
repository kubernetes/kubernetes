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

type typeNamePair struct {
	fieldType reflect.Type
	fieldName string
}

// DebugLogger allows you to get debugging messages if necessary.
type DebugLogger interface {
	Logf(format string, args ...interface{})
}

type NameFunc func(t reflect.Type) string

var DefaultNameFunc = func(t reflect.Type) string { return t.Name() }

type GenericConversionFunc func(a, b interface{}, scope Scope) (bool, error)

// Converter knows how to convert one type to another.
type Converter struct {
	// Map from the conversion pair to a function which can
	// do the conversion.
	conversionFuncs          ConversionFuncs
	generatedConversionFuncs ConversionFuncs

	// genericConversions are called during normal conversion to offer a "fast-path"
	// that avoids all reflection. These methods are not called outside of the .Convert()
	// method.
	genericConversions []GenericConversionFunc

	// Set of conversions that should be treated as a no-op
	ignoredConversions map[typePair]struct{}

	// This is a map from a source field type and name, to a list of destination
	// field type and name.
	structFieldDests map[typeNamePair][]typeNamePair

	// Allows for the opposite lookup of structFieldDests. So that SourceFromDest
	// copy flag also works. So this is a map of destination field name, to potential
	// source field name and type to look for.
	structFieldSources map[typeNamePair][]typeNamePair

	// Map from a type to a function which applies defaults.
	defaultingFuncs map[reflect.Type]reflect.Value

	// Similar to above, but function is stored as interface{}.
	defaultingInterfaces map[reflect.Type]interface{}

	// Map from an input type to a function which can apply a key name mapping
	inputFieldMappingFuncs map[reflect.Type]FieldMappingFunc

	// Map from an input type to a set of default conversion flags.
	inputDefaultFlags map[reflect.Type]FieldMatchingFlags

	// If non-nil, will be called to print helpful debugging info. Quite verbose.
	Debug DebugLogger

	// nameFunc is called to retrieve the name of a type; this name is used for the
	// purpose of deciding whether two types match or not (i.e., will we attempt to
	// do a conversion). The default returns the go type name.
	nameFunc func(t reflect.Type) string
}

// NewConverter creates a new Converter object.
func NewConverter(nameFn NameFunc) *Converter {
	c := &Converter{
		conversionFuncs:          NewConversionFuncs(),
		generatedConversionFuncs: NewConversionFuncs(),
		ignoredConversions:       make(map[typePair]struct{}),
		defaultingFuncs:          make(map[reflect.Type]reflect.Value),
		defaultingInterfaces:     make(map[reflect.Type]interface{}),
		nameFunc:                 nameFn,
		structFieldDests:         make(map[typeNamePair][]typeNamePair),
		structFieldSources:       make(map[typeNamePair][]typeNamePair),

		inputFieldMappingFuncs: make(map[reflect.Type]FieldMappingFunc),
		inputDefaultFlags:      make(map[reflect.Type]FieldMatchingFlags),
	}
	c.RegisterConversionFunc(Convert_Slice_byte_To_Slice_byte)
	return c
}

// AddGenericConversionFunc adds a function that accepts the ConversionFunc call pattern
// (for two conversion types) to the converter. These functions are checked first during
// a normal conversion, but are otherwise not called. Use AddConversionFuncs when registering
// typed conversions.
func (c *Converter) AddGenericConversionFunc(fn GenericConversionFunc) {
	c.genericConversions = append(c.genericConversions, fn)
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

	// DefaultConvert performs the default conversion, without calling a conversion func
	// on the current stack frame. This makes it safe to call from a conversion func.
	DefaultConvert(src, dest interface{}, flags FieldMatchingFlags) error

	// If registered, returns a function applying defaults for objects of a given type.
	// Used for automatically generating conversion functions.
	DefaultingInterface(inType reflect.Type) (interface{}, bool)

	// SrcTags and DestTags contain the struct tags that src and dest had, respectively.
	// If the enclosing object was not a struct, then these will contain no tags, of course.
	SrcTag() reflect.StructTag
	DestTag() reflect.StructTag

	// Flags returns the flags with which the conversion was started.
	Flags() FieldMatchingFlags

	// Meta returns any information originally passed to Convert.
	Meta() *Meta
}

// FieldMappingFunc can convert an input field value into different values, depending on
// the value of the source or destination struct tags.
type FieldMappingFunc func(key string, sourceTag, destTag reflect.StructTag) (source string, dest string)

func NewConversionFuncs() ConversionFuncs {
	return ConversionFuncs{fns: make(map[typePair]reflect.Value)}
}

type ConversionFuncs struct {
	fns map[typePair]reflect.Value
}

// Add adds the provided conversion functions to the lookup table - they must have the signature
// `func(type1, type2, Scope) error`. Functions are added in the order passed and will override
// previously registered pairs.
func (c ConversionFuncs) Add(fns ...interface{}) error {
	for _, fn := range fns {
		fv := reflect.ValueOf(fn)
		ft := fv.Type()
		if err := verifyConversionFunctionSignature(ft); err != nil {
			return err
		}
		c.fns[typePair{ft.In(0).Elem(), ft.In(1).Elem()}] = fv
	}
	return nil
}

// Merge returns a new ConversionFuncs that contains all conversions from
// both other and c, with other conversions taking precedence.
func (c ConversionFuncs) Merge(other ConversionFuncs) ConversionFuncs {
	merged := NewConversionFuncs()
	for k, v := range c.fns {
		merged.fns[k] = v
	}
	for k, v := range other.fns {
		merged.fns[k] = v
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

	// srcStack & destStack are separate because they may not have a 1:1
	// relationship.
	srcStack  scopeStack
	destStack scopeStack
}

type scopeStackElem struct {
	tag   reflect.StructTag
	value reflect.Value
	key   string
}

type scopeStack []scopeStackElem

func (s *scopeStack) pop() {
	n := len(*s)
	*s = (*s)[:n-1]
}

func (s *scopeStack) push(e scopeStackElem) {
	*s = append(*s, e)
}

func (s *scopeStack) top() *scopeStackElem {
	return &(*s)[len(*s)-1]
}

func (s scopeStack) describe() string {
	desc := ""
	if len(s) > 1 {
		desc = "(" + s[1].value.Type().String() + ")"
	}
	for i, v := range s {
		if i < 2 {
			// First layer on stack is not real; second is handled specially above.
			continue
		}
		if v.key == "" {
			desc += fmt.Sprintf(".%v", v.value.Type())
		} else {
			desc += fmt.Sprintf(".%v", v.key)
		}
	}
	return desc
}

func (s *scope) DefaultingInterface(inType reflect.Type) (interface{}, bool) {
	value, found := s.converter.defaultingInterfaces[inType]
	return value, found
}

// Formats src & dest as indices for printing.
func (s *scope) setIndices(src, dest int) {
	s.srcStack.top().key = fmt.Sprintf("[%v]", src)
	s.destStack.top().key = fmt.Sprintf("[%v]", dest)
}

// Formats src & dest as map keys for printing.
func (s *scope) setKeys(src, dest interface{}) {
	s.srcStack.top().key = fmt.Sprintf(`["%v"]`, src)
	s.destStack.top().key = fmt.Sprintf(`["%v"]`, dest)
}

// Convert continues a conversion.
func (s *scope) Convert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.Convert(src, dest, flags, s.meta)
}

// DefaultConvert continues a conversion, performing a default conversion (no conversion func)
// for the current stack frame.
func (s *scope) DefaultConvert(src, dest interface{}, flags FieldMatchingFlags) error {
	return s.converter.DefaultConvert(src, dest, flags, s.meta)
}

// SrcTag returns the tag of the struct containing the current source item, if any.
func (s *scope) SrcTag() reflect.StructTag {
	return s.srcStack.top().tag
}

// DestTag returns the tag of the struct containing the current dest item, if any.
func (s *scope) DestTag() reflect.StructTag {
	return s.destStack.top().tag
}

// Flags returns the flags with which the current conversion was started.
func (s *scope) Flags() FieldMatchingFlags {
	return s.flags
}

// Meta returns the meta object that was originally passed to Convert.
func (s *scope) Meta() *Meta {
	return s.meta
}

// describe prints the path to get to the current (source, dest) values.
func (s *scope) describe() (src, dest string) {
	return s.srcStack.describe(), s.destStack.describe()
}

// error makes an error that includes information about where we were in the objects
// we were asked to convert.
func (s *scope) errorf(message string, args ...interface{}) error {
	srcPath, destPath := s.describe()
	where := fmt.Sprintf("converting %v to %v: ", srcPath, destPath)
	return fmt.Errorf(where+message, args...)
}

// Verifies whether a conversion function has a correct signature.
func verifyConversionFunctionSignature(ft reflect.Type) error {
	if ft.Kind() != reflect.Func {
		return fmt.Errorf("expected func, got: %v", ft)
	}
	if ft.NumIn() != 3 {
		return fmt.Errorf("expected three 'in' params, got: %v", ft)
	}
	if ft.NumOut() != 1 {
		return fmt.Errorf("expected one 'out' param, got: %v", ft)
	}
	if ft.In(0).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 0, got: %v", ft)
	}
	if ft.In(1).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 1, got: %v", ft)
	}
	scopeType := Scope(nil)
	if e, a := reflect.TypeOf(&scopeType).Elem(), ft.In(2); e != a {
		return fmt.Errorf("expected '%v' arg for 'in' param 2, got '%v' (%v)", e, a, ft)
	}
	var forErrorType error
	// This convolution is necessary, otherwise TypeOf picks up on the fact
	// that forErrorType is nil.
	errorType := reflect.TypeOf(&forErrorType).Elem()
	if ft.Out(0) != errorType {
		return fmt.Errorf("expected error return, got: %v", ft)
	}
	return nil
}

// RegisterConversionFunc registers a conversion func with the
// Converter. conversionFunc must take three parameters: a pointer to the input
// type, a pointer to the output type, and a conversion.Scope (which should be
// used if recursive conversion calls are desired).  It must return an error.
//
// Example:
// c.RegisterConversionFunc(
//         func(in *Pod, out *v1.Pod, s Scope) error {
//                 // conversion logic...
//                 return nil
//          })
func (c *Converter) RegisterConversionFunc(conversionFunc interface{}) error {
	return c.conversionFuncs.Add(conversionFunc)
}

// Similar to RegisterConversionFunc, but registers conversion function that were
// automatically generated.
func (c *Converter) RegisterGeneratedConversionFunc(conversionFunc interface{}) error {
	return c.generatedConversionFuncs.Add(conversionFunc)
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
	return nil
}

// IsConversionIgnored returns true if the specified objects should be dropped during
// conversion.
func (c *Converter) IsConversionIgnored(inType, outType reflect.Type) bool {
	_, found := c.ignoredConversions[typePair{inType, outType}]
	return found
}

func (c *Converter) HasConversionFunc(inType, outType reflect.Type) bool {
	_, found := c.conversionFuncs.fns[typePair{inType, outType}]
	return found
}

func (c *Converter) ConversionFuncValue(inType, outType reflect.Type) (reflect.Value, bool) {
	value, found := c.conversionFuncs.fns[typePair{inType, outType}]
	return value, found
}

// SetStructFieldCopy registers a correspondence. Whenever a struct field is encountered
// which has a type and name matching srcFieldType and srcFieldName, it wil be copied
// into the field in the destination struct matching destFieldType & Name, if such a
// field exists.
// May be called multiple times, even for the same source field & type--all applicable
// copies will be performed.
func (c *Converter) SetStructFieldCopy(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	st := reflect.TypeOf(srcFieldType)
	dt := reflect.TypeOf(destFieldType)
	srcKey := typeNamePair{st, srcFieldName}
	destKey := typeNamePair{dt, destFieldName}
	c.structFieldDests[srcKey] = append(c.structFieldDests[srcKey], destKey)
	c.structFieldSources[destKey] = append(c.structFieldSources[destKey], srcKey)
	return nil
}

// RegisterDefaultingFunc registers a value-defaulting func with the Converter.
// defaultingFunc must take one parameter: a pointer to the input type.
//
// Example:
// c.RegisterDefaultingFunc(
//         func(in *v1.Pod) {
//                 // defaulting logic...
//          })
func (c *Converter) RegisterDefaultingFunc(defaultingFunc interface{}) error {
	fv := reflect.ValueOf(defaultingFunc)
	ft := fv.Type()
	if ft.Kind() != reflect.Func {
		return fmt.Errorf("expected func, got: %v", ft)
	}
	if ft.NumIn() != 1 {
		return fmt.Errorf("expected one 'in' param, got: %v", ft)
	}
	if ft.NumOut() != 0 {
		return fmt.Errorf("expected zero 'out' params, got: %v", ft)
	}
	if ft.In(0).Kind() != reflect.Ptr {
		return fmt.Errorf("expected pointer arg for 'in' param 0, got: %v", ft)
	}
	inType := ft.In(0).Elem()
	c.defaultingFuncs[inType] = fv
	c.defaultingInterfaces[inType] = defaultingFunc
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
	if len(c.genericConversions) > 0 {
		// TODO: avoid scope allocation
		s := &scope{converter: c, flags: flags, meta: meta}
		for _, fn := range c.genericConversions {
			if ok, err := fn(src, dest, s); ok {
				return err
			}
		}
	}
	return c.doConversion(src, dest, flags, meta, c.convert)
}

// DefaultConvert will translate src to dest if it knows how. Both must be pointers.
// No conversion func is used. If the default copying mechanism
// doesn't work on this type pair, an error will be returned.
// Read the comments on the various FieldMatchingFlags constants to understand
// what the 'flags' parameter does.
// 'meta' is given to allow you to pass information to conversion functions,
// it is not used by DefaultConvert() other than storing it in the scope.
// Not safe for objects with cyclic references!
func (c *Converter) DefaultConvert(src, dest interface{}, flags FieldMatchingFlags, meta *Meta) error {
	return c.doConversion(src, dest, flags, meta, c.defaultConvert)
}

type conversionFunc func(sv, dv reflect.Value, scope *scope) error

func (c *Converter) doConversion(src, dest interface{}, flags FieldMatchingFlags, meta *Meta, f conversionFunc) error {
	dv, err := EnforcePtr(dest)
	if err != nil {
		return err
	}
	if !dv.CanAddr() && !dv.CanSet() {
		return fmt.Errorf("can't write to dest")
	}
	sv, err := EnforcePtr(src)
	if err != nil {
		return err
	}
	s := &scope{
		converter: c,
		flags:     flags,
		meta:      meta,
	}
	// Leave something on the stack, so that calls to struct tag getters never fail.
	s.srcStack.push(scopeStackElem{})
	s.destStack.push(scopeStackElem{})
	return f(sv, dv, s)
}

// callCustom calls 'custom' with sv & dv. custom must be a conversion function.
func (c *Converter) callCustom(sv, dv, custom reflect.Value, scope *scope) error {
	if !sv.CanAddr() {
		sv2 := reflect.New(sv.Type())
		sv2.Elem().Set(sv)
		sv = sv2
	} else {
		sv = sv.Addr()
	}
	if !dv.CanAddr() {
		if !dv.CanSet() {
			return scope.errorf("can't addr or set dest.")
		}
		dvOrig := dv
		dv := reflect.New(dvOrig.Type())
		defer func() { dvOrig.Set(dv) }()
	} else {
		dv = dv.Addr()
	}
	args := []reflect.Value{sv, dv, reflect.ValueOf(scope)}
	ret := custom.Call(args)[0].Interface()
	// This convolution is necessary because nil interfaces won't convert
	// to errors.
	if ret == nil {
		return nil
	}
	return ret.(error)
}

// convert recursively copies sv into dv, calling an appropriate conversion function if
// one is registered.
func (c *Converter) convert(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()
	// Apply default values.
	if fv, ok := c.defaultingFuncs[st]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Applying defaults for '%v'", st)
		}
		args := []reflect.Value{sv.Addr()}
		fv.Call(args)
	}

	pair := typePair{st, dt}

	// ignore conversions of this type
	if _, ok := c.ignoredConversions[pair]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Ignoring conversion of '%v' to '%v'", st, dt)
		}
		return nil
	}

	// Convert sv to dv.
	if fv, ok := c.conversionFuncs.fns[pair]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Calling custom conversion of '%v' to '%v'", st, dt)
		}
		return c.callCustom(sv, dv, fv, scope)
	}
	if fv, ok := c.generatedConversionFuncs.fns[pair]; ok {
		if c.Debug != nil {
			c.Debug.Logf("Calling generated conversion of '%v' to '%v'", st, dt)
		}
		return c.callCustom(sv, dv, fv, scope)
	}

	return c.defaultConvert(sv, dv, scope)
}

// defaultConvert recursively copies sv into dv. no conversion function is called
// for the current stack frame (but conversion functions may be called for nested objects)
func (c *Converter) defaultConvert(sv, dv reflect.Value, scope *scope) error {
	dt, st := dv.Type(), sv.Type()

	if !dv.CanSet() {
		return scope.errorf("Cannot set dest. (Tried to deep copy something with unexported fields?)")
	}

	if !scope.flags.IsSet(AllowDifferentFieldTypeNames) && c.nameFunc(dt) != c.nameFunc(st) {
		return scope.errorf(
			"type names don't match (%v, %v), and no conversion 'func (%v, %v) error' registered.",
			c.nameFunc(st), c.nameFunc(dt), st, dt)
	}

	switch st.Kind() {
	case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
		// Don't copy these via assignment/conversion!
	default:
		// This should handle all simple types.
		if st.AssignableTo(dt) {
			dv.Set(sv)
			return nil
		}
		if st.ConvertibleTo(dt) {
			dv.Set(sv.Convert(dt))
			return nil
		}
	}

	if c.Debug != nil {
		c.Debug.Logf("Trying to convert '%v' to '%v'", st, dt)
	}

	scope.srcStack.push(scopeStackElem{value: sv})
	scope.destStack.push(scopeStackElem{value: dv})
	defer scope.srcStack.pop()
	defer scope.destStack.pop()

	switch dv.Kind() {
	case reflect.Struct:
		return c.convertKV(toKVValue(sv), toKVValue(dv), scope)
	case reflect.Slice:
		if sv.IsNil() {
			// Don't make a zero-length slice.
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
		for i := 0; i < sv.Len(); i++ {
			scope.setIndices(i, i)
			if err := c.convert(sv.Index(i), dv.Index(i), scope); err != nil {
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
		switch st.Kind() {
		case reflect.Ptr, reflect.Interface:
			return c.convert(sv.Elem(), dv.Elem(), scope)
		default:
			return c.convert(sv, dv.Elem(), scope)
		}
	case reflect.Map:
		if sv.IsNil() {
			// Don't copy a nil ptr!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeMap(dt))
		for _, sk := range sv.MapKeys() {
			dk := reflect.New(dt.Key()).Elem()
			if err := c.convert(sk, dk, scope); err != nil {
				return err
			}
			dkv := reflect.New(dt.Elem()).Elem()
			scope.setKeys(sk.Interface(), dk.Interface())
			// TODO:  sv.MapIndex(sk) may return a value with CanAddr() == false,
			// because a map[string]struct{} does not allow a pointer reference.
			// Calling a custom conversion function defined for the map value
			// will panic. Example is PodInfo map[string]ContainerStatus.
			if err := c.convert(sv.MapIndex(sk), dkv, scope); err != nil {
				return err
			}
			dv.SetMapIndex(dk, dkv)
		}
	case reflect.Interface:
		if sv.IsNil() {
			// Don't copy a nil interface!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		tmpdv := reflect.New(sv.Elem().Type()).Elem()
		if err := c.convert(sv.Elem(), tmpdv, scope); err != nil {
			return err
		}
		dv.Set(reflect.ValueOf(tmpdv.Interface()))
		return nil
	default:
		return scope.errorf("couldn't copy '%v' into '%v'; didn't understand types", st, dt)
	}
	return nil
}

var stringType = reflect.TypeOf("")

func toKVValue(v reflect.Value) kvValue {
	switch v.Kind() {
	case reflect.Struct:
		return structAdaptor(v)
	case reflect.Map:
		if v.Type().Key().AssignableTo(stringType) {
			return stringMapAdaptor(v)
		}
	}

	return nil
}

// kvValue lets us write the same conversion logic to work with both maps
// and structs. Only maps with string keys make sense for this.
type kvValue interface {
	// returns all keys, as a []string.
	keys() []string
	// Will just return "" for maps.
	tagOf(key string) reflect.StructTag
	// Will return the zero Value if the key doesn't exist.
	value(key string) reflect.Value
	// Maps require explicit setting-- will do nothing for structs.
	// Returns false on failure.
	confirmSet(key string, v reflect.Value) bool
}

type stringMapAdaptor reflect.Value

func (a stringMapAdaptor) len() int {
	return reflect.Value(a).Len()
}

func (a stringMapAdaptor) keys() []string {
	v := reflect.Value(a)
	keys := make([]string, v.Len())
	for i, v := range v.MapKeys() {
		if v.IsNil() {
			continue
		}
		switch t := v.Interface().(type) {
		case string:
			keys[i] = t
		}
	}
	return keys
}

func (a stringMapAdaptor) tagOf(key string) reflect.StructTag {
	return ""
}

func (a stringMapAdaptor) value(key string) reflect.Value {
	return reflect.Value(a).MapIndex(reflect.ValueOf(key))
}

func (a stringMapAdaptor) confirmSet(key string, v reflect.Value) bool {
	return true
}

type structAdaptor reflect.Value

func (a structAdaptor) len() int {
	v := reflect.Value(a)
	return v.Type().NumField()
}

func (a structAdaptor) keys() []string {
	v := reflect.Value(a)
	t := v.Type()
	keys := make([]string, t.NumField())
	for i := range keys {
		keys[i] = t.Field(i).Name
	}
	return keys
}

func (a structAdaptor) tagOf(key string) reflect.StructTag {
	v := reflect.Value(a)
	field, ok := v.Type().FieldByName(key)
	if ok {
		return field.Tag
	}
	return ""
}

func (a structAdaptor) value(key string) reflect.Value {
	v := reflect.Value(a)
	return v.FieldByName(key)
}

func (a structAdaptor) confirmSet(key string, v reflect.Value) bool {
	return true
}

// convertKV can convert things that consist of key/value pairs, like structs
// and some maps.
func (c *Converter) convertKV(skv, dkv kvValue, scope *scope) error {
	if skv == nil || dkv == nil {
		// TODO: add keys to stack to support really understandable error messages.
		return fmt.Errorf("Unable to convert %#v to %#v", skv, dkv)
	}

	lister := dkv
	if scope.flags.IsSet(SourceToDest) {
		lister = skv
	}

	var mapping FieldMappingFunc
	if scope.meta != nil && scope.meta.KeyNameMapping != nil {
		mapping = scope.meta.KeyNameMapping
	}

	for _, key := range lister.keys() {
		if found, err := c.checkField(key, skv, dkv, scope); found {
			if err != nil {
				return err
			}
			continue
		}
		stag := skv.tagOf(key)
		dtag := dkv.tagOf(key)
		skey := key
		dkey := key
		if mapping != nil {
			skey, dkey = scope.meta.KeyNameMapping(key, stag, dtag)
		}

		df := dkv.value(dkey)
		sf := skv.value(skey)
		if !df.IsValid() || !sf.IsValid() {
			switch {
			case scope.flags.IsSet(IgnoreMissingFields):
				// No error.
			case scope.flags.IsSet(SourceToDest):
				return scope.errorf("%v not present in dest", dkey)
			default:
				return scope.errorf("%v not present in src", skey)
			}
			continue
		}
		scope.srcStack.top().key = skey
		scope.srcStack.top().tag = stag
		scope.destStack.top().key = dkey
		scope.destStack.top().tag = dtag
		if err := c.convert(sf, df, scope); err != nil {
			return err
		}
	}
	return nil
}

// checkField returns true if the field name matches any of the struct
// field copying rules. The error should be ignored if it returns false.
func (c *Converter) checkField(fieldName string, skv, dkv kvValue, scope *scope) (bool, error) {
	replacementMade := false
	if scope.flags.IsSet(DestFromSource) {
		df := dkv.value(fieldName)
		if !df.IsValid() {
			return false, nil
		}
		destKey := typeNamePair{df.Type(), fieldName}
		// Check each of the potential source (type, name) pairs to see if they're
		// present in sv.
		for _, potentialSourceKey := range c.structFieldSources[destKey] {
			sf := skv.value(potentialSourceKey.fieldName)
			if !sf.IsValid() {
				continue
			}
			if sf.Type() == potentialSourceKey.fieldType {
				// Both the source's name and type matched, so copy.
				scope.srcStack.top().key = potentialSourceKey.fieldName
				scope.destStack.top().key = fieldName
				if err := c.convert(sf, df, scope); err != nil {
					return true, err
				}
				dkv.confirmSet(fieldName, df)
				replacementMade = true
			}
		}
		return replacementMade, nil
	}

	sf := skv.value(fieldName)
	if !sf.IsValid() {
		return false, nil
	}
	srcKey := typeNamePair{sf.Type(), fieldName}
	// Check each of the potential dest (type, name) pairs to see if they're
	// present in dv.
	for _, potentialDestKey := range c.structFieldDests[srcKey] {
		df := dkv.value(potentialDestKey.fieldName)
		if !df.IsValid() {
			continue
		}
		if df.Type() == potentialDestKey.fieldType {
			// Both the dest's name and type matched, so copy.
			scope.srcStack.top().key = fieldName
			scope.destStack.top().key = potentialDestKey.fieldName
			if err := c.convert(sf, df, scope); err != nil {
				return true, err
			}
			dkv.confirmSet(potentialDestKey.fieldName, df)
			replacementMade = true
		}
	}
	return replacementMade, nil
}
