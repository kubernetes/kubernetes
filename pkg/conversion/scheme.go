/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

type groupVersion struct {
	Group   string
	Version string
}

// Scheme defines an entire encoding and decoding scheme.
type Scheme struct {
	// versionMap allows one to figure out the go type of an object with
	// the given groupVersion and name.
	groupVersionMap map[groupVersion]map[string]reflect.Type

	// typeToGroup allows one to figure out the group for a given go object.
	// The reflect.Type we index by should *not* be a pointer. If the same type
	// is registered for multiple groups, the last one wins.
	typeToGroup map[reflect.Type]string

	// typeToVersion allows one to figure out the desired "apiVersion" field for a given
	// go object. Requirements and caveats are the same as typeToGroup.
	typeToVersion map[reflect.Type]string

	// typeToKind allows one to figure out the desired "kind" field for a given
	// go object. Requirements and caveats are the same as typeToGroup.
	typeToKind map[reflect.Type][]string

	// converter stores all registered conversion functions. It also has
	// default coverting behavior.
	converter *Converter

	// cloner stores all registered copy functions. It also has default
	// deep copy behavior.
	cloner *Cloner

	// defaulGroup is the assumed api group when no such field is present.
	defaultGroup string

	// Indent will cause the JSON output from Encode to be indented, iff it is true.
	Indent bool

	// InternalVersion is the default internal version. It is recommended that
	// you use "" for the internal version.
	InternalVersion string

	// MetaInsertionFactory is used to create an object to store and retrieve
	// the version and kind information for all objects. The default uses the
	// keys "apiVersion" and "kind" respectively.
	MetaFactory MetaFactory
}

// NewScheme manufactures a new scheme.
func NewScheme(defaultGroup string) *Scheme {
	s := &Scheme{
		groupVersionMap: map[groupVersion]map[string]reflect.Type{},
		typeToGroup:     map[reflect.Type]string{},
		typeToVersion:   map[reflect.Type]string{},
		typeToKind:      map[reflect.Type][]string{},
		converter:       NewConverter(),
		cloner:          NewCloner(),
		defaultGroup:    defaultGroup,
		InternalVersion: "",
		MetaFactory:     DefaultMetaFactory,
	}
	s.converter.nameFunc = s.nameFunc
	return s
}

// Log sets a logger on the scheme. For test purposes only
func (s *Scheme) Log(l DebugLogger) {
	s.converter.Debug = l
}

// nameFunc returns the name of the type that we wish to use to determine when two types attempt
// a conversion. Defaults to the go name of the type if the type is not registered.
func (s *Scheme) nameFunc(t reflect.Type) string {
	// find the preferred names for this type
	names, ok := s.typeToKind[t]
	if !ok {
		return t.Name()
	}
	group := s.typeToGroup[t] // must be in a group since we know a kind for t
	if internal, ok := s.groupVersionMap[groupVersion{group, ""}]; ok {
		for _, name := range names {
			if t, ok := internal[name]; ok {
				return s.typeToKind[t][0]
			}
		}
	}
	return names[0]
}

// AddKnownTypes registers all types passed in 'types' as being members of version 'version.
// Encode() will refuse objects unless their type has been registered with AddKnownTypes.
// All objects passed to types should be pointers to structs. The name that go reports for
// the struct becomes the "kind" field when encoding.
func (s *Scheme) AddKnownTypes(group string, version string, types ...interface{}) {
	gv := groupVersion{group, version}
	knownTypes, found := s.groupVersionMap[gv]
	if !found {
		knownTypes = map[string]reflect.Type{}
		s.groupVersionMap[gv] = knownTypes
	}
	for _, obj := range types {
		t := reflect.TypeOf(obj)
		if t.Kind() != reflect.Ptr {
			panic("All types must be pointers to structs.")
		}
		t = t.Elem()
		if t.Kind() != reflect.Struct {
			panic("All types must be pointers to structs.")
		}
		knownTypes[t.Name()] = t
		s.typeToGroup[t] = group
		s.typeToVersion[t] = version
		s.typeToKind[t] = append(s.typeToKind[t], t.Name())
	}
}

// AddKnownTypeWithName is like AddKnownTypes, but it lets you specify what this type should
// be encoded as. Useful for testing when you don't want to make multiple packages to define
// your structs.
func (s *Scheme) AddKnownTypeWithName(group, version, kind string, obj interface{}) {
	gv := groupVersion{group, version}
	knownTypes, found := s.groupVersionMap[gv]
	if !found {
		knownTypes = map[string]reflect.Type{}
		s.groupVersionMap[gv] = knownTypes
	}
	t := reflect.TypeOf(obj)
	if t.Kind() != reflect.Ptr {
		panic("All types must be pointers to structs.")
	}
	t = t.Elem()
	if t.Kind() != reflect.Struct {
		panic("All types must be pointers to structs.")
	}
	knownTypes[kind] = t
	s.typeToGroup[t] = group
	s.typeToVersion[t] = version
	s.typeToKind[t] = append(s.typeToKind[t], kind)
}

// KnownTypes returns an array of the types that are known for a particular version.
func (s *Scheme) KnownTypes(group, version string) map[string]reflect.Type {
	all, ok := s.groupVersionMap[groupVersion{group, version}]
	if !ok {
		return map[string]reflect.Type{}
	}
	types := make(map[string]reflect.Type)
	for k, v := range all {
		types[k] = v
	}
	return types
}

// NewObject returns a new object of the given version and name,
// or an error if it hasn't been registered.
func (s *Scheme) NewObject(group, version, kind string) (interface{}, error) {
	if types, ok := s.groupVersionMap[groupVersion{group, version}]; ok {
		if t, ok := types[kind]; ok {
			return reflect.New(t).Interface(), nil
		}
		return nil, &notRegisteredErr{group: group, version: version, kind: kind}
	}
	return nil, &notRegisteredErr{group: group, version: version, kind: kind}
}

// AddConversionFuncs adds functions to the list of conversion functions. The given
// functions should know how to convert between two of your API objects, or their
// sub-objects. We deduce how to call these functions from the types of their two
// parameters; see the comment for Converter.Register.
//
// Note that, if you need to copy sub-objects that didn't change, you can use the
// conversion.Scope object that will be passed to your conversion function.
// Additionally, all conversions started by Scheme will set the SrcVersion and
// DestVersion fields on the Meta object. Example:
//
// s.AddConversionFuncs(
//	func(in *InternalObject, out *ExternalObject, scope conversion.Scope) error {
//		// You can depend on Meta() being non-nil, and this being set to
//		// the source version, e.g., ""
//		s.Meta().SrcVersion
//		// You can depend on this being set to the destination version,
//		// e.g., "v1beta1".
//		s.Meta().DestVersion
//		// Call scope.Convert to copy sub-fields.
//		s.Convert(&in.SubFieldThatMoved, &out.NewLocation.NewName, 0)
//		return nil
//	},
// )
//
// (For more detail about conversion functions, see Converter.Register's comment.)
//
// Also note that the default behavior, if you don't add a conversion function, is to
// sanely copy fields that have the same names and same type names. It's OK if the
// destination type has extra fields, but it must not remove any. So you only need to
// add conversion functions for things with changed/removed fields.
func (s *Scheme) AddConversionFuncs(conversionFuncs ...interface{}) error {
	for _, f := range conversionFuncs {
		if err := s.converter.RegisterConversionFunc(f); err != nil {
			return err
		}
	}
	return nil
}

// Similar to AddConversionFuncs, but registers conversion functions that were
// automatically generated.
func (s *Scheme) AddGeneratedConversionFuncs(conversionFuncs ...interface{}) error {
	for _, f := range conversionFuncs {
		if err := s.converter.RegisterGeneratedConversionFunc(f); err != nil {
			return err
		}
	}
	return nil
}

// AddDeepCopyFuncs adds functions to the list of deep copy functions.
// Note that to copy sub-objects, you can use the conversion.Cloner object that
// will be passed to your deep-copy function.
func (s *Scheme) AddDeepCopyFuncs(deepCopyFuncs ...interface{}) error {
	for _, f := range deepCopyFuncs {
		if err := s.cloner.RegisterDeepCopyFunc(f); err != nil {
			return err
		}
	}
	return nil
}

// Similar to AddDeepCopyFuncs, but registers deep copy functions that were
// automatically generated.
func (s *Scheme) AddGeneratedDeepCopyFuncs(deepCopyFuncs ...interface{}) error {
	for _, f := range deepCopyFuncs {
		if err := s.cloner.RegisterGeneratedDeepCopyFunc(f); err != nil {
			return err
		}
	}
	return nil
}

// AddStructFieldConversion allows you to specify a mechanical copy for a moved
// or renamed struct field without writing an entire conversion function. See
// the comment in Converter.SetStructFieldCopy for parameter details.
// Call as many times as needed, even on the same fields.
func (s *Scheme) AddStructFieldConversion(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	return s.converter.SetStructFieldCopy(srcFieldType, srcFieldName, destFieldType, destFieldName)
}

// AddDefaultingFuncs adds functions to the list of default-value functions.
// Each of the given functions is responsible for applying default values
// when converting an instance of a versioned API object into an internal
// API object.  These functions do not need to handle sub-objects. We deduce
// how to call these functions from the types of their two parameters.
//
// s.AddDefaultingFuncs(
//	func(obj *v1beta1.Pod) {
//		if obj.OptionalField == "" {
//			obj.OptionalField = "DefaultValue"
//		}
//	},
// )
func (s *Scheme) AddDefaultingFuncs(defaultingFuncs ...interface{}) error {
	for _, f := range defaultingFuncs {
		err := s.converter.RegisterDefaultingFunc(f)
		if err != nil {
			return err
		}
	}
	return nil
}

// Recognizes returns true if the scheme is able to handle the provided version and kind
// of an object.
func (s *Scheme) Recognizes(group, version, kind string) bool {
	if group == "" {
		group = s.defaultGroup
	}
	m, ok := s.groupVersionMap[groupVersion{group, version}]
	if !ok {
		return false
	}
	_, ok = m[kind]
	return ok
}

// RegisterInputDefaults sets the provided field mapping function and field matching
// as the defaults for the provided input type.  The fn may be nil, in which case no
// mapping will happen by default. Use this method to register a mechanism for handling
// a specific input type in conversion, such as a map[string]string to structs.
func (s *Scheme) RegisterInputDefaults(in interface{}, fn FieldMappingFunc, defaultFlags FieldMatchingFlags) error {
	return s.converter.RegisterInputDefaults(in, fn, defaultFlags)
}

// Performs a deep copy of the given object.
func (s *Scheme) DeepCopy(in interface{}) (interface{}, error) {
	return s.cloner.DeepCopy(in)
}

// Convert will attempt to convert in into out. Both must be pointers. For easy
// testing of conversion functions. Returns an error if the conversion isn't
// possible. You can call this with types that haven't been registered (for example,
// a to test conversion of types that are nested within registered types), but in
// that case, the conversion.Scope object passed to your conversion functions won't
// have SrcVersion or DestVersion fields set correctly in Meta().
func (s *Scheme) Convert(in, out interface{}) error {
	inVersion := "unknown"
	outVersion := "unknown"
	if _, version, _, err := s.ObjectTypeMeta(in); err == nil {
		inVersion = version
	}
	if _, version, _, err := s.ObjectTypeMeta(out); err == nil {
		outVersion = version
	}
	flags, meta := s.generateConvertMeta(inVersion, outVersion, in)
	if flags == 0 {
		flags = AllowDifferentFieldTypeNames
	}
	return s.converter.Convert(in, out, flags, meta)
}

// ConvertToVersion attempts to convert an input object to its matching Kind in another
// version within this scheme. Will return an error if the provided version does not
// contain the inKind (or a mapping by name defined with AddKnownTypeWithName).
func (s *Scheme) ConvertToVersion(in interface{}, outVersion string) (interface{}, error) {
	t := reflect.TypeOf(in)
	if t.Kind() != reflect.Ptr {
		return nil, fmt.Errorf("only pointer types may be converted: %v", t)
	}
	t = t.Elem()
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("only pointers to struct types may be converted: %v", t)
	}

	kinds, ok := s.typeToKind[t]
	if !ok {
		return nil, fmt.Errorf("%v cannot be converted into version %q", t, outVersion)
	}
	outKind := kinds[0]

	inGroup, inVersion, _, err := s.ObjectTypeMeta(in)
	if err != nil {
		return nil, err
	}

	out, err := s.NewObject(inGroup, outVersion, outKind)
	if err != nil {
		return nil, err
	}

	flags, meta := s.generateConvertMeta(inVersion, outVersion, in)
	if err := s.converter.Convert(in, out, flags, meta); err != nil {
		return nil, err
	}

	if err := s.SetTypeMeta(inGroup, outVersion, outKind, out); err != nil {
		return nil, err
	}

	return out, nil
}

// Converter allows access to the converter for the scheme
func (s *Scheme) Converter() *Converter {
	return s.converter
}

// generateConvertMeta constructs the meta value we pass to Convert.
func (s *Scheme) generateConvertMeta(srcVersion, destVersion string, in interface{}) (FieldMatchingFlags, *Meta) {
	t := reflect.TypeOf(in)
	return s.converter.inputDefaultFlags[t], &Meta{
		SrcVersion:     srcVersion,
		DestVersion:    destVersion,
		KeyNameMapping: s.converter.inputFieldMappingFuncs[t],
	}
}

// DataTypeMeta will return the APIVersion and Kind of the given wire-format
// encoding of an API Object, or an error.
func (s *Scheme) DataTypeMeta(data []byte) (group, version, kind string, err error) {
	group, version, kind, err = s.MetaFactory.Interpret(data)
	if err != nil {
		return
	}
	if group == "" {
		group = s.defaultGroup
	}
	return
}

// ObjectTypeMeta returns the API version and kind of the go object,
// or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectTypeMeta(obj interface{}) (group, version, kind string, err error) {
	v, err := EnforcePtr(obj)
	if err != nil {
		return "", "", "", err
	}
	t := v.Type()
	group, gOK := s.typeToGroup[t]
	version, vOK := s.typeToVersion[t]
	kinds, kOK := s.typeToKind[t]
	if !gOK || !vOK || !kOK {
		return "", "", "", &notRegisteredErr{t: t}
	}
	return group, version, kinds[0], nil
}

// SetVersionAndKind sets the version and kind fields (with help from
// MetaInsertionFactory). Returns an error if this isn't possible. obj
// must be a pointer.
func (s *Scheme) SetTypeMeta(group, version, kind string, obj interface{}) error {
	return s.MetaFactory.Update(group, version, kind, obj)
}

// maybeCopy copies obj if it is not a pointer, to get a settable/addressable
// object. Guaranteed to return a pointer.
func maybeCopy(obj interface{}) interface{} {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		return obj
	}
	v2 := reflect.New(v.Type())
	v2.Elem().Set(v)
	return v2.Interface()
}
