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

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Scheme defines an entire encoding and decoding scheme.
type Scheme struct {
	// versionMap allows one to figure out the go type of an object with
	// the given version and name.
	gvkToType map[unversioned.GroupVersionKind]reflect.Type

	// typeToGroupVersion allows one to find metadata for a given go object.
	// The reflect.Type we index by should *not* be a pointer.
	typeToGVK map[reflect.Type][]unversioned.GroupVersionKind

	// unversionedTypes are transformed without conversion in ConvertToVersion.
	unversionedTypes map[reflect.Type]unversioned.GroupVersionKind
	// unversionedKinds are the names of kinds that exist in all versions
	unversionedKinds map[string]reflect.Type

	// converter stores all registered conversion functions. It also has
	// default coverting behavior.
	converter *Converter

	// cloner stores all registered copy functions. It also has default
	// deep copy behavior.
	cloner *Cloner

	// Indent will cause the JSON output from Encode to be indented,
	// if and only if it is true.
	Indent bool
}

// NewScheme manufactures a new scheme.
func NewScheme() *Scheme {
	s := &Scheme{
		gvkToType:        map[unversioned.GroupVersionKind]reflect.Type{},
		typeToGVK:        map[reflect.Type][]unversioned.GroupVersionKind{},
		unversionedTypes: map[reflect.Type]unversioned.GroupVersionKind{},
		unversionedKinds: map[string]reflect.Type{},
		converter:        NewConverter(),
		cloner:           NewCloner(),
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
	gvks, ok := s.typeToGVK[t]
	if !ok {
		return t.Name()
	}

	for _, gvk := range gvks {
		internalGV := gvk.GroupVersion()
		internalGV.Version = "__internal" // this is hacky and maybe should be passed in
		internalGVK := internalGV.WithKind(gvk.Kind)

		if internalType, exists := s.gvkToType[internalGVK]; exists {
			return s.typeToGVK[internalType][0].Kind
		}
	}

	return gvks[0].Kind
}

// AddUnversionedTypes registers all types passed in 'types' as being members of version 'version',
// and marks them as being convertible to all API versions.
// All objects passed to types should be pointers to structs. The name that go reports for
// the struct becomes the "kind" field when encoding.
func (s *Scheme) AddUnversionedTypes(gv unversioned.GroupVersion, types ...interface{}) {
	s.AddKnownTypes(gv, types...)
	for _, obj := range types {
		t := reflect.TypeOf(obj).Elem()
		gvk := gv.WithKind(t.Name())
		s.unversionedTypes[t] = gvk
		if _, ok := s.unversionedKinds[gvk.Kind]; ok {
			panic(fmt.Sprintf("%v has already been registered as unversioned kind %q - kind name must be unique", reflect.TypeOf(t), gvk.Kind))
		}
		s.unversionedKinds[gvk.Kind] = t
	}
}

// AddKnownTypes registers all types passed in 'types' as being members of version 'version'.
// All objects passed to types should be pointers to structs. The name that go reports for
// the struct becomes the "kind" field when encoding.
func (s *Scheme) AddKnownTypes(gv unversioned.GroupVersion, types ...interface{}) {
	if len(gv.Version) == 0 {
		panic(fmt.Sprintf("version is required on all types: %s %v", gv, types[0]))
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

		gvk := gv.WithKind(t.Name())
		s.gvkToType[gvk] = t
		s.typeToGVK[t] = append(s.typeToGVK[t], gvk)
	}
}

// AddKnownTypeWithName is like AddKnownTypes, but it lets you specify what this type should
// be encoded as. Useful for testing when you don't want to make multiple packages to define
// your structs.
func (s *Scheme) AddKnownTypeWithName(gvk unversioned.GroupVersionKind, obj interface{}) {
	t := reflect.TypeOf(obj)
	if len(gvk.Version) == 0 {
		panic(fmt.Sprintf("version is required on all types: %s %v", gvk, t))
	}
	if t.Kind() != reflect.Ptr {
		panic("All types must be pointers to structs.")
	}
	t = t.Elem()
	if t.Kind() != reflect.Struct {
		panic("All types must be pointers to structs.")
	}

	s.gvkToType[gvk] = t
	s.typeToGVK[t] = append(s.typeToGVK[t], gvk)

}

// KnownTypes returns an array of the types that are known for a particular version.
func (s *Scheme) KnownTypes(gv unversioned.GroupVersion) map[string]reflect.Type {
	types := map[string]reflect.Type{}

	for gvk, t := range s.gvkToType {
		if gv != gvk.GroupVersion() {
			continue
		}

		types[gvk.Kind] = t
	}

	return types
}

// NewObject returns a new object of the given version and name,
// or an error if it hasn't been registered.
func (s *Scheme) NewObject(kind unversioned.GroupVersionKind) (interface{}, error) {
	if t, exists := s.gvkToType[kind]; exists {
		return reflect.New(t).Interface(), nil
	}

	if t, exists := s.unversionedKinds[kind.Kind]; exists {
		return reflect.New(t).Interface(), nil
	}
	return nil, &notRegisteredErr{gvk: kind}
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
//		// e.g., "v1".
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

func (s *Scheme) AddIgnoredConversionType(from, to interface{}) error {
	return s.converter.RegisterIgnoredConversion(from, to)
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
//	func(obj *v1.Pod) {
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

// Recognizes returns true if the scheme is able to handle the provided group,version,kind
// of an object.
func (s *Scheme) Recognizes(gvk unversioned.GroupVersionKind) bool {
	_, exists := s.gvkToType[gvk]
	return exists
}

func (s *Scheme) IsUnversioned(obj interface{}) (bool, bool) {
	v, err := EnforcePtr(obj)
	if err != nil {
		return false, false
	}
	t := v.Type()

	if _, ok := s.typeToGVK[t]; !ok {
		return false, false
	}
	_, ok := s.unversionedTypes[t]
	return ok, true
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
	inVersion := unversioned.GroupVersion{Group: "unknown", Version: "unknown"}
	outVersion := unversioned.GroupVersion{Group: "unknown", Version: "unknown"}
	if gvk, err := s.ObjectKind(in); err == nil {
		inVersion = gvk.GroupVersion()
	}
	if gvk, err := s.ObjectKind(out); err == nil {
		outVersion = gvk.GroupVersion()
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
func (s *Scheme) ConvertToVersion(in interface{}, outGroupVersionString string) (interface{}, error) {
	t := reflect.TypeOf(in)
	if t.Kind() != reflect.Ptr {
		return nil, fmt.Errorf("only pointer types may be converted: %v", t)
	}
	t = t.Elem()
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("only pointers to struct types may be converted: %v", t)
	}

	outVersion, err := unversioned.ParseGroupVersion(outGroupVersionString)
	if err != nil {
		return nil, err
	}

	var kind unversioned.GroupVersionKind
	if unversionedKind, ok := s.unversionedTypes[t]; ok {
		kind = unversionedKind
	} else {
		kinds, ok := s.typeToGVK[t]
		if !ok || len(kinds) == 0 {
			return nil, fmt.Errorf("%v is not a registered type and cannot be converted into version %q", t, outGroupVersionString)
		}
		kind = kinds[0]
	}

	outKind := outVersion.WithKind(kind.Kind)

	inKind, err := s.ObjectKind(in)
	if err != nil {
		return nil, err
	}

	out, err := s.NewObject(outKind)
	if err != nil {
		return nil, err
	}

	flags, meta := s.generateConvertMeta(inKind.GroupVersion(), outVersion, in)
	if err := s.converter.Convert(in, out, flags, meta); err != nil {
		return nil, err
	}

	return out, nil
}

// Converter allows access to the converter for the scheme
func (s *Scheme) Converter() *Converter {
	return s.converter
}

// WithConversions returns a scheme with additional conversion functions
func (s *Scheme) WithConversions(fns ConversionFuncs) *Scheme {
	c := s.converter.WithConversions(fns)
	copied := *s
	copied.converter = c
	return &copied
}

// generateConvertMeta constructs the meta value we pass to Convert.
func (s *Scheme) generateConvertMeta(srcGroupVersion, destGroupVersion unversioned.GroupVersion, in interface{}) (FieldMatchingFlags, *Meta) {
	t := reflect.TypeOf(in)
	return s.converter.inputDefaultFlags[t], &Meta{
		SrcVersion:     srcGroupVersion.String(),
		DestVersion:    destGroupVersion.String(),
		KeyNameMapping: s.converter.inputFieldMappingFuncs[t],
	}
}

// ObjectKind returns the group,version,kind of the go object,
// or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectKind(obj interface{}) (unversioned.GroupVersionKind, error) {
	gvks, err := s.ObjectKinds(obj)
	if err != nil {
		return unversioned.GroupVersionKind{}, err
	}
	return gvks[0], nil
}

// ObjectKinds returns all possible group,version,kind of the go object,
// or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectKinds(obj interface{}) ([]unversioned.GroupVersionKind, error) {
	v, err := EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	t := v.Type()

	gvks, ok := s.typeToGVK[t]
	if !ok {
		return nil, &notRegisteredErr{t: t}
	}
	return gvks, nil
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
