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

package runtime

import (
	"fmt"
	"net/url"
	"reflect"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Scheme defines methods for serializing and deserializing API objects, a type
// registry for converting group, version, and kind information to and from Go
// schemas, and mappings between Go schemas of different versions. A scheme is the
// foundation for a versioned API and versioned configuration over time.
//
// In a Scheme, a Type is a particular Go struct, a Version is a point-in-time
// identifier for a particular representation of that Type (typically backwards
// compatible), a Kind is the unique name for that Type within the Version, and a
// Group identifies a set of Versions, Kinds, and Types that evolve over time. An
// Unversioned Type is one that is not yet formally bound to a type and is promised
// to be backwards compatible (effectively a "v1" of a Type that does not expect
// to break in the future).
//
// Schemes are not expected to change at runtime and are only threadsafe after
// registration is complete.
type Scheme struct {
	// versionMap allows one to figure out the go type of an object with
	// the given version and name.
	gvkToType map[schema.GroupVersionKind]reflect.Type

	// typeToGroupVersion allows one to find metadata for a given go object.
	// The reflect.Type we index by should *not* be a pointer.
	typeToGVK map[reflect.Type][]schema.GroupVersionKind

	// unversionedTypes are transformed without conversion in ConvertToVersion.
	unversionedTypes map[reflect.Type]schema.GroupVersionKind

	// unversionedKinds are the names of kinds that can be created in the context of any group
	// or version
	// TODO: resolve the status of unversioned types.
	unversionedKinds map[string]reflect.Type

	// Map from version and resource to the corresponding func to convert
	// resource field labels in that version to internal version.
	fieldLabelConversionFuncs map[string]map[string]FieldLabelConversionFunc

	// defaulterFuncs is an array of interfaces to be called with an object to provide defaulting
	// the provided object must be a pointer.
	defaulterFuncs map[reflect.Type]func(interface{})

	// converter stores all registered conversion functions. It also has
	// default coverting behavior.
	converter *conversion.Converter

	// cloner stores all registered copy functions. It also has default
	// deep copy behavior.
	cloner *conversion.Cloner
}

// Function to convert a field selector to internal representation.
type FieldLabelConversionFunc func(label, value string) (internalLabel, internalValue string, err error)

// NewScheme creates a new Scheme. This scheme is pluggable by default.
func NewScheme() *Scheme {
	s := &Scheme{
		gvkToType:        map[schema.GroupVersionKind]reflect.Type{},
		typeToGVK:        map[reflect.Type][]schema.GroupVersionKind{},
		unversionedTypes: map[reflect.Type]schema.GroupVersionKind{},
		unversionedKinds: map[string]reflect.Type{},
		cloner:           conversion.NewCloner(),
		fieldLabelConversionFuncs: map[string]map[string]FieldLabelConversionFunc{},
		defaulterFuncs:            map[reflect.Type]func(interface{}){},
	}
	s.converter = conversion.NewConverter(s.nameFunc)

	s.AddConversionFuncs(DefaultEmbeddedConversions()...)

	// Enable map[string][]string conversions by default
	if err := s.AddConversionFuncs(DefaultStringConversions...); err != nil {
		panic(err)
	}
	if err := s.RegisterInputDefaults(&map[string][]string{}, JSONKeyMapper, conversion.AllowDifferentFieldTypeNames|conversion.IgnoreMissingFields); err != nil {
		panic(err)
	}
	if err := s.RegisterInputDefaults(&url.Values{}, JSONKeyMapper, conversion.AllowDifferentFieldTypeNames|conversion.IgnoreMissingFields); err != nil {
		panic(err)
	}
	return s
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

// fromScope gets the input version, desired output version, and desired Scheme
// from a conversion.Scope.
func (s *Scheme) fromScope(scope conversion.Scope) *Scheme {
	return s
}

// Converter allows access to the converter for the scheme
func (s *Scheme) Converter() *conversion.Converter {
	return s.converter
}

// AddUnversionedTypes registers the provided types as "unversioned", which means that they follow special rules.
// Whenever an object of this type is serialized, it is serialized with the provided group version and is not
// converted. Thus unversioned objects are expected to remain backwards compatible forever, as if they were in an
// API group and version that would never be updated.
//
// TODO: there is discussion about removing unversioned and replacing it with objects that are manifest into
//   every version with particular schemas. Resolve this method at that point.
func (s *Scheme) AddUnversionedTypes(version schema.GroupVersion, types ...Object) {
	s.AddKnownTypes(version, types...)
	for _, obj := range types {
		t := reflect.TypeOf(obj).Elem()
		gvk := version.WithKind(t.Name())
		s.unversionedTypes[t] = gvk
		if _, ok := s.unversionedKinds[gvk.Kind]; ok {
			panic(fmt.Sprintf("%v has already been registered as unversioned kind %q - kind name must be unique", reflect.TypeOf(t), gvk.Kind))
		}
		s.unversionedKinds[gvk.Kind] = t
	}
}

// AddKnownTypes registers all types passed in 'types' as being members of version 'version'.
// All objects passed to types should be pointers to structs. The name that go reports for
// the struct becomes the "kind" field when encoding. Version may not be empty - use the
// APIVersionInternal constant if you have a type that does not have a formal version.
func (s *Scheme) AddKnownTypes(gv schema.GroupVersion, types ...Object) {
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
// your structs. Version may not be empty - use the APIVersionInternal constant if you have a
// type that does not have a formal version.
func (s *Scheme) AddKnownTypeWithName(gvk schema.GroupVersionKind, obj Object) {
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

// KnownTypes returns the types known for the given version.
func (s *Scheme) KnownTypes(gv schema.GroupVersion) map[string]reflect.Type {
	types := make(map[string]reflect.Type)
	for gvk, t := range s.gvkToType {
		if gv != gvk.GroupVersion() {
			continue
		}

		types[gvk.Kind] = t
	}
	return types
}

// AllKnownTypes returns the all known types.
func (s *Scheme) AllKnownTypes() map[schema.GroupVersionKind]reflect.Type {
	return s.gvkToType
}

// ObjectKind returns the group,version,kind of the go object and true if this object
// is considered unversioned, or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectKind(obj Object) (schema.GroupVersionKind, bool, error) {
	gvks, unversionedType, err := s.ObjectKinds(obj)
	if err != nil {
		return schema.GroupVersionKind{}, false, err
	}
	return gvks[0], unversionedType, nil
}

// ObjectKinds returns all possible group,version,kind of the go object, true if the
// object is considered unversioned, or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectKinds(obj Object) ([]schema.GroupVersionKind, bool, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, false, err
	}
	t := v.Type()

	gvks, ok := s.typeToGVK[t]
	if !ok {
		return nil, false, NewNotRegisteredErr(schema.GroupVersionKind{}, t)
	}
	_, unversionedType := s.unversionedTypes[t]

	return gvks, unversionedType, nil
}

// Recognizes returns true if the scheme is able to handle the provided group,version,kind
// of an object.
func (s *Scheme) Recognizes(gvk schema.GroupVersionKind) bool {
	_, exists := s.gvkToType[gvk]
	return exists
}

func (s *Scheme) IsUnversioned(obj Object) (bool, bool) {
	v, err := conversion.EnforcePtr(obj)
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

// New returns a new API object of the given version and name, or an error if it hasn't
// been registered. The version and kind fields must be specified.
func (s *Scheme) New(kind schema.GroupVersionKind) (Object, error) {
	if t, exists := s.gvkToType[kind]; exists {
		return reflect.New(t).Interface().(Object), nil
	}

	if t, exists := s.unversionedKinds[kind.Kind]; exists {
		return reflect.New(t).Interface().(Object), nil
	}
	return nil, NewNotRegisteredErr(kind, nil)
}

// AddGenericConversionFunc adds a function that accepts the ConversionFunc call pattern
// (for two conversion types) to the converter. These functions are checked first during
// a normal conversion, but are otherwise not called. Use AddConversionFuncs when registering
// typed conversions.
func (s *Scheme) AddGenericConversionFunc(fn conversion.GenericConversionFunc) {
	s.converter.AddGenericConversionFunc(fn)
}

// Log sets a logger on the scheme. For test purposes only
func (s *Scheme) Log(l conversion.DebugLogger) {
	s.converter.Debug = l
}

// AddIgnoredConversionType identifies a pair of types that should be skipped by
// conversion (because the data inside them is explicitly dropped during
// conversion).
func (s *Scheme) AddIgnoredConversionType(from, to interface{}) error {
	return s.converter.RegisterIgnoredConversion(from, to)
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

// AddDeepCopyFuncs adds a function to the list of deep-copy functions.
// For the expected format of deep-copy function, see the comment for
// Copier.RegisterDeepCopyFunction.
func (s *Scheme) AddDeepCopyFuncs(deepCopyFuncs ...interface{}) error {
	for _, f := range deepCopyFuncs {
		if err := s.cloner.RegisterDeepCopyFunc(f); err != nil {
			return err
		}
	}
	return nil
}

// Similar to AddDeepCopyFuncs, but registers deep-copy functions that were
// automatically generated.
func (s *Scheme) AddGeneratedDeepCopyFuncs(deepCopyFuncs ...conversion.GeneratedDeepCopyFunc) error {
	for _, fn := range deepCopyFuncs {
		if err := s.cloner.RegisterGeneratedDeepCopyFunc(fn); err != nil {
			return err
		}
	}
	return nil
}

// AddFieldLabelConversionFunc adds a conversion function to convert field selectors
// of the given kind from the given version to internal version representation.
func (s *Scheme) AddFieldLabelConversionFunc(version, kind string, conversionFunc FieldLabelConversionFunc) error {
	if s.fieldLabelConversionFuncs[version] == nil {
		s.fieldLabelConversionFuncs[version] = map[string]FieldLabelConversionFunc{}
	}

	s.fieldLabelConversionFuncs[version][kind] = conversionFunc
	return nil
}

// AddStructFieldConversion allows you to specify a mechanical copy for a moved
// or renamed struct field without writing an entire conversion function. See
// the comment in conversion.Converter.SetStructFieldCopy for parameter details.
// Call as many times as needed, even on the same fields.
func (s *Scheme) AddStructFieldConversion(srcFieldType interface{}, srcFieldName string, destFieldType interface{}, destFieldName string) error {
	return s.converter.SetStructFieldCopy(srcFieldType, srcFieldName, destFieldType, destFieldName)
}

// RegisterInputDefaults sets the provided field mapping function and field matching
// as the defaults for the provided input type.  The fn may be nil, in which case no
// mapping will happen by default. Use this method to register a mechanism for handling
// a specific input type in conversion, such as a map[string]string to structs.
func (s *Scheme) RegisterInputDefaults(in interface{}, fn conversion.FieldMappingFunc, defaultFlags conversion.FieldMatchingFlags) error {
	return s.converter.RegisterInputDefaults(in, fn, defaultFlags)
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

// AddTypeDefaultingFuncs registers a function that is passed a pointer to an
// object and can default fields on the object. These functions will be invoked
// when Default() is called. The function will never be called unless the
// defaulted object matches srcType. If this function is invoked twice with the
// same srcType, the fn passed to the later call will be used instead.
func (s *Scheme) AddTypeDefaultingFunc(srcType Object, fn func(interface{})) {
	s.defaulterFuncs[reflect.TypeOf(srcType)] = fn
}

// Default sets defaults on the provided Object.
func (s *Scheme) Default(src Object) {
	if fn, ok := s.defaulterFuncs[reflect.TypeOf(src)]; ok {
		fn(src)
	}
}

// Copy does a deep copy of an API object.
func (s *Scheme) Copy(src Object) (Object, error) {
	dst, err := s.DeepCopy(src)
	if err != nil {
		return nil, err
	}
	return dst.(Object), nil
}

// Performs a deep copy of the given object.
func (s *Scheme) DeepCopy(src interface{}) (interface{}, error) {
	return s.cloner.DeepCopy(src)
}

// Convert will attempt to convert in into out. Both must be pointers. For easy
// testing of conversion functions. Returns an error if the conversion isn't
// possible. You can call this with types that haven't been registered (for example,
// a to test conversion of types that are nested within registered types). The
// context interface is passed to the convertor.
// TODO: identify whether context should be hidden, or behind a formal context/scope
//   interface
func (s *Scheme) Convert(in, out interface{}, context interface{}) error {
	flags, meta := s.generateConvertMeta(in)
	meta.Context = context
	if flags == 0 {
		flags = conversion.AllowDifferentFieldTypeNames
	}
	return s.converter.Convert(in, out, flags, meta)
}

// Converts the given field label and value for an kind field selector from
// versioned representation to an unversioned one.
func (s *Scheme) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	if s.fieldLabelConversionFuncs[version] == nil {
		return "", "", fmt.Errorf("No field label conversion function found for version: %s", version)
	}
	conversionFunc, ok := s.fieldLabelConversionFuncs[version][kind]
	if !ok {
		return "", "", fmt.Errorf("No field label conversion function found for version %s and kind %s", version, kind)
	}
	return conversionFunc(label, value)
}

// ConvertToVersion attempts to convert an input object to its matching Kind in another
// version within this scheme. Will return an error if the provided version does not
// contain the inKind (or a mapping by name defined with AddKnownTypeWithName). Will also
// return an error if the conversion does not result in a valid Object being
// returned. Passes target down to the conversion methods as the Context on the scope.
func (s *Scheme) ConvertToVersion(in Object, target GroupVersioner) (Object, error) {
	return s.convertToVersion(true, in, target)
}

// UnsafeConvertToVersion will convert in to the provided target if such a conversion is possible,
// but does not guarantee the output object does not share fields with the input object. It attempts to be as
// efficient as possible when doing conversion.
func (s *Scheme) UnsafeConvertToVersion(in Object, target GroupVersioner) (Object, error) {
	return s.convertToVersion(false, in, target)
}

// convertToVersion handles conversion with an optional copy.
func (s *Scheme) convertToVersion(copy bool, in Object, target GroupVersioner) (Object, error) {
	// determine the incoming kinds with as few allocations as possible.
	t := reflect.TypeOf(in)
	if t.Kind() != reflect.Ptr {
		return nil, fmt.Errorf("only pointer types may be converted: %v", t)
	}
	t = t.Elem()
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("only pointers to struct types may be converted: %v", t)
	}
	kinds, ok := s.typeToGVK[t]
	if !ok || len(kinds) == 0 {
		return nil, NewNotRegisteredErr(schema.GroupVersionKind{}, t)
	}

	gvk, ok := target.KindForGroupVersionKinds(kinds)
	if !ok {
		// try to see if this type is listed as unversioned (for legacy support)
		// TODO: when we move to server API versions, we should completely remove the unversioned concept
		if unversionedKind, ok := s.unversionedTypes[t]; ok {
			if gvk, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{unversionedKind}); ok {
				return copyAndSetTargetKind(copy, s, in, gvk)
			}
			return copyAndSetTargetKind(copy, s, in, unversionedKind)
		}

		// TODO: should this be a typed error?
		return nil, fmt.Errorf("%v is not suitable for converting to %q", t, target)
	}

	// target wants to use the existing type, set kind and return (no conversion necessary)
	for _, kind := range kinds {
		if gvk == kind {
			return copyAndSetTargetKind(copy, s, in, gvk)
		}
	}

	// type is unversioned, no conversion necessary
	if unversionedKind, ok := s.unversionedTypes[t]; ok {
		if gvk, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{unversionedKind}); ok {
			return copyAndSetTargetKind(copy, s, in, gvk)
		}
		return copyAndSetTargetKind(copy, s, in, unversionedKind)
	}

	out, err := s.New(gvk)
	if err != nil {
		return nil, err
	}

	if copy {
		copied, err := s.Copy(in)
		if err != nil {
			return nil, err
		}
		in = copied
	}

	flags, meta := s.generateConvertMeta(in)
	meta.Context = target
	if err := s.converter.Convert(in, out, flags, meta); err != nil {
		return nil, err
	}

	setTargetKind(out, gvk)
	return out, nil
}

// generateConvertMeta constructs the meta value we pass to Convert.
func (s *Scheme) generateConvertMeta(in interface{}) (conversion.FieldMatchingFlags, *conversion.Meta) {
	return s.converter.DefaultMeta(reflect.TypeOf(in))
}

// copyAndSetTargetKind performs a conditional copy before returning the object, or an error if copy was not successful.
func copyAndSetTargetKind(copy bool, copier ObjectCopier, obj Object, kind schema.GroupVersionKind) (Object, error) {
	if copy {
		copied, err := copier.Copy(obj)
		if err != nil {
			return nil, err
		}
		obj = copied
	}
	setTargetKind(obj, kind)
	return obj, nil
}

// setTargetKind sets the kind on an object, taking into account whether the target kind is the internal version.
func setTargetKind(obj Object, kind schema.GroupVersionKind) {
	if kind.Version == APIVersionInternal {
		// internal is a special case
		// TODO: look at removing the need to special case this
		obj.GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})
		return
	}
	obj.GetObjectKind().SetGroupVersionKind(kind)
}
