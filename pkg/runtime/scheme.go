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

package runtime

import (
	"fmt"
	"net/url"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
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
	gvkToType map[unversioned.GroupVersionKind]reflect.Type

	// typeToGroupVersion allows one to find metadata for a given go object.
	// The reflect.Type we index by should *not* be a pointer.
	typeToGVK map[reflect.Type][]unversioned.GroupVersionKind

	// unversionedTypes are transformed without conversion in ConvertToVersion.
	unversionedTypes map[reflect.Type]unversioned.GroupVersionKind

	// unversionedKinds are the names of kinds that can be created in the context of any group
	// or version
	// TODO: resolve the status of unversioned types.
	unversionedKinds map[string]reflect.Type

	// Map from version and resource to the corresponding func to convert
	// resource field labels in that version to internal version.
	fieldLabelConversionFuncs map[string]map[string]FieldLabelConversionFunc

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
		gvkToType:        map[unversioned.GroupVersionKind]reflect.Type{},
		typeToGVK:        map[reflect.Type][]unversioned.GroupVersionKind{},
		unversionedTypes: map[reflect.Type]unversioned.GroupVersionKind{},
		unversionedKinds: map[string]reflect.Type{},
		cloner:           conversion.NewCloner(),
		fieldLabelConversionFuncs: map[string]map[string]FieldLabelConversionFunc{},
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
func (s *Scheme) AddUnversionedTypes(version unversioned.GroupVersion, types ...Object) {
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
func (s *Scheme) AddKnownTypes(gv unversioned.GroupVersion, types ...Object) {
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
func (s *Scheme) AddKnownTypeWithName(gvk unversioned.GroupVersionKind, obj Object) {
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
func (s *Scheme) KnownTypes(gv unversioned.GroupVersion) map[string]reflect.Type {
	types := make(map[string]reflect.Type)
	for gvk, t := range s.gvkToType {
		if gv != gvk.GroupVersion() {
			continue
		}

		types[gvk.Kind] = t
	}
	return types
}

// ObjectKind returns the group,version,kind of the go object and true if this object
// is considered unversioned, or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectKind(obj Object) (unversioned.GroupVersionKind, bool, error) {
	gvks, unversionedType, err := s.ObjectKinds(obj)
	if err != nil {
		return unversioned.GroupVersionKind{}, false, err
	}
	return gvks[0], unversionedType, nil
}

// ObjectKinds returns all possible group,version,kind of the go object, true if the
// object is considered unversioned, or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectKinds(obj Object) ([]unversioned.GroupVersionKind, bool, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, false, err
	}
	t := v.Type()

	gvks, ok := s.typeToGVK[t]
	if !ok {
		return nil, false, &notRegisteredErr{t: t}
	}
	_, unversionedType := s.unversionedTypes[t]

	return gvks, unversionedType, nil
}

// Recognizes returns true if the scheme is able to handle the provided group,version,kind
// of an object.
func (s *Scheme) Recognizes(gvk unversioned.GroupVersionKind) bool {
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
func (s *Scheme) New(kind unversioned.GroupVersionKind) (Object, error) {
	if t, exists := s.gvkToType[kind]; exists {
		return reflect.New(t).Interface().(Object), nil
	}

	if t, exists := s.unversionedKinds[kind.Kind]; exists {
		return reflect.New(t).Interface().(Object), nil
	}
	return nil, &notRegisteredErr{gvk: kind}
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
func (s *Scheme) AddGeneratedDeepCopyFuncs(deepCopyFuncs ...interface{}) error {
	for _, f := range deepCopyFuncs {
		if err := s.cloner.RegisterGeneratedDeepCopyFunc(f); err != nil {
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
// a to test conversion of types that are nested within registered types), but in
// that case, the conversion.Scope object passed to your conversion functions won't
// have SrcVersion or DestVersion fields set correctly in Meta().
func (s *Scheme) Convert(in, out interface{}) error {
	inVersion := unversioned.GroupVersion{Group: "unknown", Version: "unknown"}
	outVersion := unversioned.GroupVersion{Group: "unknown", Version: "unknown"}
	if inObj, ok := in.(Object); ok {
		if gvks, _, err := s.ObjectKinds(inObj); err == nil {
			inVersion = gvks[0].GroupVersion()
		}
	}
	if outObj, ok := out.(Object); ok {
		if gvks, _, err := s.ObjectKinds(outObj); err == nil {
			outVersion = gvks[0].GroupVersion()
		}
	}
	flags, meta := s.generateConvertMeta(inVersion, outVersion, in)
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
// returned. The serializer handles loading/serializing nested objects.
func (s *Scheme) ConvertToVersion(in Object, outVersion unversioned.GroupVersion) (Object, error) {
	switch in.(type) {
	case *Unknown, *Unstructured, *UnstructuredList:
		old := in.GetObjectKind().GroupVersionKind()
		defer in.GetObjectKind().SetGroupVersionKind(old)
		setTargetVersion(in, s, outVersion)
		return in, nil
	}
	t := reflect.TypeOf(in)
	if t.Kind() != reflect.Ptr {
		return nil, fmt.Errorf("only pointer types may be converted: %v", t)
	}

	t = t.Elem()
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("only pointers to struct types may be converted: %v", t)
	}

	var kind unversioned.GroupVersionKind
	if unversionedKind, ok := s.unversionedTypes[t]; ok {
		kind = unversionedKind
	} else {
		kinds, ok := s.typeToGVK[t]
		if !ok || len(kinds) == 0 {
			return nil, fmt.Errorf("%v is not a registered type and cannot be converted into version %q", t, outVersion)
		}
		kind = kinds[0]
	}

	outKind := outVersion.WithKind(kind.Kind)

	inKinds, _, err := s.ObjectKinds(in)
	if err != nil {
		return nil, err
	}

	out, err := s.New(outKind)
	if err != nil {
		return nil, err
	}

	flags, meta := s.generateConvertMeta(inKinds[0].GroupVersion(), outVersion, in)
	if err := s.converter.Convert(in, out, flags, meta); err != nil {
		return nil, err
	}

	setTargetVersion(out, s, outVersion)
	return out, nil
}

// UnsafeConvertToVersion will convert in to the provided outVersion if such a conversion is possible,
// but does not guarantee the output object does not share fields with the input object. It attempts to be as
// efficient as possible when doing conversion.
func (s *Scheme) UnsafeConvertToVersion(in Object, outVersion unversioned.GroupVersion) (Object, error) {
	switch t := in.(type) {
	case *Unknown:
		t.APIVersion = outVersion.String()
		return t, nil
	case *Unstructured:
		t.SetAPIVersion(outVersion.String())
		return t, nil
	case *UnstructuredList:
		t.SetAPIVersion(outVersion.String())
		return t, nil
	}

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
		return nil, fmt.Errorf("%v is not a registered type and cannot be converted into version %q", t, outVersion)
	}

	// if the Go type is also registered to the destination kind, no conversion is necessary
	for i := range kinds {
		if kinds[i].Version == outVersion.Version && kinds[i].Group == outVersion.Group {
			setTargetKind(in, kinds[i])
			return in, nil
		}
	}

	// type is unversioned, no conversion necessary
	// it should be possible to avoid this allocation
	if unversionedKind, ok := s.unversionedTypes[t]; ok {
		kind := unversionedKind
		outKind := outVersion.WithKind(kind.Kind)
		setTargetKind(in, outKind)
		return in, nil
	}

	// allocate a new object as the target using the target kind
	// TODO: this should look in the target group version and find the first kind that matches, rather than the
	//   first kind registered in typeToGVK
	kind := kinds[0]
	kind.Version = outVersion.Version
	kind.Group = outVersion.Group
	out, err := s.New(kind)
	if err != nil {
		return nil, err
	}

	// TODO: try to avoid the allocations here - in fast paths we are not likely to need these flags or meta
	flags, meta := s.converter.DefaultMeta(t)
	if err := s.converter.Convert(in, out, flags, meta); err != nil {
		return nil, err
	}

	setTargetKind(out, kind)
	return out, nil
}

// generateConvertMeta constructs the meta value we pass to Convert.
func (s *Scheme) generateConvertMeta(srcGroupVersion, destGroupVersion unversioned.GroupVersion, in interface{}) (conversion.FieldMatchingFlags, *conversion.Meta) {
	return s.converter.DefaultMeta(reflect.TypeOf(in))
}

// setTargetVersion is deprecated and should be replaced by use of setTargetKind
func setTargetVersion(obj Object, raw *Scheme, gv unversioned.GroupVersion) {
	if gv.Version == APIVersionInternal {
		// internal is a special case
		obj.GetObjectKind().SetGroupVersionKind(unversioned.GroupVersionKind{})
		return
	}
	if gvks, _, _ := raw.ObjectKinds(obj); len(gvks) > 0 {
		obj.GetObjectKind().SetGroupVersionKind(unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: gvks[0].Kind})
	} else {
		obj.GetObjectKind().SetGroupVersionKind(unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version})
	}
}

// setTargetKind sets the kind on an object, taking into account whether the target kind is the internal version.
func setTargetKind(obj Object, kind unversioned.GroupVersionKind) {
	if kind.Version == APIVersionInternal {
		// internal is a special case
		// TODO: look at removing the need to special case this
		obj.GetObjectKind().SetGroupVersionKind(unversioned.GroupVersionKind{})
		return
	}
	obj.GetObjectKind().SetGroupVersionKind(kind)
}
