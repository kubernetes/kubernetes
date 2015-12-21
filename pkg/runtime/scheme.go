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

// Scheme defines methods for serializing and deserializing API objects. It
// is an adaptation of conversion's Scheme for our API objects.
type Scheme struct {
	raw *conversion.Scheme
	// Map from version and resource to the corresponding func to convert
	// resource field labels in that version to internal version.
	fieldLabelConversionFuncs map[string]map[string]FieldLabelConversionFunc
}

// Function to convert a field selector to internal representation.
type FieldLabelConversionFunc func(label, value string) (internalLabel, internalValue string, err error)

func (self *Scheme) Raw() *conversion.Scheme {
	return self.raw
}

// fromScope gets the input version, desired output version, and desired Scheme
// from a conversion.Scope.
func (self *Scheme) fromScope(s conversion.Scope) (inVersion, outVersion string, scheme *Scheme) {
	scheme = self
	inVersion = s.Meta().SrcVersion
	outVersion = s.Meta().DestVersion
	return inVersion, outVersion, scheme
}

// NewScheme creates a new Scheme. This scheme is pluggable by default.
func NewScheme() *Scheme {
	s := &Scheme{conversion.NewScheme(), map[string]map[string]FieldLabelConversionFunc{}}
	s.AddConversionFuncs(DefaultEmbeddedConversions()...)

	// Enable map[string][]string conversions by default
	if err := s.raw.AddConversionFuncs(DefaultStringConversions...); err != nil {
		panic(err)
	}
	if err := s.raw.RegisterInputDefaults(&map[string][]string{}, JSONKeyMapper, conversion.AllowDifferentFieldTypeNames|conversion.IgnoreMissingFields); err != nil {
		panic(err)
	}
	if err := s.raw.RegisterInputDefaults(&url.Values{}, JSONKeyMapper, conversion.AllowDifferentFieldTypeNames|conversion.IgnoreMissingFields); err != nil {
		panic(err)
	}
	return s
}

// AddKnownTypes registers the types of the arguments to the marshaller of the package api.
func (s *Scheme) AddUnversionedTypes(gv unversioned.GroupVersion, types ...Object) {
	interfaces := make([]interface{}, len(types))
	for i := range types {
		interfaces[i] = types[i]
	}
	s.raw.AddUnversionedTypes(gv, interfaces...)
}

// AddKnownTypes registers the types of the arguments to the marshaller of the package api.
func (s *Scheme) AddKnownTypes(gv unversioned.GroupVersion, types ...Object) {
	interfaces := make([]interface{}, len(types))
	for i := range types {
		interfaces[i] = types[i]
	}
	s.raw.AddKnownTypes(gv, interfaces...)
}

func (s *Scheme) AddIgnoredConversionType(from, to interface{}) error {
	return s.raw.AddIgnoredConversionType(from, to)
}

// AddKnownTypeWithName is like AddKnownTypes, but it lets you specify what this type should
// be encoded as. Useful for testing when you don't want to make multiple packages to define
// your structs.
func (s *Scheme) AddKnownTypeWithName(gvk unversioned.GroupVersionKind, obj Object) {
	s.raw.AddKnownTypeWithName(gvk, obj)
}

// KnownTypes returns the types known for the given version.
// Return value must be treated as read-only.
func (s *Scheme) KnownTypes(gv unversioned.GroupVersion) map[string]reflect.Type {
	return s.raw.KnownTypes(gv)
}

// ObjectKind returns the default group,version,kind of the given Object.
func (s *Scheme) ObjectKind(obj Object) (unversioned.GroupVersionKind, error) {
	return s.raw.ObjectKind(obj)
}

// ObjectKinds returns the all possible group,version,kind of the given Object.
func (s *Scheme) ObjectKinds(obj Object) ([]unversioned.GroupVersionKind, error) {
	return s.raw.ObjectKinds(obj)
}

// Recognizes returns true if the scheme is able to handle the provided group,version,kind
// of an object.
func (s *Scheme) Recognizes(gvk unversioned.GroupVersionKind) bool {
	return s.raw.Recognizes(gvk)
}

func (s *Scheme) IsUnversioned(obj Object) (bool, bool) {
	return s.raw.IsUnversioned(obj)
}

// New returns a new API object of the given version ("" for internal
// representation) and name, or an error if it hasn't been registered.
func (s *Scheme) New(kind unversioned.GroupVersionKind) (Object, error) {
	obj, err := s.raw.NewObject(kind)
	if err != nil {
		return nil, err
	}
	return obj.(Object), nil
}

// Log sets a logger on the scheme. For test purposes only
func (s *Scheme) Log(l conversion.DebugLogger) {
	s.raw.Log(l)
}

// AddConversionFuncs adds a function to the list of conversion functions. The given
// function should know how to convert between two API objects. We deduce how to call
// it from the types of its two parameters; see the comment for
// Converter.RegisterConversionFunction.
//
// Note that, if you need to copy sub-objects that didn't change, it's safe to call
// Convert() inside your conversionFuncs, as long as you don't start a conversion
// chain that's infinitely recursive.
//
// Also note that the default behavior, if you don't add a conversion function, is to
// sanely copy fields that have the same names. It's OK if the destination type has
// extra fields, but it must not remove any. So you only need to add a conversion
// function for things with changed/removed fields.
func (s *Scheme) AddConversionFuncs(conversionFuncs ...interface{}) error {
	return s.raw.AddConversionFuncs(conversionFuncs...)
}

// Similar to AddConversionFuncs, but registers conversion functions that were
// automatically generated.
func (s *Scheme) AddGeneratedConversionFuncs(conversionFuncs ...interface{}) error {
	return s.raw.AddGeneratedConversionFuncs(conversionFuncs...)
}

// AddDeepCopyFuncs adds a function to the list of deep-copy functions.
// For the expected format of deep-copy function, see the comment for
// Copier.RegisterDeepCopyFunction.
func (s *Scheme) AddDeepCopyFuncs(deepCopyFuncs ...interface{}) error {
	return s.raw.AddDeepCopyFuncs(deepCopyFuncs...)
}

// Similar to AddDeepCopyFuncs, but registers deep-copy functions that were
// automatically generated.
func (s *Scheme) AddGeneratedDeepCopyFuncs(deepCopyFuncs ...interface{}) error {
	return s.raw.AddGeneratedDeepCopyFuncs(deepCopyFuncs...)
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
	return s.raw.AddStructFieldConversion(srcFieldType, srcFieldName, destFieldType, destFieldName)
}

// AddDefaultingFuncs adds a function to the list of value-defaulting functions.
// We deduce how to call it from the types of its two parameters; see the
// comment for Converter.RegisterDefaultingFunction.
func (s *Scheme) AddDefaultingFuncs(defaultingFuncs ...interface{}) error {
	return s.raw.AddDefaultingFuncs(defaultingFuncs...)
}

// Copy does a deep copy of an API object.
func (s *Scheme) Copy(src Object) (Object, error) {
	dst, err := s.raw.DeepCopy(src)
	if err != nil {
		return nil, err
	}
	return dst.(Object), nil
}

// Performs a deep copy of the given object.
func (s *Scheme) DeepCopy(src interface{}) (interface{}, error) {
	return s.raw.DeepCopy(src)
}

// ObjectVersioner returns an interface that performs conversions with the additional provided
// functions.
func (s *Scheme) WithConversions(fns *conversion.ConversionFuncs) ObjectConvertor {
	if fns == nil {
		return s
	}
	copied := *s
	copied.raw = s.raw.WithConversions(*fns)
	return &copied
}

// Convert will attempt to convert in into out. Both must be pointers.
// For easy testing of conversion functions. Returns an error if the conversion isn't
// possible.
func (s *Scheme) Convert(in, out interface{}) error {
	return s.raw.Convert(in, out)
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
func (s *Scheme) ConvertToVersion(in Object, outVersion string) (Object, error) {
	gv, err := unversioned.ParseGroupVersion(outVersion)
	if err != nil {
		return nil, err
	}
	switch in.(type) {
	case *Unknown, *Unstructured:
		old := in.GetObjectKind().GroupVersionKind()
		defer in.GetObjectKind().SetGroupVersionKind(old)
		setTargetVersion(in, s.raw, gv)
		return in, nil
	}
	unknown, err := s.raw.ConvertToVersion(in, outVersion)
	if err != nil {
		return nil, err
	}
	obj, ok := unknown.(Object)
	if !ok {
		return nil, fmt.Errorf("the provided object cannot be converted to a runtime.Object: %#v", unknown)
	}
	setTargetVersion(obj, s.raw, gv)
	return obj, nil
}

func setTargetVersion(obj Object, raw *conversion.Scheme, gv unversioned.GroupVersion) {
	if gv.Version == APIVersionInternal {
		// internal is a special case
		obj.GetObjectKind().SetGroupVersionKind(nil)
	} else {
		gvk, _ := raw.ObjectKind(obj)
		obj.GetObjectKind().SetGroupVersionKind(&unversioned.GroupVersionKind{Group: gv.Group, Version: gv.Version, Kind: gvk.Kind})
	}
}
