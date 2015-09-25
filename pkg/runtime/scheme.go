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
	"encoding/json"
	"fmt"
	"net/url"
	"reflect"

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

// emptyPlugin is used to copy the Kind field to and from plugin objects.
type emptyPlugin struct {
	PluginBase `json:",inline"`
}

// embeddedObjectToRawExtension does the conversion you would expect from the name, using the information
// given in conversion.Scope. It's placed in the DefaultScheme as a ConversionFunc to enable plugins;
// see the comment for RawExtension.
func (self *Scheme) embeddedObjectToRawExtension(in *EmbeddedObject, out *RawExtension, s conversion.Scope) error {
	if in.Object == nil {
		out.RawJSON = []byte("null")
		return nil
	}

	// Figure out the type and kind of the output object.
	_, outVersion, scheme := self.fromScope(s)
	_, kind, err := scheme.raw.ObjectVersionAndKind(in.Object)
	if err != nil {
		return err
	}

	// Manufacture an object of this type and kind.
	outObj, err := scheme.New(outVersion, kind)
	if err != nil {
		return err
	}

	// Manually do the conversion.
	err = s.Convert(in.Object, outObj, 0)
	if err != nil {
		return err
	}

	// Copy the kind field into the output object.
	err = s.Convert(
		&emptyPlugin{PluginBase: PluginBase{Kind: kind}},
		outObj,
		conversion.SourceToDest|conversion.IgnoreMissingFields|conversion.AllowDifferentFieldTypeNames,
	)
	if err != nil {
		return err
	}
	// Because we provide the correct version, EncodeToVersion will not attempt a conversion.
	raw, err := scheme.EncodeToVersion(outObj, outVersion)
	if err != nil {
		// TODO: if this fails, create an Unknown-- maybe some other
		// component will understand it.
		return err
	}
	out.RawJSON = raw
	return nil
}

// rawExtensionToEmbeddedObject does the conversion you would expect from the name, using the information
// given in conversion.Scope. It's placed in all schemes as a ConversionFunc to enable plugins;
// see the comment for RawExtension.
func (self *Scheme) rawExtensionToEmbeddedObject(in *RawExtension, out *EmbeddedObject, s conversion.Scope) error {
	if len(in.RawJSON) == 0 || (len(in.RawJSON) == 4 && string(in.RawJSON) == "null") {
		out.Object = nil
		return nil
	}
	// Figure out the type and kind of the output object.
	inVersion, outVersion, scheme := self.fromScope(s)
	_, kind, err := scheme.raw.DataVersionAndKind(in.RawJSON)
	if err != nil {
		return err
	}

	// We have to make this object ourselves because we don't store the version field for
	// plugin objects.
	inObj, err := scheme.New(inVersion, kind)
	if err != nil {
		return err
	}

	err = scheme.DecodeInto(in.RawJSON, inObj)
	if err != nil {
		return err
	}

	// Make the desired internal version, and do the conversion.
	outObj, err := scheme.New(outVersion, kind)
	if err != nil {
		return err
	}
	err = scheme.Convert(inObj, outObj)
	if err != nil {
		return err
	}
	// Last step, clear the Kind field; that should always be blank in memory.
	err = s.Convert(
		&emptyPlugin{PluginBase: PluginBase{Kind: ""}},
		outObj,
		conversion.SourceToDest|conversion.IgnoreMissingFields|conversion.AllowDifferentFieldTypeNames,
	)
	if err != nil {
		return err
	}
	out.Object = outObj
	return nil
}

// runtimeObjectToRawExtensionArray takes a list of objects and encodes them as RawExtension in the output version
// defined by the conversion.Scope. If objects must be encoded to different schema versions than the default, you
// should encode them yourself with runtime.Unknown, or convert the object prior to invoking conversion. Objects
// outside of the current scheme must be added as runtime.Unknown.
func (self *Scheme) runtimeObjectToRawExtensionArray(in *[]Object, out *[]RawExtension, s conversion.Scope) error {
	src := *in
	dest := make([]RawExtension, len(src))

	_, outVersion, scheme := self.fromScope(s)

	for i := range src {
		switch t := src[i].(type) {
		case *Unknown:
			// TODO: this should be decoupled from the scheme (since it is JSON specific)
			dest[i].RawJSON = t.RawJSON
		case *Unstructured:
			// TODO: this should be decoupled from the scheme (since it is JSON specific)
			data, err := json.Marshal(t.Object)
			if err != nil {
				return err
			}
			dest[i].RawJSON = data
		default:
			version := outVersion
			// if the object exists
			if inVersion, _, err := scheme.ObjectVersionAndKind(src[i]); err == nil && len(inVersion) != 0 {
				version = inVersion
			}
			data, err := scheme.EncodeToVersion(src[i], version)
			if err != nil {
				return err
			}
			dest[i].RawJSON = data
		}
	}
	*out = dest
	return nil
}

// rawExtensionToRuntimeObjectArray attempts to decode objects from the array - if they are unrecognized objects,
// they are added as Unknown.
func (self *Scheme) rawExtensionToRuntimeObjectArray(in *[]RawExtension, out *[]Object, s conversion.Scope) error {
	src := *in
	dest := make([]Object, len(src))

	_, _, scheme := self.fromScope(s)

	for i := range src {
		data := src[i].RawJSON
		version, kind, err := scheme.raw.DataVersionAndKind(data)
		if err != nil {
			return err
		}
		dest[i] = &Unknown{
			TypeMeta: TypeMeta{
				APIVersion: version,
				Kind:       kind,
			},
			RawJSON: data,
		}
	}
	*out = dest
	return nil
}

// NewScheme creates a new Scheme. This scheme is pluggable by default.
func NewScheme() *Scheme {
	s := &Scheme{conversion.NewScheme(), map[string]map[string]FieldLabelConversionFunc{}}
	s.raw.InternalVersion = ""
	s.raw.MetaFactory = conversion.SimpleMetaFactory{BaseFields: []string{"TypeMeta"}, VersionField: "APIVersion", KindField: "Kind"}
	if err := s.raw.AddConversionFuncs(
		s.embeddedObjectToRawExtension,
		s.rawExtensionToEmbeddedObject,
		s.runtimeObjectToRawExtensionArray,
		s.rawExtensionToRuntimeObjectArray,
	); err != nil {
		panic(err)
	}
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
// Encode() refuses the object unless its type is registered with AddKnownTypes.
func (s *Scheme) AddKnownTypes(version string, types ...Object) {
	interfaces := make([]interface{}, len(types))
	for i := range types {
		interfaces[i] = types[i]
	}
	s.raw.AddKnownTypes(version, interfaces...)
}

// AddKnownTypeWithName is like AddKnownTypes, but it lets you specify what this type should
// be encoded as. Useful for testing when you don't want to make multiple packages to define
// your structs.
func (s *Scheme) AddKnownTypeWithName(version, kind string, obj Object) {
	s.raw.AddKnownTypeWithName(version, kind, obj)
}

// KnownTypes returns the types known for the given version.
// Return value must be treated as read-only.
func (s *Scheme) KnownTypes(version string) map[string]reflect.Type {
	return s.raw.KnownTypes(version)
}

// DataVersionAndKind will return the APIVersion and Kind of the given wire-format
// encoding of an API Object, or an error.
func (s *Scheme) DataVersionAndKind(data []byte) (version, kind string, err error) {
	return s.raw.DataVersionAndKind(data)
}

// ObjectVersionAndKind returns the version and kind of the given Object.
func (s *Scheme) ObjectVersionAndKind(obj Object) (version, kind string, err error) {
	return s.raw.ObjectVersionAndKind(obj)
}

// Recognizes returns true if the scheme is able to handle the provided version and kind
// of an object.
func (s *Scheme) Recognizes(version, kind string) bool {
	return s.raw.Recognizes(version, kind)
}

// New returns a new API object of the given version ("" for internal
// representation) and name, or an error if it hasn't been registered.
func (s *Scheme) New(versionName, typeName string) (Object, error) {
	obj, err := s.raw.NewObject(versionName, typeName)
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

// Performs a deep copy of the given object.
func (s *Scheme) DeepCopy(src interface{}) (interface{}, error) {
	return s.raw.DeepCopy(src)
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
// returned.
func (s *Scheme) ConvertToVersion(in Object, outVersion string) (Object, error) {
	unknown, err := s.raw.ConvertToVersion(in, outVersion)
	if err != nil {
		return nil, err
	}
	obj, ok := unknown.(Object)
	if !ok {
		return nil, fmt.Errorf("the provided object cannot be converted to a runtime.Object: %#v", unknown)
	}
	return obj, nil
}

// EncodeToVersion turns the given api object into an appropriate JSON string.
// Will return an error if the object doesn't have an embedded TypeMeta.
// Obj may be a pointer to a struct, or a struct. If a struct, a copy
// must be made. If a pointer, the object may be modified before encoding,
// but will be put back into its original state before returning.
//
// Memory/wire format differences:
//  * Having to keep track of the Kind and APIVersion fields makes tests
//    very annoying, so the rule is that they are set only in wire format
//    (json), not when in native (memory) format. This is possible because
//    both pieces of information are implicit in the go typed object.
//     * An exception: note that, if there are embedded API objects of known
//       type, for example, PodList{... Items []Pod ...}, these embedded
//       objects must be of the same version of the object they are embedded
//       within, and their APIVersion and Kind must both be empty.
//     * Note that the exception does not apply to the APIObject type, which
//       recursively does Encode()/Decode(), and is capable of expressing any
//       API object.
//  * Only versioned objects should be encoded. This means that, if you pass
//    a native object, Encode will convert it to a versioned object. For
//    example, an api.Pod will get converted to a v1beta1.Pod. However, if
//    you pass in an object that's already versioned (v1beta1.Pod), Encode
//    will not modify it.
//
// The purpose of the above complex conversion behavior is to allow us to
// change the memory format yet not break compatibility with any stored
// objects, whether they be in our storage layer (e.g., etcd), or in user's
// config files.
func (s *Scheme) EncodeToVersion(obj Object, destVersion string) (data []byte, err error) {
	return s.raw.EncodeToVersion(obj, destVersion)
}

// Decode converts a YAML or JSON string back into a pointer to an api object.
// Deduces the type based upon the APIVersion and Kind fields, which are set
// by Encode. Only versioned objects (APIVersion != "") are accepted. The object
// will be converted into the in-memory unversioned type before being returned.
func (s *Scheme) Decode(data []byte) (Object, error) {
	obj, err := s.raw.Decode(data)
	if err != nil {
		return nil, err
	}
	return obj.(Object), nil
}

// DecodeToVersion converts a YAML or JSON string back into a pointer to an api
// object.  Deduces the type based upon the APIVersion and Kind fields, which
// are set by Encode. Only versioned objects (APIVersion != "") are
// accepted. The object will be converted into the in-memory versioned type
// requested before being returned.
func (s *Scheme) DecodeToVersion(data []byte, version string) (Object, error) {
	obj, err := s.raw.DecodeToVersion(data, version)
	if err != nil {
		return nil, err
	}
	return obj.(Object), nil
}

// DecodeInto parses a YAML or JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
// If obj's APIVersion doesn't match that in data, an attempt will be made to convert
// data into obj's version.
// TODO: allow Decode/DecodeInto to take a default apiVersion and a default kind, to
// be applied if the provided object does not have either field (integrate external
// apis into the decoding scheme).
func (s *Scheme) DecodeInto(data []byte, obj Object) error {
	return s.raw.DecodeInto(data, obj)
}

func (s *Scheme) DecodeIntoWithSpecifiedVersionKind(data []byte, obj Object, version, kind string) error {
	return s.raw.DecodeIntoWithSpecifiedVersionKind(data, obj, version, kind)
}

// Copy does a deep copy of an API object.  Useful mostly for tests.
func (s *Scheme) Copy(src Object) (Object, error) {
	dst, err := s.raw.DeepCopy(src)
	if err != nil {
		return nil, err
	}
	return dst.(Object), nil
}

func (s *Scheme) CopyOrDie(obj Object) Object {
	newObj, err := s.Copy(obj)
	if err != nil {
		panic(err)
	}
	return newObj
}
