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

package runtime

import (
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"gopkg.in/v1/yaml"
)

// codec defines methods for serializing and deserializing API
// objects.
type codec interface {
	Encode(obj Object) (data []byte, err error)
	Decode(data []byte) (Object, error)
	DecodeInto(data []byte, obj Object) error
}

// resourceVersioner provides methods for setting and retrieving
// the resource version from an API object.
type resourceVersioner interface {
	SetResourceVersion(obj Object, version uint64) error
	ResourceVersion(obj Object) (uint64, error)
}

var ResourceVersioner resourceVersioner = NewJSONBaseResourceVersioner()
var DefaultScheme = NewScheme("", "v1beta1")
var Codec codec = DefaultScheme

// Scheme defines methods for serializing and deserializing API objects. It
// is an adaptation of conversion's Scheme for our API objects.
type Scheme struct {
	raw *conversion.Scheme
}

// NewScheme creates a new Scheme. A default scheme is provided and accessible
// as the "DefaultScheme" variable.
func NewScheme(internalVersion, externalVersion string) *Scheme {
	s := &Scheme{conversion.NewScheme()}
	s.raw.InternalVersion = internalVersion
	s.raw.ExternalVersion = externalVersion
	s.raw.MetaInsertionFactory = metaInsertion{}
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

// New returns a new API object of the given version ("" for internal
// representation) and name, or an error if it hasn't been registered.
func (s *Scheme) New(versionName, typeName string) (Object, error) {
	obj, err := s.raw.NewObject(versionName, typeName)
	if err != nil {
		return nil, err
	}
	return obj.(Object), nil
}

// AddConversionFuncs adds a function to the list of conversion functions. The given
// function should know how to convert between two API objects. We deduce how to call
// it from the types of its two parameters; see the comment for Converter.Register.
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

// Convert will attempt to convert in into out. Both must be pointers.
// For easy testing of conversion functions. Returns an error if the conversion isn't
// possible.
func (s *Scheme) Convert(in, out interface{}) error {
	return s.raw.Convert(in, out)
}

// FindJSONBase takes an arbitary api type, returns pointer to its JSONBase field.
// obj must be a pointer to an api type.
func FindJSONBase(obj Object) (JSONBaseInterface, error) {
	v, err := enforcePtr(obj)
	if err != nil {
		return nil, err
	}
	t := v.Type()
	name := t.Name()
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), name, v.Interface())
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return nil, fmt.Errorf("struct %v lacks embedded JSON type", name)
	}
	g, err := newGenericJSONBase(jsonBase)
	if err != nil {
		return nil, err
	}
	return g, nil
}

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func (s *Scheme) EncodeOrDie(obj Object) string {
	return s.raw.EncodeOrDie(obj)
}

// Encode turns the given api object into an appropriate JSON string.
// Will return an error if the object doesn't have an embedded JSONBase.
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
//
// TODO/next steps: When we add our second versioned type, this package will
// need a version of Encode that lets you choose the wire version. A configurable
// default will be needed, to allow operating in clusters that haven't yet
// upgraded.
//
func (s *Scheme) Encode(obj Object) (data []byte, err error) {
	return s.raw.Encode(obj)
}

// enforcePtr ensures that obj is a pointer of some sort. Returns a reflect.Value of the
// dereferenced pointer, ensuring that it is settable/addressable.
// Returns an error if this is not possible.
func enforcePtr(obj Object) (reflect.Value, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		return reflect.Value{}, fmt.Errorf("expected pointer, but got %v", v.Type().Name())
	}
	return v.Elem(), nil
}

// VersionAndKind will return the APIVersion and Kind of the given wire-format
// enconding of an APIObject, or an error.
func VersionAndKind(data []byte) (version, kind string, err error) {
	findKind := struct {
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, &findKind)
	if err != nil {
		return "", "", fmt.Errorf("couldn't get version/kind: %v", err)
	}
	return findKind.APIVersion, findKind.Kind, nil
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

// DecodeInto parses a YAML or JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
// If obj's APIVersion doesn't match that in data, an attempt will be made to convert
// data into obj's version.
func (s *Scheme) DecodeInto(data []byte, obj Object) error {
	return s.raw.DecodeInto(data, obj)
}

// Does a deep copy of an API object.  Useful mostly for tests.
// TODO(dbsmith): implement directly instead of via Encode/Decode
func (s *Scheme) Copy(obj Object) (Object, error) {
	data, err := s.Encode(obj)
	if err != nil {
		return nil, err
	}
	return s.Decode(data)
}

func (s *Scheme) CopyOrDie(obj Object) Object {
	newObj, err := s.Copy(obj)
	if err != nil {
		panic(err)
	}
	return newObj
}

// metaInsertion implements conversion.MetaInsertionFactory, which lets the conversion
// package figure out how to encode our object's types and versions. These fields are
// located in our JSONBase.
type metaInsertion struct {
	JSONBase struct {
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
	} `json:",inline" yaml:",inline"`
}

// Create returns a new metaInsertion with the version and kind fields set.
func (metaInsertion) Create(version, kind string) interface{} {
	m := metaInsertion{}
	m.JSONBase.APIVersion = version
	m.JSONBase.Kind = kind
	return &m
}

// Interpret returns the version and kind information from in, which must be
// a metaInsertion pointer object.
func (metaInsertion) Interpret(in interface{}) (version, kind string) {
	m := in.(*metaInsertion)
	return m.JSONBase.APIVersion, m.JSONBase.Kind
}
