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

package conversion

import (
	"fmt"
	"reflect"

	"gopkg.in/v1/yaml"
)

// MetaInsertionFactory is used to create an object to store and retrieve
// the version and kind information for all objects. The default uses the
// keys "version" and "kind" respectively. The object produced by this
// factory is used to clear the version and kind fields in memory, so it
// must match the layout of your actual api structs. (E.g., if you have your
// version and kind field inside an inlined struct, this must produce an
// inlined struct with the same field name.)
type MetaInsertionFactory interface {
	// Create should make a new object with two fields.
	// This object will be used to encode this metadata along with your
	// API objects, so the tags on the fields you use shouldn't conflict.
	Create(version, kind string) interface{}
	// Interpret should take the same type of object that Create creates.
	// It should return the version and kind information from this object.
	Interpret(interface{}) (version, kind string)
}

// Scheme defines an entire encoding and decoding scheme.
type Scheme struct {
	// versionMap allows one to figure out the go type of an object with
	// the given version and name.
	versionMap map[string]map[string]reflect.Type

	// typeToVersion allows one to figure out the version for a given go object.
	// The reflect.Type we index by should *not* be a pointer. If the same type
	// is registered for multiple versions, the last one wins.
	typeToVersion map[reflect.Type]string

	// converter stores all registered conversion functions. It also has
	// default coverting behavior.
	converter *Converter

	// Indent will cause the JSON output from Encode to be indented, iff it is true.
	Indent bool

	// InternalVersion is the default internal version. It is recommended that
	// you use "" for the internal version.
	InternalVersion string

	// ExternalVersion is the default external version.
	ExternalVersion string

	// MetaInsertionFactory is used to create an object to store and retrieve
	// the version and kind information for all objects. The default uses the
	// keys "version" and "kind" respectively.
	MetaInsertionFactory MetaInsertionFactory
}

// NewScheme manufactures a new scheme.
func NewScheme() *Scheme {
	return &Scheme{
		versionMap:           map[string]map[string]reflect.Type{},
		typeToVersion:        map[reflect.Type]string{},
		converter:            NewConverter(),
		InternalVersion:      "",
		ExternalVersion:      "v1",
		MetaInsertionFactory: metaInsertion{},
	}
}

// AddKnownTypes registers all types passed in 'types' as being members of version 'version.
// Encode() will refuse objects unless their type has been registered with AddKnownTypes.
// All objects passed to types should be structs, not pointers to structs. The name that go
// reports for the struct becomes the "kind" field when encoding.
func (s *Scheme) AddKnownTypes(version string, types ...interface{}) {
	knownTypes, found := s.versionMap[version]
	if !found {
		knownTypes = map[string]reflect.Type{}
		s.versionMap[version] = knownTypes
	}
	for _, obj := range types {
		t := reflect.TypeOf(obj)
		if t.Kind() != reflect.Struct {
			panic("All types must be structs.")
		}
		knownTypes[t.Name()] = t
		s.typeToVersion[t] = version
	}
}

// NewObject returns a new object of the given version and name,
// or an error if it hasn't been registered.
func (s *Scheme) NewObject(versionName, typeName string) (interface{}, error) {
	if types, ok := s.versionMap[versionName]; ok {
		if t, ok := types[typeName]; ok {
			return reflect.New(t).Interface(), nil
		}
		return nil, fmt.Errorf("No type '%v' for version '%v'", typeName, versionName)
	}
	return nil, fmt.Errorf("No version '%v'", versionName)
}

// AddConversionFuncs adds functions to the list of conversion functions. The given
// functions should know how to convert between two of your API objects, or their
// sub-objects. We deduce how to call these functions from the types of their two
// parameters; see the comment for Converter.Register.
//
// Note that, if you need to copy sub-objects that didn't change, it's safe to call
// s.Convert() inside your conversionFuncs, as long as you don't start a conversion
// chain that's infinitely recursive.
//
// Also note that the default behavior, if you don't add a conversion function, is to
// sanely copy fields that have the same names and same type names. It's OK if the
// destination type has extra fields, but it must not remove any. So you only need to
// add conversion functions for things with changed/removed fields.
func (s *Scheme) AddConversionFuncs(conversionFuncs ...interface{}) error {
	for _, f := range conversionFuncs {
		err := s.converter.Register(f)
		if err != nil {
			return err
		}
	}
	return nil
}

// Convert will attempt to convert in into out. Both must be pointers. For easy
// testing of conversion functions. Returns an error if the conversion isn't
// possible.
func (s *Scheme) Convert(in, out interface{}) error {
	return s.converter.Convert(in, out, 0)
}

// metaInsertion provides a default implementation of MetaInsertionFactory.
type metaInsertion struct {
	Version string `json:"version,omitempty" yaml:"version,omitempty"`
	Kind    string `json:"kind,omitempty" yaml:"kind,omitempty"`
}

// Create should make a new object with two fields.
// This object will be used to encode this metadata along with your
// API objects, so the tags on the fields you use shouldn't conflict.
func (metaInsertion) Create(version, kind string) interface{} {
	m := metaInsertion{}
	m.Version = version
	m.Kind = kind
	return &m
}

// Interpret should take the same type of object that Create creates.
// It should return the version and kind information from this object.
func (metaInsertion) Interpret(in interface{}) (version, kind string) {
	m := in.(*metaInsertion)
	return m.Version, m.Kind
}

// DataAPIVersionAndKind will return the APIVersion and Kind of the given wire-format
// enconding of an API Object, or an error.
func (s *Scheme) DataVersionAndKind(data []byte) (version, kind string, err error) {
	findKind := s.MetaInsertionFactory.Create("", "")
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, findKind)
	if err != nil {
		return "", "", fmt.Errorf("couldn't get version/kind: %v", err)
	}
	version, kind = s.MetaInsertionFactory.Interpret(findKind)
	return version, kind, nil
}

// ObjectVersionAndKind returns the API version and kind of the go object,
// or an error if it's not a pointer or is unregistered.
func (s *Scheme) ObjectVersionAndKind(obj interface{}) (apiVersion, kind string, err error) {
	v, err := enforcePtr(obj)
	if err != nil {
		return "", "", err
	}
	t := v.Type()
	if version, ok := s.typeToVersion[t]; !ok {
		return "", "", fmt.Errorf("Unregistered type: %v", t)
	} else {
		return version, t.Name(), nil
	}
}

// SetVersionAndKind sets the version and kind fields (with help from
// MetaInsertionFactory). Returns an error if this isn't possible. obj
// must be a pointer.
func (s *Scheme) SetVersionAndKind(version, kind string, obj interface{}) error {
	versionAndKind := s.MetaInsertionFactory.Create(version, kind)
	return s.converter.Convert(versionAndKind, obj, SourceToDest|IgnoreMissingFields|AllowDifferentFieldTypeNames)
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

// Ensures that obj is a pointer of some sort. Returns a reflect.Value of the
// dereferenced pointer, ensuring that it is settable/addressable.
// Returns an error if this is not possible.
func enforcePtr(obj interface{}) (reflect.Value, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		return reflect.Value{}, fmt.Errorf("expected pointer, but got %v", v.Type().Name())
	}
	return v.Elem(), nil
}
