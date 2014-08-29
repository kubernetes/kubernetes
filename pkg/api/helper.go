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

package api

import (
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"gopkg.in/v1/yaml"
)

// codec defines methods for serializing and deserializing API
// objects
type codec interface {
	Encode(obj interface{}) (data []byte, err error)
	Decode(data []byte) (interface{}, error)
	DecodeInto(data []byte, obj interface{}) error
}

// resourceVersioner provides methods for setting and retrieving
// the resource version from an API object
type resourceVersioner interface {
	SetResourceVersion(obj interface{}, version uint64) error
	ResourceVersion(obj interface{}) (uint64, error)
}

var Codec codec
var ResourceVersioner resourceVersioner

var conversionScheme = conversion.NewScheme()

func init() {
	conversionScheme.InternalVersion = ""
	conversionScheme.ExternalVersion = "v1beta1"
	conversionScheme.MetaInsertionFactory = metaInsertion{}
	AddKnownTypes("",
		PodList{},
		Pod{},
		ReplicationControllerList{},
		ReplicationController{},
		ServiceList{},
		Service{},
		MinionList{},
		Minion{},
		Status{},
		ServerOpList{},
		ServerOp{},
		ContainerManifestList{},
		Endpoints{},
		Binding{},
	)

	Codec = conversionScheme
	ResourceVersioner = NewJSONBaseResourceVersioner()
}

// AddKnownTypes registers the types of the arguments to the marshaller of the package api.
// Encode() refuses the object unless its type is registered with AddKnownTypes.
func AddKnownTypes(version string, types ...interface{}) {
	conversionScheme.AddKnownTypes(version, types...)
}

// New returns a new API object of the given version ("" for internal
// representation) and name, or an error if it hasn't been registered.
func New(versionName, typeName string) (interface{}, error) {
	return conversionScheme.NewObject(versionName, typeName)
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
func AddConversionFuncs(conversionFuncs ...interface{}) error {
	return conversionScheme.AddConversionFuncs(conversionFuncs...)
}

// Convert will attempt to convert in into out. Both must be pointers to API objects.
// For easy testing of conversion functions. Returns an error if the conversion isn't
// possible.
func Convert(in, out interface{}) error {
	return conversionScheme.Convert(in, out)
}

// FindJSONBase takes an arbitary api type, returns pointer to its JSONBase field.
// obj must be a pointer to an api type.
func FindJSONBase(obj interface{}) (JSONBaseInterface, error) {
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

// FindJSONBaseRO takes an arbitary api type, return a copy of its JSONBase field.
// obj may be a pointer to an api type, or a non-pointer struct api type.
func FindJSONBaseRO(obj interface{}) (JSONBase, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return JSONBase{}, fmt.Errorf("expected struct, but got %v (%#v)", v.Type().Name(), v.Interface())
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return JSONBase{}, fmt.Errorf("struct %v lacks embedded JSON type", v.Type().Name())
	}
	return jsonBase.Interface().(JSONBase), nil
}

// EncodeOrDie is a version of Encode which will panic instead of returning an error. For tests.
func EncodeOrDie(obj interface{}) string {
	return conversionScheme.EncodeOrDie(obj)
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
func Encode(obj interface{}) (data []byte, err error) {
	return conversionScheme.Encode(obj)
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
func Decode(data []byte) (interface{}, error) {
	return conversionScheme.Decode(data)
}

// DecodeInto parses a YAML or JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
// If obj's APIVersion doesn't match that in data, an attempt will be made to convert
// data into obj's version.
func DecodeInto(data []byte, obj interface{}) error {
	return conversionScheme.DecodeInto(data, obj)
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
