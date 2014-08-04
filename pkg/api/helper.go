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
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"gopkg.in/v1/yaml"
)

// versionMap allows one to figure out the go type of an object with
// the given version and name.
var versionMap = map[string]map[string]reflect.Type{}

// typeToVersion allows one to figure out the version for a given go object.
// The reflect.Type we index by should *not* be a pointer. If the same type
// is registered for multiple versions, the last one wins.
var typeToVersion = map[reflect.Type]string{}

// theConverter stores all registered conversion functions. It also has
// default coverting behavior.
var theConverter = NewConverter()

func init() {
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
	)
	AddKnownTypes("v1beta1",
		v1beta1.PodList{},
		v1beta1.Pod{},
		v1beta1.ReplicationControllerList{},
		v1beta1.ReplicationController{},
		v1beta1.ServiceList{},
		v1beta1.Service{},
		v1beta1.MinionList{},
		v1beta1.Minion{},
		v1beta1.Status{},
		v1beta1.ServerOpList{},
		v1beta1.ServerOp{},
		v1beta1.ContainerManifestList{},
		v1beta1.Endpoints{},
	)

	// TODO: when we get more of this stuff, move to its own file. This is not a
	// good home for lots of conversion functions.
	// TODO: Consider inverting dependency chain-- imagine v1beta1 package
	// registering all of these functions. Then, if you want to be able to understand
	// v1beta1 objects, you just import that package for its side effects.
	AddConversionFuncs(
		// EnvVar's Name is depricated in favor of Key.
		func(in *EnvVar, out *v1beta1.EnvVar) error {
			out.Value = in.Value
			out.Key = in.Name
			out.Name = in.Name
			return nil
		},
		func(in *v1beta1.EnvVar, out *EnvVar) error {
			out.Value = in.Value
			if in.Name != "" {
				out.Name = in.Name
			} else {
				out.Name = in.Key
			}
			return nil
		},
	)
}

// AddKnownTypes registers the types of the arguments to the marshaller of the package api.
// Encode() refuses the object unless its type is registered with AddKnownTypes.
func AddKnownTypes(version string, types ...interface{}) {
	knownTypes, found := versionMap[version]
	if !found {
		knownTypes = map[string]reflect.Type{}
		versionMap[version] = knownTypes
	}
	for _, obj := range types {
		t := reflect.TypeOf(obj)
		if t.Kind() != reflect.Struct {
			panic("All types must be structs.")
		}
		knownTypes[t.Name()] = t
		typeToVersion[t] = version
	}
}

// New returns a new API object of the given version ("" for internal
// representation) and name, or an error if it hasn't been registered.
func New(versionName, typeName string) (interface{}, error) {
	if types, ok := versionMap[versionName]; ok {
		if t, ok := types[typeName]; ok {
			return reflect.New(t).Interface(), nil
		}
		return nil, fmt.Errorf("No type '%v' for version '%v'", typeName, versionName)
	}
	return nil, fmt.Errorf("No version '%v'", versionName)
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
	for _, f := range conversionFuncs {
		err := theConverter.Register(f)
		if err != nil {
			return err
		}
	}
	return nil
}

// Convert will attempt to convert in into out. Both must be pointers to API objects.
// For easy testing of conversion functions. Returns an error if the conversion isn't
// possible.
func Convert(in, out interface{}) error {
	return theConverter.Convert(in, out)
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
	bytes, err := Encode(obj)
	if err != nil {
		panic(err)
	}
	return string(bytes)
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
	obj = maybeCopy(obj)
	obj, err = maybeExternalize(obj)
	if err != nil {
		return nil, err
	}

	jsonBase, err := prepareEncode(obj)
	if err != nil {
		return nil, err
	}

	data, err = json.MarshalIndent(obj, "", "	")
	if err != nil {
		return nil, err
	}
	// Leave these blank in memory.
	jsonBase.SetKind("")
	jsonBase.SetAPIVersion("")
	return data, err
}

// Returns the API version of the go object, or an error if it's not a
// pointer or is unregistered.
func objAPIVersionAndName(obj interface{}) (apiVersion, name string, err error) {
	v, err := enforcePtr(obj)
	if err != nil {
		return "", "", err
	}
	t := v.Type()
	if version, ok := typeToVersion[t]; !ok {
		return "", "", fmt.Errorf("Unregistered type: %v", t)
	} else {
		return version, t.Name(), nil
	}
}

// maybeExternalize converts obj to an external object if it isn't one already.
// obj must be a pointer.
func maybeExternalize(obj interface{}) (interface{}, error) {
	version, _, err := objAPIVersionAndName(obj)
	if err != nil {
		return nil, err
	}
	if version != "" {
		// Object is already of an external versioned type.
		return obj, nil
	}
	return externalize(obj)
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

// prepareEncode sets the APIVersion and Kind fields to match the go type in obj.
// Returns an error if the (version, name) pair isn't registered for the type or
// if the type is an internal, non-versioned object.
func prepareEncode(obj interface{}) (JSONBaseInterface, error) {
	version, name, err := objAPIVersionAndName(obj)
	if err != nil {
		return nil, err
	}
	if version == "" {
		return nil, fmt.Errorf("No version for '%v' (%#v); extremely inadvisable to write it in wire format.", name, obj)
	}
	jsonBase, err := FindJSONBase(obj)
	if err != nil {
		return nil, err
	}
	knownTypes, found := versionMap[version]
	if !found {
		return nil, fmt.Errorf("struct %s, %s won't be unmarshalable because it's not in known versions", version, name)
	}
	if _, contains := knownTypes[name]; !contains {
		return nil, fmt.Errorf("struct %s, %s won't be unmarshalable because it's not in knownTypes", version, name)
	}
	jsonBase.SetAPIVersion(version)
	jsonBase.SetKind(name)
	return jsonBase, nil
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
	version, kind, err := VersionAndKind(data)
	if err != nil {
		return nil, err
	}
	if version == "" {
		return nil, fmt.Errorf("API Version not set in '%s'", string(data))
	}
	obj, err := New(version, kind)
	if err != nil {
		return nil, fmt.Errorf("Unable to create new object of type ('%s', '%s')", version, kind)
	}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}
	obj, err = internalize(obj)
	if err != nil {
		return nil, err
	}
	jsonBase, err := FindJSONBase(obj)
	if err != nil {
		return nil, err
	}
	// Don't leave these set. Type and version info is deducible from go's type.
	jsonBase.SetKind("")
	jsonBase.SetAPIVersion("")
	return obj, nil
}

// DecodeInto parses a YAML or JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
// If obj's APIVersion doesn't match that in data, an attempt will be made to convert
// data into obj's version.
func DecodeInto(data []byte, obj interface{}) error {
	dataVersion, dataKind, err := VersionAndKind(data)
	if err != nil {
		return err
	}
	objVersion, objKind, err := objAPIVersionAndName(obj)
	if err != nil {
		return err
	}
	if dataKind == "" {
		// Assume objects with unset Kind fields are being unmarshalled into the
		// correct type.
		dataKind = objKind
	}
	if dataKind != objKind {
		return fmt.Errorf("data of kind '%v', obj of type '%v'", dataKind, objKind)
	}
	if dataVersion == "" {
		// Assume objects with unset Version fields are being unmarshalled into the
		// correct type.
		dataVersion = objVersion
	}

	if objVersion == dataVersion {
		// Easy case!
		err = yaml.Unmarshal(data, obj)
		if err != nil {
			return err
		}
	} else {
		// TODO: look up in our map to see if we can do this dataVersion -> objVersion
		// conversion.
		if objVersion != "" || dataVersion != "v1beta1" {
			return fmt.Errorf("Can't convert from '%v' to '%v' for type '%v'", dataVersion, objVersion, dataKind)
		}

		external, err := New(dataVersion, dataKind)
		if err != nil {
			return fmt.Errorf("Unable to create new object of type ('%s', '%s')", dataVersion, dataKind)
		}
		// yaml is a superset of json, so we use it to decode here. That way,
		// we understand both.
		err = yaml.Unmarshal(data, external)
		if err != nil {
			return err
		}
		internal, err := internalize(external)
		if err != nil {
			return err
		}
		// Copy to the provided object.
		vObj := reflect.ValueOf(obj)
		vInternal := reflect.ValueOf(internal)
		if !vInternal.Type().AssignableTo(vObj.Type()) {
			return fmt.Errorf("%s is not assignable to %s", vInternal.Type(), vObj.Type())
		}
		vObj.Elem().Set(vInternal.Elem())
	}

	jsonBase, err := FindJSONBase(obj)
	if err != nil {
		return err
	}
	// Don't leave these set. Type and version info is deducible from go's type.
	jsonBase.SetKind("")
	jsonBase.SetAPIVersion("")
	return nil
}

func internalize(obj interface{}) (interface{}, error) {
	_, objKind, err := objAPIVersionAndName(obj)
	if err != nil {
		return nil, err
	}
	objOut, err := New("", objKind)
	if err != nil {
		return nil, err
	}
	err = theConverter.Convert(obj, objOut)
	if err != nil {
		return nil, err
	}
	return objOut, nil
}

func externalize(obj interface{}) (interface{}, error) {
	_, objKind, err := objAPIVersionAndName(obj)
	if err != nil {
		return nil, err
	}
	objOut, err := New("v1beta1", objKind)
	if err != nil {
		return nil, err
	}
	err = theConverter.Convert(obj, objOut)
	if err != nil {
		return nil, err
	}
	return objOut, nil
}
