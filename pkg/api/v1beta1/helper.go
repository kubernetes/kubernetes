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

	"gopkg.in/v1/yaml"
)

var knownTypes = map[string]reflect.Type{}

func init() {
	AddKnownTypes(
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
	)
}

// AddKnownTypes registers the types of the arguments to the marshaller of the package api.
// Encode() refuses the object unless its type is registered with AddKnownTypes.
func AddKnownTypes(types ...interface{}) {
	for _, obj := range types {
		t := reflect.TypeOf(obj)
		knownTypes[t.Name()] = t
	}
}

// FindJSONBase takes an arbitary api type, returns pointer to its JSONBase field.
// obj must be a pointer to an api type.
func FindJSONBase(obj interface{}) (*JSONBase, error) {
	_, jsonBase, err := nameAndJSONBase(obj)
	return jsonBase, err
}

// FindJSONBaseRO takes an arbitary api type, return a copy of its JSONBase field.
// obj may be a pointer to an api type, or a non-pointer struct api type.
func FindJSONBaseRO(obj interface{}) (JSONBase, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return JSONBase{}, fmt.Errorf("expected struct, but got %v", v.Type().Name())
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return JSONBase{}, fmt.Errorf("struct %v lacks embedded JSON type", v.Type().Name())
	}
	return jsonBase.Interface().(JSONBase), nil
}

// Encode turns the given api object into an appropriate JSON string.
// Will return an error if the object doesn't have an embedded JSONBase.
// Obj may be a pointer to a struct, or a struct. If a struct, a copy
// will be made so that the object's Kind field can be set. If a pointer,
// we change the Kind field, marshal, and then set the kind field back to
// "". Having to keep track of the kind field makes tests very annoying,
// so the rule is it's set only in wire format (json), not when in native
// format.
func Encode(obj interface{}) (data []byte, err error) {
	obj = checkPtr(obj)
	jsonBase, err := prepareEncode(obj)
	if err != nil {
		return nil, err
	}
	data, err = json.MarshalIndent(obj, "", "	")
	jsonBase.Kind = ""
	return data, err
}

func checkPtr(obj interface{}) interface{} {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		return obj
	}
	v2 := reflect.New(v.Type())
	v2.Elem().Set(v)
	return v2.Interface()
}

func prepareEncode(obj interface{}) (*JSONBase, error) {
	name, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return nil, err
	}
	if _, contains := knownTypes[name]; !contains {
		return nil, fmt.Errorf("struct %v won't be unmarshalable because it's not in knownTypes", name)
	}
	jsonBase.Kind = name
	return jsonBase, nil
}

// Returns the name of the type (sans pointer), and its kind field. Takes pointer-to-struct..
func nameAndJSONBase(obj interface{}) (string, *JSONBase, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		return "", nil, fmt.Errorf("expected pointer, but got %v", v.Type().Name())
	}
	v = v.Elem()
	name := v.Type().Name()
	if v.Kind() != reflect.Struct {
		return "", nil, fmt.Errorf("expected struct, but got %v", name)
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return "", nil, fmt.Errorf("struct %v lacks embedded JSON type", name)
	}
	return name, jsonBase.Addr().Interface().(*JSONBase), nil
}

// Decode converts a JSON string back into a pointer to an api object. Deduces the type
// based upon the Kind field (set by encode).
func Decode(data []byte) (interface{}, error) {
	findKind := struct {
		Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way, we understand both.
	err := yaml.Unmarshal(data, &findKind)
	if err != nil {
		return nil, fmt.Errorf("couldn't get kind: %#v", err)
	}
	objType, found := knownTypes[findKind.Kind]
	if !found {
		return nil, fmt.Errorf("%v is not a known type", findKind.Kind)
	}
	obj := reflect.New(objType).Interface()
	err = yaml.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}
	_, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return nil, err
	}
	// Don't leave these set. Track type with go's type.
	jsonBase.Kind = ""
	return obj, nil
}

// DecodeInto parses a JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
func DecodeInto(data []byte, obj interface{}) error {
	err := yaml.Unmarshal(data, obj)
	if err != nil {
		return err
	}
	name, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return err
	}
	if jsonBase.Kind != "" && jsonBase.Kind != name {
		return fmt.Errorf("data had kind %v, but passed object was of type %v", jsonBase.Kind, name)
	}
	// Don't leave these set. Track type with go's type.
	jsonBase.Kind = ""
	return nil
}
