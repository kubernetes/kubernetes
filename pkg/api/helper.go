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
)

var knownTypes = map[string]reflect.Type{}

func init() {
	types := []interface{}{
		PodList{}, Pod{}, ReplicationControllerList{},
		ReplicationController{}, ServiceList{}, Service{},
	}
	for _, obj := range types {
		t := reflect.TypeOf(obj)
		knownTypes[t.Name()] = t
	}
}

// Returns the name of the type (sans pointer), and its kind field. Takes pointer-to-struct..
func nameAndJSONBase(obj interface{}) (string, reflect.Value, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		return "", reflect.Value{}, fmt.Errorf("expected pointer, but got %v", v.Type().Name())
	}
	v = v.Elem()
	name := v.Type().Name()
	if v.Kind() != reflect.Struct {
		return "", reflect.Value{}, fmt.Errorf("expected struct, but got %v", name)
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return "", reflect.Value{}, fmt.Errorf("struct %v lacks embedded JSON type", name)
	}
	return name, jsonBase, nil
}

// Encode turns the given api object into an appropriate JSON string.
// Will return an error if the object doesn't have an embedded JSONBase.
// Obj must be a pointer to a struct. Note, this sets the object's Kind
// field.
func Encode(obj interface{}) (data []byte, err error) {
	name, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return nil, err
	}
	if _, contains := knownTypes[name]; !contains {
		return nil, fmt.Errorf("struct %v can't be unmarshalled because it's not in knownTypes", name)
	}
	jsonBase.FieldByName("Kind").SetString(name)
	return json.Marshal(obj)
}

// Decode converts a JSON string back into a pointer to an api object. Deduces the type
// based upon the Kind field (set by encode).
func Decode(data []byte) (interface{}, error) {
	findKind := struct {
		Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`
	}{}
	err := json.Unmarshal(data, &findKind)
	if err != nil {
		return nil, fmt.Errorf("Couldn't get kind: %#v", err)
	}
	objType, found := knownTypes[findKind.Kind]
	if !found {
		return nil, fmt.Errorf("%v is not a known type", findKind.Kind)
	}
	obj := reflect.New(objType).Interface()
	err = json.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}
	return obj, nil
}

// DecodeInto parses a JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
func DecodeInto(data []byte, obj interface{}) error {
	err := json.Unmarshal(data, obj)
	if err != nil {
		return err
	}
	name, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return err
	}
	foundName := jsonBase.FieldByName("Kind").Interface().(string)
	if foundName == "" {
		jsonBase.FieldByName("Kind").SetString(name)
	} else if foundName != name {
		return fmt.Errorf("data had kind %v, but passed object was of type %v", foundName, name)
	}
	return nil
}
