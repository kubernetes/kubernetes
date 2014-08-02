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

package internal

import (
	"fmt"
	"reflect"
)

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// AddTypes adds the types in this package to a conversion scheme
func AddTypes(scheme *conversion.Scheme) {
	scheme.MetaInsertionFactory = metaInsertion{}
	scheme.AddKnownTypes("",
		PodList{},
		Pod{},
		ReplicationControllerList{},
		ReplicationController{},
		ServiceList{},
		Service{},
		MinionList{},
		Minion{},
		ContainerManifestList{},
		Endpoints{},
	)
}

// Implements a helper interface for setting and retrieving version info
type Versioning struct {
}

func (v Versioning) ResourceVersion(obj interface{}) (uint64, error) {
	jsonBase, err := FindJSONBase(obj)
	if err != nil {
		return 0, err
	}
	return jsonBase.ResourceVersion(), nil
}

func (v Versioning) SetResourceVersion(obj interface{}, version uint64) error {
	jsonBase, err := FindJSONBase(obj)
	if err != nil {
		return err
	}
	jsonBase.SetResourceVersion(version)
	return nil
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
