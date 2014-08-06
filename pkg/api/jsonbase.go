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
)

// NewJSONBaseVersioner returns a resourceVersioner that can set or retrieve
// ResourceVersion on objects derived from JSONBase.
func NewJSONBaseResourceVersioner() resourceVersioner {
	return &jsonBaseResourceVersioner{}
}

type jsonBaseResourceVersioner struct{}

func (v jsonBaseResourceVersioner) ResourceVersion(obj interface{}) (uint64, error) {
	json, err := FindJSONBaseRO(obj)
	if err != nil {
		return 0, err
	}
	return json.ResourceVersion, nil
}

func (v jsonBaseResourceVersioner) SetResourceVersion(obj interface{}, version uint64) error {
	json, err := FindJSONBase(obj)
	if err != nil {
		return err
	}
	json.SetResourceVersion(version)
	return nil
}

// JSONBase lets you work with a JSONBase from any of the versioned or
// internal APIObjects.
type JSONBaseInterface interface {
	ID() string
	SetID(ID string)
	APIVersion() string
	SetAPIVersion(version string)
	Kind() string
	SetKind(kind string)
	ResourceVersion() uint64
	SetResourceVersion(version uint64)
}

type genericJSONBase struct {
	id              *string
	apiVersion      *string
	kind            *string
	resourceVersion *uint64
}

func (g genericJSONBase) ID() string {
	return *g.id
}

func (g genericJSONBase) SetID(id string) {
	*g.id = id
}

func (g genericJSONBase) APIVersion() string {
	return *g.apiVersion
}

func (g genericJSONBase) SetAPIVersion(version string) {
	*g.apiVersion = version
}

func (g genericJSONBase) Kind() string {
	return *g.kind
}

func (g genericJSONBase) SetKind(kind string) {
	*g.kind = kind
}

func (g genericJSONBase) ResourceVersion() uint64 {
	return *g.resourceVersion
}

func (g genericJSONBase) SetResourceVersion(version uint64) {
	*g.resourceVersion = version
}

// fieldPtr puts the address address of fieldName, which must be a member of v,
// into dest, which must be an address of a variable to which this field's address
// can be assigned.
func fieldPtr(v reflect.Value, fieldName string, dest interface{}) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("Couldn't find %v field in %#v", fieldName, v.Interface())
	}
	v = reflect.ValueOf(dest)
	if v.Kind() != reflect.Ptr {
		return fmt.Errorf("dest should be ptr")
	}
	v = v.Elem()
	field = field.Addr()
	if field.Type().AssignableTo(v.Type()) {
		v.Set(field)
		return nil
	}
	if field.Type().ConvertibleTo(v.Type()) {
		v.Set(field.Convert(v.Type()))
		return nil
	}
	return fmt.Errorf("Couldn't assign/convert %v to %v", field.Type(), v.Type())
}

// newGenericJSONBase makes a new generic JSONBase from v, which must be an
// addressable/setable reflect.Value having the same fields as api.JSONBase.
// Returns an error if this isn't the case.
func newGenericJSONBase(v reflect.Value) (genericJSONBase, error) {
	g := genericJSONBase{}
	if err := fieldPtr(v, "ID", &g.id); err != nil {
		return g, err
	}
	if err := fieldPtr(v, "APIVersion", &g.apiVersion); err != nil {
		return g, err
	}
	if err := fieldPtr(v, "Kind", &g.kind); err != nil {
		return g, err
	}
	if err := fieldPtr(v, "ResourceVersion", &g.resourceVersion); err != nil {
		return g, err
	}
	return g, nil
}
