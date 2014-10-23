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

package meta

import (
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// FindAccessor takes an arbitary api type, returns pointer to its TypeMeta field.
// obj must be a pointer to an api type.
func FindAccessor(obj runtime.Object) (Accessor, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	t := v.Type()
	name := t.Name()
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), name, v.Interface())
	}
	typeMeta := v.FieldByName("TypeMeta")
	if !typeMeta.IsValid() {
		return nil, fmt.Errorf("struct %v lacks embedded TypeMeta type", name)
	}
	g, err := newGenericTypeMeta(typeMeta)
	if err != nil {
		return nil, err
	}
	return g, nil
}

// NewResourceVersioner returns a ResourceVersioner that can set or
// retrieve ResourceVersion on objects derived from TypeMeta.
func NewResourceVersioner() runtime.ResourceVersioner {
	return typeMetaModifier{}
}

// typeMetaModifier implements ResourceVersioner and SelfLinker.
type typeMetaModifier struct{}

func (v typeMetaModifier) ResourceVersion(obj runtime.Object) (string, error) {
	accessor, err := FindAccessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.ResourceVersion(), nil
}

func (v typeMetaModifier) SetResourceVersion(obj runtime.Object, version string) error {
	accessor, err := FindAccessor(obj)
	if err != nil {
		return err
	}
	accessor.SetResourceVersion(version)
	return nil
}

func (v typeMetaModifier) Name(obj runtime.Object) (string, error) {
	accessor, err := FindAccessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.Name(), nil
}

func (v typeMetaModifier) SelfLink(obj runtime.Object) (string, error) {
	accessor, err := FindAccessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.SelfLink(), nil
}

func (v typeMetaModifier) SetSelfLink(obj runtime.Object, selfLink string) error {
	accessor, err := FindAccessor(obj)
	if err != nil {
		return err
	}
	accessor.SetSelfLink(selfLink)
	return nil
}

// NewSelfLinker returns a SelfLinker that works on all TypeMeta SelfLink fields.
func NewSelfLinker() runtime.SelfLinker {
	return typeMetaModifier{}
}

// Accessor lets you work with object metadata from any of the versioned or
// internal APIruntime.Objects.
type Accessor interface {
	Name() string
	SetName(name string)
	UID() string
	SetUID(uid string)
	APIVersion() string
	SetAPIVersion(version string)
	Kind() string
	SetKind(kind string)
	ResourceVersion() string
	SetResourceVersion(version string)
	SelfLink() string
	SetSelfLink(selfLink string)
}

type genericTypeMeta struct {
	name            *string
	uid             *string
	apiVersion      *string
	kind            *string
	resourceVersion *string
	selfLink        *string
}

func (g genericTypeMeta) Name() string {
	return *g.name
}

func (g genericTypeMeta) SetName(name string) {
	*g.name = name
}

func (g genericTypeMeta) UID() string {
	return *g.uid
}

func (g genericTypeMeta) SetUID(uid string) {
	*g.uid = uid
}

func (g genericTypeMeta) APIVersion() string {
	return *g.apiVersion
}

func (g genericTypeMeta) SetAPIVersion(version string) {
	*g.apiVersion = version
}

func (g genericTypeMeta) Kind() string {
	return *g.kind
}

func (g genericTypeMeta) SetKind(kind string) {
	*g.kind = kind
}

func (g genericTypeMeta) ResourceVersion() string {
	return *g.resourceVersion
}

func (g genericTypeMeta) SetResourceVersion(version string) {
	*g.resourceVersion = version
}

func (g genericTypeMeta) SelfLink() string {
	return *g.selfLink
}

func (g genericTypeMeta) SetSelfLink(selfLink string) {
	*g.selfLink = selfLink
}

// fieldPtr puts the address of fieldName, which must be a member of v,
// into dest, which must be an address of a variable to which this field's
// address can be assigned.
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

// newGenericTypeMeta creates a new generic TypeMeta from v, which must be an
// addressable/setable reflect.Value having the same fields as api.TypeMeta.
// Returns an error if this isn't the case.
func newGenericTypeMeta(v reflect.Value) (genericTypeMeta, error) {
	g := genericTypeMeta{}
	if err := fieldPtr(v, "Name", &g.name); err != nil {
		return g, err
	}
	if err := fieldPtr(v, "UID", &g.uid); err != nil {
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
	if err := fieldPtr(v, "SelfLink", &g.selfLink); err != nil {
		return g, err
	}
	return g, nil
}
