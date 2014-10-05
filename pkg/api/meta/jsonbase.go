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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// findJSONBase takes an arbitary api type, returns pointer to its JSONBase field.
// obj must be a pointer to an api type.
//
// DEPRECATED: Will be removed when support for v1beta2 is dropped
func findJSONBase(obj runtime.Object) (JSONBaseInterface, error) {
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

// NewJSONBaseResourceVersioner returns a ResourceVersioner that can set or
// retrieve ResourceVersion on objects derived from JSONBase.
//
// DEPRECATED: Will be removed when support for v1beta2 is dropped
func NewJSONBaseResourceVersioner() runtime.ResourceVersioner {
	return jsonBaseModifier{}
}

// jsonBaseModifier implements ResourceVersioner and SelfLinker.
type jsonBaseModifier struct{}

func (v jsonBaseModifier) ResourceVersion(obj runtime.Object) (uint64, error) {
	json, err := findJSONBase(obj)
	if err != nil {
		return 0, err
	}
	return json.ResourceVersion(), nil
}

func (v jsonBaseModifier) SetResourceVersion(obj runtime.Object, version uint64) error {
	json, err := findJSONBase(obj)
	if err != nil {
		return err
	}
	json.SetResourceVersion(version)
	return nil
}

func (v jsonBaseModifier) ID(obj runtime.Object) (string, error) {
	json, err := findJSONBase(obj)
	if err != nil {
		return "", err
	}
	return json.ID(), nil
}

func (v jsonBaseModifier) SelfLink(obj runtime.Object) (string, error) {
	json, err := findJSONBase(obj)
	if err != nil {
		return "", err
	}
	return json.SelfLink(), nil
}

func (v jsonBaseModifier) SetSelfLink(obj runtime.Object, selfLink string) error {
	json, err := findJSONBase(obj)
	if err != nil {
		return err
	}
	json.SetSelfLink(selfLink)
	return nil
}

// NewJSONBaseSelfLinker returns a SelfLinker that works on all JSONBase SelfLink fields.
func NewJSONBaseSelfLinker() runtime.SelfLinker {
	return jsonBaseModifier{}
}

// JSONBaseInterface lets you work with a JSONBase from any of the versioned or
// internal APIObjects.
//
// DEPRECATED: Will be removed when support for v1beta2 is dropped.
type JSONBaseInterface interface {
	ID() string
	SetID(ID string)
	APIVersion() string
	SetAPIVersion(version string)
	Kind() string
	SetKind(kind string)
	ResourceVersion() uint64
	SetResourceVersion(version uint64)
	SelfLink() string
	SetSelfLink(selfLink string)
}

type genericJSONBase struct {
	id              *string
	apiVersion      *string
	kind            *string
	resourceVersion *uint64
	selfLink        *string
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

func (g genericJSONBase) SelfLink() string {
	return *g.selfLink
}

func (g genericJSONBase) SetSelfLink(selfLink string) {
	*g.selfLink = selfLink
}

// newGenericJSONBase creates a new generic JSONBase from v, which must be an
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
	if err := fieldPtr(v, "SelfLink", &g.selfLink); err != nil {
		return g, err
	}
	return g, nil
}
