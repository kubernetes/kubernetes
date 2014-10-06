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
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"gopkg.in/v1/yaml"
)

// MetaFactory provides methods for retrieving the type and version of API objects.
type MetaFactory struct{}

// Interpret will return the APIVersion and Kind of the given wire-format
// encoding of an APIObject, or an error.
func (MetaFactory) Interpret(data []byte) (version, kind string, err error) {
	findKind := struct {
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, &findKind)
	if err != nil {
		return "", "", fmt.Errorf("couldn't get version/kind: %v", err)
	}
	return findKind.APIVersion, findKind.Kind, nil
}

func (MetaFactory) Update(version, kind string, obj interface{}) error {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return err
	}
	t := v.Type()
	name := t.Name()
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), name, v.Interface())
	}

	var value reflect.Value
	if jsonBase := v.FieldByName("JSONBase"); jsonBase.IsValid() {
		value = jsonBase
	} else if typeBase := v.FieldByName("TypeMeta"); typeBase.IsValid() {
		value = typeBase
	} else {
		return fmt.Errorf("struct %v lacks JSONBase or TypeMeta struct", name)
	}

	if err := setStringValue(value, "APIVersion", version); err != nil {
		return err
	}
	if err := setStringValue(value, "Kind", kind); err != nil {
		return err
	}
	return nil
}

func setStringValue(v reflect.Value, fieldName string, value string) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %#v", fieldName, v.Interface())
	}
	field.SetString(value)
	return nil
}

type ObjectInterface interface {
	TypeMetaInterface
	ObjectMetaInterface
}

var EmptyObjectMeta ObjectMetaInterface = &emptyObjectMeta{}

// FindObjectMeta takes an arbitary type satisfying the ObjectMeta interface
// obj must be a pointer to an api type.
func FindObjectMeta(obj runtime.Object) (ObjectInterface, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	t := v.Type()
	name := t.Name()
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), name, v.Interface())
	}

	if jsonBase := v.FieldByName("JSONBase"); jsonBase.IsValid() {
		json, err := findJSONBase(obj)
		if err != nil {
			return nil, err
		}
		return objectMeta{json, objectMetaFromJSONBase{json}}, nil
	}

	typeBase := v.FieldByName("TypeMeta")
	if !typeBase.IsValid() {
		return nil, fmt.Errorf("struct %v lacks TypeMeta type", name)
	}
	tm, err := newGenericTypeMeta(typeBase)
	if err != nil {
		return nil, err
	}

	var om ObjectMetaInterface
	objectMetaBase := v.FieldByName("Metadata")
	if objectMetaBase.IsValid() {
		obj, err := newGenericObjectMeta(objectMetaBase)
		if err != nil {
			return nil, err
		}
		om = obj
	} else {
		om = EmptyObjectMeta
	}

	return objectMeta{tm, om}, nil
}

// NewJSONBaseResourceVersioner returns a ResourceVersioner that can set or
// retrieve ResourceVersion on objects derived from ObjectMeta.
func NewObjectMetaResourceVersioner() runtime.ResourceVersioner {
	return objectMetaModifier{}
}

// objectMetaModifier implements ResourceVersioner and SelfLinker.
type objectMetaModifier struct{}

func (v objectMetaModifier) ResourceVersion(obj runtime.Object) (uint64, error) {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return 0, err
	}
	version := meta.ResourceVersion()
	return strconv.ParseUint(version, 10, 64)
}

func (v objectMetaModifier) SetResourceVersion(obj runtime.Object, version uint64) error {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return err
	}
	meta.SetResourceVersion(strconv.FormatUint(version, 10))
	return nil
}

func (v objectMetaModifier) Namespace(obj runtime.Object) (string, error) {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return "", err
	}
	return meta.Namespace(), nil
}

func (v objectMetaModifier) Name(obj runtime.Object) (string, error) {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return "", err
	}
	return meta.Name(), nil
}

func (v objectMetaModifier) UID(obj runtime.Object) (string, error) {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return "", err
	}
	return meta.UID(), nil
}

func (v objectMetaModifier) SelfLink(obj runtime.Object) (string, error) {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return "", err
	}
	return meta.SelfLink(), nil
}

func (v objectMetaModifier) SetSelfLink(obj runtime.Object, selfLink string) error {
	meta, err := FindObjectMeta(obj)
	if err != nil {
		return err
	}
	meta.SetSelfLink(selfLink)
	return nil
}

// NewJSONBaseSelfLinker returns a SelfLinker that works on all JSONBase SelfLink fields.
func NewObjectMetaSelfLinker() runtime.SelfLinker {
	return objectMetaModifier{}
}

type TypeMetaInterface interface {
	APIVersion() string
	SetAPIVersion(version string)
	Kind() string
	SetKind(kind string)
}

type genericTypeMeta struct {
	apiVersion *string
	kind       *string
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

func newGenericTypeMeta(v reflect.Value) (genericTypeMeta, error) {
	g := genericTypeMeta{}

	if err := fieldPtr(v, "APIVersion", &g.apiVersion); err != nil {
		return g, err
	}
	if err := fieldPtr(v, "Kind", &g.kind); err != nil {
		return g, err
	}
	return g, nil
}

// ObjectMetaInterface lets you work with a ObjectMeta from any of the versioned or
// internal APIObjects.
type ObjectMetaInterface interface {
	Namespace() string
	SetNamespace(namespace string)
	Name() string
	SetName(name string)
	UID() string
	SetUID(name string)
	ResourceVersion() string
	SetResourceVersion(version string)
	SelfLink() string
	SetSelfLink(selfLink string)
}

type genericObjectMeta struct {
	namespace       *string
	name            *string
	uid             *string
	resourceVersion *string
	selfLink        *string
}

func (g genericObjectMeta) Namespace() string {
	if g.namespace == nil {
		return ""
	}
	return *g.namespace
}

func (g genericObjectMeta) SetNamespace(namespace string) {
	if g.namespace == nil {
		return
	}
	*g.namespace = namespace
}

func (g genericObjectMeta) Name() string {
	if g.name == nil {
		return ""
	}
	return *g.name
}

func (g genericObjectMeta) SetName(name string) {
	if g.name == nil {
		return
	}
	*g.name = name
}

func (g genericObjectMeta) UID() string {
	if g.uid == nil {
		return ""
	}
	return *g.uid
}

func (g genericObjectMeta) SetUID(uid string) {
	if g.uid == nil {
		return
	}
	*g.uid = uid
}

func (g genericObjectMeta) ResourceVersion() string {
	return *g.resourceVersion
}

func (g genericObjectMeta) SetResourceVersion(version string) {
	*g.resourceVersion = version
}

func (g genericObjectMeta) SelfLink() string {
	return *g.selfLink
}

func (g genericObjectMeta) SetSelfLink(selfLink string) {
	*g.selfLink = selfLink
}

// fieldPtr puts the address of fieldName, which must be a member of v,
// into dest, which must be an address of a variable to which this field's
// address can be assigned.
func fieldPtr(v reflect.Value, fieldName string, dest interface{}) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %#v", fieldName, v.Interface())
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
	return fmt.Errorf("couldn't assign/convert %v to %v", field.Type(), v.Type())
}

// newGenericObjectMeta creates a new generic ObjectMeta from v, which must be an
// addressable/setable reflect.Value having the same fields as api.ObjectMeta.
// Returns an error if this isn't the case.
func newGenericObjectMeta(v reflect.Value) (genericObjectMeta, error) {
	g := genericObjectMeta{}

	if err := fieldPtr(v, "ResourceVersion", &g.resourceVersion); err != nil {
		return g, err
	}
	if err := fieldPtr(v, "SelfLink", &g.selfLink); err != nil {
		return g, err
	}

	// these pointers may be nil
	fieldPtr(v, "Namespace", &g.namespace)
	fieldPtr(v, "Name", &g.name)
	fieldPtr(v, "UID", &g.uid)

	return g, nil
}

// objectMeta exposes TypeMetaInterface and ObjectMetaInterface
type objectMeta struct {
	TypeMetaInterface
	ObjectMetaInterface
}

// objectMetaFromJSONBase adapts JSONBase to ObjectMetaInterface
type objectMetaFromJSONBase struct {
	json JSONBaseInterface
}

func (m objectMetaFromJSONBase) Namespace() string {
	return ""
}

func (m objectMetaFromJSONBase) SetNamespace(namespace string) {
}

func (m objectMetaFromJSONBase) Name() string {
	return m.json.ID()
}

func (m objectMetaFromJSONBase) SetName(name string) {
	m.json.SetID(name)
}

func (m objectMetaFromJSONBase) UID() string {
	return ""
}

func (m objectMetaFromJSONBase) SetUID(uid string) {
}

func (m objectMetaFromJSONBase) ResourceVersion() string {
	if m.json.ResourceVersion() == 0 {
		return ""
	}
	return strconv.FormatUint(m.json.ResourceVersion(), 10)
}

func (m objectMetaFromJSONBase) SetResourceVersion(version string) {
	v, err := strconv.ParseUint(version, 10, 64)
	if err != nil {
		//TODO: may need to change signature of SetResourceVersion to handle errors
		v = 0
	}
	m.json.SetResourceVersion(v)
}

func (m objectMetaFromJSONBase) SelfLink() string {
	return m.json.SelfLink()
}

func (m objectMetaFromJSONBase) SetSelfLink(selfLink string) {
	m.json.SetSelfLink(selfLink)
}

type emptyObjectMeta struct{}

func (emptyObjectMeta) Namespace() string {
	return ""
}

func (emptyObjectMeta) SetNamespace(namespace string) {
}

func (emptyObjectMeta) Name() string {
	return ""
}

func (emptyObjectMeta) SetName(name string) {
}

func (emptyObjectMeta) UID() string {
	return ""
}

func (emptyObjectMeta) SetUID(uid string) {
}

func (emptyObjectMeta) ResourceVersion() string {
	return ""
}

func (emptyObjectMeta) SetResourceVersion(version string) {
}

func (emptyObjectMeta) SelfLink() string {
	return ""
}

func (emptyObjectMeta) SetSelfLink(selfLink string) {
}
