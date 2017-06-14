/*
Copyright 2017 The Kubernetes Authors.

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

package openapi

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type Resources interface {
	LookupResource(gvk schema.GroupVersionKind) Schema
	ListGroupVersionKinds() []schema.GroupVersionKind
}

// SchemaVisitor is an interface that you need to implement if you want
// to "visit" an openapi schema.
type SchemaVisitor interface {
	VisitArray(*Array)
	VisitSimpleObject(*SimpleObject)
	VisitPrimitive(*Primitive)
	VisitPropertiesMap(*PropertiesMap)
}

// Schema is the base definition of an openapi type.
type Schema interface {
	Accept(SchemaVisitor)

	GetDescription() string
	GetExtensions() map[string]interface{}
}

type BaseSchema struct {
	Description string
	Extensions  map[string]interface{}
}

func (b *BaseSchema) GetDescription() string {
	return b.Description
}

func (b *BaseSchema) GetExtensions() map[string]interface{} {
	return b.Extensions
}

// Array must have all its element of the same `SubType`.
type Array struct {
	BaseSchema

	SubType Schema
}

var _ Schema = &Array{}

func (a *Array) Accept(v SchemaVisitor) {
	v.VisitArray(a)
}

// PropertiesMap is a complex object.  It can have multiple different
// subtypes for each field, as defined in the `Fields` field. Mandatory
// fields are listed in `RequiredFields`.  THe key of the object is
// always of type `string`.
type PropertiesMap struct {
	BaseSchema

	RequiredFields []string
	Fields         map[string]Schema
}

var _ Schema = &PropertiesMap{}

func (m *PropertiesMap) Accept(v SchemaVisitor) {
	v.VisitPropertiesMap(m)
}

// SimpleObject is an object who values must all be of the same `SubType`.
// The key of the object is always of type `string`.
type SimpleObject struct {
	BaseSchema

	SubType Schema
}

var _ Schema = &SimpleObject{}

func (o *SimpleObject) Accept(v SchemaVisitor) {
	v.VisitSimpleObject(o)
}

// Primitive is a literal. There can be multiple types of primitives,
// and this subtype can be visited through the `subType` field.
type Primitive struct {
	BaseSchema

	Type   string
	Format string
}

var _ Schema = &Primitive{}

func (p *Primitive) Accept(v SchemaVisitor) {
	v.VisitPrimitive(p)
}
