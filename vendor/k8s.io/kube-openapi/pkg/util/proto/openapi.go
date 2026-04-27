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

package proto

import (
	"fmt"
	"sort"
	"strings"
)

// Defines openapi types.
const (
	Integer = "integer"
	Number  = "number"
	String  = "string"
	Boolean = "boolean"

	// These types are private as they should never leak, and are
	// represented by actual structs.
	array  = "array"
	object = "object"
)

// Models interface describe a model provider. They can give you the
// schema for a specific model.
type Models interface {
	LookupModel(string) Schema
	ListModels() []string
}

// SchemaVisitor is an interface that you need to implement if you want
// to "visit" an openapi schema. A dispatch on the Schema type will call
// the appropriate function based on its actual type:
// - Array is a list of one and only one given subtype
// - Map is a map of string to one and only one given subtype
// - Primitive can be string, integer, number and boolean.
// - Kind is an object with specific fields mapping to specific types.
// - Reference is a link to another definition.
type SchemaVisitor interface {
	VisitArray(*Array)
	VisitMap(*Map)
	VisitPrimitive(*Primitive)
	VisitKind(*Kind)
	VisitReference(Reference)
}

// SchemaVisitorArbitrary is an additional visitor interface which handles
// arbitrary types. For backwards compatibility, it's a separate interface
// which is checked for at runtime.
type SchemaVisitorArbitrary interface {
	SchemaVisitor
	VisitArbitrary(*Arbitrary)
}

// Schema is the base definition of an openapi type.
type Schema interface {
	// Giving a visitor here will let you visit the actual type.
	Accept(SchemaVisitor)

	// Pretty print the name of the type.
	GetName() string
	// Describes how to access this field.
	GetPath() *Path
	// Describes the field.
	GetDescription() string
	// Default for that schema.
	GetDefault() interface{}
	// Returns type extensions.
	GetExtensions() map[string]interface{}
}

// Path helps us keep track of type paths
type Path struct {
	parent *Path
	key    string
}

func NewPath(key string) Path {
	return Path{key: key}
}

func (p *Path) Get() []string {
	if p == nil {
		return []string{}
	}
	if p.key == "" {
		return p.parent.Get()
	}
	return append(p.parent.Get(), p.key)
}

func (p *Path) Len() int {
	return len(p.Get())
}

func (p *Path) String() string {
	return strings.Join(p.Get(), "")
}

// ArrayPath appends an array index and creates a new path
func (p *Path) ArrayPath(i int) Path {
	return Path{
		parent: p,
		key:    fmt.Sprintf("[%d]", i),
	}
}

// FieldPath appends a field name and creates a new path
func (p *Path) FieldPath(field string) Path {
	return Path{
		parent: p,
		key:    fmt.Sprintf(".%s", field),
	}
}

// BaseSchema holds data used by each types of schema.
type BaseSchema struct {
	Description string
	Extensions  map[string]interface{}
	Default     interface{}

	Path Path
}

func (b *BaseSchema) GetDescription() string {
	return b.Description
}

func (b *BaseSchema) GetExtensions() map[string]interface{} {
	return b.Extensions
}

func (b *BaseSchema) GetDefault() interface{} {
	return b.Default
}

func (b *BaseSchema) GetPath() *Path {
	return &b.Path
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

func (a *Array) GetName() string {
	return fmt.Sprintf("Array of %s", a.SubType.GetName())
}

// Kind is a complex object. It can have multiple different
// subtypes for each field, as defined in the `Fields` field. Mandatory
// fields are listed in `RequiredFields`. The key of the object is
// always of type `string`.
type Kind struct {
	BaseSchema

	// Lists names of required fields.
	RequiredFields []string
	// Maps field names to types.
	Fields map[string]Schema
	// FieldOrder reports the canonical order for the fields.
	FieldOrder []string
}

var _ Schema = &Kind{}

func (k *Kind) Accept(v SchemaVisitor) {
	v.VisitKind(k)
}

func (k *Kind) GetName() string {
	properties := []string{}
	for key := range k.Fields {
		properties = append(properties, key)
	}
	return fmt.Sprintf("Kind(%v)", properties)
}

// IsRequired returns true if `field` is a required field for this type.
func (k *Kind) IsRequired(field string) bool {
	for _, f := range k.RequiredFields {
		if f == field {
			return true
		}
	}
	return false
}

// Keys returns a alphabetically sorted list of keys.
func (k *Kind) Keys() []string {
	keys := make([]string, 0)
	for key := range k.Fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// Map is an object who values must all be of the same `SubType`.
// The key of the object is always of type `string`.
type Map struct {
	BaseSchema

	SubType Schema
}

var _ Schema = &Map{}

func (m *Map) Accept(v SchemaVisitor) {
	v.VisitMap(m)
}

func (m *Map) GetName() string {
	return fmt.Sprintf("Map of %s", m.SubType.GetName())
}

// Primitive is a literal. There can be multiple types of primitives,
// and this subtype can be visited through the `subType` field.
type Primitive struct {
	BaseSchema

	// Type of a primitive must be one of: integer, number, string, boolean.
	Type   string
	Format string
}

var _ Schema = &Primitive{}

func (p *Primitive) Accept(v SchemaVisitor) {
	v.VisitPrimitive(p)
}

func (p *Primitive) GetName() string {
	if p.Format == "" {
		return p.Type
	}
	return fmt.Sprintf("%s (%s)", p.Type, p.Format)
}

// Arbitrary is a value of any type (primitive, object or array)
type Arbitrary struct {
	BaseSchema
}

var _ Schema = &Arbitrary{}

func (a *Arbitrary) Accept(v SchemaVisitor) {
	if visitor, ok := v.(SchemaVisitorArbitrary); ok {
		visitor.VisitArbitrary(a)
	}
}

func (a *Arbitrary) GetName() string {
	return "Arbitrary value (primitive, object or array)"
}

// Reference implementation depends on the type of document.
type Reference interface {
	Schema

	Reference() string
	SubSchema() Schema
}
