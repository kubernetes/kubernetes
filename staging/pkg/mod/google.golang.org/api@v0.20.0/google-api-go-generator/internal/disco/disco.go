// Copyright 2016 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package disco represents Google API discovery documents.
package disco

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// A Document is an API discovery document.
type Document struct {
	ID                string             `json:"id"`
	Name              string             `json:"name"`
	Version           string             `json:"version"`
	Title             string             `json:"title"`
	RootURL           string             `json:"rootUrl"`
	ServicePath       string             `json:"servicePath"`
	BasePath          string             `json:"basePath"`
	DocumentationLink string             `json:"documentationLink"`
	Auth              Auth               `json:"auth"`
	Features          []string           `json:"features"`
	Methods           MethodList         `json:"methods"`
	Schemas           map[string]*Schema `json:"schemas"`
	Resources         ResourceList       `json:"resources"`
}

// init performs additional initialization and checks that
// were not done during unmarshaling.
func (d *Document) init() error {
	schemasByID := map[string]*Schema{}
	for _, s := range d.Schemas {
		schemasByID[s.ID] = s
	}
	for name, s := range d.Schemas {
		if s.Ref != "" {
			return fmt.Errorf("top level schema %q is a reference", name)
		}
		s.Name = name
		if err := s.init(schemasByID); err != nil {
			return err
		}
	}
	for _, m := range d.Methods {
		if err := m.init(schemasByID); err != nil {
			return err
		}
	}
	for _, r := range d.Resources {
		if err := r.init("", schemasByID); err != nil {
			return err
		}
	}
	return nil
}

// NewDocument unmarshals the bytes into a Document.
// It also validates the document to make sure it is error-free.
func NewDocument(bytes []byte) (*Document, error) {
	// The discovery service returns JSON with this format if there's an error, e.g.
	// the document isn't found.
	var errDoc struct {
		Error struct {
			Code    int
			Message string
			Status  string
		}
	}
	if err := json.Unmarshal(bytes, &errDoc); err == nil && errDoc.Error.Code != 0 {
		return nil, fmt.Errorf("bad discovery doc: %+v", errDoc.Error)
	}

	var doc Document
	if err := json.Unmarshal(bytes, &doc); err != nil {
		return nil, err
	}
	if err := doc.init(); err != nil {
		return nil, err
	}
	return &doc, nil
}

// Auth represents the auth section of a discovery document.
// Only OAuth2 information is retained.
type Auth struct {
	OAuth2Scopes []Scope
}

// A Scope is an OAuth2 scope.
type Scope struct {
	ID          string
	Description string
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (a *Auth) UnmarshalJSON(data []byte) error {
	// Pull out the oauth2 scopes and turn them into nice structs.
	// Ignore other auth information.
	var m struct {
		OAuth2 struct {
			Scopes map[string]struct {
				Description string
			}
		}
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	// Sort keys to provide a deterministic ordering, mainly for testing.
	for _, k := range sortedKeys(m.OAuth2.Scopes) {
		a.OAuth2Scopes = append(a.OAuth2Scopes, Scope{
			ID:          k,
			Description: m.OAuth2.Scopes[k].Description,
		})
	}
	return nil
}

// A Schema holds a JSON Schema as defined by
// https://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.1.
// We only support the subset of JSON Schema needed for Google API generation.
type Schema struct {
	ID                   string // union types not supported
	Type                 string // union types not supported
	Format               string
	Description          string
	Properties           PropertyList
	ItemSchema           *Schema `json:"items"` // array of schemas not supported
	AdditionalProperties *Schema // boolean not supported
	Ref                  string  `json:"$ref"`
	Default              string
	Pattern              string
	Enums                []string `json:"enum"`
	// Google extensions to JSON Schema
	EnumDescriptions []string
	Variant          *Variant

	RefSchema *Schema `json:"-"` // Schema referred to by $ref
	Name      string  `json:"-"` // Schema name, if top level
	Kind      Kind    `json:"-"`
}

type Variant struct {
	Discriminant string
	Map          []*VariantMapItem
}

type VariantMapItem struct {
	TypeValue string `json:"type_value"`
	Ref       string `json:"$ref"`
}

func (s *Schema) init(topLevelSchemas map[string]*Schema) error {
	if s == nil {
		return nil
	}
	var err error
	if s.Ref != "" {
		if s.RefSchema, err = resolveRef(s.Ref, topLevelSchemas); err != nil {
			return err
		}
	}
	s.Kind, err = s.initKind()
	if err != nil {
		return err
	}
	if s.Kind == ArrayKind && s.ItemSchema == nil {
		return fmt.Errorf("schema %+v: array does not have items", s)
	}
	if s.Kind != ArrayKind && s.ItemSchema != nil {
		return fmt.Errorf("schema %+v: non-array has items", s)
	}
	if err := s.AdditionalProperties.init(topLevelSchemas); err != nil {
		return err
	}
	if err := s.ItemSchema.init(topLevelSchemas); err != nil {
		return err
	}
	for _, p := range s.Properties {
		if err := p.Schema.init(topLevelSchemas); err != nil {
			return err
		}
	}
	return nil
}

func resolveRef(ref string, topLevelSchemas map[string]*Schema) (*Schema, error) {
	rs, ok := topLevelSchemas[ref]
	if !ok {
		return nil, fmt.Errorf("could not resolve schema reference %q", ref)
	}
	return rs, nil
}

func (s *Schema) initKind() (Kind, error) {
	if s.Ref != "" {
		return ReferenceKind, nil
	}
	switch s.Type {
	case "string", "number", "integer", "boolean", "any":
		return SimpleKind, nil
	case "object":
		if s.AdditionalProperties != nil {
			if s.AdditionalProperties.Type == "any" {
				return AnyStructKind, nil
			}
			return MapKind, nil
		}
		return StructKind, nil
	case "array":
		return ArrayKind, nil
	default:
		return 0, fmt.Errorf("unknown type %q for schema %q", s.Type, s.ID)
	}
}

// ElementSchema returns the schema for the element type of s. For maps,
// this is the schema of the map values. For arrays, it is the schema
// of the array item type.
//
// ElementSchema panics if called on a schema that is not of kind map or array.
func (s *Schema) ElementSchema() *Schema {
	switch s.Kind {
	case MapKind:
		return s.AdditionalProperties
	case ArrayKind:
		return s.ItemSchema
	default:
		panic("ElementSchema called on schema of type " + s.Type)
	}
}

// IsIntAsString reports whether the schema represents an integer value
// formatted as a string.
func (s *Schema) IsIntAsString() bool {
	return s.Type == "string" && strings.Contains(s.Format, "int")
}

// Kind classifies a Schema.
type Kind int

const (
	// SimpleKind is the category for any JSON Schema that maps to a
	// primitive Go type: strings, numbers, booleans, and "any" (since it
	// maps to interface{}).
	SimpleKind Kind = iota

	// StructKind is the category for a JSON Schema that declares a JSON
	// object without any additional (arbitrary) properties.
	StructKind

	// MapKind is the category for a JSON Schema that declares a JSON
	// object with additional (arbitrary) properties that have a non-"any"
	// schema type.
	MapKind

	// AnyStructKind is the category for a JSON Schema that declares a
	// JSON object with additional (arbitrary) properties that can be any
	// type.
	AnyStructKind

	// ArrayKind is the category for a JSON Schema that declares an
	// "array" type.
	ArrayKind

	// ReferenceKind is the category for a JSON Schema that is a reference
	// to another JSON Schema.  During code generation, these references
	// are resolved using the API.schemas map.
	// See https://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.28
	// for more details on the format.
	ReferenceKind
)

type Property struct {
	Name   string
	Schema *Schema
}

type PropertyList []*Property

func (pl *PropertyList) UnmarshalJSON(data []byte) error {
	// In the discovery doc, properties are a map. Convert to a list.
	var m map[string]*Schema
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	for _, k := range sortedKeys(m) {
		*pl = append(*pl, &Property{
			Name:   k,
			Schema: m[k],
		})
	}
	return nil
}

type ResourceList []*Resource

func (rl *ResourceList) UnmarshalJSON(data []byte) error {
	// In the discovery doc, resources are a map. Convert to a list.
	var m map[string]*Resource
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	for _, k := range sortedKeys(m) {
		r := m[k]
		r.Name = k
		*rl = append(*rl, r)
	}
	return nil
}

// A Resource holds information about a Google API Resource.
type Resource struct {
	Name      string
	FullName  string // {parent.FullName}.{Name}
	Methods   MethodList
	Resources ResourceList
}

func (r *Resource) init(parentFullName string, topLevelSchemas map[string]*Schema) error {
	r.FullName = fmt.Sprintf("%s.%s", parentFullName, r.Name)
	for _, m := range r.Methods {
		if err := m.init(topLevelSchemas); err != nil {
			return err
		}
	}
	for _, r2 := range r.Resources {
		if err := r2.init(r.FullName, topLevelSchemas); err != nil {
			return err
		}
	}
	return nil
}

type MethodList []*Method

func (ml *MethodList) UnmarshalJSON(data []byte) error {
	// In the discovery doc, resources are a map. Convert to a list.
	var m map[string]*Method
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	for _, k := range sortedKeys(m) {
		meth := m[k]
		meth.Name = k
		*ml = append(*ml, meth)
	}
	return nil
}

// A Method holds information about a resource method.
type Method struct {
	Name                  string
	ID                    string
	Path                  string
	HTTPMethod            string
	Description           string
	Parameters            ParameterList
	ParameterOrder        []string
	Request               *Schema
	Response              *Schema
	Scopes                []string
	MediaUpload           *MediaUpload
	SupportsMediaDownload bool

	JSONMap map[string]interface{} `json:"-"`
}

type MediaUpload struct {
	Accept    []string
	MaxSize   string
	Protocols map[string]Protocol
}

type Protocol struct {
	Multipart bool
	Path      string
}

func (m *Method) init(topLevelSchemas map[string]*Schema) error {
	if err := m.Request.init(topLevelSchemas); err != nil {
		return err
	}
	if err := m.Response.init(topLevelSchemas); err != nil {
		return err
	}
	return nil
}

func (m *Method) UnmarshalJSON(data []byte) error {
	type T Method // avoid a recursive call to UnmarshalJSON
	if err := json.Unmarshal(data, (*T)(m)); err != nil {
		return err
	}
	// Keep the unmarshalled map around, because the generator
	// outputs it as a comment after the method body.
	// TODO(jba): make this unnecessary.
	return json.Unmarshal(data, &m.JSONMap)
}

type ParameterList []*Parameter

func (pl *ParameterList) UnmarshalJSON(data []byte) error {
	// In the discovery doc, resources are a map. Convert to a list.
	var m map[string]*Parameter
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	for _, k := range sortedKeys(m) {
		p := m[k]
		p.Name = k
		*pl = append(*pl, p)
	}
	return nil
}

// A Parameter holds information about a method parameter.
type Parameter struct {
	Name string
	Schema
	Required bool
	Repeated bool
	Location string
}

// sortedKeys returns the keys of m, which must be a map[string]T, in sorted order.
func sortedKeys(m interface{}) []string {
	vkeys := reflect.ValueOf(m).MapKeys()
	var keys []string
	for _, vk := range vkeys {
		keys = append(keys, vk.Interface().(string))
	}
	sort.Strings(keys)
	return keys
}
