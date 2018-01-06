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

package explain

import (
	"fmt"

	"k8s.io/kube-openapi/pkg/util/proto"
)

// fieldLookup walks through a schema by following a path, and returns
// the final schema.
type fieldLookup struct {
	// Path to walk
	Path []string

	// Return information: Schema found, or error.
	Schema proto.Schema
	Error  error
}

// SaveLeafSchema is used to detect if we are done walking the path, and
// saves the schema as a match.
func (f *fieldLookup) SaveLeafSchema(schema proto.Schema) bool {
	if len(f.Path) != 0 {
		return false
	}

	f.Schema = schema

	return true
}

// VisitArray is mostly a passthrough.
func (f *fieldLookup) VisitArray(a *proto.Array) {
	if f.SaveLeafSchema(a) {
		return
	}

	// Passthrough arrays.
	a.SubType.Accept(f)
}

// VisitMap is mostly a passthrough.
func (f *fieldLookup) VisitMap(m *proto.Map) {
	if f.SaveLeafSchema(m) {
		return
	}

	// Passthrough maps.
	m.SubType.Accept(f)
}

// VisitPrimitive stops the operation and returns itself as the found
// schema, even if it had more path to walk.
func (f *fieldLookup) VisitPrimitive(p *proto.Primitive) {
	// Even if Path is not empty (we're not expecting a leaf),
	// return that primitive.
	f.Schema = p
}

// VisitKind unstacks fields as it finds them.
func (f *fieldLookup) VisitKind(k *proto.Kind) {
	if f.SaveLeafSchema(k) {
		return
	}

	subSchema, ok := k.Fields[f.Path[0]]
	if !ok {
		f.Error = fmt.Errorf("field %q does not exist", f.Path[0])
		return
	}

	f.Path = f.Path[1:]
	subSchema.Accept(f)
}

// VisitReference is mostly a passthrough.
func (f *fieldLookup) VisitReference(r proto.Reference) {
	if f.SaveLeafSchema(r) {
		return
	}

	// Passthrough references.
	r.SubSchema().Accept(f)
}

// LookupSchemaForField looks for the schema of a given path in a base schema.
func LookupSchemaForField(schema proto.Schema, path []string) (proto.Schema, error) {
	f := &fieldLookup{Path: path}
	schema.Accept(f)
	return f.Schema, f.Error
}
