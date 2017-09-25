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

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

type walker struct {
	// Path to walk
	Path []string

	// Return information: Schema found, or error.
	Schema openapi.Schema
	Error  error
}

func (w *walker) SaveLeafSchema(schema openapi.Schema) bool {
	if len(w.Path) != 0 {
		return false
	}

	w.Schema = schema

	return true
}

func (w *walker) VisitArray(a *openapi.Array) {
	if w.SaveLeafSchema(a) {
		return
	}

	// Passthrough arrays.
	a.SubType.Accept(w)
}

func (w *walker) VisitMap(m *openapi.Map) {
	if w.SaveLeafSchema(m) {
		return
	}

	// Passthrough maps.
	m.SubType.Accept(w)
}

func (w *walker) VisitPrimitive(p *openapi.Primitive) {
	// Even if Path is not empty (we're not expecting a leaf),
	// return that primitive.
	w.Schema = p
}

func (w *walker) VisitKind(k *openapi.Kind) {
	if w.SaveLeafSchema(k) {
		return
	}

	subSchema, ok := k.Fields[w.Path[0]]
	if !ok {
		w.Error = fmt.Errorf("field %q does not exist", w.Path[0])
		return
	}

	w.Path = w.Path[1:]
	subSchema.Accept(w)
}

func (w *walker) VisitReference(r openapi.Reference) {
	if w.SaveLeafSchema(r) {
		return
	}

	// Passthrough references.
	r.SubSchema().Accept(w)
}

// FindFieldSchema looks for the schema of a given path in a base schema.
func FindFieldSchema(schema openapi.Schema, path []string) (openapi.Schema, error) {
	w := &walker{Path: path}
	schema.Accept(w)
	return w.Schema, w.Error
}
