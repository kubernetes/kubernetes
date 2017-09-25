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

type typeName struct {
	Name string
}

var _ openapi.SchemaVisitor = &typeName{}

func (t *typeName) VisitArray(a *openapi.Array) {
	s := &typeName{}
	a.SubType.Accept(s)
	t.Name = fmt.Sprintf("[]%s", s.Name)
}

func (t *typeName) VisitKind(k *openapi.Kind) {
	t.Name = "Object"
}

func (t *typeName) VisitMap(m *openapi.Map) {
	s := &typeName{}
	m.SubType.Accept(s)
	t.Name = fmt.Sprintf("map[string]%s", s.Name)
}

func (t *typeName) VisitPrimitive(p *openapi.Primitive) {
	t.Name = p.Type
}

func (t *typeName) VisitReference(r openapi.Reference) {
	r.SubSchema().Accept(t)
}

// GetTypeName returns the type of a schema.
func GetTypeName(schema openapi.Schema) string {
	t := &typeName{}
	schema.Accept(t)
	return t.Name
}
