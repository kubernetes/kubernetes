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
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

const indentPerLevel = 3

type recursiveFields struct {
	Writer *Formatter
	Error  error
}

var _ openapi.SchemaVisitor = &recursiveFields{}
var _ fields = &recursiveFields{}

func (f *recursiveFields) VisitArray(a *openapi.Array) {
	a.SubType.Accept(f)
}

func (f *recursiveFields) VisitKind(k *openapi.Kind) {
	for _, key := range k.Keys() {
		v := k.Fields[key]
		f.Writer.Write("%s\t<%s>", key, GetTypeName(v))
		subFields := &recursiveFields{
			Writer: f.Writer.Indent(indentPerLevel),
		}
		if err := subFields.Print(v); err != nil {
			f.Error = err
			return
		}
	}
}

func (f *recursiveFields) VisitMap(m *openapi.Map) {
	m.SubType.Accept(f)
}

func (f *recursiveFields) VisitPrimitive(p *openapi.Primitive) {
	// Nothing to do.
}

func (f *recursiveFields) VisitReference(r openapi.Reference) {
	r.SubSchema().Accept(f)
}

func (f *recursiveFields) Print(schema openapi.Schema) error {
	schema.Accept(f)
	return f.Error
}
