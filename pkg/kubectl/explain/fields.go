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

const indentDesc = 2

type regularFields struct {
	Writer *Formatter
	Error  error
}

var _ openapi.SchemaVisitor = &regularFields{}
var _ fields = &regularFields{}

func (f *regularFields) VisitArray(a *openapi.Array) {
	a.SubType.Accept(f)
}

func (f *regularFields) VisitKind(k *openapi.Kind) {
	for _, key := range k.Keys() {
		v := k.Fields[key]
		required := ""
		if k.IsRequired(key) {
			required = " -required-"
		}

		if err := f.Writer.Write("%s\t<%s>%s", key, GetTypeName(v), required); err != nil {
			f.Error = err
			return
		}
		if err := f.Writer.Indent(indentDesc).WriteWrapped("%s", v.GetDescription()); err != nil {
			f.Error = err
			return
		}
		if err := f.Writer.Write(""); err != nil {
			f.Error = err
			return
		}
	}
}

func (f *regularFields) VisitMap(m *openapi.Map) {
	m.SubType.Accept(f)
}

func (f *regularFields) VisitPrimitive(p *openapi.Primitive) {
	// Nothing to do. Shouldn't really happen.
}

func (f *regularFields) VisitReference(r openapi.Reference) {
	r.SubSchema().Accept(f)
}

func (f *regularFields) Print(schema openapi.Schema) error {
	schema.Accept(f)
	return f.Error
}
