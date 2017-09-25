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

const fieldIndentLevel = 3
const descriptionIndentLevel = 5

type model struct {
	Name         string
	Type         string
	Descriptions []string
	Writer       *Formatter
	Builder      fieldsBuilder
	Error        error
}

var _ openapi.SchemaVisitor = &model{}

func (m *model) PrintDescription(schema openapi.Schema) error {
	if err := m.Writer.Write("DESCRIPTION:"); err != nil {
		return err
	}
	for i, desc := range append(m.Descriptions, schema.GetDescription()) {
		if desc == "" {
			continue
		}
		if i != 0 {
			if err := m.Writer.Write(""); err != nil {
				return err
			}
		}
		if err := m.Writer.Indent(descriptionIndentLevel).WriteWrapped(desc); err != nil {
			return err
		}
	}
	return nil
}

func (m *model) VisitArray(a *openapi.Array) {
	m.Descriptions = append(m.Descriptions, a.GetDescription())
	if m.Type == "" {
		m.Type = GetTypeName(a)
	}
	a.SubType.Accept(m)
}

func (m *model) VisitKind(k *openapi.Kind) {
	if m.Type == "" {
		m.Type = GetTypeName(k)
	}
	if m.Name != "" {
		m.Writer.Write("RESOURCE: %s <%s>\n", m.Name, m.Type)
	}

	if err := m.PrintDescription(k); err != nil {
		m.Error = err
		return
	}
	if err := m.Writer.Write("\nFIELDS:"); err != nil {
		m.Error = err
		return
	}
	m.Error = m.Builder.Build(m.Writer.Indent(fieldIndentLevel)).Print(k)
}

func (m *model) VisitMap(om *openapi.Map) {
	m.Descriptions = append(m.Descriptions, om.GetDescription())
	if m.Type == "" {
		m.Type = GetTypeName(om)
	}
	om.SubType.Accept(m)
}

func (m *model) VisitPrimitive(p *openapi.Primitive) {
	if m.Type == "" {
		m.Type = GetTypeName(p)
	}
	if err := m.Writer.Write("FIELD: %s <%s>\n", m.Name, m.Type); err != nil {
		m.Error = err
		return
	}
	m.Error = m.PrintDescription(p)
}

func (m *model) VisitReference(r openapi.Reference) {
	m.Descriptions = append(m.Descriptions, r.GetDescription())
	r.SubSchema().Accept(m)
}

// PrintModel prints the description of a specific model.
func PrintModel(name string, writer *Formatter, builder fieldsBuilder, schema openapi.Schema) error {
	m := &model{Name: name, Writer: writer, Builder: builder}
	schema.Accept(m)
	return m.Error
}
