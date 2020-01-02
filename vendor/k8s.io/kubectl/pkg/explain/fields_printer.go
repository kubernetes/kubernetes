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

import "k8s.io/kube-openapi/pkg/util/proto"

// indentDesc is the level of indentation for descriptions.
const indentDesc = 2

// regularFieldsPrinter prints fields with their type and description.
type regularFieldsPrinter struct {
	Writer *Formatter
	Error  error
}

var _ proto.SchemaVisitor = &regularFieldsPrinter{}
var _ fieldsPrinter = &regularFieldsPrinter{}

// VisitArray prints a Array type. It is just a passthrough.
func (f *regularFieldsPrinter) VisitArray(a *proto.Array) {
	a.SubType.Accept(f)
}

// VisitKind prints a Kind type. It prints each key in the kind, with
// the type, the required flag, and the description.
func (f *regularFieldsPrinter) VisitKind(k *proto.Kind) {
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

// VisitMap prints a Map type. It is just a passthrough.
func (f *regularFieldsPrinter) VisitMap(m *proto.Map) {
	m.SubType.Accept(f)
}

// VisitPrimitive prints a Primitive type. It stops the recursion.
func (f *regularFieldsPrinter) VisitPrimitive(p *proto.Primitive) {
	// Nothing to do. Shouldn't really happen.
}

// VisitReference prints a Reference type. It is just a passthrough.
func (f *regularFieldsPrinter) VisitReference(r proto.Reference) {
	r.SubSchema().Accept(f)
}

// PrintFields will write the types from schema.
func (f *regularFieldsPrinter) PrintFields(schema proto.Schema) error {
	schema.Accept(f)
	return f.Error
}
