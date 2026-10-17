/*
Copyright 2026 The Kubernetes Authors.

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

package protobuf

import (
	"bytes"
	"go/format"
	"go/parser"
	"go/token"
	"testing"
)

// Trimmed from protoc-gen-gogo output for a message with interface fields.
const anyFieldsInput = `package foo

import (
	fmt "fmt"
	strings "strings"
	anypb "google.golang.org/protobuf/types/known/anypb"
)

type Holder struct {
	Single *anypb.Any
	List   []*anypb.Any
	Other  *Other
	Others []*Other
}

type Other struct {
	Single *Other
}

func (m *Holder) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	for iNdEx := len(m.List) - 1; iNdEx >= 0; iNdEx-- {
		size, err := m.List[iNdEx].MarshalToSizedBuffer(dAtA[:i])
		_, _ = size, err
	}
	if m.Single != nil {
		size, err := m.Single.MarshalToSizedBuffer(dAtA[:i])
		_, _ = size, err
	}
	if m.Other != nil {
		size, err := m.Other.MarshalToSizedBuffer(dAtA[:i])
		_, _ = size, err
	}
	return 0, nil
}

func (m *Holder) Size() (n int) {
	if m.Single != nil {
		n += m.Single.Size()
	}
	for _, e := range m.List {
		n += e.Size()
	}
	for _, e := range m.Others {
		n += e.Size()
	}
	return n
}

func (this *Holder) String() string {
	repeatedStringForList := "[]*Any{"
	for _, f := range this.List {
		repeatedStringForList += strings.Replace(fmt.Sprintf("%v", f), "Any", "anypb.Any", 1) + ","
	}
	return strings.Replace(fmt.Sprintf("%v", this.Single), "Any", "anypb.Any", 1) +
		strings.Replace(fmt.Sprintf("%v", this.Other), "Other", "Other", 1)
}

func (m *Holder) Unmarshal(dAtA []byte) error {
	switch fieldNum {
	case 1:
		if m.Single == nil {
			m.Single = &anypb.Any{}
		}
		if err := m.Single.Unmarshal(dAtA); err != nil {
			return err
		}
	case 2:
		m.List = append(m.List, &anypb.Any{})
		if err := m.List[len(m.List)-1].Unmarshal(dAtA); err != nil {
			return err
		}
	case 3:
		if m.Other == nil {
			m.Other = &Other{}
		}
		if err := m.Other.Unmarshal(dAtA); err != nil {
			return err
		}
	}
	return nil
}

func (m *Other) Size() (n int) {
	return m.Single.Size()
}
`

const anyFieldsExpected = `package foo

import (
	fmt "fmt"
	protoany "k8s.io/apimachinery/pkg/runtime/protoany"
	strings "strings"
)

type Holder struct {
	Single *anypb.Any
	List   []*anypb.Any
	Other  *Other
	Others []*Other
}

type Other struct {
	Single *Other
}

func (m *Holder) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	for iNdEx := len(m.List) - 1; iNdEx >= 0; iNdEx-- {
		size, err := protoany.MarshalToSizedBuffer(m.List[iNdEx], dAtA[:i])
		_, _ = size, err
	}
	if m.Single != nil {
		size, err := protoany.MarshalToSizedBuffer(m.Single, dAtA[:i])
		_, _ = size, err
	}
	if m.Other != nil {
		size, err := m.Other.MarshalToSizedBuffer(dAtA[:i])
		_, _ = size, err
	}
	return 0, nil
}

func (m *Holder) Size() (n int) {
	if m.Single != nil {
		n += protoany.Size(m.Single)
	}
	for _, e := range m.List {
		n += protoany.Size(e)
	}
	for _, e := range m.Others {
		n += e.Size()
	}
	return n
}

func (this *Holder) String() string {
	repeatedStringForList := "[]*Any{"
	for _, f := range this.List {
		repeatedStringForList += fmt.Sprintf("%v", f) + ","
	}
	return fmt.Sprintf("%v", this.Single) +
		strings.Replace(fmt.Sprintf("%v", this.Other), "Other", "Other", 1)
}

func (m *Holder) Unmarshal(dAtA []byte) error {
	switch fieldNum {
	case 1:

		if err := protoany.Unmarshal(dAtA, &m.Single); err != nil {
			return err
		}
	case 2:
		m.List = append(m.List, nil)
		if err := protoany.Unmarshal(dAtA, &m.List[len(m.List)-1]); err != nil {
			return err
		}
	case 3:
		if m.Other == nil {
			m.Other = &Other{}
		}
		if err := m.Other.Unmarshal(dAtA); err != nil {
			return err
		}
	}
	return nil
}

func (m *Other) Size() (n int) {
	return m.Single.Size()
}
`

func TestRewriteAnyFields(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "generated.pb.go", anyFieldsInput, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	if !rewriteAnyFields(file) {
		t.Fatal("expected file to be rewritten")
	}
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != anyFieldsExpected {
		t.Errorf("unexpected output:\n%s", got)
	}
}

func TestRewriteAnyFieldsNoop(t *testing.T) {
	for name, src := range map[string]string{
		"no any import":           "package foo\n\ntype Holder struct{ A *Other }\n",
		"import unused by fields": "package foo\n\nimport anypb \"google.golang.org/protobuf/types/known/anypb\"\n\ntype Holder struct{ A *Other }\n",
	} {
		t.Run(name, func(t *testing.T) {
			file, err := parser.ParseFile(token.NewFileSet(), "generated.pb.go", src, 0)
			if err != nil {
				t.Fatal(err)
			}
			if rewriteAnyFields(file) {
				t.Error("expected no rewrite")
			}
		})
	}
}
