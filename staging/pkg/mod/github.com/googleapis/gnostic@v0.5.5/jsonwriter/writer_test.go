// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package jsonwriter_test

import (
	"testing"

	"github.com/googleapis/gnostic/compiler"
	"github.com/googleapis/gnostic/jsonwriter"

	"gopkg.in/yaml.v3"
)

type MarshalTestCase struct {
	Name     string
	Node     *yaml.Node
	Expected string
	Err      bool
}

func TestMarshal(t *testing.T) {
	t.Parallel()
	tests := []*MarshalTestCase{
		scalarNodeTestCase(),
		scalarBoolTestCase(),
		scalarFloatTestCase(),
		scalarIntTestCase(),
		sequenceStringArrayTestCase(),
		sequenceBoolArrayTestCase(),
		sequenceFloatArrayTestCase(),
		sequenceIntArrayTestCase(),
		sequenceSequenceStringArrayTestCase(),
		sequenceMappingNodeTestCase(),
		mappingNodeTestCase(),
		documentNodeTestCase(),
		aliasNodeTestCase(),
	}

	for _, test := range tests {
		t.Run(test.Name, func(s *testing.T) {
			b, err := jsonwriter.Marshal(test.Node)
			if err != nil && !test.Err {
				s.Errorf("expected %v to be nil", err)
			}
			if err == nil && test.Err {
				s.Error("expected error")
			}
			if string(b) != test.Expected {
				s.Errorf("expected %v to equal %v", string(b), test.Expected)
			}
		})
	}
}

func scalarNodeTestCase() *MarshalTestCase {
	return &MarshalTestCase{
		Name:     "scalar string",
		Node:     compiler.NewScalarNodeForString("expected"),
		Expected: "\"expected\"\n",
	}
}

func scalarBoolTestCase() *MarshalTestCase {
	return &MarshalTestCase{
		Name:     "scalar bool",
		Node:     compiler.NewScalarNodeForBool(true),
		Expected: "true\n",
	}
}

func scalarFloatTestCase() *MarshalTestCase {
	return &MarshalTestCase{
		Name:     "scalar float",
		Node:     compiler.NewScalarNodeForFloat(42.1),
		Expected: "42.1\n",
	}
}

func scalarIntTestCase() *MarshalTestCase {
	return &MarshalTestCase{
		Name:     "scalar int",
		Node:     compiler.NewScalarNodeForInt(42),
		Expected: "42\n",
	}
}

func sequenceStringArrayTestCase() *MarshalTestCase {
	return &MarshalTestCase{
		Name:     "sequence string array",
		Node:     compiler.NewSequenceNodeForStringArray([]string{"a", "b", "c"}),
		Expected: "[\n  \"a\",\n  \"b\",\n  \"c\"\n]\n",
	}
}

func sequenceBoolArrayTestCase() *MarshalTestCase {
	node := compiler.NewSequenceNode()
	for _, b := range []bool{true, false, true} {
		node.Content = append(node.Content, compiler.NewScalarNodeForBool(b))
	}
	return &MarshalTestCase{
		Name:     "sequence bool array",
		Node:     node,
		Expected: "[\n  true,\n  false,\n  true\n]\n",
	}
}

func sequenceFloatArrayTestCase() *MarshalTestCase {
	node := compiler.NewSequenceNode()
	for _, f := range []float64{1.1, 2.2, 3.3} {
		node.Content = append(node.Content, compiler.NewScalarNodeForFloat(f))
	}
	return &MarshalTestCase{
		Name:     "sequence float array",
		Node:     node,
		Expected: "[\n  1.1,\n  2.2,\n  3.3\n]\n",
	}
}

func sequenceIntArrayTestCase() *MarshalTestCase {
	node := compiler.NewSequenceNode()
	for _, i := range []int64{1, 2, 3} {
		node.Content = append(node.Content, compiler.NewScalarNodeForInt(i))
	}
	return &MarshalTestCase{
		Name:     "sequence int array",
		Node:     node,
		Expected: "[\n  1,\n  2,\n  3\n]\n",
	}
}

func sequenceSequenceStringArrayTestCase() *MarshalTestCase {
	node := compiler.NewSequenceNode()
	node.Content = append(node.Content, compiler.NewSequenceNodeForStringArray([]string{"a", "b", "c"}))
	node.Content = append(node.Content, compiler.NewSequenceNodeForStringArray([]string{"e", "f"}))
	return &MarshalTestCase{
		Name:     "sequence sequence string array",
		Node:     node,
		Expected: "[\n  [\n    \"a\",\n    \"b\",\n    \"c\"\n  ],\n  [\n    \"e\",\n    \"f\"\n  ]\n]\n",
	}
}

func sequenceMappingNodeTestCase() *MarshalTestCase {
	m := compiler.NewMappingNode()
	m.Content = append(m.Content, compiler.NewScalarNodeForString("required"))
	m.Content = append(m.Content, compiler.NewSequenceNodeForStringArray([]string{"a", "b", "c"}))
	node := compiler.NewSequenceNode()
	node.Content = append(node.Content, m)
	return &MarshalTestCase{
		Name:     "sequence mapping node array",
		Node:     node,
		Expected: "[\n  {\n    \"required\": [\n      \"a\",\n      \"b\",\n      \"c\"\n    ]\n  }\n]\n",
	}
}

func mappingNodeTestCase() *MarshalTestCase {
	node := compiler.NewMappingNode()
	node.Content = append(node.Content, compiler.NewScalarNodeForString("required"))
	node.Content = append(node.Content, compiler.NewSequenceNodeForStringArray([]string{"a", "b", "c"}))
	return &MarshalTestCase{
		Name:     "Mapping node",
		Node:     node,
		Expected: "{\n  \"required\": [\n    \"a\",\n    \"b\",\n    \"c\"\n  ]\n}\n",
	}
}

func documentNodeTestCase() *MarshalTestCase {
	m := compiler.NewMappingNode()
	m.Content = append(m.Content, compiler.NewScalarNodeForString("version"))
	m.Content = append(m.Content, compiler.NewScalarNodeForString("1.0.0"))
	node := &yaml.Node{
		Kind:    yaml.DocumentNode,
		Content: []*yaml.Node{m},
	}
	return &MarshalTestCase{
		Name:     "Document node",
		Node:     node,
		Expected: "{\n  \"version\": \"1.0.0\"\n}\n",
	}
}

func aliasNodeTestCase() *MarshalTestCase {
	return &MarshalTestCase{
		Name: "unsupported alias node",
		Node: &yaml.Node{Kind: yaml.AliasNode},
		Err:  true,
	}
}
