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

package jsonwriter

import (
	"bytes"
	"errors"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

const indentation = "  "

// basic escaping, will need to be improved or replaced
func escape(s string) string {
	s = strings.Replace(s, "\n", "\\n", -1)
	s = strings.Replace(s, "\"", "\\\"", -1)
	return s
}

type writer struct {
	b bytes.Buffer
}

func (w *writer) bytes() []byte {
	return w.b.Bytes()
}

func (w *writer) writeString(s string) {
	w.b.Write([]byte(s))
}

func (w *writer) writeMap(node *yaml.Node, indent string) {
	if node.Kind == yaml.DocumentNode {
		w.writeMap(node.Content[0], indent)
		return
	}
	if node.Kind != yaml.MappingNode {
		w.writeString(fmt.Sprintf("invalid node for map: %+v", node))
		return
	}
	w.writeString("{\n")
	innerIndent := indent + indentation
	for i := 0; i < len(node.Content); i += 2 {
		// first print the key
		key := node.Content[i].Value
		w.writeString(fmt.Sprintf("%s\"%+v\": ", innerIndent, key))
		// then the value
		value := node.Content[i+1]
		switch value.Kind {
		case yaml.MappingNode:
			w.writeMap(value, innerIndent)
		case yaml.SequenceNode:
			w.writeSequence(value, innerIndent)
		case yaml.ScalarNode:
			w.writeScalar(value, innerIndent)
		}
		if i < len(node.Content)-2 {
			w.writeString(",")
		}
		w.writeString("\n")
	}
	w.writeString(indent)
	w.writeString("}")
}

func (w *writer) writeScalar(node *yaml.Node, indent string) {
	if node.Kind != yaml.ScalarNode {
		w.writeString(fmt.Sprintf("invalid node for scalar: %+v", node))
		return
	}
	switch node.Tag {
	case "!!str":
		w.writeString("\"")
		w.writeString(escape(node.Value))
		w.writeString("\"")
	case "!!int":
		w.writeString(node.Value)
	case "!!float":
		w.writeString(node.Value)
	case "!!bool":
		w.writeString(node.Value)
	}
}

func (w *writer) writeSequence(node *yaml.Node, indent string) {
	if node.Kind != yaml.SequenceNode {
		w.writeString(fmt.Sprintf("invalid node for sequence: %+v", node))
		return
	}
	w.writeString("[\n")
	innerIndent := indent + indentation
	for i, value := range node.Content {
		w.writeString(innerIndent)
		switch value.Kind {
		case yaml.MappingNode:
			w.writeMap(value, innerIndent)
		case yaml.SequenceNode:
			w.writeSequence(value, innerIndent)
		case yaml.ScalarNode:
			w.writeScalar(value, innerIndent)
		}
		if i < len(node.Content)-1 {
			w.writeString(",")
		}
		w.writeString("\n")
	}
	w.writeString(indent)
	w.writeString("]")
}

// Marshal writes a yaml.Node as JSON
func Marshal(in *yaml.Node) (out []byte, err error) {
	var w writer

	switch in.Kind {
	case yaml.DocumentNode:
		w.writeMap(in.Content[0], "")
		w.writeString("\n")
	case yaml.MappingNode:
		w.writeMap(in, "")
		w.writeString("\n")
	case yaml.SequenceNode:
		w.writeSequence(in, "")
		w.writeString("\n")
	case yaml.ScalarNode:
		w.writeScalar(in, "")
		w.writeString("\n")
	default:
		return nil, errors.New("invalid type passed to Marshal")
	}

	return w.bytes(), err
}
