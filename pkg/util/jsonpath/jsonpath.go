/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package jsonpath

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"strconv"
)

type JSONPath struct {
	name   string
	parser *Parser
}

func NewJSONPath(n string) *JSONPath {
	return &JSONPath{
		name: n,
	}
}

func (j *JSONPath) Parse(text string) (err error) {
	j.parser, err = Parse(j.name, text)
	return
}

func (j *JSONPath) Execute(wr io.Writer, data interface{}) error {
	if j.parser == nil {
		return fmt.Errorf("%s is an incomplete jsonpath template", j.name)
	}
	return j.walkTree(wr, reflect.ValueOf(data))
}

// walkTree visits the parsed tree from root
func (j *JSONPath) walkTree(wr io.Writer, value reflect.Value) (err error) {
	for _, node := range j.parser.Root.Nodes {
		_, err = j.walk(wr, value, node)
		if err != nil {
			return err
		}
	}
	return nil
}

// walk visits subtree rooted at the given node in DFS order
func (j *JSONPath) walk(wr io.Writer, value reflect.Value, node Node) (reflect.Value, error) {
	switch node := node.(type) {
	case *ListNode:
		return j.evalList(wr, value, node)
	case *TextNode:
		return j.evalText(wr, value, node)
	case *FieldNode:
		return j.evalField(value, node)
	case *ArrayNode:
		return j.evalArray(value, node)
	case *FilterNode:
		return j.evalFilter(value, node)
	default:
		return reflect.Value{}, fmt.Errorf("unexpect Node %v", node)
	}
}

// evalText evaluate TextNode
func (j *JSONPath) evalText(wr io.Writer, value reflect.Value, node *TextNode) (reflect.Value, error) {
	if _, err := wr.Write(node.Text); err != nil {
		return reflect.Value{}, err
	}
	return reflect.Value{}, nil
}

// evalText evaluate ListNode
func (j *JSONPath) evalList(wr io.Writer, value reflect.Value, node *ListNode) (reflect.Value, error) {
	var err error
	curValue := value
	for _, node := range node.Nodes {
		curValue, err = j.walk(wr, curValue, node)
		if err != nil {
			return value, err
		}
	}
	text, err := j.evalToText(curValue)
	if err != nil {
		return value, err
	}
	if _, err = wr.Write(text); err != nil {
		return value, err
	}
	return value, nil
}

// evalText evaluate ArrayNode
func (j *JSONPath) evalArray(value reflect.Value, node *ArrayNode) (reflect.Value, error) {
	if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
		return value, fmt.Errorf("%v is not array or slice", value)
	}
	if !node.Params[0].Exists {
		node.Params[0].Value = 0
	}
	if !node.Params[1].Exists {
		node.Params[1].Value = value.Len()
	}
	if !node.Params[2].Exists {
		return value.Slice(node.Params[0].Value, node.Params[1].Value), nil
	} else {
		return value.Slice3(node.Params[0].Value, node.Params[1].Value, node.Params[2].Value), nil
	}
}

// evalField evaluate filed from struct value
func (j *JSONPath) evalField(value reflect.Value, node *FieldNode) (reflect.Value, error) {
	result := value.FieldByName(node.Value)
	if !result.IsValid() {
		return result, fmt.Errorf("%s is not found", node.Value)
	}
	return result, nil
}

// evalFilter filter array according to FilterNode
func (j *JSONPath) evalFilter(value reflect.Value, node *FilterNode) (reflect.Value, error) {
	result := reflect.ValueOf([]interface{}{})
	return result, nil
}

// evalToText translate reflect value to corresponding text
func (j *JSONPath) evalToText(v reflect.Value) ([]byte, error) {
	var text string
	switch v.Kind() {
	case reflect.Invalid:
		//pass
	case reflect.Bool:
		if variable := v.Bool(); variable {
			text = "True"
		} else {
			text = "False"
		}
	case reflect.Float32:
		text = strconv.FormatFloat(v.Float(), 'f', -1, 32)
	case reflect.Float64:
		text = strconv.FormatFloat(v.Float(), 'f', -1, 64)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		text = strconv.FormatInt(v.Int(), 10)
	case reflect.String:
		text = v.String()
	case reflect.Array, reflect.Slice:
		buffer := bytes.NewBufferString("[")
		for i := 0; i < v.Len(); i++ {
			text, err := j.evalToText(v.Index(i))
			if err != nil {
				return nil, err
			}
			buffer.Write(text)
			if i != v.Len()-1 {
				buffer.WriteString(", ")
			}
		}
		buffer.WriteString("]")
		return buffer.Bytes(), nil
	default:
		return nil, fmt.Errorf("%v is not a printable", v)
	}
	return []byte(text), nil
}
