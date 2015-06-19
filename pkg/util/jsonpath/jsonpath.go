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
	wr     io.Writer
}

func NewJSONPath(n string) *JSONPath {
	return &JSONPath{
		name: n,
	}
}

// Parse parse the given template, return error
func (j *JSONPath) Parse(text string) (err error) {
	j.parser, err = Parse(j.name, text)
	return
}

func (j *JSONPath) Execute(wr io.Writer, data interface{}) error {
	if j.parser == nil {
		return fmt.Errorf("%s is an incomplete jsonpath template", j.name)
	}
	j.wr = wr
	return j.walkTree(reflect.ValueOf(data))
}

// walkTree visits the parsed tree from root and write the result
func (j *JSONPath) walkTree(value reflect.Value) error {
	for _, node := range j.parser.Root.Nodes {
		result, err := j.walk(value, node)
		if err != nil {
			return err
		}
		text, err := j.evalToText(result)
		if err != nil {
			return err
		}
		if _, err = j.wr.Write(text); err != nil {
			return err
		}
	}
	return nil
}

// walk visits subtree rooted at the given node in DFS order
func (j *JSONPath) walk(value reflect.Value, node Node) (reflect.Value, error) {
	switch node := node.(type) {
	case *ListNode:
		return j.evalList(value, node)
	case *TextNode:
		return reflect.ValueOf(string(node.Text)), nil
	case *FieldNode:
		return j.evalField(value, node)
	case *ArrayNode:
		return j.evalArray(value, node)
	case *FilterNode:
		return j.evalFilter(value, node)
	case *IntNode:
		return reflect.ValueOf(node.Value), nil
	case *FloatNode:
		return reflect.ValueOf(node.Value), nil
	default:
		return reflect.Value{}, fmt.Errorf("unexpect Node %v", node)
	}
}

// evalList evaluates ListNode
func (j *JSONPath) evalList(value reflect.Value, node *ListNode) (reflect.Value, error) {
	var err error
	curValue := value
	for _, node := range node.Nodes {
		curValue, err = j.walk(curValue, node)
		if err != nil {
			return value, err
		}
	}
	return curValue, nil
}

// evalArray evaluates ArrayNode
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

// evalField evaluates filed of struct or key of map
func (j *JSONPath) evalField(value reflect.Value, node *FieldNode) (reflect.Value, error) {
	var result reflect.Value
	if value.Kind() == reflect.Struct {
		result = value.FieldByName(node.Value)
	} else if value.Kind() == reflect.Map {
		result = value.MapIndex(reflect.ValueOf(node.Value))
	}
	if !result.IsValid() {
		return result, fmt.Errorf("in %v, %s is not found", value, node.Value)
	}
	return result, nil
}

// evalFilter filter array according to FilterNode
func (j *JSONPath) evalFilter(value reflect.Value, node *FilterNode) (reflect.Value, error) {
	if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
		return value, fmt.Errorf("%v is not array or slice", value)
	}
	var result reflect.Value
	if value.Len() == 0 {
		return result, nil
	}

	result = reflect.MakeSlice(reflect.SliceOf(value.Index(0).Type()), 0, 0)
	for i := 0; i < value.Len(); i++ {
		cur := value.Index(i)
		left, err := j.evalList(cur, node.Left)
		if err != nil {
			return value, err
		}
		right, err := j.evalList(cur, node.Right)
		if err != nil {
			return value, err
		}
		pass := false
		switch node.Operator {
		case "<":
			pass, err = less(left, right)
		case ">":
			pass, err = greater(left, right)
		case "==":
			pass, err = equal(left, right)
		case "!=":
			pass, err = notEqual(left, right)
		case "<=":
			pass, err = lessEqual(left, right)
		case ">=":
			pass, err = greaterEqual(left, right)
		case "exists":
			pass = left.IsValid()
		default:
			return result, fmt.Errorf("unrecognized filter operator %s", node.Operator)
		}
		if err != nil {
			return result, err
		}
		if pass {
			result = reflect.Append(result, cur)
		}
	}
	return result, nil
}

// evalToText translates reflect value to corresponding text
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
	case reflect.Struct:
		buffer := bytes.NewBufferString("{")
		for i := 0; i < v.NumField(); i++ {
			text, err := j.evalToText(v.Field(i))
			if err != nil {
				return nil, err
			}
			buffer.Write(text)
			if i != v.NumField()-1 {
				buffer.WriteString(", ")
			}
		}
		buffer.WriteString("}")
		return buffer.Bytes(), nil
	default:
		return nil, fmt.Errorf("%v is not printable", v.Kind())
	}
	return []byte(text), nil
}
