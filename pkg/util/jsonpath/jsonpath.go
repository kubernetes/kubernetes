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

// Parse parse the given template, return error
func (j *JSONPath) Parse(text string) (err error) {
	j.parser, err = Parse(j.name, text)
	return
}

// Execute bounds data into template and write the result
func (j *JSONPath) Execute(wr io.Writer, data interface{}) error {
	if j.parser == nil {
		return fmt.Errorf("%s is an incomplete jsonpath template", j.name)
	}
	for _, node := range j.parser.Root.Nodes {
		results, err := j.walk([]reflect.Value{reflect.ValueOf(data)}, node)
		if err != nil {
			return nil
		}
		err = j.PrintResults(wr, results)
		if err != nil {
			return nil
		}
	}
	return nil
}

// PrintResults write the results into writer
func (j *JSONPath) PrintResults(wr io.Writer, results []reflect.Value) error {
	for i, r := range results {
		text, err := j.evalToText(r)
		if err != nil {
			return err
		}
		if i != len(results)-1 {
			text = append(text, ' ')
		}
		if _, err = wr.Write(text); err != nil {
			return err
		}
	}
	return nil
}

// walk visits tree rooted at the given node in DFS order
func (j *JSONPath) walk(value []reflect.Value, node Node) ([]reflect.Value, error) {
	switch node := node.(type) {
	case *ListNode:
		return j.evalList(value, node)
	case *TextNode:
		return []reflect.Value{reflect.ValueOf(string(node.Text))}, nil
	case *FieldNode:
		return j.evalField(value, node)
	case *ArrayNode:
		return j.evalArray(value, node)
	case *FilterNode:
		return j.evalFilter(value, node)
	case *IntNode:
		return j.evalInt(value, node)
	case *FloatNode:
		return j.evalFloat(value, node)
	case *WildcardNode:
		return j.evalWildcard(value, node)
	default:
		return value, fmt.Errorf("unexpect Node %v", node)
	}
}

// evalInt evalutes IntNode
func (j *JSONPath) evalInt(input []reflect.Value, node *IntNode) ([]reflect.Value, error) {
	result := []reflect.Value{}
	for range input {
		result = append(result, reflect.ValueOf(node.Value))
	}
	return result, nil
}

// evalFloat evalutes FloatNode
func (j *JSONPath) evalFloat(input []reflect.Value, node *FloatNode) ([]reflect.Value, error) {
	result := []reflect.Value{}
	for range input {
		result = append(result, reflect.ValueOf(node.Value))
	}
	return result, nil
}

// evalList evaluates ListNode
func (j *JSONPath) evalList(value []reflect.Value, node *ListNode) ([]reflect.Value, error) {
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
func (j *JSONPath) evalArray(input []reflect.Value, node *ArrayNode) ([]reflect.Value, error) {
	result := []reflect.Value{}
	for _, value := range input {
		if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
			return input, fmt.Errorf("%v is not array or slice", value)
		}
		if !node.Params[0].Exists {
			node.Params[0].Value = 0
		}
		if !node.Params[1].Exists {
			node.Params[1].Value = value.Len()
		}
		if !node.Params[2].Exists {
			value = value.Slice(node.Params[0].Value, node.Params[1].Value)
		} else {
			value = value.Slice3(node.Params[0].Value, node.Params[1].Value, node.Params[2].Value)
		}
		for i := 0; i < value.Len(); i++ {
			result = append(result, value.Index(i))
		}
	}
	return result, nil
}

// evalField evaluates filed of struct or key of map.
func (j *JSONPath) evalField(input []reflect.Value, node *FieldNode) ([]reflect.Value, error) {
	results := []reflect.Value{}
	for _, value := range input {
		var result reflect.Value
		if value.Kind() == reflect.Struct {
			result = value.FieldByName(node.Value)
		} else if value.Kind() == reflect.Map {
			result = value.MapIndex(reflect.ValueOf(node.Value))
		}
		if !result.IsValid() {
			return results, fmt.Errorf("in %v, %s is not found", value, node.Value)
		}
		results = append(results, result)
	}
	return results, nil
}

// evalWildcard extrac all contents of the given value
func (j *JSONPath) evalWildcard(input []reflect.Value, node *WildcardNode) ([]reflect.Value, error) {
	results := []reflect.Value{}
	for _, value := range input {
		kind := value.Kind()
		if kind == reflect.Struct {
			for i := 0; i < value.NumField(); i++ {
				results = append(results, value.Field(i))
			}
		} else if kind == reflect.Map {
			for _, key := range value.MapKeys() {
				results = append(results, value.MapIndex(key))
			}
		} else if kind == reflect.Array || kind == reflect.Slice || kind == reflect.String {
			for i := 0; i < value.Len(); i++ {
				results = append(results, value.Index(i))
			}
		}
	}
	return results, nil
}

// evalFilter filter array according to FilterNode
func (j *JSONPath) evalFilter(input []reflect.Value, node *FilterNode) ([]reflect.Value, error) {
	results := []reflect.Value{}
	for _, value := range input {
		if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
			return input, fmt.Errorf("%v is not array or slice", value)
		}
		temp := []reflect.Value{}
		for i := 0; i < value.Len(); i++ {
			temp = append(temp, value.Index(i))
		}
		lefts, err := j.evalList(temp, node.Left)
		if err != nil {
			return input, err
		}
		rights, err := j.evalList(temp, node.Right)
		if err != nil {
			return input, err
		}
		for i := 0; i < value.Len(); i++ {
			left := lefts[i]
			right := rights[i]
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
				return results, fmt.Errorf("unrecognized filter operator %s", node.Operator)
			}
			if err != nil {
				return results, err
			}
			if pass {
				results = append(results, value.Index(i))
			}
		}
	}
	return results, nil
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
