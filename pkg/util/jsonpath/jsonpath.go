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

	"github.com/GoogleCloudPlatform/kubernetes/third_party/golang/template"
)

type JSONPath struct {
	name       string
	parser     *Parser
	stack      [][]reflect.Value //push and pop values in different scopes
	cur        []reflect.Value   //current scope values
	beginRange int
	inRange    int
	endRange   int
}

func New(name string) *JSONPath {
	return &JSONPath{
		name:       name,
		beginRange: 0,
		inRange:    0,
		endRange:   0,
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

	j.cur = []reflect.Value{reflect.ValueOf(data)}
	nodes := j.parser.Root.Nodes
	for i := 0; i < len(nodes); i++ {
		node := nodes[i]
		results, err := j.walk(j.cur, node)
		if err != nil {
			return err
		}

		//encounter an end node, break the current block
		if j.endRange > 0 && j.endRange <= j.inRange {
			j.endRange -= 1
			break
		}
		//encounter a range node, start a range loop
		if j.beginRange > 0 {
			j.beginRange -= 1
			j.inRange += 1
			for k, value := range results {
				j.parser.Root.Nodes = nodes[i+1:]
				if k == len(results)-1 {
					j.inRange -= 1
				}
				err := j.Execute(wr, value.Interface())
				if err != nil {
					return err
				}

			}
			break
		}
		err = j.PrintResults(wr, results)
		if err != nil {
			return err
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
	case *RecursiveNode:
		return j.evalRecursive(value, node)
	case *UnionNode:
		return j.evalUnion(value, node)
	case *IdentifierNode:
		return j.evalIdentifier(value, node)
	default:
		return value, fmt.Errorf("unexpected Node %v", node)
	}
}

// evalInt evaluates IntNode
func (j *JSONPath) evalInt(input []reflect.Value, node *IntNode) ([]reflect.Value, error) {
	result := make([]reflect.Value, len(input))
	for i := range input {
		result[i] = reflect.ValueOf(node.Value)
	}
	return result, nil
}

// evalFloat evaluates FloatNode
func (j *JSONPath) evalFloat(input []reflect.Value, node *FloatNode) ([]reflect.Value, error) {
	result := make([]reflect.Value, len(input))
	for i := range input {
		result[i] = reflect.ValueOf(node.Value)
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
			return curValue, err
		}
	}
	return curValue, nil
}

// evalIdentifier evaluates IdentifierNode
func (j *JSONPath) evalIdentifier(input []reflect.Value, node *IdentifierNode) ([]reflect.Value, error) {
	results := []reflect.Value{}
	switch node.Name {
	case "range":
		j.stack = append(j.stack, j.cur)
		j.beginRange += 1
		results = input
	case "end":
		if j.endRange < j.inRange { //inside a loop, break the current block
			j.endRange += 1
			break
		}
		// the loop is about to end, pop value and continue the following execution
		if len(j.stack) > 0 {
			j.cur, j.stack = j.stack[len(j.stack)-1], j.stack[:len(j.stack)-1]
		} else {
			return results, fmt.Errorf("not in range, nothing to end")
		}
	default:
		return input, fmt.Errorf("unrecongnized identifier %v", node.Name)
	}
	return results, nil
}

// evalArray evaluates ArrayNode
func (j *JSONPath) evalArray(input []reflect.Value, node *ArrayNode) ([]reflect.Value, error) {
	result := []reflect.Value{}
	for _, value := range input {
		if value.Kind() == reflect.Interface {
			value = reflect.ValueOf(value.Interface())
		}
		if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
			return input, fmt.Errorf("%v is not array or slice", value)
		}
		params := node.Params
		if !params[0].Known {
			params[0].Value = 0
		}
		if params[0].Value < 0 {
			params[0].Value += value.Len()
		}
		if !params[1].Known {
			params[1].Value = value.Len()
		}

		if params[1].Value < 0 {
			params[1].Value += value.Len()
		}

		if !params[2].Known {
			value = value.Slice(params[0].Value, params[1].Value)
		} else {
			value = value.Slice3(params[0].Value, params[1].Value, params[2].Value)
		}
		for i := 0; i < value.Len(); i++ {
			result = append(result, value.Index(i))
		}
	}
	return result, nil
}

// evalUnion evaluates UnionNode
func (j *JSONPath) evalUnion(input []reflect.Value, node *UnionNode) ([]reflect.Value, error) {
	result := []reflect.Value{}
	for _, listNode := range node.Nodes {
		temp, err := j.evalList(input, listNode)
		if err != nil {
			return input, err
		}
		result = append(result, temp...)
	}
	return result, nil
}

// evalField evaluates filed of struct or key of map.
func (j *JSONPath) evalField(input []reflect.Value, node *FieldNode) ([]reflect.Value, error) {
	results := []reflect.Value{}
	for _, value := range input {
		var result reflect.Value
		if value.Kind() == reflect.Interface {
			value = reflect.ValueOf(value.Interface())
		}
		if value.Kind() == reflect.Struct {
			result = value.FieldByName(node.Value)
		} else if value.Kind() == reflect.Map {
			result = value.MapIndex(reflect.ValueOf(node.Value))
		}
		if result.IsValid() {
			results = append(results, result)
		}
	}
	if len(results) == 0 {
		return results, fmt.Errorf("%s is not found", node.Value)
	}
	return results, nil
}

// evalWildcard extract all contents of the given value
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

// evalRecursive visit the given value recursively and push all of them to result
func (j *JSONPath) evalRecursive(input []reflect.Value, node *RecursiveNode) ([]reflect.Value, error) {
	result := []reflect.Value{}
	for _, value := range input {
		results := []reflect.Value{}
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
		if len(results) != 0 {
			result = append(result, value)
			output, err := j.evalRecursive(results, node)
			if err != nil {
				return result, err
			}
			result = append(result, output...)
		}
	}
	return result, nil
}

// evalFilter filter array according to FilterNode
func (j *JSONPath) evalFilter(input []reflect.Value, node *FilterNode) ([]reflect.Value, error) {
	results := []reflect.Value{}
	for _, value := range input {
		if value.Kind() == reflect.Interface {
			value = reflect.ValueOf(value.Interface())
		}
		if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
			return input, fmt.Errorf("%v is not array or slice", value)
		}
		for i := 0; i < value.Len(); i++ {
			temp := []reflect.Value{value.Index(i)}
			lefts, err := j.evalList(temp, node.Left)

			//case exists
			if node.Operator == "exists" {
				if len(lefts) > 0 {
					results = append(results, value.Index(i))
				}
				continue
			}

			if err != nil {
				return input, err
			}

			var left, right interface{}
			if len(lefts) != 1 {
				return input, fmt.Errorf("can only compare one element at a time")
			}
			left = lefts[0].Interface()

			rights, err := j.evalList(temp, node.Right)
			if err != nil {
				return input, err
			}
			if len(rights) != 1 {
				return input, fmt.Errorf("can only compare one element at a time")
			}
			right = rights[0].Interface()

			pass := false
			switch node.Operator {
			case "<":
				pass, err = template.Less(left, right)
			case ">":
				pass, err = template.Greater(left, right)
			case "==":
				pass, err = template.Equal(left, right)
			case "!=":
				pass, err = template.NotEqual(left, right)
			case "<=":
				pass, err = template.LessEqual(left, right)
			case ">=":
				pass, err = template.GreaterEqual(left, right)
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
	if v.Kind() == reflect.Interface {
		v = reflect.ValueOf(v.Interface())
	}
	var buffer bytes.Buffer
	switch v.Kind() {
	case reflect.Invalid:
		//pass
	case reflect.Ptr:
		text, err := j.evalToText(reflect.Indirect(v))
		if err != nil {
			return nil, err
		}
		buffer.Write(text)
	case reflect.Bool:
		if variable := v.Bool(); variable {
			buffer.WriteString("True")
		} else {
			buffer.WriteString("False")
		}
	case reflect.Float32:
		buffer.WriteString(strconv.FormatFloat(v.Float(), 'f', -1, 32))
	case reflect.Float64:
		buffer.WriteString(strconv.FormatFloat(v.Float(), 'f', -1, 64))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		buffer.WriteString(strconv.FormatInt(v.Int(), 10))
	case reflect.String:
		buffer.WriteString(v.String())
	case reflect.Array, reflect.Slice:
		buffer.WriteString("[")
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
	case reflect.Struct:
		buffer.WriteString("{")
		for i := 0; i < v.NumField(); i++ {
			text, err := j.evalToText(v.Field(i))
			if err != nil {
				return nil, err
			}
			pair := fmt.Sprintf("%s: %s", v.Type().Field(i).Name, text)
			buffer.WriteString(pair)
			if i != v.NumField()-1 {
				buffer.WriteString(", ")
			}
		}
		buffer.WriteString("}")
	case reflect.Map:
		buffer.WriteString("{")
		for i, key := range v.MapKeys() {
			text, err := j.evalToText(v.MapIndex(key))
			if err != nil {
				return nil, err
			}
			pair := fmt.Sprintf("%s: %s", key, text)
			buffer.WriteString(pair)
			if i != len(v.MapKeys())-1 {
				buffer.WriteString(", ")
			}
		}
		buffer.WriteString("}")

	default:
		return nil, fmt.Errorf("%v is not printable", v.Kind())
	}
	return buffer.Bytes(), nil
}
