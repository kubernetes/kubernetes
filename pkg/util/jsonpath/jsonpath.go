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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/third_party/golang/parse"
	"github.com/golang/glog"
)

type JSONPath struct {
	name string
	tree *parse.Tree
}

func NewJSONPath(n string) *JSONPath {
	return &JSONPath{
		name: n,
	}
}

func (j *JSONPath) Parse(text string) (err error) {
	j.tree, err = parse.Parse(j.name, text)
	return
}

func (j *JSONPath) Execute(wr io.Writer, data interface{}) error {
	value := reflect.ValueOf(data)
	if j.tree == nil {
		return fmt.Errorf("%s is an incomplete jsonpath template", j.name)
	}
	j.walkTree(wr, value)
	return nil
}

// walkTree visit the parsed tree from root
func (j *JSONPath) walkTree(wr io.Writer, value reflect.Value) {
	for _, node := range j.tree.Root.Nodes {
		j.walk(wr, value, node)
	}
}

// walk visit subtree rooted at the gived node in DFS order
func (j *JSONPath) walk(wr io.Writer, value reflect.Value, node parse.Node) reflect.Value {
	switch node := node.(type) {
	case *parse.ListNode:
		return j.evalList(wr, value, node)
	case *parse.TextNode:
		return j.evalText(wr, value, node)
	case *parse.VariableNode:
		return j.evalVariable(wr, value, node)
	case *parse.ArrayNode:
		return j.evalArray(wr, value, node)
	}
	return reflect.Value{}
}

// evalText evaluate TextNode
func (j *JSONPath) evalText(wr io.Writer, value reflect.Value, node *parse.TextNode) reflect.Value {
	if _, err := wr.Write(node.Text); err != nil {
		glog.Errorf("%s", err)
	}
	return reflect.Value{}
}

// evalText evaluate ListNode
func (j *JSONPath) evalList(wr io.Writer, value reflect.Value, node *parse.ListNode) reflect.Value {
	curValue := value
	for _, node := range node.Nodes {
		curValue = j.walk(wr, curValue, node)
	}
	text := j.evalToText(curValue)
	if _, err := wr.Write(text); err != nil {
		glog.Errorf("%s", err)
	}
	return value
}

// evalText evaluate ArrayNode
func (j *JSONPath) evalArray(wr io.Writer, value reflect.Value, node *parse.ArrayNode) reflect.Value {
	if value.Kind() != reflect.Array && value.Kind() != reflect.Slice {
		glog.Errorf("%v is not array", value)
	}
	indexes := strings.Split(node.Range, ":")
	params := []int{0, value.Len(), 1}
	var err error
	for i := 0; i < len(indexes); i++ {
		params[i], err = strconv.Atoi(indexes[i])
		if err != nil {
			glog.Errorf("parse range %s to integer", node.Range)
		}
	}
	if len(indexes) == 1 {
		return value.Index(params[0])
	} else if len(indexes) == 2 {
		return value.Slice(params[0], params[1])
	} else if len(indexes) == 3 {
		return value.Slice3(params[0], params[1], params[2])
	} else {
		glog.Errorf("unsupported filter")
	}
	return reflect.Value{}
}

// evalVariable evaluate VariableNode
func (j *JSONPath) evalVariable(wr io.Writer, value reflect.Value, node *parse.VariableNode) reflect.Value {
	return value.FieldByName(node.Name)
}

// evalToText translate reflect value to corresponding text
func (j *JSONPath) evalToText(v reflect.Value) []byte {
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
		for i := 0; i < v.Len()-1; i++ {
			buffer.Write(j.evalToText(v.Index(i)))
			buffer.WriteString(", ")
		}
		buffer.Write(j.evalToText(v.Index(v.Len() - 1)))
		buffer.WriteString("]")
		return buffer.Bytes()
	default:
		glog.Errorf("%v is not a printable", v)
	}
	return []byte(text)
}
