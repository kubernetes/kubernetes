/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package jq

import (
	"errors"
	"github.com/golang/glog"
	"io"
	"reflect"
	"strconv"
)

type Jq struct {
	name string
	tree *Tree
}

func New(n string) *Jq {
	return &Jq{
		name: n,
	}
}

func (j *Jq) Parse(text string) (*Jq, error) {
	var err error
	j.tree, err = Parse(j.name, text)
	if err != nil {
		return nil, err
	}
	return j, nil
}

func (j *Jq) Execute(wr io.Writer, data interface{}) error {
	value := reflect.ValueOf(data)
	if j.tree == nil {
		return errors.New(j.name + " is an incomplete jq template")
	}
	j.walk(wr, value, j.tree.Root)
	return nil
}

func (j *Jq) walk(wr io.Writer, value reflect.Value, node Node) {
	var text []byte
	switch node := node.(type) {
	case *ListNode:
		for _, node := range node.Nodes {
			j.walk(wr, value, node)
		}
	case *TextNode:
		text = node.Text
	case *VariableNode:
		text = j.evalVariable(value, node.Name)
	default:
		return
	}
	if _, err := wr.Write(text); err != nil {
		glog.Errorf("%s", err)
	}

}

func (j *Jq) evalVariable(value reflect.Value, name string) []byte {
	var text string
	v := value.FieldByName(name)
	switch v.Kind() {
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
	default:
		glog.Errorf("%s is not a printable variable", name)
	}
	return []byte(text)
}
