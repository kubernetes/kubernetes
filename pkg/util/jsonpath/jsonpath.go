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
	"errors"
	"github.com/GoogleCloudPlatform/kubernetes/third_party/golang/parse"
	"github.com/golang/glog"
	"io"
	"reflect"
	"strconv"
)

type Jsonpath struct {
	name string
	tree *parse.Tree
}

func New(n string) *Jsonpath {
	return &Jsonpath{
		name: n,
	}
}

func (j *Jsonpath) Parse(text string) error {
	var err error
	j.tree, err = parse.Parse(j.name, text)
	return err
}

func (j *Jsonpath) Execute(wr io.Writer, data interface{}) error {
	value := reflect.ValueOf(data)
	if j.tree == nil {
		return errors.New(j.name + " is an incomplete jsonpath template")
	}
	j.walk(wr, value, j.tree.Root)
	return nil
}

func (j *Jsonpath) walk(wr io.Writer, value reflect.Value, node parse.Node) reflect.Value {
	var text []byte
	switch node := node.(type) {
	case *parse.ListNode:
		curValue := value
		for _, node := range node.Nodes {
			curValue = j.walk(wr, curValue, node)
		}
		return value
	case *parse.TextNode:
		text = node.Text
		if _, err := wr.Write(text); err != nil {
			glog.Errorf("%s", err)
		}
		return value
	case *parse.VariableNode:
		value = j.evalVariable(wr, value, node.Name)
		return value
	default:
		return reflect.Value{}
	}
}

func (j *Jsonpath) evalVariable(wr io.Writer, value reflect.Value, name string) reflect.Value {
	var text string
	v := value.FieldByName(name)
	switch v.Kind() {
	case reflect.Struct:
		return v
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
	if _, err := wr.Write([]byte(text)); err != nil {
		glog.Errorf("%s", err)
	}
	return reflect.Value{}
}
