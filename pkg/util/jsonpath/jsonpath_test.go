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
	"testing"
)

func TestPlainText(t *testing.T) {
	text := "hello jsonpath"
	j := New("plain")
	err := j.Parse(text)
	if err != nil {
		t.Errorf("parse plain text %s error %v", text, err)
	}
	buf := new(bytes.Buffer)
	err = j.Execute(buf, nil)
	if err != nil {
		t.Errorf("execute plain text error %v", err)
	}
	out := buf.String()
	if out != text {
		t.Errorf("expect to get %s, got %s", text, out)
	}
}

func TestVariable(t *testing.T) {
	text := "hello ${.jsonpath}"
	j := New("variable")
	err := j.Parse(text)
	if err != nil {
		t.Errorf("parse variable %s error %v", text, err)
	}
	buf := new(bytes.Buffer)
	err = j.Execute(buf, struct{ jsonpath string }{jsonpath: "world"})
	if err != nil {
		t.Errorf("execute variable error %v", err)
	}
	out := buf.String()
	expect := "hello world"
	if out != expect {
		t.Errorf("expect to get %s, got %s", expect, out)
	}
}

func TestNestedDict(t *testing.T) {
	text := "hello ${.jsonpath.title}"
	j := New("nestedDict")
	err := j.Parse(text)
	if err != nil {
		t.Errorf("parse nested dict %s error %v", text, err)
	}
	buf := new(bytes.Buffer)

	type inner struct {
		title string
	}
	type outer struct {
		jsonpath inner
	}
	err = j.Execute(buf, outer{jsonpath: inner{title: "world"}})
	if err != nil {
		t.Errorf("execute variable error %v", err)
	}
	out := buf.String()
	expect := "hello world"
	if out != expect {
		t.Errorf("expect to get %s, got %s", expect, out)
	}
}

func TestQuote(t *testing.T) {
	text := `hello ${"${"}`
	j := New("quote")
	err := j.Parse(text)
	if err != nil {
		t.Errorf("parse quote %s error %v", text, err)
	}
	buf := new(bytes.Buffer)
	err = j.Execute(buf, nil)
	if err != nil {
		t.Errorf("execute quote error %v", err)
	}
	out := buf.String()
	expect := "hello ${"
	if out != expect {
		t.Errorf("expect to get %s, got %s", text, expect)
	}

}
