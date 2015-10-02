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

package resource

import (
	"bytes"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

type testTransform struct {
	search  string
	replace string
}

func (t testTransform) TransformStream(in io.Reader) (io.Reader, error) {
	data, err := ioutil.ReadAll(in)
	if err != nil {
		return nil, err
	}
	output := strings.Replace(string(data), t.search, t.replace, -1)
	return bytes.NewBufferString(output), nil
}

func TestTransformsList(t *testing.T) {
	t1 := testTransform{"foo", "bar"}
	t2 := testTransform{"bar", "baz"}
	t3 := testTransform{"baz", "blah"}

	list := StreamTransformList([]StreamTransform{t1.TransformStream, t2.TransformStream, t3.TransformStream})
	reader, err := list.Transform(bytes.NewBufferString("foo"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(data) != "blah" {
		t.Errorf("expected: %s, saw: %s", "blah", string(data))
	}
}
