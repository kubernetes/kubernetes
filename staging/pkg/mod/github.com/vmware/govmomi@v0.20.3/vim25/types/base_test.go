/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package types

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/vmware/govmomi/vim25/xml"
)

func TestAnyType(t *testing.T) {
	x := func(s string) []byte {
		s = `<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">` + s
		s += `</root>`
		return []byte(s)
	}

	tests := []struct {
		Input []byte
		Value interface{}
	}{
		{
			Input: x(`<name xsi:type="xsd:string">test</name>`),
			Value: "test",
		},
		{
			Input: x(`<name xsi:type="ArrayOfString"><string>AA</string><string>BB</string></name>`),
			Value: ArrayOfString{String: []string{"AA", "BB"}},
		},
	}

	for _, test := range tests {
		var r struct {
			A interface{} `xml:"name,typeattr"`
		}

		dec := xml.NewDecoder(bytes.NewReader(test.Input))
		dec.TypeFunc = TypeFunc()

		err := dec.Decode(&r)
		if err != nil {
			t.Fatalf("Decode: %s", err)
		}

		if !reflect.DeepEqual(r.A, test.Value) {
			t.Errorf("Expected: %#v, actual: %#v", r.A, test.Value)
		}
	}
}
