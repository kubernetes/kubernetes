/*
Copyright 2014 Google Inc. All rights reserved.

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

package cloudcfg

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"gopkg.in/v1/yaml"
)

func TestYAMLPrinterPrint(t *testing.T) {
	type testStruct struct {
		Key        string         `yaml:"Key" json:"Key"`
		Map        map[string]int `yaml:"Map" json:"Map"`
		StringList []string       `yaml:"StringList" json:"StringList"`
		IntList    []int          `yaml:"IntList" json:"IntList"`
	}
	testData := testStruct{
		"testValue",
		map[string]int{"TestSubkey": 1},
		[]string{"a", "b", "c"},
		[]int{1, 2, 3},
	}
	printer := &YAMLPrinter{}
	buf := bytes.NewBuffer([]byte{})

	err := printer.Print([]byte("invalidJSON"), buf)
	if err == nil {
		t.Error("Error: didn't fail on invalid JSON data")
	}

	jTestData, err := json.Marshal(&testData)
	if err != nil {
		t.Fatal("Unexpected error: couldn't marshal test data")
	}
	err = printer.Print(jTestData, buf)
	if err != nil {
		t.Fatal(err)
	}
	var poutput testStruct
	err = yaml.Unmarshal(buf.Bytes(), &poutput)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testData, poutput) {
		t.Error("Test data and unmarshaled data are not equal")
	}
}
