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

package kubectl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"gopkg.in/v1/yaml"
)

type testStruct struct {
	api.TypeMeta   `yaml:",inline" json:",inline"`
	api.ObjectMeta `yaml:"metadata,omitempty" json:"metadata,omitempty"`
	Key            string         `yaml:"Key" json:"Key"`
	Map            map[string]int `yaml:"Map" json:"Map"`
	StringList     []string       `yaml:"StringList" json:"StringList"`
	IntList        []int          `yaml:"IntList" json:"IntList"`
}

func (ts *testStruct) IsAnAPIObject() {}

const TestVersion = latest.Version

func init() {
	api.Scheme.AddKnownTypes("", &testStruct{})
	api.Scheme.AddKnownTypes(TestVersion, &testStruct{})
}

var testData = testStruct{
	Key:        "testValue",
	Map:        map[string]int{"TestSubkey": 1},
	StringList: []string{"a", "b", "c"},
	IntList:    []int{1, 2, 3},
}

func TestYAMLPrinter(t *testing.T) {
	testPrinter(t, &YAMLPrinter{TestVersion}, yaml.Unmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, &JSONPrinter{TestVersion}, json.Unmarshal)
}

func TestPrintJSON(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter(TestVersion, "json", "", nil)
	if err != nil {
		t.Errorf("unexpected error: %#v", err)
	}
	if !printer.IsVersioned() {
		t.Errorf("printer should be versioned")
	}
	printer.PrintObj(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}, buf)
	obj := map[string]interface{}{}
	if err := json.Unmarshal(buf.Bytes(), &obj); err != nil {
		t.Errorf("unexpected error: %#v\n%s", err, buf.String())
	}
}

func TestPrintYAML(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter(TestVersion, "yaml", "", nil)
	if err != nil {
		t.Errorf("unexpected error: %#v", err)
	}
	if !printer.IsVersioned() {
		t.Errorf("printer should be versioned")
	}
	printer.PrintObj(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}, buf)
	obj := map[string]interface{}{}
	if err := yaml.Unmarshal(buf.Bytes(), &obj); err != nil {
		t.Errorf("unexpected error: %#v\n%s", err, buf.String())
	}
}

func TestPrintTemplate(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter(TestVersion, "template", "{{.id}}", nil)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if !printer.IsVersioned() {
		t.Errorf("printer should be versioned")
	}
	err = printer.PrintObj(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}, buf)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if buf.String() != "foo" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPrintEmptyTemplate(t *testing.T) {
	if _, err := GetPrinter(TestVersion, "template", "", nil); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintBadTemplate(t *testing.T) {
	if _, err := GetPrinter(TestVersion, "template", "{{ .Name", nil); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintBadTemplateFile(t *testing.T) {
	if _, err := GetPrinter(TestVersion, "templatefile", "", nil); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func testPrinter(t *testing.T, printer ResourcePrinter, unmarshalFunc func(data []byte, v interface{}) error) {
	buf := bytes.NewBuffer([]byte{})

	err := printer.PrintObj(&testData, buf)
	if err != nil {
		t.Fatal(err)
	}
	var poutput testStruct
	// Verify that given function runs without error.
	err = unmarshalFunc(buf.Bytes(), &poutput)
	if err != nil {
		t.Fatal(err)
	}
	// Use real decode function to undo the versioning process.
	poutput = testStruct{}
	err = latest.Codec.DecodeInto(buf.Bytes(), &poutput)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testData, poutput) {
		t.Errorf("Test data and unmarshaled data are not equal: %v", util.ObjectDiff(poutput, testData))
	}

	obj := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	buf.Reset()
	printer.PrintObj(obj, buf)
	var objOut api.Pod
	// Verify that given function runs without error.
	err = unmarshalFunc(buf.Bytes(), &objOut)
	if err != nil {
		t.Errorf("Unexpeted error: %#v", err)
	}
	// Use real decode function to undo the versioning process.
	objOut = api.Pod{}
	err = latest.Codec.DecodeInto(buf.Bytes(), &objOut)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(obj, &objOut) {
		t.Errorf("Unexpected inequality:\n%v", util.ObjectDiff(obj, &objOut))
	}
}

type TestPrintType struct {
	Data string
}

func (*TestPrintType) IsAnAPIObject() {}

type TestUnknownType struct{}

func (*TestUnknownType) IsAnAPIObject() {}

func PrintCustomType(obj *TestPrintType, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s", obj.Data)
	return err
}

func ErrorPrintHandler(obj *TestPrintType, w io.Writer) error {
	return fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(false)
	printer.Handler(columns, PrintCustomType)

	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err != nil {
		t.Errorf("An error occurred printing the custom type: %#v", err)
	}
	expectedOutput := "Data\ntest object"
	if buffer.String() != expectedOutput {
		t.Errorf("The data was not printed as expected. Expected:\n%s\nGot:\n%s", expectedOutput, buffer.String())
	}
}

func TestPrintHandlerError(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(false)
	printer.Handler(columns, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := NewHumanReadablePrinter(false)
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&TestUnknownType{}, buffer)
	if err == nil {
		t.Errorf("An error was expected from printing unknown type")
	}
}

func TestTemplateEmitsVersionedObjects(t *testing.T) {
	// kind is always blank in memory and set on the wire
	printer, err := NewTemplatePrinter(TestVersion, []byte(`{{.kind}}`))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}
	buffer := &bytes.Buffer{}
	err = printer.PrintObj(&api.Pod{}, buffer)
	if err != nil {
		t.Fatalf("print fail: %v", err)
	}
	if e, a := "Pod", string(buffer.Bytes()); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}
