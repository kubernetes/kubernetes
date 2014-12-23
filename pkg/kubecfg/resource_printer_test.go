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

package kubecfg

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

	"github.com/ghodss/yaml"
)

func TestYAMLPrinterPrint(t *testing.T) {
	type testStruct struct {
		Key        string         `json:"Key"`
		Map        map[string]int `json:"Map"`
		StringList []string       `json:"StringList"`
		IntList    []int          `json:"IntList"`
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
		t.Errorf("Test data and unmarshaled data are not equal: %#v vs %#v", poutput, testData)
	}

	obj := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	buf.Reset()
	printer.PrintObj(obj, buf)
	var objOut api.Pod
	err = yaml.Unmarshal([]byte(buf.String()), &objOut)
	if err != nil {
		t.Errorf("Unexpeted error: %#v", err)
	}
	if !reflect.DeepEqual(obj, &objOut) {
		t.Errorf("Unexpected inequality: %#v vs %#v", obj, &objOut)
	}
}

func TestIdentityPrinter(t *testing.T) {
	printer := &IdentityPrinter{}
	buff := bytes.NewBuffer([]byte{})
	str := "this is a test string"
	printer.Print([]byte(str), buff)
	if buff.String() != str {
		t.Errorf("Bytes are not equal: %s vs %s", str, buff.String())
	}

	obj := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	buff.Reset()
	printer.PrintObj(obj, buff)
	objOut, err := latest.Codec.Decode([]byte(buff.String()))
	if err != nil {
		t.Errorf("Unexpeted error: %#v", err)
	}
	if !reflect.DeepEqual(obj, objOut) {
		t.Errorf("Unexpected inequality: %#v vs %#v", obj, objOut)
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
	printer := NewHumanReadablePrinter()
	printer.Handler(columns, PrintCustomType)

	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err != nil {
		t.Errorf("An error occurred printing the custom type: %#v", err)
	}
	expectedOutput := "Data\n----------\ntest object"
	if buffer.String() != expectedOutput {
		t.Errorf("The data was not printed as expected. Expected:\n%s\nGot:\n%s", expectedOutput, buffer.String())
	}
}

func TestPrintHandlerError(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter()
	printer.Handler(columns, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := NewHumanReadablePrinter()
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&TestUnknownType{}, buffer)
	if err == nil {
		t.Errorf("An error was expected from printing unknown type")
	}
}

func TestTemplateEmitsVersionedObjects(t *testing.T) {
	// kind is always blank in memory and set on the wire
	printer, err := NewTemplatePrinter([]byte(`{{.kind}}`))
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

func TestTemplatePanic(t *testing.T) {
	tmpl := `{{and ((index .currentState.info "update-demo").state.running.startedAt) .currentState.info.net.state.running.startedAt}}`
	printer, err := NewTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}
	buffer := &bytes.Buffer{}
	err = printer.PrintObj(&api.Pod{}, buffer)
	if err == nil {
		t.Fatalf("expected that template to crash")
	}
	if buffer.String() == "" {
		t.Errorf("no debugging info was printed")
	}
}

func TestTemplateStrings(t *testing.T) {
	// This unit tests the "exists" function as well as the template from update.sh
	table := map[string]struct {
		pod    api.Pod
		expect string
	}{
		"nilInfo":   {api.Pod{}, "false"},
		"emptyInfo": {api.Pod{Status: api.PodStatus{Info: api.PodInfo{}}}, "false"},
		"containerExists": {
			api.Pod{
				Status: api.PodStatus{
					Info: api.PodInfo{"update-demo": api.ContainerStatus{}},
				},
			},
			"false",
		},
		"netExists": {
			api.Pod{
				Status: api.PodStatus{
					Info: api.PodInfo{"net": api.ContainerStatus{}},
				},
			},
			"false",
		},
		"bothExist": {
			api.Pod{
				Status: api.PodStatus{
					Info: api.PodInfo{
						"update-demo": api.ContainerStatus{},
						"net":         api.ContainerStatus{},
					},
				},
			},
			"false",
		},
		"oneValid": {
			api.Pod{
				Status: api.PodStatus{
					Info: api.PodInfo{
						"update-demo": api.ContainerStatus{},
						"net": api.ContainerStatus{
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: util.Time{},
								},
							},
						},
					},
				},
			},
			"false",
		},
		"bothValid": {
			api.Pod{
				Status: api.PodStatus{
					Info: api.PodInfo{
						"update-demo": api.ContainerStatus{
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: util.Time{},
								},
							},
						},
						"net": api.ContainerStatus{
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: util.Time{},
								},
							},
						},
					},
				},
			},
			"true",
		},
	}

	// The point of this test is to verify that the below template works. If you change this
	// template, you need to update hack/e2e-suite/update.sh.
	tmpl :=
		`{{and (exists . "currentState" "info" "update-demo" "state" "running") (exists . "currentState" "info" "net" "state" "running")}}`
	useThisToDebug := `
a: {{exists . "currentState"}}
b: {{exists . "currentState" "info"}}
c: {{exists . "currentState" "info" "update-demo"}}
d: {{exists . "currentState" "info" "update-demo" "state"}}
e: {{exists . "currentState" "info" "update-demo" "state" "running"}}
f: {{exists . "currentState" "info" "update-demo" "state" "running" "startedAt"}}`
	_ = useThisToDebug // don't complain about unused var

	printer, err := NewTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}

	for name, item := range table {
		buffer := &bytes.Buffer{}
		err = printer.PrintObj(&item.pod, buffer)
		if err != nil {
			t.Errorf("%v: unexpected err: %v", name, err)
			continue
		}
		if e, a := item.expect, buffer.String(); e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
		}
	}
}
