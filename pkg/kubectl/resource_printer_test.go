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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/ghodss/yaml"
)

type testStruct struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty"`
	Key            string         `json:"Key"`
	Map            map[string]int `json:"Map"`
	StringList     []string       `json:"StringList"`
	IntList        []int          `json:"IntList"`
}

func (ts *testStruct) IsAnAPIObject() {}

func init() {
	api.Scheme.AddKnownTypes("", &testStruct{})
	api.Scheme.AddKnownTypes(testapi.Version(), &testStruct{})
}

var testData = testStruct{
	Key:        "testValue",
	Map:        map[string]int{"TestSubkey": 1},
	StringList: []string{"a", "b", "c"},
	IntList:    []int{1, 2, 3},
}

func TestYAMLPrinter(t *testing.T) {
	testPrinter(t, &YAMLPrinter{testapi.Version(), api.Scheme}, yaml.Unmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, &JSONPrinter{testapi.Version(), api.Scheme}, json.Unmarshal)
}

type internalType struct {
	Name string
	Kind string
}

type externalType struct {
	Name string
	Kind string `json:"kind"`
}

func (*internalType) IsAnAPIObject() {}
func (*externalType) IsAnAPIObject() {}

func newExternalScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName("", "Type", &internalType{})
	scheme.AddKnownTypeWithName("unlikelyversion", "Type", &externalType{})
	return scheme
}

func TestPrintJSONForUnknownSchema(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter("json", "", "unlikelyversion", newExternalScheme(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if err := printer.PrintObj(&internalType{Name: "foo"}, buf); err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	obj := map[string]interface{}{}
	if err := json.Unmarshal(buf.Bytes(), &obj); err != nil {
		t.Fatalf("unexpected error: %#v\n%s", err, buf.String())
	}
	if obj["Name"] != "foo" {
		t.Errorf("unexpected field: %#v", obj)
	}
}

func TestPrintJSONForUnknownSchemaAndWrongVersion(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter("json", "", "badversion", newExternalScheme(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if err := printer.PrintObj(&internalType{Name: "foo"}, buf); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintJSON(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter("json", "", testapi.Version(), api.Scheme, nil)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	printer.PrintObj(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}, buf)
	obj := map[string]interface{}{}
	if err := json.Unmarshal(buf.Bytes(), &obj); err != nil {
		t.Errorf("unexpected error: %#v\n%s", err, buf.String())
	}
}

func TestPrintYAML(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter("yaml", "", testapi.Version(), api.Scheme, nil)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	printer.PrintObj(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}, buf)
	obj := map[string]interface{}{}
	if err := yaml.Unmarshal(buf.Bytes(), &obj); err != nil {
		t.Errorf("unexpected error: %#v\n%s", err, buf.String())
	}
}

func TestPrintTemplate(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, err := GetPrinter("template", "{{.id}}", "v1beta1", api.Scheme, nil)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
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
	if _, err := GetPrinter("template", "", testapi.Version(), api.Scheme, nil); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintBadTemplate(t *testing.T) {
	if _, err := GetPrinter("template", "{{ .Name", testapi.Version(), api.Scheme, nil); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintBadTemplateFile(t *testing.T) {
	if _, err := GetPrinter("templatefile", "", testapi.Version(), api.Scheme, nil); err == nil {
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
	err = testapi.Codec().DecodeInto(buf.Bytes(), &poutput)
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
		t.Fatalf("unexpected error: %#v", err)
	}
	// Use real decode function to undo the versioning process.
	objOut = api.Pod{}
	err = testapi.Codec().DecodeInto(buf.Bytes(), &objOut)
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
		t.Fatalf("An error occurred printing the custom type: %#v", err)
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
	printer, err := NewTemplatePrinter([]byte(`{{.kind}}`), testapi.Version(), api.Scheme)
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
	printer, err := NewTemplatePrinter([]byte(tmpl), testapi.Version(), api.Scheme)
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

	printer, err := NewTemplatePrinter([]byte(tmpl), "v1beta1", api.Scheme)
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

func TestPrinters(t *testing.T) {
	om := func(name string) api.ObjectMeta { return api.ObjectMeta{Name: name} }
	templatePrinter, err := NewTemplatePrinter([]byte("{{.name}}"), testapi.Version(), api.Scheme)
	if err != nil {
		t.Fatal(err)
	}
	templatePrinter2, err := NewTemplatePrinter([]byte("{{len .items}}"), testapi.Version(), api.Scheme)
	if err != nil {
		t.Fatal(err)
	}
	printers := map[string]ResourcePrinter{
		"humanReadable":        NewHumanReadablePrinter(true),
		"humanReadableHeaders": NewHumanReadablePrinter(false),
		"json":                 &JSONPrinter{testapi.Version(), api.Scheme},
		"yaml":                 &YAMLPrinter{testapi.Version(), api.Scheme},
		"template":             templatePrinter,
		"template2":            templatePrinter2,
	}
	objects := map[string]runtime.Object{
		"pod":             &api.Pod{ObjectMeta: om("pod")},
		"emptyPodList":    &api.PodList{},
		"nonEmptyPodList": &api.PodList{Items: []api.Pod{{}}},
	}
	// map of printer name to set of objects it should fail on.
	expectedErrors := map[string]util.StringSet{
		"template2": util.NewStringSet("pod", "emptyPodList"),
	}

	for pName, p := range printers {
		for oName, obj := range objects {
			b := &bytes.Buffer{}
			if err := p.PrintObj(obj, b); err != nil {
				if set, found := expectedErrors[pName]; found && set.Has(oName) {
					// expected error
					continue
				}
				t.Errorf("printer '%v', object '%v'; error: '%v'", pName, oName, err)
			}
		}
	}
}

func TestPrintEventsResultSorted(t *testing.T) {
	// Arrange
	printer := NewHumanReadablePrinter(false /* noHeaders */)

	obj := api.EventList{
		Items: []api.Event{
			{
				Source:    api.EventSource{Component: "kubelet"},
				Message:   "Item 1",
				Timestamp: util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
			},
			{
				Source:    api.EventSource{Component: "scheduler"},
				Message:   "Item 2",
				Timestamp: util.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
			},
			{
				Source:    api.EventSource{Component: "kubelet"},
				Message:   "Item 3",
				Timestamp: util.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
			},
		},
	}
	buffer := &bytes.Buffer{}

	// Act
	err := printer.PrintObj(&obj, buffer)

	// Assert
	if err != nil {
		t.Fatalf("An error occurred printing the EventList: %#v", err)
	}
	out := buffer.String()
	VerifyDatesInOrder(out, "\n" /* rowDelimiter */, "  " /* columnDelimiter */, t)
}
