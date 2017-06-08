/*
Copyright 2014 The Kubernetes Authors.

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

package internalversion

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	yamlserializer "k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	kubectltesting "k8s.io/kubernetes/pkg/kubectl/testing"
	"k8s.io/kubernetes/pkg/printers"
)

func init() {
	api.Scheme.AddKnownTypes(testapi.Default.InternalGroupVersion(), &kubectltesting.TestStruct{})
	api.Scheme.AddKnownTypes(api.Registry.GroupOrDie(api.GroupName).GroupVersion, &kubectltesting.TestStruct{})
}

var testData = kubectltesting.TestStruct{
	Key:        "testValue",
	Map:        map[string]int{"TestSubkey": 1},
	StringList: []string{"a", "b", "c"},
	IntList:    []int{1, 2, 3},
}

func TestVersionedPrinter(t *testing.T) {
	original := &kubectltesting.TestStruct{Key: "value"}
	p := printers.NewVersionedPrinter(
		printers.ResourcePrinterFunc(func(obj runtime.Object, w io.Writer) error {
			if obj == original {
				t.Fatalf("object should not be identical: %#v", obj)
			}
			if obj.(*kubectltesting.TestStruct).Key != "value" {
				t.Fatalf("object was not converted: %#v", obj)
			}
			return nil
		}),
		api.Scheme,
		api.Registry.GroupOrDie(api.GroupName).GroupVersion,
	)
	if err := p.PrintObj(original, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPrintDefault(t *testing.T) {
	printerTests := []struct {
		Name   string
		Format string
	}{
		{"test wide", "wide"},
		{"test blank format", ""},
	}

	for _, test := range printerTests {
		printer, err := printers.GetStandardPrinter(&printers.OutputOptions{AllowMissingKeys: false}, false, nil, nil, api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), []runtime.Decoder{api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme}, printers.PrintOptions{})
		if err != nil {
			t.Errorf("in %s, unexpected error: %#v", test.Name, err)
		}
		if printer.IsGeneric() {
			t.Errorf("in %s, printer should not be generic: %#v", test.Name, printer)
		}
	}
}

type TestPrintType struct {
	Data string
}

func (obj *TestPrintType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

type TestUnknownType struct{}

func (obj *TestUnknownType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

func TestPrinter(t *testing.T) {
	//test inputs
	simpleTest := &TestPrintType{"foo"}
	podTest := &api.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	podListTest := &api.PodList{
		Items: []api.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
		},
	}
	emptyListTest := &api.PodList{}
	testapi, err := api.Scheme.ConvertToVersion(podTest, api.Registry.GroupOrDie(api.GroupName).GroupVersion)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	printerTests := []struct {
		Name           string
		OutputOpts     *printers.OutputOptions
		Input          runtime.Object
		OutputVersions []schema.GroupVersion
		Expect         string
	}{
		{"test json", &printers.OutputOptions{FmtType: "json", AllowMissingKeys: true}, simpleTest, nil, "{\n    \"Data\": \"foo\"\n}\n"},
		{"test yaml", &printers.OutputOptions{FmtType: "yaml", AllowMissingKeys: true}, simpleTest, nil, "Data: foo\n"},
		{"test template", &printers.OutputOptions{FmtType: "template", FmtArg: "{{if .id}}{{.id}}{{end}}{{if .metadata.name}}{{.metadata.name}}{{end}}", AllowMissingKeys: true},
			podTest, []schema.GroupVersion{v1.SchemeGroupVersion}, "foo"},
		{"test jsonpath", &printers.OutputOptions{FmtType: "jsonpath", FmtArg: "{.metadata.name}", AllowMissingKeys: true}, podTest, []schema.GroupVersion{v1.SchemeGroupVersion}, "foo"},
		{"test jsonpath list", &printers.OutputOptions{FmtType: "jsonpath", FmtArg: "{.items[*].metadata.name}", AllowMissingKeys: true}, podListTest, []schema.GroupVersion{v1.SchemeGroupVersion}, "foo bar"},
		{"test jsonpath empty list", &printers.OutputOptions{FmtType: "jsonpath", FmtArg: "{.items[*].metadata.name}", AllowMissingKeys: true}, emptyListTest, []schema.GroupVersion{v1.SchemeGroupVersion}, ""},
		{"test name", &printers.OutputOptions{FmtType: "name", AllowMissingKeys: true}, podTest, []schema.GroupVersion{v1.SchemeGroupVersion}, "pods/foo\n"},
		{"emits versioned objects", &printers.OutputOptions{FmtType: "template", FmtArg: "{{.kind}}", AllowMissingKeys: true}, testapi, []schema.GroupVersion{v1.SchemeGroupVersion}, "Pod"},
	}
	for _, test := range printerTests {
		buf := bytes.NewBuffer([]byte{})
		printer, err := printers.GetStandardPrinter(test.OutputOpts, false, api.Registry.RESTMapper(api.Registry.EnabledVersions()...), api.Scheme, api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), []runtime.Decoder{api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme}, printers.PrintOptions{})
		if err != nil {
			t.Errorf("in %s, unexpected error: %#v", test.Name, err)
		}
		if printer.IsGeneric() && len(test.OutputVersions) > 0 {
			printer = printers.NewVersionedPrinter(printer, api.Scheme, test.OutputVersions...)
		}
		if err := printer.PrintObj(test.Input, buf); err != nil {
			t.Errorf("in %s, unexpected error: %#v", test.Name, err)
		}
		if buf.String() != test.Expect {
			t.Errorf("in %s, expect %q, got %q", test.Name, test.Expect, buf.String())
		}
	}

}

func TestBadPrinter(t *testing.T) {
	badPrinterTests := []struct {
		Name       string
		OutputOpts *printers.OutputOptions
		Error      error
	}{
		{"empty template", &printers.OutputOptions{FmtType: "template", AllowMissingKeys: false}, fmt.Errorf("template format specified but no template given")},
		{"bad template", &printers.OutputOptions{FmtType: "template", FmtArg: "{{ .Name", AllowMissingKeys: false}, fmt.Errorf("error parsing template {{ .Name, template: output:1: unclosed action\n")},
		{"bad templatefile", &printers.OutputOptions{FmtType: "templatefile", AllowMissingKeys: false}, fmt.Errorf("templatefile format specified but no template file given")},
		{"bad jsonpath", &printers.OutputOptions{FmtType: "jsonpath", FmtArg: "{.Name", AllowMissingKeys: false}, fmt.Errorf("error parsing jsonpath {.Name, unclosed action\n")},
		{"unknown format", &printers.OutputOptions{FmtType: "anUnknownFormat", FmtArg: "", AllowMissingKeys: false}, fmt.Errorf("output format \"anUnknownFormat\" not recognized")},
	}
	for _, test := range badPrinterTests {
		_, err := printers.GetStandardPrinter(test.OutputOpts, false, api.Registry.RESTMapper(api.Registry.EnabledVersions()...), api.Scheme, api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), []runtime.Decoder{api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme}, printers.PrintOptions{})
		if err == nil || err.Error() != test.Error.Error() {
			t.Errorf("in %s, expect %s, got %s", test.Name, test.Error, err)
		}
	}
}

func testPrinter(t *testing.T, printer printers.ResourcePrinter, unmarshalFunc func(data []byte, v interface{}) error) {
	buf := bytes.NewBuffer([]byte{})

	err := printer.PrintObj(&testData, buf)
	if err != nil {
		t.Fatal(err)
	}
	var poutput kubectltesting.TestStruct
	// Verify that given function runs without error.
	err = unmarshalFunc(buf.Bytes(), &poutput)
	if err != nil {
		t.Fatal(err)
	}
	// Use real decode function to undo the versioning process.
	poutput = kubectltesting.TestStruct{}
	s := yamlserializer.NewDecodingSerializer(testapi.Default.Codec())
	if err := runtime.DecodeInto(s, buf.Bytes(), &poutput); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testData, poutput) {
		t.Errorf("Test data and unmarshaled data are not equal: %v", diff.ObjectDiff(poutput, testData))
	}

	obj := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
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
	if err := runtime.DecodeInto(s, buf.Bytes(), &objOut); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(obj, &objOut) {
		t.Errorf("Unexpected inequality:\n%v", diff.ObjectDiff(obj, &objOut))
	}
}

func TestYAMLPrinter(t *testing.T) {
	testPrinter(t, &printers.YAMLPrinter{}, yaml.Unmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, &printers.JSONPrinter{}, json.Unmarshal)
}

func TestFormatResourceName(t *testing.T) {
	tests := []struct {
		kind, name string
		want       string
	}{
		{"", "", ""},
		{"", "name", "name"},
		{"kind", "", "kind/"}, // should not happen in practice
		{"kind", "name", "kind/name"},
	}
	for _, tt := range tests {
		if got := printers.FormatResourceName(tt.kind, tt.name, true); got != tt.want {
			t.Errorf("formatResourceName(%q, %q) = %q, want %q", tt.kind, tt.name, got, tt.want)
		}
	}
}

func PrintCustomType(obj *TestPrintType, w io.Writer, options printers.PrintOptions) error {
	data := obj.Data
	kind := options.Kind
	if options.WithKind {
		data = kind + "/" + data
	}
	_, err := fmt.Fprintf(w, "%s", data)
	return err
}

func ErrorPrintHandler(obj *TestPrintType, w io.Writer, options printers.PrintOptions) error {
	return fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []string{"Data"}
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{})
	printer.Handler(columns, nil, PrintCustomType)

	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err != nil {
		t.Fatalf("An error occurred printing the custom type: %#v", err)
	}
	expectedOutput := "DATA\ntest object"
	if buffer.String() != expectedOutput {
		t.Errorf("The data was not printed as expected. Expected:\n%s\nGot:\n%s", expectedOutput, buffer.String())
	}
}

func TestCustomTypePrintingWithKind(t *testing.T) {
	columns := []string{"Data"}
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{})
	printer.Handler(columns, nil, PrintCustomType)
	printer.EnsurePrintWithKind("test")

	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err != nil {
		t.Fatalf("An error occurred printing the custom type: %#v", err)
	}
	expectedOutput := "DATA\ntest/test object"
	if buffer.String() != expectedOutput {
		t.Errorf("The data was not printed as expected. Expected:\n%s\nGot:\n%s", expectedOutput, buffer.String())
	}
}

func TestPrintHandlerError(t *testing.T) {
	columns := []string{"Data"}
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{})
	printer.Handler(columns, nil, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{})
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&TestUnknownType{}, buffer)
	if err == nil {
		t.Errorf("An error was expected from printing unknown type")
	}
}

func TestTemplatePanic(t *testing.T) {
	tmpl := `{{and ((index .currentState.info "foo").state.running.startedAt) .currentState.info.net.state.running.startedAt}}`
	printer, err := printers.NewTemplatePrinter([]byte(tmpl))
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

func TestNamePrinter(t *testing.T) {
	tests := map[string]struct {
		obj    runtime.Object
		expect string
	}{
		"singleObject": {
			&api.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind: "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			"pods/foo\n"},
		"List": {
			&v1.List{
				TypeMeta: metav1.TypeMeta{
					Kind: "List",
				},
				Items: []runtime.RawExtension{
					{
						Raw: []byte(`{"kind": "Pod", "apiVersion": "v1", "metadata": { "name": "foo"}}`),
					},
					{
						Raw: []byte(`{"kind": "Pod", "apiVersion": "v1", "metadata": { "name": "bar"}}`),
					},
				},
			},
			"pods/foo\npods/bar\n"},
	}
	outputOpts := &printers.OutputOptions{FmtType: "name", AllowMissingKeys: false}
	printer, _ := printers.GetStandardPrinter(outputOpts, false, api.Registry.RESTMapper(api.Registry.EnabledVersions()...), api.Scheme, api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), []runtime.Decoder{api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme}, printers.PrintOptions{})
	for name, item := range tests {
		buff := &bytes.Buffer{}
		err := printer.PrintObj(item.obj, buff)
		if err != nil {
			t.Errorf("%v: unexpected err: %v", name, err)
			continue
		}
		got := buff.String()
		if item.expect != got {
			t.Errorf("%v: expected %v, got %v", name, item.expect, got)
		}
	}
}

func TestTemplateStrings(t *testing.T) {
	// This unit tests the "exists" function as well as the template from update.sh
	table := map[string]struct {
		pod    api.Pod
		expect string
	}{
		"nilInfo":   {api.Pod{}, "false"},
		"emptyInfo": {api.Pod{Status: api.PodStatus{ContainerStatuses: []api.ContainerStatus{}}}, "false"},
		"fooExists": {
			api.Pod{
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Name: "foo",
						},
					},
				},
			},
			"false",
		},
		"barExists": {
			api.Pod{
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Name: "bar",
						},
					},
				},
			},
			"false",
		},
		"bothExist": {
			api.Pod{
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Name: "foo",
						},
						{
							Name: "bar",
						},
					},
				},
			},
			"false",
		},
		"barValid": {
			api.Pod{
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Name: "foo",
						},
						{
							Name: "bar",
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: metav1.Time{},
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
					ContainerStatuses: []api.ContainerStatus{
						{
							Name: "foo",
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
						{
							Name: "bar",
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
					},
				},
			},
			"true",
		},
	}
	// The point of this test is to verify that the below template works.
	tmpl := `{{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "foo") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`
	p, err := printers.NewTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}

	printer := printers.NewVersionedPrinter(p, api.Scheme, api.Registry.GroupOrDie(api.GroupName).GroupVersion)

	for name, item := range table {
		buffer := &bytes.Buffer{}
		err = printer.PrintObj(&item.pod, buffer)
		if err != nil {
			t.Errorf("%v: unexpected err: %v", name, err)
			continue
		}
		actual := buffer.String()
		if len(actual) == 0 {
			actual = "false"
		}
		if e := item.expect; e != actual {
			t.Errorf("%v: expected %v, got %v", name, e, actual)
		}
	}
}

func TestPrinters(t *testing.T) {
	om := func(name string) metav1.ObjectMeta { return metav1.ObjectMeta{Name: name} }

	var (
		err              error
		templatePrinter  printers.ResourcePrinter
		templatePrinter2 printers.ResourcePrinter
		jsonpathPrinter  printers.ResourcePrinter
	)

	templatePrinter, err = printers.NewTemplatePrinter([]byte("{{.name}}"))
	if err != nil {
		t.Fatal(err)
	}
	templatePrinter = printers.NewVersionedPrinter(templatePrinter, api.Scheme, v1.SchemeGroupVersion)

	templatePrinter2, err = printers.NewTemplatePrinter([]byte("{{len .items}}"))
	if err != nil {
		t.Fatal(err)
	}
	templatePrinter2 = printers.NewVersionedPrinter(templatePrinter2, api.Scheme, v1.SchemeGroupVersion)

	jsonpathPrinter, err = printers.NewJSONPathPrinter("{.metadata.name}")
	if err != nil {
		t.Fatal(err)
	}
	jsonpathPrinter = printers.NewVersionedPrinter(jsonpathPrinter, api.Scheme, v1.SchemeGroupVersion)

	allPrinters := map[string]printers.ResourcePrinter{
		"humanReadable": printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
			NoHeaders: true,
		}),
		"humanReadableHeaders": printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{}),
		"json":                 &printers.JSONPrinter{},
		"yaml":                 &printers.YAMLPrinter{},
		"template":             templatePrinter,
		"template2":            templatePrinter2,
		"jsonpath":             jsonpathPrinter,
		"name": &printers.NamePrinter{
			Typer:    api.Scheme,
			Decoders: []runtime.Decoder{api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme},
			Mapper:   api.Registry.RESTMapper(api.Registry.EnabledVersions()...),
		},
	}
	AddHandlers((allPrinters["humanReadable"]).(*printers.HumanReadablePrinter))
	AddHandlers((allPrinters["humanReadableHeaders"]).(*printers.HumanReadablePrinter))
	objects := map[string]runtime.Object{
		"pod":             &api.Pod{ObjectMeta: om("pod")},
		"emptyPodList":    &api.PodList{},
		"nonEmptyPodList": &api.PodList{Items: []api.Pod{{}}},
		"endpoints": &api.Endpoints{
			Subsets: []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
				Ports:     []api.EndpointPort{{Port: 8080}},
			}}},
	}
	// map of printer name to set of objects it should fail on.
	expectedErrors := map[string]sets.String{
		"template2": sets.NewString("pod", "emptyPodList", "endpoints"),
		"jsonpath":  sets.NewString("emptyPodList", "nonEmptyPodList", "endpoints"),
	}

	for pName, p := range allPrinters {
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
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{})
	AddHandlers(printer)

	obj := api.EventList{
		Items: []api.Event{
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
			{
				Source:         api.EventSource{Component: "scheduler"},
				Message:        "Item 2",
				FirstTimestamp: metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 3",
				FirstTimestamp: metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
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

func TestPrintNodeStatus(t *testing.T) {
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{})
	AddHandlers(printer)
	table := []struct {
		node   api.Node
		status string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
				Status: api.NodeStatus{Conditions: []api.NodeCondition{
					{Type: api.NodeReady, Status: api.ConditionTrue},
					{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo4"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionFalse}}},
			},
			status: "NotReady",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo5"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionFalse}}},
			},
			status: "NotReady,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo6"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: "InvalidValue", Status: api.ConditionTrue}}},
			},
			status: "Unknown",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo7"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{}}},
			},
			status: "Unknown",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo8"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: "InvalidValue", Status: api.ConditionTrue}}},
			},
			status: "Unknown,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo9"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{}}},
			},
			status: "Unknown,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo12",
					Labels: map[string]string{"kubeadm.alpha.kubernetes.io/role": "node"},
				},
				Status: api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready,node",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.status) {
			t.Fatalf("Expect printing node %s with status %#v, got: %#v", test.node.Name, test.status, buffer.String())
		}
	}
}

func TestPrintNodeOSImage(t *testing.T) {
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
		ColumnLabels: []string{},
		Wide:         true,
	})
	AddHandlers(printer)

	table := []struct {
		node    api.Node
		osImage string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status: api.NodeStatus{
					NodeInfo:  api.NodeSystemInfo{OSImage: "fake-os-image"},
					Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			osImage: "fake-os-image",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status: api.NodeStatus{
					NodeInfo:  api.NodeSystemInfo{KernelVersion: "fake-kernel-version"},
					Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			osImage: "<unknown>",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.osImage) {
			t.Fatalf("Expect printing node %s with os image %#v, got: %#v", test.node.Name, test.osImage, buffer.String())
		}
	}
}

func TestPrintNodeKernelVersion(t *testing.T) {
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
		ColumnLabels: []string{},
		Wide:         true,
	})
	AddHandlers(printer)

	table := []struct {
		node          api.Node
		kernelVersion string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status: api.NodeStatus{
					NodeInfo:  api.NodeSystemInfo{KernelVersion: "fake-kernel-version"},
					Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			kernelVersion: "fake-kernel-version",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status: api.NodeStatus{
					NodeInfo:  api.NodeSystemInfo{OSImage: "fake-os-image"},
					Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			kernelVersion: "<unknown>",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.kernelVersion) {
			t.Fatalf("Expect printing node %s with kernel version %#v, got: %#v", test.node.Name, test.kernelVersion, buffer.String())
		}
	}
}

func TestPrintNodeName(t *testing.T) {
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
		Wide: true,
	})
	AddHandlers(printer)
	table := []struct {
		node api.Node
		Name string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "127.0.0.1"},
				Status:     api.NodeStatus{},
			},
			Name: "127.0.0.1",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: ""},
				Status:     api.NodeStatus{},
			},
			Name: "<unknown>",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.Name) {
			t.Fatalf("Expect printing node %s with node name %#v, got: %#v", test.node.Name, test.Name, buffer.String())
		}
	}
}

func TestPrintNodeExternalIP(t *testing.T) {
	printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
		Wide: true,
	})
	AddHandlers(printer)
	table := []struct {
		node       api.Node
		externalIP string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status:     api.NodeStatus{Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}}},
			},
			externalIP: "1.1.1.1",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status:     api.NodeStatus{Addresses: []api.NodeAddress{{Type: api.NodeInternalIP, Address: "1.1.1.1"}}},
			},
			externalIP: "<none>",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
				Status: api.NodeStatus{Addresses: []api.NodeAddress{
					{Type: api.NodeExternalIP, Address: "2.2.2.2"},
					{Type: api.NodeInternalIP, Address: "3.3.3.3"},
					{Type: api.NodeExternalIP, Address: "4.4.4.4"},
				}},
			},
			externalIP: "2.2.2.2",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.externalIP) {
			t.Fatalf("Expect printing node %s with external ip %#v, got: %#v", test.node.Name, test.externalIP, buffer.String())
		}
	}
}

func contains(fields []string, field string) bool {
	for _, v := range fields {
		if v == field {
			return true
		}
	}
	return false
}

func TestPrintHunmanReadableIngressWithColumnLabels(t *testing.T) {
	ingress := extensions.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test1",
			CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
			Labels: map[string]string{
				"app_name": "kubectl_test_ingress",
			},
		},
		Spec: extensions.IngressSpec{
			Backend: &extensions.IngressBackend{
				ServiceName: "svc",
				ServicePort: intstr.FromInt(93),
			},
		},
		Status: extensions.IngressStatus{
			LoadBalancer: api.LoadBalancerStatus{
				Ingress: []api.LoadBalancerIngress{
					{
						IP:       "2.3.4.5",
						Hostname: "localhost.localdomain",
					},
				},
			},
		},
	}
	buff := bytes.Buffer{}
	printIngress(&ingress, &buff, printers.PrintOptions{
		ColumnLabels: []string{"app_name"},
	})
	output := string(buff.Bytes())
	appName := ingress.ObjectMeta.Labels["app_name"]
	if !strings.Contains(output, appName) {
		t.Errorf("expected to container app_name label value %s, but doesn't %s", appName, output)
	}
}

func TestPrintHumanReadableService(t *testing.T) {
	tests := []api.Service{
		{
			Spec: api.ServiceSpec{
				ClusterIP: "1.2.3.4",
				Type:      "LoadBalancer",
				Ports: []api.ServicePort{
					{
						Port:     80,
						Protocol: "TCP",
					},
				},
			},
			Status: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{
							IP: "2.3.4.5",
						},
						{
							IP: "3.4.5.6",
						},
					},
				},
			},
		},
		{
			Spec: api.ServiceSpec{
				ClusterIP: "1.3.4.5",
				Ports: []api.ServicePort{
					{
						Port:     80,
						Protocol: "TCP",
					},
					{
						Port:     8090,
						Protocol: "UDP",
					},
					{
						Port:     8000,
						Protocol: "TCP",
					},
				},
			},
		},
		{
			Spec: api.ServiceSpec{
				ClusterIP: "1.4.5.6",
				Type:      "LoadBalancer",
				Ports: []api.ServicePort{
					{
						Port:     80,
						Protocol: "TCP",
					},
					{
						Port:     8090,
						Protocol: "UDP",
					},
					{
						Port:     8000,
						Protocol: "TCP",
					},
				},
			},
			Status: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{
							IP: "2.3.4.5",
						},
					},
				},
			},
		},
		{
			Spec: api.ServiceSpec{
				ClusterIP: "1.5.6.7",
				Type:      "LoadBalancer",
				Ports: []api.ServicePort{
					{
						Port:     80,
						Protocol: "TCP",
					},
					{
						Port:     8090,
						Protocol: "UDP",
					},
					{
						Port:     8000,
						Protocol: "TCP",
					},
				},
			},
			Status: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{
							IP: "2.3.4.5",
						},
						{
							IP: "3.4.5.6",
						},
						{
							IP:       "5.6.7.8",
							Hostname: "host5678",
						},
					},
				},
			},
		},
	}

	for _, svc := range tests {
		for _, wide := range []bool{false, true} {
			buff := bytes.Buffer{}
			printService(&svc, &buff, printers.PrintOptions{Wide: wide})
			output := string(buff.Bytes())
			ip := svc.Spec.ClusterIP
			if !strings.Contains(output, ip) {
				t.Errorf("expected to contain ClusterIP %s, but doesn't: %s", ip, output)
			}

			for n, ingress := range svc.Status.LoadBalancer.Ingress {
				ip = ingress.IP
				// For non-wide output, we only guarantee the first IP to be printed
				if (n == 0 || wide) && !strings.Contains(output, ip) {
					t.Errorf("expected to contain ingress ip %s with wide=%v, but doesn't: %s", ip, wide, output)
				}
			}

			for _, port := range svc.Spec.Ports {
				portSpec := fmt.Sprintf("%d/%s", port.Port, port.Protocol)
				if !strings.Contains(output, portSpec) {
					t.Errorf("expected to contain port: %s, but doesn't: %s", portSpec, output)
				}
			}
			// Each service should print on one line
			if 1 != strings.Count(output, "\n") {
				t.Errorf("expected a single newline, found %d", strings.Count(output, "\n"))
			}
		}
	}
}

func TestPrintHumanReadableWithNamespace(t *testing.T) {
	namespaceName := "testnamespace"
	name := "test"
	table := []struct {
		obj          runtime.Object
		isNamespaced bool
	}{
		{
			obj: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"name": "foo",
								"type": "production",
							},
						},
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Image: "foo/bar",
									TerminationMessagePath: api.TerminationMessagePathDefault,
									ImagePullPolicy:        api.PullIfNotPresent,
								},
							},
							RestartPolicy: api.RestartPolicyAlways,
							DNSPolicy:     api.DNSDefault,
							NodeSelector: map[string]string{
								"baz": "blah",
							},
						},
					},
				},
			},
			isNamespaced: true,
		},
		{
			obj: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Ports: []api.ServicePort{
						{
							Port:     80,
							Protocol: "TCP",
						},
					},
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{
								IP: "2.3.4.5",
							},
						},
					},
				},
			},
			isNamespaced: true,
		},
		{
			obj: &api.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
					Ports:     []api.EndpointPort{{Port: 8080}},
				},
				}},
			isNamespaced: true,
		},
		{
			obj: &api.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: name},
			},
			isNamespaced: false,
		},
		{
			obj: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Secrets:    []api.ObjectReference{},
			},
			isNamespaced: true,
		},
		{
			obj: &api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Status:     api.NodeStatus{},
			},
			isNamespaced: false,
		},
		{
			obj: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       api.PersistentVolumeSpec{},
			},
			isNamespaced: false,
		},
		{
			obj: &api.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       api.PersistentVolumeClaimSpec{},
			},
			isNamespaced: true,
		},
		{
			obj: &api.Event{
				ObjectMeta:     metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
			isNamespaced: true,
		},
		{
			obj: &api.LimitRange{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ComponentStatus{
				Conditions: []api.ComponentCondition{
					{Type: api.ComponentHealthy, Status: api.ConditionTrue, Message: "ok", Error: ""},
				},
			},
			isNamespaced: false,
		},
	}

	for i, test := range table {
		if test.isNamespaced {
			// Expect output to include namespace when requested.
			printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
				WithNamespace: true,
			})
			AddHandlers(printer)
			buffer := &bytes.Buffer{}
			err := printer.PrintObj(test.obj, buffer)
			if err != nil {
				t.Fatalf("An error occurred printing object: %#v", err)
			}
			matched := contains(strings.Fields(buffer.String()), fmt.Sprintf("%s", namespaceName))
			if !matched {
				t.Errorf("%d: Expect printing object to contain namespace: %#v", i, test.obj)
			}
		} else {
			// Expect error when trying to get all namespaces for un-namespaced object.
			printer := printers.NewHumanReadablePrinter(nil, nil, printers.PrintOptions{
				WithNamespace: true,
			})
			buffer := &bytes.Buffer{}
			err := printer.PrintObj(test.obj, buffer)
			if err == nil {
				t.Errorf("Expected error when printing un-namespaced type")
			}
		}
	}
}

func TestPrintPodTable(t *testing.T) {
	runningPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test1", Labels: map[string]string{"a": "1", "b": "2"}},
		Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
		Status: api.PodStatus{
			Phase: "Running",
			ContainerStatuses: []api.ContainerStatus{
				{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
				{RestartCount: 3},
			},
		},
	}
	failedPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test2", Labels: map[string]string{"b": "2"}},
		Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
		Status: api.PodStatus{
			Phase: "Failed",
			ContainerStatuses: []api.ContainerStatus{
				{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
				{RestartCount: 3},
			},
		},
	}
	tests := []struct {
		obj          runtime.Object
		opts         printers.PrintOptions
		expect       string
		ignoreLegacy bool
	}{
		{
			obj: runningPod, opts: printers.PrintOptions{},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\ntest1\t1/2\tRunning\t6\t<unknown>\n",
		},
		{
			obj: runningPod, opts: printers.PrintOptions{WithKind: true, Kind: "pods"},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\npods/test1\t1/2\tRunning\t6\t<unknown>\n",
		},
		{
			obj: runningPod, opts: printers.PrintOptions{ShowLabels: true},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\tLABELS\ntest1\t1/2\tRunning\t6\t<unknown>\ta=1,b=2\n",
		},
		{
			obj: &api.PodList{Items: []api.Pod{*runningPod, *failedPod}}, opts: printers.PrintOptions{ShowAll: true, ColumnLabels: []string{"a"}},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\tA\ntest1\t1/2\tRunning\t6\t<unknown>\t1\ntest2\t1/2\tFailed\t6\t<unknown>\t\n",
		},
		{
			obj: runningPod, opts: printers.PrintOptions{NoHeaders: true},
			expect: "test1\t1/2\tRunning\t6\t<unknown>\n",
		},
		{
			obj: failedPod, opts: printers.PrintOptions{},
			expect:       "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\n",
			ignoreLegacy: true, // filtering is not done by the printer in the legacy path
		},
		{
			obj: failedPod, opts: printers.PrintOptions{ShowAll: true},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\ntest2\t1/2\tFailed\t6\t<unknown>\n",
		},
	}

	for i, test := range tests {
		table, err := printers.NewTablePrinter().With(AddHandlers).PrintTable(test.obj, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		buf := &bytes.Buffer{}
		p := printers.NewHumanReadablePrinter(nil, nil, test.opts).With(AddHandlers).AddTabWriter(false)
		if err := p.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if test.expect != buf.String() {
			t.Errorf("%d mismatch:\n%s\n%s", i, strconv.Quote(test.expect), strconv.Quote(buf.String()))
		}
		if test.ignoreLegacy {
			continue
		}

		buf.Reset()
		if err := p.PrintObj(test.obj, buf); err != nil {
			t.Fatal(err)
		}
		if test.expect != buf.String() {
			t.Errorf("%d legacy mismatch:\n%s\n%s", i, strconv.Quote(test.expect), strconv.Quote(buf.String()))
		}
	}
}
func TestPrintPod(t *testing.T) {
	tests := []struct {
		pod    api.Pod
		expect []metav1alpha1.TableRow
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", 6, "<unknown>"}}},
		},
		{
			// Test container error overwrites pod phase
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test2", "1/2", "ContainerWaitingReason", 6, "<unknown>"}}},
		},
		{
			// Test the same as the above but with Terminated state and the first container overwrites the rest
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test3"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
						{State: api.ContainerState{Terminated: &api.ContainerStateTerminated{Reason: "ContainerTerminatedReason"}}, RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test3", "0/2", "ContainerWaitingReason", 6, "<unknown>"}}},
		},
		{
			// Test ready is not enough for reporting running
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test4"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test4", "1/2", "podPhase", 6, "<unknown>"}}},
		},
		{
			// Test ready is not enough for reporting running
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Reason: "OutOfDisk",
					Phase:  "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test5", "1/2", "OutOfDisk", 6, "<unknown>"}}},
		},
	}

	for i, test := range tests {
		rows, err := printPod(&test.pod, printers.PrintOptions{ShowAll: true})
		if err != nil {
			t.Fatal(err)
		}
		for i := range rows {
			rows[i].Object.Object = nil
		}
		if !reflect.DeepEqual(test.expect, rows) {
			t.Errorf("%d mismatch: %s", i, diff.ObjectReflectDiff(test.expect, rows))
		}
	}
}

func TestPrintPodList(t *testing.T) {
	tests := []struct {
		pods   api.PodList
		expect []metav1alpha1.TableRow
	}{
		// Test podList's pod: name, num of containers, restarts, container ready status
		{
			api.PodList{
				Items: []api.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "test1"},
						Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
						Status: api.PodStatus{
							Phase: "podPhase",
							ContainerStatuses: []api.ContainerStatus{
								{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
								{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "test2"},
						Spec:       api.PodSpec{Containers: make([]api.Container, 1)},
						Status: api.PodStatus{
							Phase: "podPhase",
							ContainerStatuses: []api.ContainerStatus{
								{Ready: true, RestartCount: 1, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
							},
						},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "2/2", "podPhase", 6, "<unknown>"}}, {Cells: []interface{}{"test2", "1/1", "podPhase", 1, "<unknown>"}}},
		},
	}

	for _, test := range tests {
		rows, err := printPodList(&test.pods, printers.PrintOptions{ShowAll: true})

		if err != nil {
			t.Fatal(err)
		}
		for i := range rows {
			rows[i].Object.Object = nil
		}
		if !reflect.DeepEqual(test.expect, rows) {
			t.Errorf("mismatch: %s", diff.ObjectReflectDiff(test.expect, rows))
		}
	}
}

func TestPrintNonTerminatedPod(t *testing.T) {
	tests := []struct {
		pod    api.Pod
		expect []metav1alpha1.TableRow
	}{
		{
			// Test pod phase Running should be printed
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodRunning,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "1/2", "Running", 6, "<unknown>"}}},
		},
		{
			// Test pod phase Pending should be printed
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodPending,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test2", "1/2", "Pending", 6, "<unknown>"}}},
		},
		{
			// Test pod phase Unknown should be printed
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test3"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodUnknown,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test3", "1/2", "Unknown", 6, "<unknown>"}}},
		},
		{
			// Test pod phase Succeeded shouldn't be printed
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test4"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodSucceeded,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test4", "1/2", "Succeeded", 6, "<unknown>"}, Conditions: podSuccessConditions}},
		},
		{
			// Test pod phase Failed shouldn't be printed
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodFailed,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test5", "1/2", "Failed", 6, "<unknown>"}, Conditions: podFailedConditions}},
		},
	}

	for i, test := range tests {
		table, err := printers.NewTablePrinter().With(AddHandlers).PrintTable(&test.pod, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		rows := table.Rows
		for i := range rows {
			rows[i].Object.Object = nil
		}
		if !reflect.DeepEqual(test.expect, rows) {
			t.Errorf("%d mismatch: %s", i, diff.ObjectReflectDiff(test.expect, rows))
		}
	}
}

func TestPrintPodWithLabels(t *testing.T) {
	tests := []struct {
		pod          api.Pod
		labelColumns []string
		expect       []metav1alpha1.TableRow
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col1": "asd", "COL2": "zxc"},
				},
				Spec: api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]string{"col1", "COL2"},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", 6, "<unknown>", "asd", "zxc"}}},
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col1": "asd", "COL2": "zxc"},
				},
				Spec: api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]string{},
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", 6, "<unknown>"}}},
		},
	}

	for i, test := range tests {
		table, err := printers.NewTablePrinter().With(AddHandlers).PrintTable(&test.pod, printers.PrintOptions{ColumnLabels: test.labelColumns})
		if err != nil {
			t.Fatal(err)
		}
		rows := table.Rows
		for i := range rows {
			rows[i].Object.Object = nil
		}
		if !reflect.DeepEqual(test.expect, rows) {
			t.Errorf("%d mismatch: %s", i, diff.ObjectReflectDiff(test.expect, rows))
		}
	}
}

type stringTestList []struct {
	name, got, exp string
}

func TestTranslateTimestamp(t *testing.T) {
	tl := stringTestList{
		{"a while from now", translateTimestamp(metav1.Time{Time: time.Now().Add(2.1e9)}), "<invalid>"},
		{"almost now", translateTimestamp(metav1.Time{Time: time.Now().Add(1.9e9)}), "0s"},
		{"now", translateTimestamp(metav1.Time{Time: time.Now()}), "0s"},
		{"unknown", translateTimestamp(metav1.Time{}), "<unknown>"},
		{"30 seconds ago", translateTimestamp(metav1.Time{Time: time.Now().Add(-3e10)}), "30s"},
		{"5 minutes ago", translateTimestamp(metav1.Time{Time: time.Now().Add(-3e11)}), "5m"},
		{"an hour ago", translateTimestamp(metav1.Time{Time: time.Now().Add(-6e12)}), "1h"},
		{"2 days ago", translateTimestamp(metav1.Time{Time: time.Now().UTC().AddDate(0, 0, -2)}), "2d"},
		{"months ago", translateTimestamp(metav1.Time{Time: time.Now().UTC().AddDate(0, 0, -90)}), "90d"},
		{"10 years ago", translateTimestamp(metav1.Time{Time: time.Now().UTC().AddDate(-10, 0, 0)}), "10y"},
	}
	for _, test := range tl {
		if test.got != test.exp {
			t.Errorf("On %v, expected '%v', but got '%v'",
				test.name, test.exp, test.got)
		}
	}
}

func TestPrintDeployment(t *testing.T) {
	tests := []struct {
		deployment extensions.Deployment
		expect     string
		wideExpect string
	}{
		{
			extensions.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: extensions.DeploymentSpec{
					Replicas: 5,
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Name:  "fake-container1",
									Image: "fake-image1",
								},
								{
									Name:  "fake-container2",
									Image: "fake-image2",
								},
							},
						},
					},
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
				Status: extensions.DeploymentStatus{
					Replicas:            10,
					UpdatedReplicas:     2,
					AvailableReplicas:   1,
					UnavailableReplicas: 4,
				},
			},
			"test1\t5\t10\t2\t1\t0s\n",
			"test1\t5\t10\t2\t1\t0s\tfake-container1,fake-container2\tfake-image1,fake-image2\tfoo=bar\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printDeployment(&test.deployment, buf, printers.PrintOptions{})
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
		// print deployment with '-o wide' option
		printDeployment(&test.deployment, buf, printers.PrintOptions{Wide: true})
		if buf.String() != test.wideExpect {
			t.Fatalf("Expected: %s, got: %s", test.wideExpect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintDaemonSet(t *testing.T) {
	tests := []struct {
		ds         extensions.DaemonSet
		startsWith string
	}{
		{
			extensions.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: extensions.DaemonSetSpec{
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{Containers: make([]api.Container, 2)},
					},
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					UpdatedNumberScheduled: 2,
					NumberAvailable:        0,
				},
			},
			"test1\t3\t2\t1\t2\t0\t<none>\t0s\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printDaemonSet(&test.ds, buf, printers.PrintOptions{})
		if !strings.HasPrefix(buf.String(), test.startsWith) {
			t.Fatalf("Expected to start with %s but got %s", test.startsWith, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintJob(t *testing.T) {
	completions := int32(2)
	tests := []struct {
		job    batch.Job
		expect string
	}{
		{
			batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: batch.JobSpec{
					Completions: &completions,
				},
				Status: batch.JobStatus{
					Succeeded: 1,
				},
			},
			"job1\t2\t1\t0s\n",
		},
		{
			batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job2",
					CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: batch.JobSpec{
					Completions: nil,
				},
				Status: batch.JobStatus{
					Succeeded: 0,
				},
			},
			"job2\t<none>\t0\t10y\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printJob(&test.job, buf, printers.PrintOptions{})
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintHPA(t *testing.T) {
	minReplicasVal := int32(2)
	targetUtilizationVal := int32(80)
	currentUtilizationVal := int32(50)
	tests := []struct {
		hpa      autoscaling.HorizontalPodAutoscaler
		expected string
	}{
		// minReplicas unset
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MaxReplicas: 10,
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<none>\t<unset>\t10\t4\t<unknown>\n",
		},
		// pods source type (no current)
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricSource{
								MetricName:         "some-pods-metric",
								TargetAverageValue: *resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown> / 100m\t2\t10\t4\t<unknown>\n",
		},
		// pods source type
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricSource{
								MetricName:         "some-pods-metric",
								TargetAverageValue: *resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricStatus{
								MetricName:          "some-pods-metric",
								CurrentAverageValue: *resource.NewMilliQuantity(50, resource.DecimalSI),
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m / 100m\t2\t10\t4\t<unknown>\n",
		},
		// object source type (no current)
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ObjectMetricSourceType,
							Object: &autoscaling.ObjectMetricSource{
								Target: autoscaling.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								MetricName:  "some-service-metric",
								TargetValue: *resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown> / 100m\t2\t10\t4\t<unknown>\n",
		},
		// object source type
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ObjectMetricSourceType,
							Object: &autoscaling.ObjectMetricSource{
								Target: autoscaling.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								MetricName:  "some-service-metric",
								TargetValue: *resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.ObjectMetricSourceType,
							Object: &autoscaling.ObjectMetricStatus{
								Target: autoscaling.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								MetricName:   "some-service-metric",
								CurrentValue: *resource.NewMilliQuantity(50, resource.DecimalSI),
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m / 100m\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetVal (no current)
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name:               api.ResourceCPU,
								TargetAverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown> / 100m\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetVal
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name:               api.ResourceCPU,
								TargetAverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricStatus{
								Name:                api.ResourceCPU,
								CurrentAverageValue: *resource.NewMilliQuantity(50, resource.DecimalSI),
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m / 100m\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetUtil (no current)
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								TargetAverageUtilization: &targetUtilizationVal,
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown> / 80%\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetUtil
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								TargetAverageUtilization: &targetUtilizationVal,
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricStatus{
								Name: api.ResourceCPU,
								CurrentAverageUtilization: &currentUtilizationVal,
								CurrentAverageValue:       *resource.NewMilliQuantity(40, resource.DecimalSI),
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50% / 80%\t2\t10\t4\t<unknown>\n",
		},
		// multiple specs
		{
			autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricSource{
								MetricName:         "some-pods-metric",
								TargetAverageValue: *resource.NewMilliQuantity(100, resource.DecimalSI),
							},
						},
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								TargetAverageUtilization: &targetUtilizationVal,
							},
						},
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricSource{
								MetricName:         "other-pods-metric",
								TargetAverageValue: *resource.NewMilliQuantity(400, resource.DecimalSI),
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricStatus{
								MetricName:          "some-pods-metric",
								CurrentAverageValue: *resource.NewMilliQuantity(50, resource.DecimalSI),
							},
						},
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricStatus{
								Name: api.ResourceCPU,
								CurrentAverageUtilization: &currentUtilizationVal,
								CurrentAverageValue:       *resource.NewMilliQuantity(40, resource.DecimalSI),
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m / 100m, 50% / 80% + 1 more...\t2\t10\t4\t<unknown>\n",
		},
	}

	buff := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		err := printHorizontalPodAutoscaler(&test.hpa, buff, printers.PrintOptions{})
		if err != nil {
			t.Errorf("expected %q, got error: %v", test.expected, err)
			buff.Reset()
			continue
		}

		if buff.String() != test.expected {
			t.Errorf("expected %q, got %q", test.expected, buff.String())
		}

		buff.Reset()
	}
}

func TestPrintPodShowLabels(t *testing.T) {
	tests := []struct {
		pod        api.Pod
		showLabels bool
		expect     []metav1alpha1.TableRow
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col1": "asd", "COL2": "zxc"},
				},
				Spec: api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			true,
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", 6, "<unknown>", "COL2=zxc,col1=asd"}}},
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col3": "asd", "COL4": "zxc"},
				},
				Spec: api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			false,
			[]metav1alpha1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", 6, "<unknown>"}}},
		},
	}

	for i, test := range tests {
		table, err := printers.NewTablePrinter().With(AddHandlers).PrintTable(&test.pod, printers.PrintOptions{ShowLabels: test.showLabels})
		if err != nil {
			t.Fatal(err)
		}
		rows := table.Rows
		for i := range rows {
			rows[i].Object.Object = nil
		}
		if !reflect.DeepEqual(test.expect, rows) {
			t.Errorf("%d mismatch: %s", i, diff.ObjectReflectDiff(test.expect, rows))
		}
	}
}

func TestPrintService(t *testing.T) {
	tests := []struct {
		service api.Service
		expect  string
	}{
		{
			// Test name, cluster ip, port with protocol
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
					Ports: []api.ServicePort{
						{Protocol: "tcp",
							Port: 2233},
					},
					ClusterIP: "0.0.0.0",
				},
			},
			"test1\t0.0.0.0\t<none>\t2233/tcp\t<unknown>\n",
		},
		{
			// Test name, cluster ip, port:nodePort with protocol
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeClusterIP,
					Ports: []api.ServicePort{
						{Protocol: "tcp",
							Port:     8888,
							NodePort: 9999,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test2\t10.9.8.7\t<none>\t8888:9999/tcp\t<unknown>\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printService(&test.service, buf, printers.PrintOptions{})
		// We ignore time
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPodDisruptionBudget(t *testing.T) {
	minAvailable := intstr.FromInt(22)
	tests := []struct {
		pdb    policy.PodDisruptionBudget
		expect string
	}{
		{
			policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "ns1",
					Name:              "pdb1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MinAvailable: &minAvailable,
				},
				Status: policy.PodDisruptionBudgetStatus{
					PodDisruptionsAllowed: 5,
				},
			},
			"pdb1\t22\tN/A\t5\t0s\n",
		}}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printPodDisruptionBudget(&test.pdb, buf, printers.PrintOptions{})
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestAllowMissingKeys(t *testing.T) {
	tests := []struct {
		Name       string
		OutputOpts *printers.OutputOptions
		Input      runtime.Object
		Expect     string
		Error      string
	}{
		{"test template, allow missing keys", &printers.OutputOptions{FmtType: "template", FmtArg: "{{.blarg}}", AllowMissingKeys: true}, &api.Pod{}, "<no value>", ""},
		{"test template, strict", &printers.OutputOptions{FmtType: "template", FmtArg: "{{.blarg}}", AllowMissingKeys: false}, &api.Pod{}, "", `error executing template "{{.blarg}}": template: output:1:2: executing "output" at <.blarg>: map has no entry for key "blarg"`},
		{"test jsonpath, allow missing keys", &printers.OutputOptions{FmtType: "jsonpath", FmtArg: "{.blarg}", AllowMissingKeys: true}, &api.Pod{}, "", ""},
		{"test jsonpath, strict", &printers.OutputOptions{FmtType: "jsonpath", FmtArg: "{.blarg}", AllowMissingKeys: false}, &api.Pod{}, "", "error executing jsonpath \"{.blarg}\": blarg is not found\n"},
	}
	for _, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		printer, err := printers.GetStandardPrinter(test.OutputOpts, false, api.Registry.RESTMapper(api.Registry.EnabledVersions()...), api.Scheme, api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), []runtime.Decoder{api.Codecs.UniversalDecoder(), unstructured.UnstructuredJSONScheme}, printers.PrintOptions{})
		if err != nil {
			t.Errorf("in %s, unexpected error: %#v", test.Name, err)
		}
		err = printer.PrintObj(test.Input, buf)
		if len(test.Error) == 0 && err != nil {
			t.Errorf("in %s, unexpected error: %v", test.Name, err)
			continue
		}
		if len(test.Error) > 0 {
			if err == nil {
				t.Errorf("in %s, expected to get error: %v", test.Name, test.Error)
			} else if e, a := test.Error, err.Error(); e != a {
				t.Errorf("in %s, expected error %q, got %q", test.Name, e, a)
			}
			continue
		}
		if buf.String() != test.Expect {
			t.Errorf("in %s, expect %q, got %q", test.Name, test.Expect, buf.String())
		}
	}
}

func TestPrintControllerRevision(t *testing.T) {
	tests := []struct {
		history apps.ControllerRevision
		expect  string
	}{
		{
			apps.ControllerRevision{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
					OwnerReferences: []metav1.OwnerReference{
						{
							Controller: boolP(true),
							Kind:       "DaemonSet",
							Name:       "foo",
						},
					},
				},
				Revision: 1,
			},
			"test1\tDaemonSet/foo\t1\t0s\n",
		},
		{
			apps.ControllerRevision{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
					OwnerReferences: []metav1.OwnerReference{
						{
							Controller: boolP(false),
							Kind:       "ABC",
							Name:       "foo",
						},
					},
				},
				Revision: 2,
			},
			"test2\t<none>\t2\t0s\n",
		},
		{
			apps.ControllerRevision{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test3",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
					OwnerReferences:   []metav1.OwnerReference{},
				},
				Revision: 3,
			},
			"test3\t<none>\t3\t0s\n",
		},
		{
			apps.ControllerRevision{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test4",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
					OwnerReferences:   nil,
				},
				Revision: 4,
			},
			"test4\t<none>\t4\t0s\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printControllerRevision(&test.history, buf, printers.PrintOptions{})
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func boolP(b bool) *bool {
	return &b
}
