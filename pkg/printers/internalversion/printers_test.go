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
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	yamlserializer "k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	genericprinters "k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/coordination"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/printers"
)

var testData = TestStruct{
	TypeMeta:   metav1.TypeMeta{APIVersion: "foo/bar", Kind: "TestStruct"},
	Key:        "testValue",
	Map:        map[string]int{"TestSubkey": 1},
	StringList: []string{"a", "b", "c"},
	IntList:    []int{1, 2, 3},
}

type TestStruct struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Key               string         `json:"Key"`
	Map               map[string]int `json:"Map"`
	StringList        []string       `json:"StringList"`
	IntList           []int          `json:"IntList"`
}

func (in *TestStruct) DeepCopyObject() runtime.Object {
	panic("never called")
}

func TestPrintUnstructuredObject(t *testing.T) {
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Test",
			"dummy1":     "present",
			"dummy2":     "present",
			"metadata": map[string]interface{}{
				"name":              "MyName",
				"namespace":         "MyNamespace",
				"creationTimestamp": "2017-04-01T00:00:00Z",
				"resourceVersion":   123,
				"uid":               "00000000-0000-0000-0000-000000000001",
				"dummy3":            "present",
				"labels":            map[string]interface{}{"test": "other"},
			},
			/*"items": []interface{}{
				map[string]interface{}{
					"itemBool": true,
					"itemInt":  42,
				},
			},*/
			"url":    "http://localhost",
			"status": "ok",
		},
	}

	tests := []struct {
		expected string
		options  printers.PrintOptions
		object   runtime.Object
	}{
		{
			expected: "NAME\\s+AGE\nMyName\\s+\\d+",
			object:   obj,
		},
		{
			options: printers.PrintOptions{
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+AGE\nMyNamespace\\s+MyName\\s+\\d+",
			object:   obj,
		},
		{
			options: printers.PrintOptions{
				ShowLabels:    true,
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+AGE\\s+LABELS\nMyNamespace\\s+MyName\\s+\\d+\\w+\\s+test\\=other",
			object:   obj,
		},
		{
			expected: "NAME\\s+AGE\nMyName\\s+\\d+\\w+\nMyName2\\s+\\d+",
			object: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Test",
					"dummy1":     "present",
					"dummy2":     "present",
					"items": []interface{}{
						map[string]interface{}{
							"metadata": map[string]interface{}{
								"name":              "MyName",
								"namespace":         "MyNamespace",
								"creationTimestamp": "2017-04-01T00:00:00Z",
								"resourceVersion":   123,
								"uid":               "00000000-0000-0000-0000-000000000001",
								"dummy3":            "present",
								"labels":            map[string]interface{}{"test": "other"},
							},
						},
						map[string]interface{}{
							"metadata": map[string]interface{}{
								"name":              "MyName2",
								"namespace":         "MyNamespace",
								"creationTimestamp": "2017-04-01T00:00:00Z",
								"resourceVersion":   123,
								"uid":               "00000000-0000-0000-0000-000000000001",
								"dummy3":            "present",
								"labels":            "badlabel",
							},
						},
					},
					"url":    "http://localhost",
					"status": "ok",
				},
			},
		},
	}
	out := bytes.NewBuffer([]byte{})

	for _, test := range tests {
		out.Reset()
		printer := printers.NewTablePrinter(test.options)
		printer.PrintObj(test.object, out)

		matches, err := regexp.MatchString(test.expected, out.String())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !matches {
			t.Errorf("wanted:\n%s\ngot:\n%s", test.expected, out)
		}
	}
}

type TestPrintType struct {
	Data string
}

func (obj *TestPrintType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *TestPrintType) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := *obj
	return &clone
}

type TestUnknownType struct{}

func (obj *TestUnknownType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *TestUnknownType) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := *obj
	return &clone
}

func testPrinter(t *testing.T, printer printers.ResourcePrinter, unmarshalFunc func(data []byte, v interface{}) error) {
	buf := bytes.NewBuffer([]byte{})

	err := printer.PrintObj(&testData, buf)
	if err != nil {
		t.Fatal(err)
	}
	var poutput TestStruct
	// Verify that given function runs without error.
	err = unmarshalFunc(buf.Bytes(), &poutput)
	if err != nil {
		t.Fatal(err)
	}
	// Use real decode function to undo the versioning process.
	poutput = TestStruct{}
	s := yamlserializer.NewDecodingSerializer(testapi.Default.Codec())
	if err := runtime.DecodeInto(s, buf.Bytes(), &poutput); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testData, poutput) {
		t.Errorf("Test data and unmarshaled data are not equal: %v", diff.ObjectDiff(poutput, testData))
	}

	obj := &v1.Pod{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
	}
	// our decoder defaults, so we should default our expected object as well
	legacyscheme.Scheme.Default(obj)
	buf.Reset()
	printer.PrintObj(obj, buf)
	var objOut v1.Pod
	// Verify that given function runs without error.
	err = unmarshalFunc(buf.Bytes(), &objOut)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	// Use real decode function to undo the versioning process.
	objOut = v1.Pod{}
	if err := runtime.DecodeInto(s, buf.Bytes(), &objOut); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(obj, &objOut) {
		t.Errorf("Unexpected inequality:\n%v", diff.ObjectDiff(obj, &objOut))
	}
}

func yamlUnmarshal(data []byte, v interface{}) error {
	return yaml.Unmarshal(data, v)
}

func TestYAMLPrinter(t *testing.T) {
	testPrinter(t, genericprinters.NewTypeSetter(legacyscheme.Scheme).ToPrinter(&genericprinters.YAMLPrinter{}), yamlUnmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, genericprinters.NewTypeSetter(legacyscheme.Scheme).ToPrinter(&genericprinters.JSONPrinter{}), json.Unmarshal)
}

func TestFormatResourceName(t *testing.T) {
	tests := []struct {
		kind schema.GroupKind
		name string
		want string
	}{
		{schema.GroupKind{}, "", ""},
		{schema.GroupKind{}, "name", "name"},
		{schema.GroupKind{Kind: "Kind"}, "", "kind/"}, // should not happen in practice
		{schema.GroupKind{Kind: "Kind"}, "name", "kind/name"},
		{schema.GroupKind{Group: "group", Kind: "Kind"}, "name", "kind.group/name"},
	}
	for _, tt := range tests {
		if got := formatResourceName(tt.kind, tt.name, true); got != tt.want {
			t.Errorf("formatResourceName(%q, %q) = %q, want %q", tt.kind, tt.name, got, tt.want)
		}
	}
}

func PrintCustomType(obj *TestPrintType, options printers.PrintOptions) ([]metav1beta1.TableRow, error) {
	data := obj.Data
	kind := options.Kind
	if options.WithKind {
		data = kind.String() + "/" + data
	}
	return []metav1beta1.TableRow{{Cells: []interface{}{data}}}, nil
}

func ErrorPrintHandler(obj *TestPrintType, options printers.PrintOptions) ([]metav1beta1.TableRow, error) {
	return nil, fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []metav1beta1.TableColumnDefinition{{Name: "Data"}}
	printer := printers.NewTablePrinter(printers.PrintOptions{})
	printer.TableHandler(columns, PrintCustomType)

	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err != nil {
		t.Fatalf("An error occurred printing the custom type: %#v", err)
	}
	expectedOutput := "DATA\ntest object\n"
	if buffer.String() != expectedOutput {
		t.Errorf("The data was not printed as expected. Expected:\n%s\nGot:\n%s", expectedOutput, buffer.String())
	}
}

func TestPrintHandlerError(t *testing.T) {
	columns := []metav1beta1.TableColumnDefinition{{Name: "Data"}}
	printer := printers.NewTablePrinter(printers.PrintOptions{})
	printer.TableHandler(columns, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := printers.NewTablePrinter(printers.PrintOptions{})
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&TestUnknownType{}, buffer)
	if err == nil {
		t.Errorf("An error was expected from printing unknown type")
	}
}

func TestTemplatePanic(t *testing.T) {
	tmpl := `{{and ((index .currentState.info "foo").state.running.startedAt) .currentState.info.net.state.running.startedAt}}`
	printer, err := genericprinters.NewGoTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}
	buffer := &bytes.Buffer{}
	err = printer.PrintObj(&v1.Pod{}, buffer)
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
			&v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind: "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			"pod/foo\n"},
		"List": {
			&unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"kind":       "Pod",
							"apiVersion": "v1",
							"metadata": map[string]interface{}{
								"name": "bar",
							},
						},
					},
				},
			},
			"pod/bar\n"},
	}

	printFlags := genericclioptions.NewPrintFlags("").WithTypeSetter(legacyscheme.Scheme).WithDefaultOutput("name")
	printer, err := printFlags.ToPrinter()
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

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
		pod    v1.Pod
		expect string
	}{
		"nilInfo":   {v1.Pod{}, "false"},
		"emptyInfo": {v1.Pod{Status: v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{}}}, "false"},
		"fooExists": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
						},
					},
				},
			},
			"false",
		},
		"barExists": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "bar",
						},
					},
				},
			},
			"false",
		},
		"bothExist": {
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
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
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
						},
						{
							Name: "bar",
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{
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
			v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							Name: "foo",
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
						{
							Name: "bar",
							State: v1.ContainerState{
								Running: &v1.ContainerStateRunning{
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
	printer, err := genericprinters.NewGoTemplatePrinter([]byte(tmpl))
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

	templatePrinter, err = genericprinters.NewGoTemplatePrinter([]byte("{{.name}}"))
	if err != nil {
		t.Fatal(err)
	}

	templatePrinter2, err = genericprinters.NewGoTemplatePrinter([]byte("{{len .items}}"))
	if err != nil {
		t.Fatal(err)
	}

	jsonpathPrinter, err = genericprinters.NewJSONPathPrinter("{.metadata.name}")
	if err != nil {
		t.Fatal(err)
	}

	genericPrinters := map[string]printers.ResourcePrinter{
		// TODO(juanvallejo): move "generic printer" tests to pkg/kubectl/genericclioptions/printers
		"json":      genericprinters.NewTypeSetter(legacyscheme.Scheme).ToPrinter(&genericprinters.JSONPrinter{}),
		"yaml":      genericprinters.NewTypeSetter(legacyscheme.Scheme).ToPrinter(&genericprinters.YAMLPrinter{}),
		"template":  templatePrinter,
		"template2": templatePrinter2,
		"jsonpath":  jsonpathPrinter,
	}
	objects := map[string]runtime.Object{
		"pod":             &v1.Pod{ObjectMeta: om("pod")},
		"emptyPodList":    &v1.PodList{},
		"nonEmptyPodList": &v1.PodList{Items: []v1.Pod{{}}},
		"endpoints": &v1.Endpoints{
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
				Ports:     []v1.EndpointPort{{Port: 8080}},
			}}},
	}
	// map of printer name to set of objects it should fail on.
	expectedErrors := map[string]sets.String{
		"template2": sets.NewString("pod", "emptyPodList", "endpoints"),
		"jsonpath":  sets.NewString("emptyPodList", "nonEmptyPodList", "endpoints"),
	}

	for pName, p := range genericPrinters {
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

	// a humanreadable printer deals with internal-versioned objects
	humanReadablePrinter := map[string]printers.ResourcePrinter{
		"humanReadable": printers.NewTablePrinter(printers.PrintOptions{
			NoHeaders: true,
		}),
		"humanReadableHeaders": printers.NewTablePrinter(printers.PrintOptions{}),
	}
	AddHandlers((humanReadablePrinter["humanReadable"]).(*printers.HumanReadablePrinter))
	AddHandlers((humanReadablePrinter["humanReadableHeaders"]).(*printers.HumanReadablePrinter))
	for pName, p := range humanReadablePrinter {
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
	printer := printers.NewTablePrinter(printers.PrintOptions{})
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
	printer := printers.NewTablePrinter(printers.PrintOptions{})
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

func TestPrintNodeRole(t *testing.T) {
	printer := printers.NewTablePrinter(printers.PrintOptions{})
	AddHandlers(printer)
	table := []struct {
		node     api.Node
		expected string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo9"},
			},
			expected: "<none>",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo10",
					Labels: map[string]string{"node-role.kubernetes.io/master": "", "node-role.kubernetes.io/proxy": "", "kubernetes.io/role": "node"},
				},
			},
			expected: "master,node,proxy",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo11",
					Labels: map[string]string{"kubernetes.io/role": "node"},
				},
			},
			expected: "node",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.expected) {
			t.Fatalf("Expect printing node %s with role %#v, got: %#v", test.node.Name, test.expected, buffer.String())
		}
	}
}

func TestPrintNodeOSImage(t *testing.T) {
	printer := printers.NewTablePrinter(printers.PrintOptions{
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
	printer := printers.NewTablePrinter(printers.PrintOptions{
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

func TestPrintNodeContainerRuntimeVersion(t *testing.T) {
	printer := printers.NewTablePrinter(printers.PrintOptions{
		ColumnLabels: []string{},
		Wide:         true,
	})
	AddHandlers(printer)

	table := []struct {
		node                    api.Node
		containerRuntimeVersion string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status: api.NodeStatus{
					NodeInfo:  api.NodeSystemInfo{ContainerRuntimeVersion: "foo://1.2.3"},
					Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			containerRuntimeVersion: "foo://1.2.3",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status: api.NodeStatus{
					NodeInfo:  api.NodeSystemInfo{},
					Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			containerRuntimeVersion: "<unknown>",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.containerRuntimeVersion) {
			t.Fatalf("Expect printing node %s with kernel version %#v, got: %#v", test.node.Name, test.containerRuntimeVersion, buffer.String())
		}
	}
}

func TestPrintNodeName(t *testing.T) {
	printer := printers.NewTablePrinter(printers.PrintOptions{
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
	printer := printers.NewTablePrinter(printers.PrintOptions{
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

func TestPrintNodeInternalIP(t *testing.T) {
	printer := printers.NewTablePrinter(printers.PrintOptions{
		Wide: true,
	})
	AddHandlers(printer)
	table := []struct {
		node       api.Node
		internalIP string
	}{
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status:     api.NodeStatus{Addresses: []api.NodeAddress{{Type: api.NodeInternalIP, Address: "1.1.1.1"}}},
			},
			internalIP: "1.1.1.1",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status:     api.NodeStatus{Addresses: []api.NodeAddress{{Type: api.NodeExternalIP, Address: "1.1.1.1"}}},
			},
			internalIP: "<none>",
		},
		{
			node: api.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
				Status: api.NodeStatus{Addresses: []api.NodeAddress{
					{Type: api.NodeInternalIP, Address: "2.2.2.2"},
					{Type: api.NodeExternalIP, Address: "3.3.3.3"},
					{Type: api.NodeInternalIP, Address: "4.4.4.4"},
				}},
			},
			internalIP: "2.2.2.2",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.node, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Node: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.internalIP) {
			t.Fatalf("Expect printing node %s with internal ip %#v, got: %#v", test.node.Name, test.internalIP, buffer.String())
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
	ingress := networking.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test1",
			CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
			Labels: map[string]string{
				"app_name": "kubectl_test_ingress",
			},
		},
		Spec: networking.IngressSpec{
			Backend: &networking.IngressBackend{
				ServiceName: "svc",
				ServicePort: intstr.FromInt(93),
			},
		},
		Status: networking.IngressStatus{
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
	buff := bytes.NewBuffer([]byte{})
	table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&ingress, printers.PrintOptions{ColumnLabels: []string{"app_name"}})
	if err != nil {
		t.Fatal(err)
	}
	verifyTable(t, table)
	printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
	if err := printer.PrintObj(table, buff); err != nil {
		t.Fatal(err)
	}
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
					{
						Port:     7777,
						Protocol: "SCTP",
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
			buff := bytes.NewBuffer([]byte{})
			table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&svc, printers.PrintOptions{Wide: wide})
			if err != nil {
				t.Fatal(err)
			}
			verifyTable(t, table)
			printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
			if err := printer.PrintObj(table, buff); err != nil {
				t.Fatal(err)
			}
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
									Image:                  "foo/bar",
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
				},
			},
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
				ObjectMeta: metav1.ObjectMeta{Name: name},
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
		{
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"kind":       "Foo",
					"apiVersion": "example.com/v1",
					"metadata":   map[string]interface{}{"name": "test", "namespace": namespaceName},
				},
			},
			isNamespaced: true,
		},
		{
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"kind":       "Foo",
					"apiVersion": "example.com/v1",
					"metadata":   map[string]interface{}{"name": "test"},
				},
			},
			isNamespaced: false,
		},
	}

	for i, test := range table {
		printer := printers.NewTablePrinter(printers.PrintOptions{
			WithNamespace: true,
			NoHeaders:     true,
		})
		AddHandlers(printer)
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(test.obj, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing object: %#v", err)
		}
		if test.isNamespaced {
			if !strings.HasPrefix(buffer.String(), namespaceName+" ") {
				t.Errorf("%d: Expect printing object %T to contain namespace %q, got %s", i, test.obj, namespaceName, buffer.String())
			}
		} else {
			if !strings.HasPrefix(buffer.String(), " ") {
				t.Errorf("%d: Expect printing object %T to not contain namespace got %s", i, test.obj, buffer.String())
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
			expect: "NAME    READY   STATUS    RESTARTS   AGE\ntest1   1/2     Running   6          <unknown>\n",
		},
		{
			obj: runningPod, opts: printers.PrintOptions{WithKind: true, Kind: schema.GroupKind{Kind: "Pod"}},
			expect: "NAME        READY   STATUS    RESTARTS   AGE\npod/test1   1/2     Running   6          <unknown>\n",
		},
		{
			obj: runningPod, opts: printers.PrintOptions{ShowLabels: true},
			expect: "NAME    READY   STATUS    RESTARTS   AGE         LABELS\ntest1   1/2     Running   6          <unknown>   a=1,b=2\n",
		},
		{
			obj: &api.PodList{Items: []api.Pod{*runningPod, *failedPod}}, opts: printers.PrintOptions{ColumnLabels: []string{"a"}},
			expect: "NAME    READY   STATUS    RESTARTS   AGE         A\ntest1   1/2     Running   6          <unknown>   1\ntest2   1/2     Failed    6          <unknown>   \n",
		},
		{
			obj: runningPod, opts: printers.PrintOptions{NoHeaders: true},
			expect: "test1   1/2   Running   6     <unknown>\n",
		},
		{
			obj: failedPod, opts: printers.PrintOptions{},
			expect:       "NAME    READY   STATUS   RESTARTS   AGE\ntest2   1/2     Failed   6          <unknown>\n",
			ignoreLegacy: true, // filtering is not done by the printer in the legacy path
		},
		{
			obj: failedPod, opts: printers.PrintOptions{},
			expect: "NAME    READY   STATUS   RESTARTS   AGE\ntest2   1/2     Failed   6          <unknown>\n",
		},
	}

	for i, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(test.obj, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		buf := &bytes.Buffer{}
		p := printers.NewTablePrinter(test.opts).With(AddHandlers)
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
		expect []metav1beta1.TableRow
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test2", "1/2", "ContainerWaitingReason", int64(6), "<unknown>"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test3", "0/2", "ContainerWaitingReason", int64(6), "<unknown>"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test4", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
		{
			// Test ready is not enough for reporting running
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Reason: "podReason",
					Phase:  "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test5", "1/2", "podReason", int64(6), "<unknown>"}}},
		},
		{
			// Test pod has 2 containers, one is running and the other is completed.
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test6"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase:  "Running",
					Reason: "",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Terminated: &api.ContainerStateTerminated{Reason: "Completed", ExitCode: 0}}},
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test6", "1/2", "Running", int64(6), "<unknown>"}}},
		},
	}

	for i, test := range tests {
		rows, err := printPod(&test.pod, printers.PrintOptions{})
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

func TestPrintPodwide(t *testing.T) {
	condition1 := "condition1"
	condition2 := "condition2"
	condition3 := "condition3"
	tests := []struct {
		pod    api.Pod
		expect []metav1beta1.TableRow
	}{
		{
			// Test when the NodeName and PodIP are not none
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   "test1",
					ReadinessGates: []api.PodReadinessGate{
						{
							ConditionType: api.PodConditionType(condition1),
						},
						{
							ConditionType: api.PodConditionType(condition2),
						},
						{
							ConditionType: api.PodConditionType(condition3),
						},
					},
				},
				Status: api.PodStatus{
					Conditions: []api.PodCondition{
						{
							Type:   api.PodConditionType(condition1),
							Status: api.ConditionFalse,
						},
						{
							Type:   api.PodConditionType(condition2),
							Status: api.ConditionTrue,
						},
					},
					Phase:  "podPhase",
					PodIPs: []api.PodIP{{IP: "1.1.1.1"}},
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
					NominatedNodeName: "node1",
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "1.1.1.1", "test1", "node1", "1/3"}}},
		},
		{
			// Test when the NodeName and PodIP are not none
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   "test1",
					ReadinessGates: []api.PodReadinessGate{
						{
							ConditionType: api.PodConditionType(condition1),
						},
						{
							ConditionType: api.PodConditionType(condition2),
						},
						{
							ConditionType: api.PodConditionType(condition3),
						},
					},
				},
				Status: api.PodStatus{
					Conditions: []api.PodCondition{
						{
							Type:   api.PodConditionType(condition1),
							Status: api.ConditionFalse,
						},
						{
							Type:   api.PodConditionType(condition2),
							Status: api.ConditionTrue,
						},
					},
					Phase:  "podPhase",
					PodIPs: []api.PodIP{{IP: "1.1.1.1"}, {IP: "2001:db8::"}},
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
					NominatedNodeName: "node1",
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "1.1.1.1", "test1", "node1", "1/3"}}},
		},
		{
			// Test when the NodeName and PodIP are none
			api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec: api.PodSpec{
					Containers: make([]api.Container, 2),
					NodeName:   "",
				},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test2", "1/2", "ContainerWaitingReason", int64(6), "<unknown>", "<none>", "<none>", "<none>", "<none>"}}},
		},
	}

	for i, test := range tests {
		rows, err := printPod(&test.pod, printers.PrintOptions{Wide: true})
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
		expect []metav1beta1.TableRow
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "2/2", "podPhase", int64(6), "<unknown>"}}, {Cells: []interface{}{"test2", "1/1", "podPhase", int64(1), "<unknown>"}}},
		},
	}

	for _, test := range tests {
		rows, err := printPodList(&test.pods, printers.PrintOptions{})

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
		expect []metav1beta1.TableRow
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "Running", int64(6), "<unknown>"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test2", "1/2", "Pending", int64(6), "<unknown>"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test3", "1/2", "Unknown", int64(6), "<unknown>"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test4", "1/2", "Succeeded", int64(6), "<unknown>"}, Conditions: podSuccessConditions}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test5", "1/2", "Failed", int64(6), "<unknown>"}, Conditions: podFailedConditions}},
		},
	}

	for i, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pod, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
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
		expect       []metav1beta1.TableRow
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "asd", "zxc"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
	}

	for i, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pod, printers.PrintOptions{ColumnLabels: test.labelColumns})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
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

func TestTranslateTimestampSince(t *testing.T) {
	tl := stringTestList{
		{"a while from now", translateTimestampSince(metav1.Time{Time: time.Now().Add(2.1e9)}), "<invalid>"},
		{"almost now", translateTimestampSince(metav1.Time{Time: time.Now().Add(1.9e9)}), "0s"},
		{"now", translateTimestampSince(metav1.Time{Time: time.Now()}), "0s"},
		{"unknown", translateTimestampSince(metav1.Time{}), "<unknown>"},
		{"30 seconds ago", translateTimestampSince(metav1.Time{Time: time.Now().Add(-3e10)}), "30s"},
		{"5 minutes ago", translateTimestampSince(metav1.Time{Time: time.Now().Add(-3e11)}), "5m"},
		{"an hour ago", translateTimestampSince(metav1.Time{Time: time.Now().Add(-6e12)}), "100m"},
		{"2 days ago", translateTimestampSince(metav1.Time{Time: time.Now().UTC().AddDate(0, 0, -2)}), "2d"},
		{"months ago", translateTimestampSince(metav1.Time{Time: time.Now().UTC().AddDate(0, 0, -90)}), "90d"},
		{"10 years ago", translateTimestampSince(metav1.Time{Time: time.Now().UTC().AddDate(-10, 0, 0)}), "10y"},
	}
	for _, test := range tl {
		if test.got != test.exp {
			t.Errorf("On %v, expected '%v', but got '%v'",
				test.name, test.exp, test.got)
		}
	}
}

func TestTranslateTimestampUntil(t *testing.T) {
	// Since this method compares the time with time.Now() internally,
	// small buffers of 0.1 seconds are added on comparing times to consider method call overhead.
	// Otherwise, the output strings become shorter than expected.
	const buf = 1e8
	tl := stringTestList{
		{"a while ago", translateTimestampUntil(metav1.Time{Time: time.Now().Add(-2.1e9)}), "<invalid>"},
		{"almost now", translateTimestampUntil(metav1.Time{Time: time.Now().Add(-1.9e9)}), "0s"},
		{"now", translateTimestampUntil(metav1.Time{Time: time.Now()}), "0s"},
		{"unknown", translateTimestampUntil(metav1.Time{}), "<unknown>"},
		{"in 30 seconds", translateTimestampUntil(metav1.Time{Time: time.Now().Add(3e10 + buf)}), "30s"},
		{"in 5 minutes", translateTimestampUntil(metav1.Time{Time: time.Now().Add(3e11 + buf)}), "5m"},
		{"in an hour", translateTimestampUntil(metav1.Time{Time: time.Now().Add(6e12 + buf)}), "100m"},
		{"in 2 days", translateTimestampUntil(metav1.Time{Time: time.Now().UTC().AddDate(0, 0, 2).Add(buf)}), "2d"},
		{"in months", translateTimestampUntil(metav1.Time{Time: time.Now().UTC().AddDate(0, 0, 90).Add(buf)}), "90d"},
		{"in 10 years", translateTimestampUntil(metav1.Time{Time: time.Now().UTC().AddDate(10, 0, 0).Add(buf)}), "10y"},
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
		deployment apps.Deployment
		expect     string
		wideExpect string
	}{
		{
			apps.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: apps.DeploymentSpec{
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
				Status: apps.DeploymentStatus{
					Replicas:            10,
					UpdatedReplicas:     2,
					AvailableReplicas:   1,
					UnavailableReplicas: 4,
				},
			},
			"test1   0/5   2     1     0s\n",
			"test1   0/5   2     1     0s    fake-container1,fake-container2   fake-image1,fake-image2   foo=bar\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.deployment, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
		table, err = printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.deployment, printers.PrintOptions{Wide: true})
		verifyTable(t, table)
		// print deployment with '-o wide' option
		printer = printers.NewTablePrinter(printers.PrintOptions{Wide: true, NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.wideExpect {
			t.Fatalf("Expected: %s, got: %s", test.wideExpect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintDaemonSet(t *testing.T) {
	tests := []struct {
		ds         apps.DaemonSet
		startsWith string
	}{
		{
			apps.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: apps.DaemonSetSpec{
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{Containers: make([]api.Container, 2)},
					},
				},
				Status: apps.DaemonSetStatus{
					CurrentNumberScheduled: 2,
					DesiredNumberScheduled: 3,
					NumberReady:            1,
					UpdatedNumberScheduled: 2,
					NumberAvailable:        0,
				},
			},
			"test1   3     2     1     2     0     <none>   0s",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.ds, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(buf.String(), test.startsWith) {
			t.Fatalf("Expected to start with %s but got %s", test.startsWith, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintJob(t *testing.T) {
	now := time.Now()
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
			"job1   1/2         0s\n",
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
			"job2   0/1         10y\n",
		},
		{
			batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job3",
					CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: batch.JobSpec{
					Completions: nil,
				},
				Status: batch.JobStatus{
					Succeeded:      0,
					StartTime:      &metav1.Time{Time: now.Add(time.Minute)},
					CompletionTime: &metav1.Time{Time: now.Add(31 * time.Minute)},
				},
			},
			"job3   0/1   30m   10y\n",
		},
		{
			batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job4",
					CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: batch.JobSpec{
					Completions: nil,
				},
				Status: batch.JobStatus{
					Succeeded: 0,
					StartTime: &metav1.Time{Time: time.Now().Add(-20 * time.Minute)},
				},
			},
			"job4   0/1   20m   10y\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.job, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintHPA(t *testing.T) {
	minReplicasVal := int32(2)
	targetUtilizationVal := int32(80)
	currentUtilizationVal := int32(50)
	metricLabelSelector, err := metav1.ParseToLabelSelector("label=value")
	if err != nil {
		t.Errorf("unable to parse label selector: %v", err)
	}
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
			"some-hpa   ReplicationController/some-rc   <none>   <unset>   10    4     <unknown>\n",
		},
		// external source type, target average value (no current)
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
							Type: autoscaling.ExternalMetricSourceType,
							External: &autoscaling.ExternalMetricSource{
								Metric: autoscaling.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa   ReplicationController/some-rc   <unknown>/100m (avg)   2     10    4     <unknown>\n",
		},
		// external source type, target average value
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
							Type: autoscaling.ExternalMetricSourceType,
							External: &autoscaling.ExternalMetricSource{
								Metric: autoscaling.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.ExternalMetricSourceType,
							External: &autoscaling.ExternalMetricStatus{
								Metric: autoscaling.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Current: autoscaling.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50m/100m (avg)   2     10    4     <unknown>\n",
		},
		// external source type, target value (no current)
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
							Type: autoscaling.ExternalMetricSourceType,
							External: &autoscaling.ExternalMetricSource{
								Metric: autoscaling.MetricIdentifier{
									Name:     "some-service-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscaling.MetricTarget{
									Type:  autoscaling.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa   ReplicationController/some-rc   <unknown>/100m   2     10    4     <unknown>\n",
		},
		// external source type, target value
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
							Type: autoscaling.ExternalMetricSourceType,
							External: &autoscaling.ExternalMetricSource{
								Metric: autoscaling.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscaling.MetricTarget{
									Type:  autoscaling.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscaling.MetricStatus{
						{
							Type: autoscaling.ExternalMetricSourceType,
							External: &autoscaling.ExternalMetricStatus{
								Metric: autoscaling.MetricIdentifier{
									Name: "some-external-metric",
								},
								Current: autoscaling.MetricValueStatus{
									Value: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50m/100m   2     10    4     <unknown>\n",
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
								Metric: autoscaling.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa   ReplicationController/some-rc   <unknown>/100m   2     10    4     <unknown>\n",
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
								Metric: autoscaling.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
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
								Metric: autoscaling.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscaling.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50m/100m   2     10    4     <unknown>\n",
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
								DescribedObject: autoscaling.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscaling.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscaling.MetricTarget{
									Type:  autoscaling.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa   ReplicationController/some-rc   <unknown>/100m   2     10    4     <unknown>\n",
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
								DescribedObject: autoscaling.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscaling.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscaling.MetricTarget{
									Type:  autoscaling.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
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
								DescribedObject: autoscaling.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscaling.MetricIdentifier{
									Name: "some-service-metric",
								},
								Current: autoscaling.MetricValueStatus{
									Value: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50m/100m   2     10    4     <unknown>\n",
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
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa   ReplicationController/some-rc   <unknown>/100m   2     10    4     <unknown>\n",
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
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
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
								Current: autoscaling.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50m/100m   2     10    4     <unknown>\n",
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
								Target: autoscaling.MetricTarget{
									Type:               autoscaling.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscaling.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa   ReplicationController/some-rc   <unknown>/80%   2     10    4     <unknown>\n",
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
								Target: autoscaling.MetricTarget{
									Type:               autoscaling.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
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
								Current: autoscaling.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50%/80%   2     10    4     <unknown>\n",
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
								Metric: autoscaling.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:               autoscaling.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
						{
							Type: autoscaling.PodsMetricSourceType,
							Pods: &autoscaling.PodsMetricSource{
								Metric: autoscaling.MetricIdentifier{
									Name: "other-pods-metric",
								},
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(400, resource.DecimalSI),
								},
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
								Metric: autoscaling.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscaling.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricStatus{
								Name: api.ResourceCPU,
								Current: autoscaling.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa   ReplicationController/some-rc   50m/100m, 50%/80% + 1 more...   2     10    4     <unknown>\n",
		},
	}

	buff := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.hpa, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buff); err != nil {
			t.Fatal(err)
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
		expect     []metav1beta1.TableRow
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "COL2=zxc,col1=asd"}}},
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
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
	}

	for i, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pod, printers.PrintOptions{ShowLabels: test.showLabels})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
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
	singleExternalIP := []string{"80.11.12.10"}
	mulExternalIP := []string{"80.11.12.10", "80.11.12.11"}
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
						{
							Protocol: "tcp",
							Port:     2233,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test1   ClusterIP   10.9.8.7   <none>   2233/tcp   <unknown>\n",
		},
		{
			// Test NodePort service
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeNodePort,
					Ports: []api.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
							NodePort: 9999,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test2   NodePort   10.9.8.7   <none>   8888:9999/tcp   <unknown>\n",
		},
		{
			// Test LoadBalancer service
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test3"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
					Ports: []api.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test3   LoadBalancer   10.9.8.7   <pending>   8888/tcp   <unknown>\n",
		},
		{
			// Test LoadBalancer service with single ExternalIP and no LoadBalancerStatus
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test4"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
					Ports: []api.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP:   "10.9.8.7",
					ExternalIPs: singleExternalIP,
				},
			},
			"test4   LoadBalancer   10.9.8.7   80.11.12.10   8888/tcp   <unknown>\n",
		},
		{
			// Test LoadBalancer service with single ExternalIP
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
					Ports: []api.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP:   "10.9.8.7",
					ExternalIPs: singleExternalIP,
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{
								IP:       "3.4.5.6",
								Hostname: "test.cluster.com",
							},
						},
					},
				},
			},
			"test5   LoadBalancer   10.9.8.7   3.4.5.6,80.11.12.10   8888/tcp   <unknown>\n",
		},
		{
			// Test LoadBalancer service with mul ExternalIPs
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test6"},
				Spec: api.ServiceSpec{
					Type: api.ServiceTypeLoadBalancer,
					Ports: []api.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP:   "10.9.8.7",
					ExternalIPs: mulExternalIP,
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{
								IP:       "2.3.4.5",
								Hostname: "test.cluster.local",
							},
							{
								IP:       "3.4.5.6",
								Hostname: "test.cluster.com",
							},
						},
					},
				},
			},
			"test6   LoadBalancer   10.9.8.7   2.3.4.5,3.4.5.6,80.11.12.10,80.11.12.11   8888/tcp   <unknown>\n",
		},
		{
			// Test ExternalName service
			api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test7"},
				Spec: api.ServiceSpec{
					Type:         api.ServiceTypeExternalName,
					ExternalName: "my.database.example.com",
				},
			},
			"test7   ExternalName   <none>   my.database.example.com   <none>   <unknown>\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.service, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		// We ignore time
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPodDisruptionBudget(t *testing.T) {
	minAvailable := intstr.FromInt(22)
	maxUnavailable := intstr.FromInt(11)
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
			"pdb1   22    N/A   5     0s\n",
		},
		{
			policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "ns2",
					Name:              "pdb2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: policy.PodDisruptionBudgetSpec{
					MaxUnavailable: &maxUnavailable,
				},
				Status: policy.PodDisruptionBudgetStatus{
					PodDisruptionsAllowed: 5,
				},
			},
			"pdb2   N/A   11    5     0s\n",
		}}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pdb, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
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
							APIVersion: "apps/v1",
							Kind:       "DaemonSet",
							Name:       "foo",
						},
					},
				},
				Revision: 1,
			},
			"test1   daemonset.apps/foo   1     0s\n",
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
			"test2   <none>   2     0s\n",
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
			"test3   <none>   3     0s\n",
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
			"test4   <none>   4     0s\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.history, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func boolP(b bool) *bool {
	return &b
}

func TestPrintReplicaSet(t *testing.T) {
	tests := []struct {
		replicaSet apps.ReplicaSet
		expect     string
		wideExpect string
	}{
		{
			apps.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: apps.ReplicaSetSpec{
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
				Status: apps.ReplicaSetStatus{
					Replicas:      5,
					ReadyReplicas: 2,
				},
			},
			"test1   5     5     2     0s\n",
			"test1   5     5     2     0s    fake-container1,fake-container2   fake-image1,fake-image2   foo=bar\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.replicaSet, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()

		table, err = printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.replicaSet, printers.PrintOptions{Wide: true})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer = printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true, Wide: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.wideExpect {
			t.Errorf("Expected: %s, got: %s", test.wideExpect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPersistentVolume(t *testing.T) {
	myScn := "my-scn"

	claimRef := api.ObjectReference{
		Name:      "test",
		Namespace: "default",
	}
	tests := []struct {
		pv     api.PersistentVolume
		expect string
	}{
		{
			// Test bound
			api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test1",
				},
				Spec: api.PersistentVolumeSpec{
					ClaimRef:    &claimRef,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("4Gi"),
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeBound,
				},
			},
			"test1   4Gi   ROX         Bound   default/test               <unknown>\n",
		},
		{
			// // Test failed
			api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test2",
				},
				Spec: api.PersistentVolumeSpec{
					ClaimRef:    &claimRef,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("4Gi"),
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeFailed,
				},
			},
			"test2   4Gi   ROX         Failed   default/test               <unknown>\n",
		},
		{
			// Test pending
			api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test3",
				},
				Spec: api.PersistentVolumeSpec{
					ClaimRef:    &claimRef,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteMany},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumePending,
				},
			},
			"test3   10Gi   RWX         Pending   default/test               <unknown>\n",
		},
		{
			// Test pending, storageClass
			api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test4",
				},
				Spec: api.PersistentVolumeSpec{
					ClaimRef:         &claimRef,
					StorageClassName: myScn,
					AccessModes:      []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumePending,
				},
			},
			"test4   10Gi   RWO         Pending   default/test   my-scn         <unknown>\n",
		},
		{
			// Test available
			api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test5",
				},
				Spec: api.PersistentVolumeSpec{
					ClaimRef:         &claimRef,
					StorageClassName: myScn,
					AccessModes:      []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeAvailable,
				},
			},
			"test5   10Gi   RWO         Available   default/test   my-scn         <unknown>\n",
		},
		{
			// Test released
			api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test6",
				},
				Spec: api.PersistentVolumeSpec{
					ClaimRef:         &claimRef,
					StorageClassName: myScn,
					AccessModes:      []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeReleased,
				},
			},
			"test6   10Gi   RWO         Released   default/test   my-scn         <unknown>\n",
		},
	}
	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pv, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPersistentVolumeClaim(t *testing.T) {
	volumeMode := api.PersistentVolumeFilesystem
	myScn := "my-scn"
	tests := []struct {
		pvc    api.PersistentVolumeClaim
		expect string
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test1",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "my-volume",
					VolumeMode: &volumeMode,
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase:       api.ClaimBound,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("4Gi"),
					},
				},
			},
			"test1   Bound   my-volume   4Gi   ROX         <unknown>   Filesystem\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test2",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeMode: &volumeMode,
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase:       api.ClaimLost,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("4Gi"),
					},
				},
			},
			"test2   Lost                           <unknown>   Filesystem\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test3",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "my-volume",
					VolumeMode: &volumeMode,
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase:       api.ClaimPending,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteMany},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
			},
			"test3   Pending   my-volume   10Gi   RWX         <unknown>   Filesystem\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test4",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName:       "my-volume",
					StorageClassName: &myScn,
					VolumeMode:       &volumeMode,
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase:       api.ClaimPending,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
			},
			"test4   Pending   my-volume   10Gi   RWO   my-scn   <unknown>   Filesystem\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test5",
				},
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName:       "my-volume",
					StorageClassName: &myScn,
				},
				Status: api.PersistentVolumeClaimStatus{
					Phase:       api.ClaimPending,
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: map[api.ResourceName]resource.Quantity{
						api.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
			},
			"test5   Pending   my-volume   10Gi   RWO   my-scn   <unknown>   <unset>\n",
		},
	}
	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pvc, printers.PrintOptions{Wide: true})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true, Wide: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintCronJob(t *testing.T) {
	suspend := false
	tests := []struct {
		cronjob batch.CronJob
		expect  string
	}{
		{
			batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "cronjob1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: batch.CronJobSpec{
					Schedule: "0/5 * * * ?",
					Suspend:  &suspend,
				},
				Status: batch.CronJobStatus{
					LastScheduleTime: &metav1.Time{Time: time.Now().Add(1.9e9)},
				},
			},
			"cronjob1   0/5 * * * ?   False   0     0s    0s\n",
		},
		{
			batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "cronjob2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Spec: batch.CronJobSpec{
					Schedule: "0/5 * * * ?",
					Suspend:  &suspend,
				},
				Status: batch.CronJobStatus{
					LastScheduleTime: &metav1.Time{Time: time.Now().Add(-3e10)},
				},
			},
			"cronjob2   0/5 * * * ?   False   0     30s   5m\n",
		},
		{
			batch.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "cronjob3",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Spec: batch.CronJobSpec{
					Schedule: "0/5 * * * ?",
					Suspend:  &suspend,
				},
				Status: batch.CronJobStatus{},
			},
			"cronjob3   0/5 * * * ?   False   0     <none>   5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.cronjob, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintStorageClass(t *testing.T) {
	tests := []struct {
		sc     storage.StorageClass
		expect string
	}{
		{
			storage.StorageClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "sc1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Provisioner: "kubernetes.io/glusterfs",
			},
			"sc1   kubernetes.io/glusterfs   0s\n",
		},
		{
			storage.StorageClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "sc2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Provisioner: "kubernetes.io/nfs",
			},
			"sc2   kubernetes.io/nfs   5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.sc, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintLease(t *testing.T) {
	holder1 := "holder1"
	holder2 := "holder2"
	tests := []struct {
		sc     coordination.Lease
		expect string
	}{
		{
			coordination.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "lease1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: coordination.LeaseSpec{
					HolderIdentity: &holder1,
				},
			},
			"lease1   holder1   0s\n",
		},
		{
			coordination.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "lease2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Spec: coordination.LeaseSpec{
					HolderIdentity: &holder2,
				},
			},
			"lease2   holder2   5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.sc, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPriorityClass(t *testing.T) {
	tests := []struct {
		pc     scheduling.PriorityClass
		expect string
	}{
		{
			scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pc1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Value: 1,
			},
			"pc1   1     false   0s\n",
		},
		{
			scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pc2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Value:         1000000000,
				GlobalDefault: true,
			},
			"pc2   1000000000   true   5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.pc, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintRuntimeClass(t *testing.T) {
	tests := []struct {
		rc     nodeapi.RuntimeClass
		expect string
	}{
		{
			nodeapi.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "rc1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Handler: "h1",
			},
			"rc1   h1    0s\n",
		},
		{
			nodeapi.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "rc2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Handler: "h2",
			},
			"rc2   h2    5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := printers.NewTableGenerator().With(AddHandlers).GenerateTable(&test.rc, printers.PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		printer := printers.NewTablePrinter(printers.PrintOptions{NoHeaders: true})
		if err := printer.PrintObj(table, buf); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Errorf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func verifyTable(t *testing.T, table *metav1beta1.Table) {
	var panicErr interface{}
	func() {
		defer func() {
			panicErr = recover()
		}()
		table.DeepCopyObject() // cells are untyped, better check that types are JSON types and can be deep copied
	}()

	if panicErr != nil {
		t.Errorf("unexpected panic during deepcopy of table %#v: %v", table, panicErr)
	}
}

// VerifyDatesInOrder checks the start of each line for a RFC1123Z date
// and posts error if all subsequent dates are not equal or increasing
func VerifyDatesInOrder(
	resultToTest, rowDelimiter, columnDelimiter string, t *testing.T) {
	lines := strings.Split(resultToTest, rowDelimiter)
	var previousTime time.Time
	for _, str := range lines {
		columns := strings.Split(str, columnDelimiter)
		if len(columns) > 0 {
			currentTime, err := time.Parse(time.RFC1123Z, columns[0])
			if err == nil {
				if previousTime.After(currentTime) {
					t.Errorf(
						"Output is not sorted by time. %s should be listed after %s. Complete output: %s",
						previousTime.Format(time.RFC1123Z),
						currentTime.Format(time.RFC1123Z),
						resultToTest)
				}
				previousTime = currentTime
			}
		}
	}
}
