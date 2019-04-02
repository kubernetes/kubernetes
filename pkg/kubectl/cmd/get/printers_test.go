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

package get

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	nodev1beta1 "k8s.io/api/node/v1beta1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	storagev1 "k8s.io/api/storage/v1"
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
	"k8s.io/kubernetes/pkg/kubectl/scheme"
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
		options  PrintOptions
		object   runtime.Object
	}{
		{
			expected: "NAME\\s+AGE\nMyName\\s+\\d+",
			object:   obj,
		},
		{
			options: PrintOptions{
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+AGE\nMyNamespace\\s+MyName\\s+\\d+",
			object:   obj,
		},
		{
			options: PrintOptions{
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
		printer := NewHumanReadablePrinter(nil, test.options).With(AddDefaultHandlers)
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

func testPrinter(t *testing.T, printer genericprinters.ResourcePrinter, unmarshalFunc func(data []byte, v interface{}) error) {
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
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	s := yamlserializer.NewDecodingSerializer(codec)
	if err := runtime.DecodeInto(s, buf.Bytes(), &poutput); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testData, poutput) {
		t.Errorf("Test data and unmarshaled data are not equal: %v", diff.ObjectDiff(poutput, testData))
	}

	obj := &corev1.Pod{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
	}
	buf.Reset()
	printer.PrintObj(obj, buf)
	var objOut corev1.Pod
	// Verify that given function runs without error.
	err = unmarshalFunc(buf.Bytes(), &objOut)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	// Use real decode function to undo the versioning process.
	objOut = corev1.Pod{}
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
	testPrinter(t, genericprinters.NewTypeSetter(scheme.Scheme).ToPrinter(&genericprinters.YAMLPrinter{}), yamlUnmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, genericprinters.NewTypeSetter(scheme.Scheme).ToPrinter(&genericprinters.JSONPrinter{}), json.Unmarshal)
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
		if got := FormatResourceName(tt.kind, tt.name, true); got != tt.want {
			t.Errorf("formatResourceName(%q, %q) = %q, want %q", tt.kind, tt.name, got, tt.want)
		}
	}
}

func PrintCustomType(obj *TestPrintType, w io.Writer, options PrintOptions) error {
	data := obj.Data
	kind := options.Kind
	if options.WithKind {
		data = kind.String() + "/" + data
	}
	_, err := fmt.Fprintf(w, "%s", data)
	return err
}

func ErrorPrintHandler(obj *TestPrintType, w io.Writer, options PrintOptions) error {
	return fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(nil, PrintOptions{})
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

func TestPrintHandlerError(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(nil, PrintOptions{})
	printer.Handler(columns, nil, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{})
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
	err = printer.PrintObj(&corev1.Pod{}, buffer)
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
			&corev1.Pod{
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

	printFlags := genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme).WithDefaultOutput("name")
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
		pod    corev1.Pod
		expect string
	}{
		"nilInfo":   {corev1.Pod{}, "false"},
		"emptyInfo": {corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{}}}, "false"},
		"fooExists": {
			corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							Name: "foo",
						},
					},
				},
			},
			"false",
		},
		"barExists": {
			corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							Name: "bar",
						},
					},
				},
			},
			"false",
		},
		"bothExist": {
			corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
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
			corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							Name: "foo",
						},
						{
							Name: "bar",
							State: corev1.ContainerState{
								Running: &corev1.ContainerStateRunning{
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
			corev1.Pod{
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							Name: "foo",
							State: corev1.ContainerState{
								Running: &corev1.ContainerStateRunning{
									StartedAt: metav1.Time{},
								},
							},
						},
						{
							Name: "bar",
							State: corev1.ContainerState{
								Running: &corev1.ContainerStateRunning{
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
		templatePrinter  genericprinters.ResourcePrinter
		templatePrinter2 genericprinters.ResourcePrinter
		jsonpathPrinter  genericprinters.ResourcePrinter
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

	genericPrinters := map[string]genericprinters.ResourcePrinter{
		// TODO(juanvallejo): move "generic printer" tests to pkg/kubectl/genericclioptions/printers
		"json":      genericprinters.NewTypeSetter(scheme.Scheme).ToPrinter(&genericprinters.JSONPrinter{}),
		"yaml":      genericprinters.NewTypeSetter(scheme.Scheme).ToPrinter(&genericprinters.YAMLPrinter{}),
		"template":  templatePrinter,
		"template2": templatePrinter2,
		"jsonpath":  jsonpathPrinter,
	}
	objects := map[string]runtime.Object{
		"pod":             &corev1.Pod{ObjectMeta: om("pod")},
		"emptyPodList":    &corev1.PodList{},
		"nonEmptyPodList": &corev1.PodList{Items: []corev1.Pod{{}}},
		"endpoints": &corev1.Endpoints{
			Subsets: []corev1.EndpointSubset{{
				Addresses: []corev1.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
				Ports:     []corev1.EndpointPort{{Port: 8080}},
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
	humanReadablePrinter := map[string]genericprinters.ResourcePrinter{
		"humanReadable": NewHumanReadablePrinter(nil, PrintOptions{
			NoHeaders: true,
		}),
		"humanReadableHeaders": NewHumanReadablePrinter(nil, PrintOptions{}),
	}
	AddHandlers((humanReadablePrinter["humanReadable"]).(*HumanReadablePrinter))
	AddHandlers((humanReadablePrinter["humanReadableHeaders"]).(*HumanReadablePrinter))
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
	printer := NewHumanReadablePrinter(nil, PrintOptions{})
	AddHandlers(printer)

	obj := corev1.EventList{
		Items: []corev1.Event{
			{
				Source:         corev1.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           corev1.EventTypeNormal,
			},
			{
				Source:         corev1.EventSource{Component: "scheduler"},
				Message:        "Item 2",
				FirstTimestamp: metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           corev1.EventTypeNormal,
			},
			{
				Source:         corev1.EventSource{Component: "kubelet"},
				Message:        "Item 3",
				FirstTimestamp: metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           corev1.EventTypeNormal,
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
	printer := NewHumanReadablePrinter(nil, PrintOptions{})
	AddHandlers(printer)
	table := []struct {
		node   corev1.Node
		status string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Spec:       corev1.NodeSpec{Unschedulable: true},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}},
			},
			status: "Ready,SchedulingDisabled",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
				Status: corev1.NodeStatus{Conditions: []corev1.NodeCondition{
					{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
					{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo4"},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionFalse}}},
			},
			status: "NotReady",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo5"},
				Spec:       corev1.NodeSpec{Unschedulable: true},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionFalse}}},
			},
			status: "NotReady,SchedulingDisabled",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo6"},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: "InvalidValue", Status: corev1.ConditionTrue}}},
			},
			status: "Unknown",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo7"},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{}}},
			},
			status: "Unknown",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo8"},
				Spec:       corev1.NodeSpec{Unschedulable: true},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: "InvalidValue", Status: corev1.ConditionTrue}}},
			},
			status: "Unknown,SchedulingDisabled",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo9"},
				Spec:       corev1.NodeSpec{Unschedulable: true},
				Status:     corev1.NodeStatus{Conditions: []corev1.NodeCondition{{}}},
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
		if !containsField(strings.Fields(buffer.String()), test.status) {
			t.Fatalf("Expect printing node %s with status %#v, got: %#v", test.node.Name, test.status, buffer.String())
		}
	}
}

func TestPrintNodeRole(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{})
	AddHandlers(printer)
	table := []struct {
		node     corev1.Node
		expected string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo9"},
			},
			expected: "<none>",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo10",
					Labels: map[string]string{"node-role.kubernetes.io/master": "", "node-role.kubernetes.io/proxy": "", "kubernetes.io/role": "node"},
				},
			},
			expected: "master,node,proxy",
		},
		{
			node: corev1.Node{
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
		if !containsField(strings.Fields(buffer.String()), test.expected) {
			t.Fatalf("Expect printing node %s with role %#v, got: %#v", test.node.Name, test.expected, buffer.String())
		}
	}
}

func TestPrintNodeOSImage(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{
		ColumnLabels: []string{},
		Wide:         true,
	})
	AddHandlers(printer)

	table := []struct {
		node    corev1.Node
		osImage string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status: corev1.NodeStatus{
					NodeInfo:  corev1.NodeSystemInfo{OSImage: "fake-os-image"},
					Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			osImage: "fake-os-image",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status: corev1.NodeStatus{
					NodeInfo:  corev1.NodeSystemInfo{KernelVersion: "fake-kernel-version"},
					Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}},
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
		if !containsField(strings.Fields(buffer.String()), test.osImage) {
			t.Fatalf("Expect printing node %s with os image %#v, got: %#v", test.node.Name, test.osImage, buffer.String())
		}
	}
}

func TestPrintNodeKernelVersion(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{
		ColumnLabels: []string{},
		Wide:         true,
	})
	AddHandlers(printer)

	table := []struct {
		node          corev1.Node
		kernelVersion string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status: corev1.NodeStatus{
					NodeInfo:  corev1.NodeSystemInfo{KernelVersion: "fake-kernel-version"},
					Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			kernelVersion: "fake-kernel-version",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status: corev1.NodeStatus{
					NodeInfo:  corev1.NodeSystemInfo{OSImage: "fake-os-image"},
					Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}},
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
		if !containsField(strings.Fields(buffer.String()), test.kernelVersion) {
			t.Fatalf("Expect printing node %s with kernel version %#v, got: %#v", test.node.Name, test.kernelVersion, buffer.String())
		}
	}
}

func TestPrintNodeContainerRuntimeVersion(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{
		ColumnLabels: []string{},
		Wide:         true,
	})
	AddHandlers(printer)

	table := []struct {
		node                    corev1.Node
		containerRuntimeVersion string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status: corev1.NodeStatus{
					NodeInfo:  corev1.NodeSystemInfo{ContainerRuntimeVersion: "foo://1.2.3"},
					Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}},
				},
			},
			containerRuntimeVersion: "foo://1.2.3",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status: corev1.NodeStatus{
					NodeInfo:  corev1.NodeSystemInfo{},
					Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}},
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
		if !containsField(strings.Fields(buffer.String()), test.containerRuntimeVersion) {
			t.Fatalf("Expect printing node %s with kernel version %#v, got: %#v", test.node.Name, test.containerRuntimeVersion, buffer.String())
		}
	}
}

func TestPrintNodeName(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{
		Wide: true,
	})
	AddHandlers(printer)
	table := []struct {
		node corev1.Node
		Name string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "127.0.0.1"},
				Status:     corev1.NodeStatus{},
			},
			Name: "127.0.0.1",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: ""},
				Status:     corev1.NodeStatus{},
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
		if !containsField(strings.Fields(buffer.String()), test.Name) {
			t.Fatalf("Expect printing node %s with node name %#v, got: %#v", test.node.Name, test.Name, buffer.String())
		}
	}
}

func TestPrintNodeExternalIP(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{
		Wide: true,
	})
	AddHandlers(printer)
	table := []struct {
		node       corev1.Node
		externalIP string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status:     corev1.NodeStatus{Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}}},
			},
			externalIP: "1.1.1.1",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status:     corev1.NodeStatus{Addresses: []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: "1.1.1.1"}}},
			},
			externalIP: "<none>",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
				Status: corev1.NodeStatus{Addresses: []corev1.NodeAddress{
					{Type: corev1.NodeExternalIP, Address: "2.2.2.2"},
					{Type: corev1.NodeInternalIP, Address: "3.3.3.3"},
					{Type: corev1.NodeExternalIP, Address: "4.4.4.4"},
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
		if !containsField(strings.Fields(buffer.String()), test.externalIP) {
			t.Fatalf("Expect printing node %s with external ip %#v, got: %#v", test.node.Name, test.externalIP, buffer.String())
		}
	}
}

func TestPrintNodeInternalIP(t *testing.T) {
	printer := NewHumanReadablePrinter(nil, PrintOptions{
		Wide: true,
	})
	AddHandlers(printer)
	table := []struct {
		node       corev1.Node
		internalIP string
	}{
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
				Status:     corev1.NodeStatus{Addresses: []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: "1.1.1.1"}}},
			},
			internalIP: "1.1.1.1",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
				Status:     corev1.NodeStatus{Addresses: []corev1.NodeAddress{{Type: corev1.NodeExternalIP, Address: "1.1.1.1"}}},
			},
			internalIP: "<none>",
		},
		{
			node: corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
				Status: corev1.NodeStatus{Addresses: []corev1.NodeAddress{
					{Type: corev1.NodeInternalIP, Address: "2.2.2.2"},
					{Type: corev1.NodeExternalIP, Address: "3.3.3.3"},
					{Type: corev1.NodeInternalIP, Address: "4.4.4.4"},
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
		if !containsField(strings.Fields(buffer.String()), test.internalIP) {
			t.Fatalf("Expect printing node %s with internal ip %#v, got: %#v", test.node.Name, test.internalIP, buffer.String())
		}
	}
}

func containsField(fields []string, field string) bool {
	for _, v := range fields {
		if v == field {
			return true
		}
	}
	return false
}

func TestPrintHunmanReadableIngressWithColumnLabels(t *testing.T) {
	ingress := networkingv1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test1",
			CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
			Labels: map[string]string{
				"app_name": "kubectl_test_ingress",
			},
		},
		Spec: networkingv1beta1.IngressSpec{
			Backend: &networkingv1beta1.IngressBackend{
				ServiceName: "svc",
				ServicePort: intstr.FromInt(93),
			},
		},
		Status: networkingv1beta1.IngressStatus{
			LoadBalancer: corev1.LoadBalancerStatus{
				Ingress: []corev1.LoadBalancerIngress{
					{
						IP:       "2.3.4.5",
						Hostname: "localhost.localdomain",
					},
				},
			},
		},
	}
	buff := bytes.NewBuffer([]byte{})
	table, err := NewTablePrinter().With(AddHandlers).PrintTable(&ingress, PrintOptions{ColumnLabels: []string{"app_name"}})
	if err != nil {
		t.Fatal(err)
	}
	verifyTable(t, table)
	if err := PrintTable(table, buff, PrintOptions{NoHeaders: true}); err != nil {
		t.Fatal(err)
	}
	output := string(buff.Bytes())
	appName := ingress.ObjectMeta.Labels["app_name"]
	if !strings.Contains(output, appName) {
		t.Errorf("expected to container app_name label value %s, but doesn't %s", appName, output)
	}
}

func TestPrintHumanReadableService(t *testing.T) {
	tests := []corev1.Service{
		{
			Spec: corev1.ServiceSpec{
				ClusterIP: "1.2.3.4",
				Type:      "LoadBalancer",
				Ports: []corev1.ServicePort{
					{
						Port:     80,
						Protocol: "TCP",
					},
				},
			},
			Status: corev1.ServiceStatus{
				LoadBalancer: corev1.LoadBalancerStatus{
					Ingress: []corev1.LoadBalancerIngress{
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
			Spec: corev1.ServiceSpec{
				ClusterIP: "1.3.4.5",
				Ports: []corev1.ServicePort{
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
			Spec: corev1.ServiceSpec{
				ClusterIP: "1.4.5.6",
				Type:      "LoadBalancer",
				Ports: []corev1.ServicePort{
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
			Status: corev1.ServiceStatus{
				LoadBalancer: corev1.LoadBalancerStatus{
					Ingress: []corev1.LoadBalancerIngress{
						{
							IP: "2.3.4.5",
						},
					},
				},
			},
		},
		{
			Spec: corev1.ServiceSpec{
				ClusterIP: "1.5.6.7",
				Type:      "LoadBalancer",
				Ports: []corev1.ServicePort{
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
			Status: corev1.ServiceStatus{
				LoadBalancer: corev1.LoadBalancerStatus{
					Ingress: []corev1.LoadBalancerIngress{
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
			table, err := NewTablePrinter().With(AddHandlers).PrintTable(&svc, PrintOptions{Wide: wide})
			if err != nil {
				t.Fatal(err)
			}
			verifyTable(t, table)
			if err := PrintTable(table, buff, PrintOptions{NoHeaders: true}); err != nil {
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
	numReplicas := int32(2)
	table := []struct {
		obj          runtime.Object
		isNamespaced bool
	}{
		{
			obj: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec: corev1.ReplicationControllerSpec{
					Replicas: &numReplicas,
					Template: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"name": "foo",
								"type": "production",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Image:                  "foo/bar",
									TerminationMessagePath: corev1.TerminationMessagePathDefault,
									ImagePullPolicy:        corev1.PullIfNotPresent,
								},
							},
							RestartPolicy: corev1.RestartPolicyAlways,
							DNSPolicy:     corev1.DNSDefault,
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
			obj: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec: corev1.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Ports: []corev1.ServicePort{
						{
							Port:     80,
							Protocol: "TCP",
						},
					},
				},
				Status: corev1.ServiceStatus{
					LoadBalancer: corev1.LoadBalancerStatus{
						Ingress: []corev1.LoadBalancerIngress{
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
			obj: &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Subsets: []corev1.EndpointSubset{{
					Addresses: []corev1.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
					Ports:     []corev1.EndpointPort{{Port: 8080}},
				},
				}},
			isNamespaced: true,
		},
		{
			obj: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: name},
			},
			isNamespaced: false,
		},
		{
			obj: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Secrets:    []corev1.ObjectReference{},
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Status:     corev1.NodeStatus{},
			},
			isNamespaced: false,
		},
		{
			obj: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       corev1.PersistentVolumeSpec{},
			},
			isNamespaced: false,
		},
		{
			obj: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       corev1.PersistentVolumeClaimSpec{},
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.Event{
				ObjectMeta:     metav1.ObjectMeta{Name: name, Namespace: namespaceName},
				Source:         corev1.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           corev1.EventTypeNormal,
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.LimitRange{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &corev1.ComponentStatus{
				Conditions: []corev1.ComponentCondition{
					{Type: corev1.ComponentHealthy, Status: corev1.ConditionTrue, Message: "ok", Error: ""},
				},
			},
			isNamespaced: false,
		},
	}

	for i, test := range table {
		if test.isNamespaced {
			// Expect output to include namespace when requested.
			printer := NewHumanReadablePrinter(nil, PrintOptions{
				WithNamespace: true,
			})
			AddHandlers(printer)
			buffer := &bytes.Buffer{}
			err := printer.PrintObj(test.obj, buffer)
			if err != nil {
				t.Fatalf("An error occurred printing object: %#v", err)
			}
			matched := containsField(strings.Fields(buffer.String()), fmt.Sprintf("%s", namespaceName))
			if !matched {
				t.Errorf("%d: Expect printing object to contain namespace: %#v", i, test.obj)
			}
		} else {
			// Expect error when trying to get all namespaces for un-namespaced object.
			printer := NewHumanReadablePrinter(nil, PrintOptions{
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
	runningPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test1", Labels: map[string]string{"a": "1", "b": "2"}},
		Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
		Status: corev1.PodStatus{
			Phase: "Running",
			ContainerStatuses: []corev1.ContainerStatus{
				{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
				{RestartCount: 3},
			},
		},
	}
	failedPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test2", Labels: map[string]string{"b": "2"}},
		Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
		Status: corev1.PodStatus{
			Phase: "Failed",
			ContainerStatuses: []corev1.ContainerStatus{
				{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
				{RestartCount: 3},
			},
		},
	}
	tests := []struct {
		obj          runtime.Object
		opts         PrintOptions
		expect       string
		ignoreLegacy bool
	}{
		{
			obj: runningPod, opts: PrintOptions{},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\ntest1\t1/2\tRunning\t6\t<unknown>\n",
		},
		{
			obj: runningPod, opts: PrintOptions{WithKind: true, Kind: schema.GroupKind{Kind: "Pod"}},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\npod/test1\t1/2\tRunning\t6\t<unknown>\n",
		},
		{
			obj: runningPod, opts: PrintOptions{ShowLabels: true},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\tLABELS\ntest1\t1/2\tRunning\t6\t<unknown>\ta=1,b=2\n",
		},
		{
			obj: &corev1.PodList{Items: []corev1.Pod{*runningPod, *failedPod}}, opts: PrintOptions{ColumnLabels: []string{"a"}},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\tA\ntest1\t1/2\tRunning\t6\t<unknown>\t1\ntest2\t1/2\tFailed\t6\t<unknown>\t\n",
		},
		{
			obj: runningPod, opts: PrintOptions{NoHeaders: true},
			expect: "test1\t1/2\tRunning\t6\t<unknown>\n",
		},
		{
			obj: failedPod, opts: PrintOptions{},
			expect:       "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\ntest2\t1/2\tFailed\t6\t<unknown>\n",
			ignoreLegacy: true, // filtering is not done by the printer in the legacy path
		},
		{
			obj: failedPod, opts: PrintOptions{},
			expect: "NAME\tREADY\tSTATUS\tRESTARTS\tAGE\ntest2\t1/2\tFailed\t6\t<unknown>\n",
		},
	}

	for i, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(test.obj, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		buf := &bytes.Buffer{}
		p := NewHumanReadablePrinter(nil, test.opts).With(AddHandlers).AddTabWriter(false)
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
		pod    corev1.Pod
		expect []metav1beta1.TableRow
	}{
		{
			// Test name, num of containers, restarts, container ready status
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
		{
			// Test container error overwrites pod phase
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test2", "1/2", "ContainerWaitingReason", int64(6), "<unknown>"}}},
		},
		{
			// Test the same as the above but with Terminated state and the first container overwrites the rest
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test3"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
						{State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{Reason: "ContainerTerminatedReason"}}, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test3", "0/2", "ContainerWaitingReason", int64(6), "<unknown>"}}},
		},
		{
			// Test ready is not enough for reporting running
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test4"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test4", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
		{
			// Test ready is not enough for reporting running
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Reason: "podReason",
					Phase:  "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test5", "1/2", "podReason", int64(6), "<unknown>"}}},
		},
		{
			// Test pod has 2 containers, one is running and the other is completed.
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test6"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase:  "Running",
					Reason: "",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{Reason: "Completed", ExitCode: 0}}},
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test6", "1/2", "Running", int64(6), "<unknown>"}}},
		},
	}

	for i, test := range tests {
		rows, err := printPod(&test.pod, PrintOptions{})
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
		pod    corev1.Pod
		expect []metav1beta1.TableRow
	}{
		{
			// Test when the NodeName and PodIP are not none
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec: corev1.PodSpec{
					Containers: make([]corev1.Container, 2),
					NodeName:   "test1",
					ReadinessGates: []corev1.PodReadinessGate{
						{
							ConditionType: corev1.PodConditionType(condition1),
						},
						{
							ConditionType: corev1.PodConditionType(condition2),
						},
						{
							ConditionType: corev1.PodConditionType(condition3),
						},
					},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodConditionType(condition1),
							Status: corev1.ConditionFalse,
						},
						{
							Type:   corev1.PodConditionType(condition2),
							Status: corev1.ConditionTrue,
						},
					},
					Phase: "podPhase",
					PodIP: "1.1.1.1",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
					NominatedNodeName: "node1",
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "1.1.1.1", "test1", "node1", "1/3"}}},
		},
		{
			// Test when the NodeName and PodIP are none
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec: corev1.PodSpec{
					Containers: make([]corev1.Container, 2),
					NodeName:   "",
				},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					PodIP: "",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test2", "1/2", "ContainerWaitingReason", int64(6), "<unknown>", "<none>", "<none>", "<none>", "<none>"}}},
		},
	}

	for i, test := range tests {
		rows, err := printPod(&test.pod, PrintOptions{Wide: true})
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
		pods   corev1.PodList
		expect []metav1beta1.TableRow
	}{
		// Test podList's pod: name, num of containers, restarts, container ready status
		{
			corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "test1"},
						Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
						Status: corev1.PodStatus{
							Phase: "podPhase",
							ContainerStatuses: []corev1.ContainerStatus{
								{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
								{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "test2"},
						Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 1)},
						Status: corev1.PodStatus{
							Phase: "podPhase",
							ContainerStatuses: []corev1.ContainerStatus{
								{Ready: true, RestartCount: 1, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
							},
						},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "2/2", "podPhase", int64(6), "<unknown>"}}, {Cells: []interface{}{"test2", "1/1", "podPhase", int64(1), "<unknown>"}}},
		},
	}

	for _, test := range tests {
		rows, err := printPodList(&test.pods, PrintOptions{})

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
		pod    corev1.Pod
		expect []metav1beta1.TableRow
	}{
		{
			// Test pod phase Running should be printed
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "Running", int64(6), "<unknown>"}}},
		},
		{
			// Test pod phase Pending should be printed
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: corev1.PodPending,
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test2", "1/2", "Pending", int64(6), "<unknown>"}}},
		},
		{
			// Test pod phase Unknown should be printed
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test3"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: corev1.PodUnknown,
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test3", "1/2", "Unknown", int64(6), "<unknown>"}}},
		},
		{
			// Test pod phase Succeeded shouldn't be printed
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test4"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: corev1.PodSucceeded,
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test4", "1/2", "Succeeded", int64(6), "<unknown>"}, Conditions: podSuccessConditions}},
		},
		{
			// Test pod phase Failed shouldn't be printed
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec:       corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: corev1.PodFailed,
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test5", "1/2", "Failed", int64(6), "<unknown>"}, Conditions: podFailedConditions}},
		},
	}

	for i, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.pod, PrintOptions{})
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
		pod          corev1.Pod
		labelColumns []string
		expect       []metav1beta1.TableRow
	}{
		{
			// Test name, num of containers, restarts, container ready status
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col1": "asd", "COL2": "zxc"},
				},
				Spec: corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]string{"col1", "COL2"},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "asd", "zxc"}}},
		},
		{
			// Test name, num of containers, restarts, container ready status
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col1": "asd", "COL2": "zxc"},
				},
				Spec: corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			[]string{},
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
	}

	for i, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.pod, PrintOptions{ColumnLabels: test.labelColumns})
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
	numReplicas := int32(5)
	tests := []struct {
		deployment appsv1.Deployment
		expect     string
		wideExpect string
	}{
		{
			appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: &numReplicas,
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
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
				Status: appsv1.DeploymentStatus{
					Replicas:            10,
					UpdatedReplicas:     2,
					AvailableReplicas:   1,
					UnavailableReplicas: 4,
				},
			},
			"test1\t0/5\t2\t1\t0s\n",
			"test1\t0/5\t2\t1\t0s\tfake-container1,fake-container2\tfake-image1,fake-image2\tfoo=bar\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.deployment, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
		table, err = NewTablePrinter().With(AddHandlers).PrintTable(&test.deployment, PrintOptions{Wide: true})
		verifyTable(t, table)
		// print deployment with '-o wide' option
		if err := PrintTable(table, buf, PrintOptions{Wide: true, NoHeaders: true}); err != nil {
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
		ds         appsv1.DaemonSet
		startsWith string
	}{
		{
			appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: appsv1.DaemonSetSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: make([]corev1.Container, 2)},
					},
				},
				Status: appsv1.DaemonSetStatus{
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
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.ds, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
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
		job    batchv1.Job
		expect string
	}{
		{
			batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: batchv1.JobSpec{
					Completions: &completions,
				},
				Status: batchv1.JobStatus{
					Succeeded: 1,
				},
			},
			"job1\t1/2\t\t0s\n",
		},
		{
			batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job2",
					CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: batchv1.JobSpec{
					Completions: nil,
				},
				Status: batchv1.JobStatus{
					Succeeded: 0,
				},
			},
			"job2\t0/1\t\t10y\n",
		},
		{
			batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job3",
					CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: batchv1.JobSpec{
					Completions: nil,
				},
				Status: batchv1.JobStatus{
					Succeeded:      0,
					StartTime:      &metav1.Time{Time: now.Add(time.Minute)},
					CompletionTime: &metav1.Time{Time: now.Add(31 * time.Minute)},
				},
			},
			"job3\t0/1\t30m\t10y\n",
		},
		{
			batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "job4",
					CreationTimestamp: metav1.Time{Time: time.Now().AddDate(-10, 0, 0)},
				},
				Spec: batchv1.JobSpec{
					Completions: nil,
				},
				Status: batchv1.JobStatus{
					Succeeded: 0,
					StartTime: &metav1.Time{Time: time.Now().Add(-20 * time.Minute)},
				},
			},
			"job4\t0/1\t20m\t10y\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.job, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
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
	metricLabelSelector, err := metav1.ParseToLabelSelector("label=value")
	if err != nil {
		t.Errorf("unable to parse label selector: %v", err)
	}
	tests := []struct {
		hpa      autoscalingv2beta2.HorizontalPodAutoscaler
		expected string
	}{
		// minReplicas unset
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MaxReplicas: 10,
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<none>\t<unset>\t10\t4\t<unknown>\n",
		},
		// external source type, target average value (no current)
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown>/100m (avg)\t2\t10\t4\t<unknown>\n",
		},
		// external source type, target average value
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m/100m (avg)\t2\t10\t4\t<unknown>\n",
		},
		// external source type, target value (no current)
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-service-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown>/100m\t2\t10\t4\t<unknown>\n",
		},
		// external source type, target value
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-external-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									Value: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m/100m\t2\t10\t4\t<unknown>\n",
		},
		// pods source type (no current)
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown>/100m\t2\t10\t4\t<unknown>\n",
		},
		// pods source type
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m/100m\t2\t10\t4\t<unknown>\n",
		},
		// object source type (no current)
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ObjectMetricSourceType,
							Object: &autoscalingv2beta2.ObjectMetricSource{
								DescribedObject: autoscalingv2beta2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown>/100m\t2\t10\t4\t<unknown>\n",
		},
		// object source type
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ObjectMetricSourceType,
							Object: &autoscalingv2beta2.ObjectMetricSource{
								DescribedObject: autoscalingv2beta2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ObjectMetricSourceType,
							Object: &autoscalingv2beta2.ObjectMetricStatus{
								DescribedObject: autoscalingv2beta2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									Value: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m/100m\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetVal (no current)
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown>/100m\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetVal
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m/100m\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetUtil (no current)
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:               autoscalingv2beta2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
			"some-hpa\tReplicationController/some-rc\t<unknown>/80%\t2\t10\t4\t<unknown>\n",
		},
		// resource source type, targetUtil
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:               autoscalingv2beta2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50%/80%\t2\t10\t4\t<unknown>\n",
		},
		// multiple specs
		{
			autoscalingv2beta2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: "some-hpa"},
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:               autoscalingv2beta2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "other-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(400, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
			"some-hpa\tReplicationController/some-rc\t50m/100m, 50%/80% + 1 more...\t2\t10\t4\t<unknown>\n",
		},
	}

	buff := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.hpa, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buff, PrintOptions{NoHeaders: true}); err != nil {
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
		pod        corev1.Pod
		showLabels bool
		expect     []metav1beta1.TableRow
	}{
		{
			// Test name, num of containers, restarts, container ready status
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col1": "asd", "COL2": "zxc"},
				},
				Spec: corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			true,
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>", "COL2=zxc,col1=asd"}}},
		},
		{
			// Test name, num of containers, restarts, container ready status
			corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test1",
					Labels: map[string]string{"col3": "asd", "COL4": "zxc"},
				},
				Spec: corev1.PodSpec{Containers: make([]corev1.Container, 2)},
				Status: corev1.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []corev1.ContainerStatus{
						{Ready: true, RestartCount: 3, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			false,
			[]metav1beta1.TableRow{{Cells: []interface{}{"test1", "1/2", "podPhase", int64(6), "<unknown>"}}},
		},
	}

	for i, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.pod, PrintOptions{ShowLabels: test.showLabels})
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
	single_ExternalIP := []string{"80.11.12.10"}
	mul_ExternalIP := []string{"80.11.12.10", "80.11.12.11"}
	tests := []struct {
		service corev1.Service
		expect  string
	}{
		{
			// Test name, cluster ip, port with protocol
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test1"},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeClusterIP,
					Ports: []corev1.ServicePort{
						{
							Protocol: "tcp",
							Port:     2233,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test1\tClusterIP\t10.9.8.7\t<none>\t2233/tcp\t<unknown>\n",
		},
		{
			// Test NodePort service
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test2"},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeNodePort,
					Ports: []corev1.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
							NodePort: 9999,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test2\tNodePort\t10.9.8.7\t<none>\t8888:9999/tcp\t<unknown>\n",
		},
		{
			// Test LoadBalancer service
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test3"},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP: "10.9.8.7",
				},
			},
			"test3\tLoadBalancer\t10.9.8.7\t<pending>\t8888/tcp\t<unknown>\n",
		},
		{
			// Test LoadBalancer service with single ExternalIP and no LoadBalancerStatus
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test4"},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP:   "10.9.8.7",
					ExternalIPs: single_ExternalIP,
				},
			},
			"test4\tLoadBalancer\t10.9.8.7\t80.11.12.10\t8888/tcp\t<unknown>\n",
		},
		{
			// Test LoadBalancer service with single ExternalIP
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test5"},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP:   "10.9.8.7",
					ExternalIPs: single_ExternalIP,
				},
				Status: corev1.ServiceStatus{
					LoadBalancer: corev1.LoadBalancerStatus{
						Ingress: []corev1.LoadBalancerIngress{
							{
								IP:       "3.4.5.6",
								Hostname: "test.cluster.com",
							},
						},
					},
				},
			},
			"test5\tLoadBalancer\t10.9.8.7\t3.4.5.6,80.11.12.10\t8888/tcp\t<unknown>\n",
		},
		{
			// Test LoadBalancer service with mul ExternalIPs
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test6"},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{
						{
							Protocol: "tcp",
							Port:     8888,
						},
					},
					ClusterIP:   "10.9.8.7",
					ExternalIPs: mul_ExternalIP,
				},
				Status: corev1.ServiceStatus{
					LoadBalancer: corev1.LoadBalancerStatus{
						Ingress: []corev1.LoadBalancerIngress{
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
			"test6\tLoadBalancer\t10.9.8.7\t2.3.4.5,3.4.5.6,80.11.12.10,80.11.12.11\t8888/tcp\t<unknown>\n",
		},
		{
			// Test ExternalName service
			corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "test7"},
				Spec: corev1.ServiceSpec{
					Type:         corev1.ServiceTypeExternalName,
					ExternalName: "my.database.example.com",
				},
			},
			"test7\tExternalName\t<none>\tmy.database.example.com\t<none>\t<unknown>\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.service, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		// We ignore time
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPodDisruptionBudget(t *testing.T) {
	minAvailable := intstr.FromInt(22)
	maxUnavailable := intstr.FromInt(11)
	tests := []struct {
		pdb    policyv1beta1.PodDisruptionBudget
		expect string
	}{
		{
			policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "ns1",
					Name:              "pdb1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: policyv1beta1.PodDisruptionBudgetSpec{
					MinAvailable: &minAvailable,
				},
				Status: policyv1beta1.PodDisruptionBudgetStatus{
					PodDisruptionsAllowed: 5,
				},
			},
			"pdb1\t22\tN/A\t5\t0s\n",
		},
		{
			policyv1beta1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "ns2",
					Name:              "pdb2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: policyv1beta1.PodDisruptionBudgetSpec{
					MaxUnavailable: &maxUnavailable,
				},
				Status: policyv1beta1.PodDisruptionBudgetStatus{
					PodDisruptionsAllowed: 5,
				},
			},
			"pdb2\tN/A\t11\t5\t0s\n",
		}}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.pdb, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintControllerRevision(t *testing.T) {
	tests := []struct {
		history appsv1.ControllerRevision
		expect  string
	}{
		{
			appsv1.ControllerRevision{
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
			"test1\tdaemonset.apps/foo\t1\t0s\n",
		},
		{
			appsv1.ControllerRevision{
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
			appsv1.ControllerRevision{
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
			appsv1.ControllerRevision{
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
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.history, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func boolP(b bool) *bool {
	return &b
}

func TestPrintReplicaSet(t *testing.T) {
	numReplicas := int32(5)
	tests := []struct {
		replicaSet appsv1.ReplicaSet
		expect     string
		wideExpect string
	}{
		{
			appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: appsv1.ReplicaSetSpec{
					Replicas: &numReplicas,
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
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
				Status: appsv1.ReplicaSetStatus{
					Replicas:      5,
					ReadyReplicas: 2,
				},
			},
			"test1\t5\t5\t2\t0s\n",
			"test1\t5\t5\t2\t0s\tfake-container1,fake-container2\tfake-image1,fake-image2\tfoo=bar\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.replicaSet, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()

		table, err = NewTablePrinter().With(AddHandlers).PrintTable(&test.replicaSet, PrintOptions{Wide: true})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true, Wide: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.wideExpect {
			t.Fatalf("Expected: %s, got: %s", test.wideExpect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPersistentVolumeClaim(t *testing.T) {
	myScn := "my-scn"
	tests := []struct {
		pvc    corev1.PersistentVolumeClaim
		expect string
	}{
		{
			// Test name, num of containers, restarts, container ready status
			corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test1",
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName: "my-volume",
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase:       corev1.ClaimBound,
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
					Capacity: map[corev1.ResourceName]resource.Quantity{
						corev1.ResourceStorage: resource.MustParse("4Gi"),
					},
				},
			},
			"test1\tBound\tmy-volume\t4Gi\tROX\t\t<unknown>\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test2",
				},
				Spec: corev1.PersistentVolumeClaimSpec{},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase:       corev1.ClaimLost,
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadOnlyMany},
					Capacity: map[corev1.ResourceName]resource.Quantity{
						corev1.ResourceStorage: resource.MustParse("4Gi"),
					},
				},
			},
			"test2\tLost\t\t\t\t\t<unknown>\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test3",
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName: "my-volume",
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase:       corev1.ClaimPending,
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
					Capacity: map[corev1.ResourceName]resource.Quantity{
						corev1.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
			},
			"test3\tPending\tmy-volume\t10Gi\tRWX\t\t<unknown>\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test4",
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "my-volume",
					StorageClassName: &myScn,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase:       corev1.ClaimPending,
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
					Capacity: map[corev1.ResourceName]resource.Quantity{
						corev1.ResourceStorage: resource.MustParse("10Gi"),
					},
				},
			},
			"test4\tPending\tmy-volume\t10Gi\tRWO\tmy-scn\t<unknown>\n",
		},
	}
	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.pvc, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			fmt.Println(buf.String())
			fmt.Println(test.expect)
			t.Fatalf("Expected: %s, but got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintCronJob(t *testing.T) {
	suspend := false
	tests := []struct {
		cronjob batchv1beta1.CronJob
		expect  string
	}{
		{
			batchv1beta1.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "cronjob1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: "0/5 * * * ?",
					Suspend:  &suspend,
				},
				Status: batchv1beta1.CronJobStatus{
					LastScheduleTime: &metav1.Time{Time: time.Now().Add(1.9e9)},
				},
			},
			"cronjob1\t0/5 * * * ?\tFalse\t0\t0s\t0s\n",
		},
		{
			batchv1beta1.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "cronjob2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: "0/5 * * * ?",
					Suspend:  &suspend,
				},
				Status: batchv1beta1.CronJobStatus{
					LastScheduleTime: &metav1.Time{Time: time.Now().Add(-3e10)},
				},
			},
			"cronjob2\t0/5 * * * ?\tFalse\t0\t30s\t5m\n",
		},
		{
			batchv1beta1.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "cronjob3",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Spec: batchv1beta1.CronJobSpec{
					Schedule: "0/5 * * * ?",
					Suspend:  &suspend,
				},
				Status: batchv1beta1.CronJobStatus{},
			},
			"cronjob3\t0/5 * * * ?\tFalse\t0\t<none>\t5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.cronjob, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintStorageClass(t *testing.T) {
	tests := []struct {
		sc     storagev1.StorageClass
		expect string
	}{
		{
			storagev1.StorageClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "sc1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Provisioner: "kubernetes.io/glusterfs",
			},
			"sc1\tkubernetes.io/glusterfs\t0s\n",
		},
		{
			storagev1.StorageClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "sc2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Provisioner: "kubernetes.io/nfs",
			},
			"sc2\tkubernetes.io/nfs\t5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.sc, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintLease(t *testing.T) {
	holder1 := "holder1"
	holder2 := "holder2"
	tests := []struct {
		sc     coordinationv1.Lease
		expect string
	}{
		{
			coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "lease1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity: &holder1,
				},
			},
			"lease1\tholder1\t0s\n",
		},
		{
			coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "lease2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity: &holder2,
				},
			},
			"lease2\tholder2\t5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.sc, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPriorityClass(t *testing.T) {
	tests := []struct {
		pc     schedulingv1.PriorityClass
		expect string
	}{
		{
			schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pc1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Value: 1,
			},
			"pc1\t1\tfalse\t0s\n",
		},
		{
			schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pc2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Value:         1000000000,
				GlobalDefault: true,
			},
			"pc2\t1000000000\ttrue\t5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.pc, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintRuntimeClass(t *testing.T) {
	tests := []struct {
		rc     nodev1beta1.RuntimeClass
		expect string
	}{
		{
			nodev1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "rc1",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
				},
				Handler: "h1",
			},
			"rc1\th1\t0s\n",
		},
		{
			nodev1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "rc2",
					CreationTimestamp: metav1.Time{Time: time.Now().Add(-3e11)},
				},
				Handler: "h2",
			},
			"rc2\th2\t5m\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		table, err := NewTablePrinter().With(AddHandlers).PrintTable(&test.rc, PrintOptions{})
		if err != nil {
			t.Fatal(err)
		}
		verifyTable(t, table)
		if err := PrintTable(table, buf, PrintOptions{NoHeaders: true}); err != nil {
			t.Fatal(err)
		}
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
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
