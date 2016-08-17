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

package kubectl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	kubectltesting "k8s.io/kubernetes/pkg/kubectl/testing"
	"k8s.io/kubernetes/pkg/runtime"
	yamlserializer "k8s.io/kubernetes/pkg/runtime/serializer/yaml"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/ghodss/yaml"
)

func init() {
	api.Scheme.AddKnownTypes(testapi.Default.InternalGroupVersion(), &kubectltesting.TestStruct{})
	api.Scheme.AddKnownTypes(*testapi.Default.GroupVersion(), &kubectltesting.TestStruct{})
}

var testData = kubectltesting.TestStruct{
	Key:        "testValue",
	Map:        map[string]int{"TestSubkey": 1},
	StringList: []string{"a", "b", "c"},
	IntList:    []int{1, 2, 3},
}

func TestVersionedPrinter(t *testing.T) {
	original := &kubectltesting.TestStruct{Key: "value"}
	p := NewVersionedPrinter(
		ResourcePrinterFunc(func(obj runtime.Object, w io.Writer) error {
			if obj == original {
				t.Fatalf("object should not be identical: %#v", obj)
			}
			if obj.(*kubectltesting.TestStruct).Key != "value" {
				t.Fatalf("object was not converted: %#v", obj)
			}
			return nil
		}),
		api.Scheme,
		*testapi.Default.GroupVersion(),
	)
	if err := p.PrintObj(original, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPrintDefault(t *testing.T) {
	printer, found, err := GetPrinter("", "", false)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if found {
		t.Errorf("no printer should have been found: %#v / %v", printer, err)
	}
}

type TestPrintType struct {
	Data string
}

func (obj *TestPrintType) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }

type TestUnknownType struct{}

func (obj *TestUnknownType) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }

func TestPrinter(t *testing.T) {
	//test inputs
	simpleTest := &TestPrintType{"foo"}
	podTest := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podListTest := &api.PodList{
		Items: []api.Pod{
			{ObjectMeta: api.ObjectMeta{Name: "foo"}},
			{ObjectMeta: api.ObjectMeta{Name: "bar"}},
		},
	}
	emptyListTest := &api.PodList{}
	testapi, err := api.Scheme.ConvertToVersion(podTest, *testapi.Default.GroupVersion())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	printerTests := []struct {
		Name           string
		Format         string
		FormatArgument string
		Input          runtime.Object
		Expect         string
	}{
		{"test json", "json", "", simpleTest, "{\n    \"Data\": \"foo\"\n}\n"},
		{"test yaml", "yaml", "", simpleTest, "Data: foo\n"},
		{"test template", "template", "{{if .id}}{{.id}}{{end}}{{if .metadata.name}}{{.metadata.name}}{{end}}",
			podTest, "foo"},
		{"test jsonpath", "jsonpath", "{.metadata.name}", podTest, "foo"},
		{"test jsonpath list", "jsonpath", "{.items[*].metadata.name}", podListTest, "foo bar"},
		{"test jsonpath empty list", "jsonpath", "{.items[*].metadata.name}", emptyListTest, ""},
		{"test name", "name", "", podTest, "pod/foo\n"},
		{"emits versioned objects", "template", "{{.kind}}", testapi, "Pod"},
	}
	for _, test := range printerTests {
		buf := bytes.NewBuffer([]byte{})
		printer, found, err := GetPrinter(test.Format, test.FormatArgument, false)
		if err != nil || !found {
			t.Errorf("in %s, unexpected error: %#v", test.Name, err)
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
		Name           string
		Format         string
		FormatArgument string
		Error          error
	}{
		{"empty template", "template", "", fmt.Errorf("template format specified but no template given")},
		{"bad template", "template", "{{ .Name", fmt.Errorf("error parsing template {{ .Name, template: output:1: unclosed action\n")},
		{"bad templatefile", "templatefile", "", fmt.Errorf("templatefile format specified but no template file given")},
		{"bad jsonpath", "jsonpath", "{.Name", fmt.Errorf("error parsing jsonpath {.Name, unclosed action\n")},
	}
	for _, test := range badPrinterTests {
		_, _, err := GetPrinter(test.Format, test.FormatArgument, false)
		if err == nil || err.Error() != test.Error.Error() {
			t.Errorf("in %s, expect %s, got %s", test.Name, test.Error, err)
		}
	}
}

func testPrinter(t *testing.T, printer ResourcePrinter, unmarshalFunc func(data []byte, v interface{}) error) {
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
	if err := runtime.DecodeInto(s, buf.Bytes(), &objOut); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(obj, &objOut) {
		t.Errorf("Unexpected inequality:\n%v", diff.ObjectDiff(obj, &objOut))
	}
}

func TestYAMLPrinter(t *testing.T) {
	testPrinter(t, &YAMLPrinter{}, yaml.Unmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, &JSONPrinter{}, json.Unmarshal)
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
		if got := formatResourceName(tt.kind, tt.name, true); got != tt.want {
			t.Errorf("formatResourceName(%q, %q) = %q, want %q", tt.kind, tt.name, got, tt.want)
		}
	}
}

func PrintCustomType(obj *TestPrintType, w io.Writer, options PrintOptions) error {
	data := obj.Data
	kind := options.Kind
	if options.WithKind {
		data = kind + "/" + data
	}
	_, err := fmt.Fprintf(w, "%s", data)
	return err
}

func ErrorPrintHandler(obj *TestPrintType, w io.Writer, options PrintOptions) error {
	return fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(PrintOptions{
		ColumnLabels: []string{},
	})
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

func TestCustomTypePrintingWithKind(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(PrintOptions{
		ColumnLabels: []string{},
	})
	printer.Handler(columns, PrintCustomType)
	printer.EnsurePrintWithKind("test")

	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err != nil {
		t.Fatalf("An error occurred printing the custom type: %#v", err)
	}
	expectedOutput := "Data\ntest/test object"
	if buffer.String() != expectedOutput {
		t.Errorf("The data was not printed as expected. Expected:\n%s\nGot:\n%s", expectedOutput, buffer.String())
	}
}

func TestPrintHandlerError(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(PrintOptions{
		ColumnLabels: []string{},
	})
	printer.Handler(columns, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := NewHumanReadablePrinter(PrintOptions{
		ColumnLabels: []string{},
	})
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&TestUnknownType{}, buffer)
	if err == nil {
		t.Errorf("An error was expected from printing unknown type")
	}
}

func TestTemplatePanic(t *testing.T) {
	tmpl := `{{and ((index .currentState.info "foo").state.running.startedAt) .currentState.info.net.state.running.startedAt}}`
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

func TestNamePrinter(t *testing.T) {
	tests := map[string]struct {
		obj    runtime.Object
		expect string
	}{
		"singleObject": {
			&api.Pod{
				TypeMeta: unversioned.TypeMeta{
					Kind: "Pod",
				},
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			"pod/foo\n"},
		"List": {
			&v1.List{
				TypeMeta: unversioned.TypeMeta{
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
			"pod/foo\npod/bar\n"},
	}
	printer, _, _ := GetPrinter("name", "", false)
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
									StartedAt: unversioned.Time{},
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
									StartedAt: unversioned.Time{},
								},
							},
						},
						{
							Name: "bar",
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: unversioned.Time{},
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
	p, err := NewTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}

	printer := NewVersionedPrinter(p, api.Scheme, *testapi.Default.GroupVersion())

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
	om := func(name string) api.ObjectMeta { return api.ObjectMeta{Name: name} }
	templatePrinter, err := NewTemplatePrinter([]byte("{{.name}}"))
	if err != nil {
		t.Fatal(err)
	}
	templatePrinter2, err := NewTemplatePrinter([]byte("{{len .items}}"))
	if err != nil {
		t.Fatal(err)
	}
	jsonpathPrinter, err := NewJSONPathPrinter("{.metadata.name}")
	if err != nil {
		t.Fatal(err)
	}
	printers := map[string]ResourcePrinter{
		"humanReadable": NewHumanReadablePrinter(PrintOptions{
			NoHeaders:    true,
			ColumnLabels: []string{},
		}),
		"humanReadableHeaders": NewHumanReadablePrinter(PrintOptions{
			ColumnLabels: []string{},
		}),
		"json":      &JSONPrinter{},
		"yaml":      &YAMLPrinter{},
		"template":  templatePrinter,
		"template2": templatePrinter2,
		"jsonpath":  jsonpathPrinter,
		"name": &NamePrinter{
			Typer:   api.Scheme,
			Decoder: api.Codecs.UniversalDecoder(),
		},
	}
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
	printer := NewHumanReadablePrinter(PrintOptions{
		ColumnLabels: []string{},
	})

	obj := api.EventList{
		Items: []api.Event{
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
			{
				Source:         api.EventSource{Component: "scheduler"},
				Message:        "Item 2",
				FirstTimestamp: unversioned.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 3",
				FirstTimestamp: unversioned.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
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
	printer := NewHumanReadablePrinter(PrintOptions{
		ColumnLabels: []string{},
	})
	table := []struct {
		node   api.Node
		status string
	}{
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo1"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo2"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo3"},
				Status: api.NodeStatus{Conditions: []api.NodeCondition{
					{Type: api.NodeReady, Status: api.ConditionTrue},
					{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo4"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionFalse}}},
			},
			status: "NotReady",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo5"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionFalse}}},
			},
			status: "NotReady,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo6"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: "InvalidValue", Status: api.ConditionTrue}}},
			},
			status: "Unknown",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo7"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{}}},
			},
			status: "Unknown",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo8"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: "InvalidValue", Status: api.ConditionTrue}}},
			},
			status: "Unknown,SchedulingDisabled",
		},
		{
			node: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo9"},
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
		ObjectMeta: api.ObjectMeta{
			Name:              "test1",
			CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
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
	printIngress(&ingress, &buff, PrintOptions{
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
			printService(&svc, &buff, PrintOptions{false, false, false, wide, false, false, false, "", []string{}})
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
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{
						ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
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
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
					Ports:     []api.EndpointPort{{Port: 8080}},
				},
				}},
			isNamespaced: true,
		},
		{
			obj: &api.Namespace{
				ObjectMeta: api.ObjectMeta{Name: name},
			},
			isNamespaced: false,
		},
		{
			obj: &api.Secret{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ServiceAccount{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Secrets:    []api.ObjectReference{},
			},
			isNamespaced: true,
		},
		{
			obj: &api.Node{
				ObjectMeta: api.ObjectMeta{Name: name},
				Status:     api.NodeStatus{},
			},
			isNamespaced: false,
		},
		{
			obj: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       api.PersistentVolumeSpec{},
			},
			isNamespaced: false,
		},
		{
			obj: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       api.PersistentVolumeClaimSpec{},
			},
			isNamespaced: true,
		},
		{
			obj: &api.Event{
				ObjectMeta:     api.ObjectMeta{Name: name, Namespace: namespaceName},
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
			isNamespaced: true,
		},
		{
			obj: &api.LimitRange{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			isNamespaced: true,
		},
		{
			obj: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
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

	for _, test := range table {
		if test.isNamespaced {
			// Expect output to include namespace when requested.
			printer := NewHumanReadablePrinter(PrintOptions{
				WithNamespace: true,
				ColumnLabels:  []string{},
			})
			buffer := &bytes.Buffer{}
			err := printer.PrintObj(test.obj, buffer)
			if err != nil {
				t.Fatalf("An error occurred printing object: %#v", err)
			}
			matched := contains(strings.Fields(buffer.String()), fmt.Sprintf("%s", namespaceName))
			if !matched {
				t.Errorf("Expect printing object to contain namespace: %#v", test.obj)
			}
		} else {
			// Expect error when trying to get all namespaces for un-namespaced object.
			printer := NewHumanReadablePrinter(PrintOptions{
				WithNamespace: true,
				ColumnLabels:  []string{},
			})
			buffer := &bytes.Buffer{}
			err := printer.PrintObj(test.obj, buffer)
			if err == nil {
				t.Errorf("Expected error when printing un-namespaced type")
			}
		}
	}
}

func TestPrintPod(t *testing.T) {
	tests := []struct {
		pod    api.Pod
		expect string
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test1"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			"test1\t1/2\tpodPhase\t6\t",
		},
		{
			// Test container error overwrites pod phase
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test2"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
					},
				},
			},
			"test2\t1/2\tContainerWaitingReason\t6\t",
		},
		{
			// Test the same as the above but with Terminated state and the first container overwrites the rest
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test3"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ContainerWaitingReason"}}, RestartCount: 3},
						{State: api.ContainerState{Terminated: &api.ContainerStateTerminated{Reason: "ContainerTerminatedReason"}}, RestartCount: 3},
					},
				},
			},
			"test3\t0/2\tContainerWaitingReason\t6\t",
		},
		{
			// Test ready is not enough for reporting running
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test4"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: "podPhase",
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			"test4\t1/2\tpodPhase\t6\t",
		},
		{
			// Test ready is not enough for reporting running
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test5"},
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
			"test5\t1/2\tOutOfDisk\t6\t",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	printer := HumanReadablePrinter{hiddenObjNum: 0}
	for _, test := range tests {
		printer.printPod(&test.pod, buf, PrintOptions{false, false, false, false, true, false, false, "", []string{}})
		// We ignore time
		if !strings.HasPrefix(buf.String(), test.expect) {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
	if printer.hiddenObjNum > 0 {
		t.Fatalf("Expected hidden pods: 0, got: %d", printer.hiddenObjNum)
	}
}

func TestPrintNonTerminatedPod(t *testing.T) {
	tests := []struct {
		pod    api.Pod
		expect string
	}{
		{
			// Test pod phase Running should be printed
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test1"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodRunning,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			"test1\t1/2\tRunning\t6\t",
		},
		{
			// Test pod phase Pending should be printed
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test2"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodPending,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			"test2\t1/2\tPending\t6\t",
		},
		{
			// Test pod phase Unknown should be printed
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test3"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodUnknown,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			"test3\t1/2\tUnknown\t6\t",
		},
		{
			// Test pod phase Succeeded shouldn't be printed
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test4"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodSucceeded,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{RestartCount: 3},
					},
				},
			},
			"",
		},
		{
			// Test pod phase Failed shouldn't be printed
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "test5"},
				Spec:       api.PodSpec{Containers: make([]api.Container, 2)},
				Status: api.PodStatus{
					Phase: api.PodFailed,
					ContainerStatuses: []api.ContainerStatus{
						{Ready: true, RestartCount: 3, State: api.ContainerState{Running: &api.ContainerStateRunning{}}},
						{Ready: true, RestartCount: 3},
					},
				},
			},
			"",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	printer := HumanReadablePrinter{hiddenObjNum: 0}
	for _, test := range tests {
		printer.printPod(&test.pod, buf, PrintOptions{false, false, false, false, false, false, false, "", []string{}})
		// We ignore time
		if !strings.HasPrefix(buf.String(), test.expect) {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
	if printer.hiddenObjNum != 2 {
		t.Fatalf("Expected hidden pods: 2, got: %d", printer.hiddenObjNum)
	}
}

func TestPrintPodWithLabels(t *testing.T) {
	tests := []struct {
		pod          api.Pod
		labelColumns []string
		startsWith   string
		endsWith     string
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: api.ObjectMeta{
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
			"test1\t1/2\tpodPhase\t6\t",
			"\tasd\tzxc\n",
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: api.ObjectMeta{
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
			"test1\t1/2\tpodPhase\t6\t",
			"\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	printer := HumanReadablePrinter{hiddenObjNum: 0}
	for _, test := range tests {
		printer.printPod(&test.pod, buf, PrintOptions{false, false, false, false, false, false, false, "", test.labelColumns})
		// We ignore time
		if !strings.HasPrefix(buf.String(), test.startsWith) || !strings.HasSuffix(buf.String(), test.endsWith) {
			t.Fatalf("Expected to start with: %s and end with: %s, but got: %s", test.startsWith, test.endsWith, buf.String())
		}
		buf.Reset()
	}
	if printer.hiddenObjNum > 0 {
		t.Fatalf("Expected hidden pods: 0, got: %d", printer.hiddenObjNum)
	}
}

type stringTestList []struct {
	name, got, exp string
}

func TestTranslateTimestamp(t *testing.T) {
	tl := stringTestList{
		{"a while from now", translateTimestamp(unversioned.Time{Time: time.Now().Add(2.1e9)}), "<invalid>"},
		{"almost now", translateTimestamp(unversioned.Time{Time: time.Now().Add(1.9e9)}), "0s"},
		{"now", translateTimestamp(unversioned.Time{Time: time.Now()}), "0s"},
		{"unknown", translateTimestamp(unversioned.Time{}), "<unknown>"},
		{"30 seconds ago", translateTimestamp(unversioned.Time{Time: time.Now().Add(-3e10)}), "30s"},
		{"5 minutes ago", translateTimestamp(unversioned.Time{Time: time.Now().Add(-3e11)}), "5m"},
		{"an hour ago", translateTimestamp(unversioned.Time{Time: time.Now().Add(-6e12)}), "1h"},
		{"2 days ago", translateTimestamp(unversioned.Time{Time: time.Now().UTC().AddDate(0, 0, -2)}), "2d"},
		{"months ago", translateTimestamp(unversioned.Time{Time: time.Now().UTC().AddDate(0, 0, -90)}), "90d"},
		{"10 years ago", translateTimestamp(unversioned.Time{Time: time.Now().UTC().AddDate(-10, 0, 0)}), "10y"},
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
	}{
		{
			extensions.Deployment{
				ObjectMeta: api.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: unversioned.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: extensions.DeploymentSpec{
					Replicas: 5,
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{Containers: make([]api.Container, 2)},
					},
				},
				Status: extensions.DeploymentStatus{
					Replicas:            10,
					UpdatedReplicas:     2,
					AvailableReplicas:   1,
					UnavailableReplicas: 4,
				},
			},
			"test1\t5\t10\t2\t1\t0s\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printDeployment(&test.deployment, buf, PrintOptions{false, false, false, false, true, false, false, "", []string{}})
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
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
				ObjectMeta: api.ObjectMeta{
					Name:              "test1",
					CreationTimestamp: unversioned.Time{Time: time.Now().Add(1.9e9)},
				},
				Spec: extensions.DaemonSetSpec{
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{Containers: make([]api.Container, 2)},
					},
				},
				Status: extensions.DaemonSetStatus{
					CurrentNumberScheduled: 2,
					DesiredNumberScheduled: 3,
				},
			},
			"test1\t3\t2\t<none>\t0s\n",
		},
	}

	buf := bytes.NewBuffer([]byte{})
	for _, test := range tests {
		printDaemonSet(&test.ds, buf, PrintOptions{false, false, false, false, false, false, false, "", []string{}})
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
				ObjectMeta: api.ObjectMeta{
					Name:              "job1",
					CreationTimestamp: unversioned.Time{Time: time.Now().Add(1.9e9)},
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
				ObjectMeta: api.ObjectMeta{
					Name:              "job2",
					CreationTimestamp: unversioned.Time{Time: time.Now().AddDate(-10, 0, 0)},
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
		printJob(&test.job, buf, PrintOptions{false, false, false, false, true, false, false, "", []string{}})
		if buf.String() != test.expect {
			t.Fatalf("Expected: %s, got: %s", test.expect, buf.String())
		}
		buf.Reset()
	}
}

func TestPrintPodShowLabels(t *testing.T) {
	tests := []struct {
		pod        api.Pod
		startsWith string
		endsWith   string
		showLabels bool
	}{
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: api.ObjectMeta{
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
			"test1\t1/2\tpodPhase\t6\t",
			"\tCOL2=zxc,col1=asd\n",
			true,
		},
		{
			// Test name, num of containers, restarts, container ready status
			api.Pod{
				ObjectMeta: api.ObjectMeta{
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
			"test1\t1/2\tpodPhase\t6\t",
			"\n",
			false,
		},
	}

	buf := bytes.NewBuffer([]byte{})
	printer := HumanReadablePrinter{hiddenObjNum: 0}

	for _, test := range tests {
		printer.printPod(&test.pod, buf, PrintOptions{false, false, false, false, false, test.showLabels, false, "", []string{}})
		// We ignore time
		if !strings.HasPrefix(buf.String(), test.startsWith) || !strings.HasSuffix(buf.String(), test.endsWith) {
			t.Fatalf("Expected to start with: %s and end with: %s, but got: %s", test.startsWith, test.endsWith, buf.String())
		}
		buf.Reset()
	}
	if printer.hiddenObjNum > 0 {
		t.Fatalf("Expected hidden pods: 0, got: %d", printer.hiddenObjNum)
	}
}
