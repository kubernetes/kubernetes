/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

func TestVersionedPrinter(t *testing.T) {
	original := &testStruct{Key: "value"}
	p := NewVersionedPrinter(
		ResourcePrinterFunc(func(obj runtime.Object, w io.Writer) error {
			if obj == original {
				t.Fatalf("object should not be identical: %#v", obj)
			}
			if obj.(*testStruct).Key != "value" {
				t.Fatalf("object was not converted: %#v", obj)
			}
			return nil
		}),
		api.Scheme,
		testapi.Version(),
	)
	if err := p.PrintObj(original, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestYAMLPrinter(t *testing.T) {
	testPrinter(t, &YAMLPrinter{}, yaml.Unmarshal)
}

func TestJSONPrinter(t *testing.T) {
	testPrinter(t, &JSONPrinter{}, json.Unmarshal)
}

func TestPrintDefault(t *testing.T) {
	printer, found, err := GetPrinter("", "")
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if found {
		t.Errorf("no printer should have been found: %#v / %v", printer, err)
	}
}

type internalType struct {
	Name string
}

func (*internalType) IsAnAPIObject() {

}

func TestPrintJSONForObject(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, found, err := GetPrinter("json", "")
	if err != nil || !found {
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

func TestPrintJSON(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	printer, found, err := GetPrinter("json", "")
	if err != nil || !found {
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
	printer, found, err := GetPrinter("yaml", "")
	if err != nil || !found {
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
	printer, found, err := GetPrinter("template", "{{if .id}}{{.id}}{{end}}{{if .metadata.name}}{{.metadata.name}}{{end}}")
	if err != nil || !found {
		t.Fatalf("unexpected error: %#v", err)
	}
	unversionedPod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	obj, err := api.Scheme.ConvertToVersion(unversionedPod, testapi.Version())
	err = printer.PrintObj(obj, buf)
	if err != nil {
		t.Fatalf("unexpected error: %#v", err)
	}
	if buf.String() != "foo" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestPrintEmptyTemplate(t *testing.T) {
	if _, _, err := GetPrinter("template", ""); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintBadTemplate(t *testing.T) {
	if _, _, err := GetPrinter("template", "{{ .Name"); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestPrintBadTemplateFile(t *testing.T) {
	if _, _, err := GetPrinter("templatefile", ""); err == nil {
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
	err = runtime.YAMLDecoder(testapi.Codec()).DecodeInto(buf.Bytes(), &poutput)
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
	err = runtime.YAMLDecoder(testapi.Codec()).DecodeInto(buf.Bytes(), &objOut)
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

func PrintCustomType(obj *TestPrintType, w io.Writer, withNamespace bool) error {
	_, err := fmt.Fprintf(w, "%s", obj.Data)
	return err
}

func ErrorPrintHandler(obj *TestPrintType, w io.Writer, withNamespace bool) error {
	return fmt.Errorf("ErrorPrintHandler error")
}

func TestCustomTypePrinting(t *testing.T) {
	columns := []string{"Data"}
	printer := NewHumanReadablePrinter(false, false)
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
	printer := NewHumanReadablePrinter(false, false)
	printer.Handler(columns, ErrorPrintHandler)
	obj := TestPrintType{"test object"}
	buffer := &bytes.Buffer{}
	err := printer.PrintObj(&obj, buffer)
	if err == nil || err.Error() != "ErrorPrintHandler error" {
		t.Errorf("Did not get the expected error: %#v", err)
	}
}

func TestUnknownTypePrinting(t *testing.T) {
	printer := NewHumanReadablePrinter(false, false)
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
	obj, err := api.Scheme.ConvertToVersion(&api.Pod{}, testapi.Version())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	buffer := &bytes.Buffer{}
	err = printer.PrintObj(obj, buffer)
	if err != nil {
		t.Fatalf("print fail: %v", err)
	}
	if e, a := "Pod", string(buffer.Bytes()); e != a {
		t.Errorf("Expected %v, got %v", e, a)
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
					ContainerStatuses: []api.ContainerStatus{
						{
							Name: "foo",
							State: api.ContainerState{
								Running: &api.ContainerStateRunning{
									StartedAt: util.Time{},
								},
							},
						},
						{
							Name: "bar",
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
	tmpl := ``
	if api.PreV1Beta3(testapi.Version()) {
		tmpl = `{{exists . "currentState" "info" "foo" "state" "running"}}`
		useThisToDebug := `
a: {{exists . "currentState"}}
b: {{exists . "currentState" "info"}}
c: {{exists . "currentState" "info" "foo"}}
d: {{exists . "currentState" "info" "foo" "state"}}
e: {{exists . "currentState" "info" "foo" "state" "running"}}
f: {{exists . "currentState" "info" "foo" "state" "running" "startedAt"}}`
		_ = useThisToDebug // don't complain about unused var
	} else {
		tmpl = `{{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "foo") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`
	}
	p, err := NewTemplatePrinter([]byte(tmpl))
	if err != nil {
		t.Fatalf("tmpl fail: %v", err)
	}

	printer := NewVersionedPrinter(p, api.Scheme, testapi.Version())

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
	printers := map[string]ResourcePrinter{
		"humanReadable":        NewHumanReadablePrinter(true, false),
		"humanReadableHeaders": NewHumanReadablePrinter(false, false),
		"json":                 &JSONPrinter{},
		"yaml":                 &YAMLPrinter{},
		"template":             templatePrinter,
		"template2":            templatePrinter2,
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
	expectedErrors := map[string]util.StringSet{
		"template2": util.NewStringSet("pod", "emptyPodList", "endpoints"),
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
	printer := NewHumanReadablePrinter(false /* noHeaders */, false)

	obj := api.EventList{
		Items: []api.Event{
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
			},
			{
				Source:         api.EventSource{Component: "scheduler"},
				Message:        "Item 2",
				FirstTimestamp: util.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  util.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				Count:          1,
			},
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 3",
				FirstTimestamp: util.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  util.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				Count:          1,
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

func TestPrintMinionStatus(t *testing.T) {
	printer := NewHumanReadablePrinter(false, false)
	table := []struct {
		minion api.Node
		status string
	}{
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo1"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo2"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready,SchedulingDisabled",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo3"},
				Status: api.NodeStatus{Conditions: []api.NodeCondition{
					{Type: api.NodeReady, Status: api.ConditionTrue},
					{Type: api.NodeReady, Status: api.ConditionTrue}}},
			},
			status: "Ready",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo4"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionFalse}}},
			},
			status: "NotReady",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo5"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: api.NodeReady, Status: api.ConditionFalse}}},
			},
			status: "NotReady,SchedulingDisabled",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo6"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: "InvalidValue", Status: api.ConditionTrue}}},
			},
			status: "Unknown",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo7"},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{}}},
			},
			status: "Unknown",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo8"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{Type: "InvalidValue", Status: api.ConditionTrue}}},
			},
			status: "Unknown,SchedulingDisabled",
		},
		{
			minion: api.Node{
				ObjectMeta: api.ObjectMeta{Name: "foo9"},
				Spec:       api.NodeSpec{Unschedulable: true},
				Status:     api.NodeStatus{Conditions: []api.NodeCondition{{}}},
			},
			status: "Unknown,SchedulingDisabled",
		},
	}

	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(&test.minion, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing Minion: %#v", err)
		}
		if !contains(strings.Fields(buffer.String()), test.status) {
			t.Fatalf("Expect printing minion %s with status %#v, got: %#v", test.minion.Name, test.status, buffer.String())
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

func TestPrintHumanReadableService(t *testing.T) {
	tests := []api.Service{
		{
			Spec: api.ServiceSpec{
				PortalIP: "1.2.3.4",
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
				PortalIP: "1.2.3.4",
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
				PortalIP: "1.2.3.4",
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
				PortalIP: "1.2.3.4",
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
		buff := bytes.Buffer{}
		printService(&svc, &buff, false)
		output := string(buff.Bytes())
		ip := svc.Spec.PortalIP
		if !strings.Contains(output, ip) {
			t.Errorf("expected to contain portal ip %s, but doesn't: %s", ip, output)
		}

		for _, ingress := range svc.Status.LoadBalancer.Ingress {
			ip = ingress.IP
			if !strings.Contains(output, ip) {
				t.Errorf("expected to contain ingress ip %s, but doesn't: %s", ip, output)
			}
		}

		for _, port := range svc.Spec.Ports {
			portSpec := fmt.Sprintf("%d/%s", port.Port, port.Protocol)
			if !strings.Contains(output, portSpec) {
				t.Errorf("expected to contain port: %s, but doesn't: %s", portSpec, output)
			}
		}
		// Max of # ports and (# public ip + portal ip)
		count := len(svc.Spec.Ports)
		if len(svc.Status.LoadBalancer.Ingress)+1 > count {
			count = len(svc.Status.LoadBalancer.Ingress) + 1
		}
		if count != strings.Count(output, "\n") {
			t.Errorf("expected %d newlines, found %d", count, strings.Count(output, "\n"))
		}
	}
}

func TestInterpretContainerStatus(t *testing.T) {
	tests := []struct {
		status          *api.ContainerStatus
		expectedState   string
		expectedMessage string
		expectErr       bool
		name            string
	}{
		{
			status: &api.ContainerStatus{
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{},
				},
				Ready: true,
			},
			expectedState:   "Running",
			expectedMessage: "",
			name:            "basic",
		},
		{
			status: &api.ContainerStatus{
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{},
				},
				Ready: false,
			},
			expectedState:   "Running *not ready*",
			expectedMessage: "",
			name:            "basic not ready",
		},
		{
			status: &api.ContainerStatus{
				State: api.ContainerState{
					Waiting: &api.ContainerStateWaiting{},
				},
				Ready: false,
			},
			expectedState:   "Waiting",
			expectedMessage: "",
			name:            "waiting not ready",
		},
		{
			status: &api.ContainerStatus{
				State: api.ContainerState{
					Waiting: &api.ContainerStateWaiting{},
				},
				Ready: true,
			},
			expectedState:   "Waiting",
			expectedMessage: "",
			name:            "waiting",
		},
		{
			status: &api.ContainerStatus{
				State: api.ContainerState{
					Termination: &api.ContainerStateTerminated{
						ExitCode: 3,
					},
				},
				Ready: false,
			},
			expectedState:   "Terminated",
			expectedMessage: "exit code 3",
			name:            "terminated not ready",
		},
		{
			status: &api.ContainerStatus{
				State: api.ContainerState{
					Termination: &api.ContainerStateTerminated{
						ExitCode: 5,
						Reason:   "test reason",
					},
				},
				Ready: true,
			},
			expectedState:   "Terminated",
			expectedMessage: "exit code 5, reason: test reason",
			name:            "terminated",
		},
	}

	for _, test := range tests {
		// TODO: test timestamp printing.
		state, _, msg, err := interpretContainerStatus(test.status)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error (%s)", test.name)
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v (%s)", err, test.name)
			continue
		}
		if state != test.expectedState {
			t.Errorf("expected: %s, got: %s", test.expectedState, state)
		}
		if msg != test.expectedMessage {
			t.Errorf("expected: %s, got: %s", test.expectedMessage, msg)
		}
	}
}

func TestPrintHumanReadableWithNamespace(t *testing.T) {
	namespaceName := "testnamespace"
	name := "test"
	table := []struct {
		obj            runtime.Object
		printNamespace bool
	}{
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			printNamespace: true,
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
			printNamespace: true,
		},
		{
			obj: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec: api.ServiceSpec{
					PortalIP: "1.2.3.4",
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
			printNamespace: true,
		},
		{
			obj: &api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Subsets: []api.EndpointSubset{{
					Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}, {IP: "localhost"}},
					Ports:     []api.EndpointPort{{Port: 8080}},
				},
				}},
			printNamespace: true,
		},
		{
			obj: &api.Namespace{
				ObjectMeta: api.ObjectMeta{Name: name},
			},
			printNamespace: false,
		},
		{
			obj: &api.Secret{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			printNamespace: true,
		},
		{
			obj: &api.ServiceAccount{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Secrets:    []api.ObjectReference{},
			},
			printNamespace: true,
		},
		{
			obj: &api.Node{
				ObjectMeta: api.ObjectMeta{Name: name},
				Status:     api.NodeStatus{},
			},
			printNamespace: false,
		},
		{
			obj: &api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       api.PersistentVolumeSpec{},
			},
			printNamespace: true,
		},
		{
			obj: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
				Spec:       api.PersistentVolumeClaimSpec{},
			},
			printNamespace: true,
		},
		{
			obj: &api.Event{
				ObjectMeta:     api.ObjectMeta{Name: name, Namespace: namespaceName},
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  util.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
			},
			printNamespace: false,
		},
		{
			obj: &api.LimitRange{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			printNamespace: true,
		},
		{
			obj: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespaceName},
			},
			printNamespace: true,
		},
		{
			obj: &api.ComponentStatus{
				Conditions: []api.ComponentCondition{
					{Type: api.ComponentHealthy, Status: api.ConditionTrue, Message: "ok", Error: ""},
				},
			},
			printNamespace: false,
		},
	}

	printer := NewHumanReadablePrinter(false, false)
	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(test.obj, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing object: %#v", err)
		}
		matched := contains(strings.Fields(buffer.String()), fmt.Sprintf("%s/%s", namespaceName, name))
		if matched {
			t.Errorf("Expect printing object not to contain namespace: %v", test.obj)
		}
	}

	printer = NewHumanReadablePrinter(false, true)
	for _, test := range table {
		buffer := &bytes.Buffer{}
		err := printer.PrintObj(test.obj, buffer)
		if err != nil {
			t.Fatalf("An error occurred printing object: %#v", err)
		}
		matched := contains(strings.Fields(buffer.String()), fmt.Sprintf("%s/%s", namespaceName, name))
		if test.printNamespace && !matched {
			t.Errorf("Expect printing object to contain namespace: %v", test.obj)
		} else if !test.printNamespace && matched {
			t.Errorf("Expect printing object not to contain namespace: %v", test.obj)
		}
	}
}
