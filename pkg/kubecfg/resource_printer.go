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
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"text/tabwriter"
	"text/template"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

// ResourcePrinter is an interface that knows how to print API resources.
type ResourcePrinter interface {
	// Print receives an arbitrary JSON body, formats it and prints it to a writer.
	Print([]byte, io.Writer) error
	PrintObj(runtime.Object, io.Writer) error
}

// IdentityPrinter is an implementation of ResourcePrinter which simply copies the body out to the output stream.
type IdentityPrinter struct{}

// Print is an implementation of ResourcePrinter.Print which simply writes the data to the Writer.
func (i *IdentityPrinter) Print(data []byte, w io.Writer) error {
	_, err := w.Write(data)
	return err
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which simply writes the object to the Writer.
func (i *IdentityPrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	data, err := latest.Codec.Encode(obj)
	if err != nil {
		return err
	}
	return i.Print(data, output)
}

// YAMLPrinter is an implementation of ResourcePrinter which parsess JSON, and re-formats as YAML.
type YAMLPrinter struct{}

// Print parses the data as JSON, re-formats as YAML and prints the YAML.
func (y *YAMLPrinter) Print(data []byte, w io.Writer) error {
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	output, err := yaml.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(w, string(output))
	return err
}

// PrintObj prints the data as YAML.
func (y *YAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	output, err := yaml.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(w, string(output))
	return err
}

type handlerEntry struct {
	columns   []string
	printFunc reflect.Value
}

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide more elegant output.
type HumanReadablePrinter struct {
	handlerMap map[reflect.Type]*handlerEntry
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter() *HumanReadablePrinter {
	printer := &HumanReadablePrinter{make(map[reflect.Type]*handlerEntry)}
	printer.addDefaultHandlers()
	return printer
}

// Handler adds a print handler with a given set of columns to HumanReadablePrinter instance.
// printFunc is the function that will be called to print an object.
// It must be of the following type:
//  func printFunc(object ObjectType, w io.Writer) error
// where ObjectType is the type of the object that will be printed.
func (h *HumanReadablePrinter) Handler(columns []string, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	if err := h.validatePrintHandlerFunc(printFuncValue); err != nil {
		glog.Errorf("Unable to add print handler: %v", err)
		return err
	}
	objType := printFuncValue.Type().In(0)
	h.handlerMap[objType] = &handlerEntry{
		columns:   columns,
		printFunc: printFuncValue,
	}
	return nil
}

func (h *HumanReadablePrinter) validatePrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function.", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 2 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 2 parameters and return 1 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*io.Writer)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, w io.Writer) error", funcType.In(0))
	}
	return nil
}

var podColumns = []string{"Name", "Image(s)", "Host", "Labels", "Status"}
var replicationControllerColumns = []string{"Name", "Image(s)", "Selector", "Replicas"}
var serviceColumns = []string{"Name", "Labels", "Selector", "IP", "Port"}
var minionColumns = []string{"Minion identifier", "Labels"}
var statusColumns = []string{"Status"}
var eventColumns = []string{"Name", "Kind", "Reason", "Message"}

// addDefaultHandlers adds print handlers for default Kubernetes types.
func (h *HumanReadablePrinter) addDefaultHandlers() {
	h.Handler(podColumns, printPod)
	h.Handler(podColumns, printPodList)
	h.Handler(replicationControllerColumns, printReplicationController)
	h.Handler(replicationControllerColumns, printReplicationControllerList)
	h.Handler(serviceColumns, printService)
	h.Handler(serviceColumns, printServiceList)
	h.Handler(minionColumns, printMinion)
	h.Handler(minionColumns, printMinionList)
	h.Handler(statusColumns, printStatus)
	h.Handler(eventColumns, printEvent)
	h.Handler(eventColumns, printEventList)
}

func (h *HumanReadablePrinter) unknown(data []byte, w io.Writer) error {
	_, err := fmt.Fprintf(w, "Unknown object: %s", string(data))
	return err
}

func (h *HumanReadablePrinter) printHeader(columnNames []string, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(columnNames, "\t")); err != nil {
		return err
	}
	var lines []string
	for range columnNames {
		lines = append(lines, "----------")
	}
	_, err := fmt.Fprintf(w, "%s\n", strings.Join(lines, "\t"))
	return err
}

func makeImageList(manifest api.PodSpec) string {
	var images []string
	for _, container := range manifest.Containers {
		images = append(images, container.Image)
	}
	return strings.Join(images, ",")
}

func makeImageListPodSpec(spec api.PodSpec) string {
	var images []string
	for _, container := range spec.Containers {
		images = append(images, container.Image)
	}
	return strings.Join(images, ",")
}

func podHostString(host, ip string) string {
	if host == "" && ip == "" {
		return "<unassigned>"
	}
	return host + "/" + ip
}

func printPod(pod *api.Pod, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
		pod.Name, makeImageList(pod.Spec),
		podHostString(pod.Status.Host, pod.Status.HostIP),
		labels.Set(pod.Labels), pod.Status.Phase)
	return err
}

func printPodList(podList *api.PodList, w io.Writer) error {
	for _, pod := range podList.Items {
		if err := printPod(&pod, w); err != nil {
			return err
		}
	}
	return nil
}

func printReplicationController(controller *api.ReplicationController, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%d\n",
		controller.Name, makeImageListPodSpec(controller.Spec.Template.Spec),
		labels.Set(controller.Spec.Selector), controller.Spec.Replicas)
	return err
}

func printReplicationControllerList(list *api.ReplicationControllerList, w io.Writer) error {
	for _, controller := range list.Items {
		if err := printReplicationController(&controller, w); err != nil {
			return err
		}
	}
	return nil
}

func printService(svc *api.Service, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\n", svc.Name, labels.Set(svc.Labels),
		labels.Set(svc.Spec.Selector), svc.Spec.PortalIP, svc.Spec.Port)
	return err
}

func printServiceList(list *api.ServiceList, w io.Writer) error {
	for _, svc := range list.Items {
		if err := printService(&svc, w); err != nil {
			return err
		}
	}
	return nil
}

func printMinion(minion *api.Node, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\n", minion.Name, labels.Set(minion.Labels))
	return err
}

func printMinionList(list *api.NodeList, w io.Writer) error {
	for _, minion := range list.Items {
		if err := printMinion(&minion, w); err != nil {
			return err
		}
	}
	return nil
}

func printStatus(status *api.Status, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%v\n", status.Status)
	return err
}

func printEvent(event *api.Event, w io.Writer) error {
	_, err := fmt.Fprintf(
		w, "%s\t%s\t%s\t%s\n",
		event.InvolvedObject.Name,
		event.InvolvedObject.Kind,
		event.Reason,
		event.Message,
	)
	return err
}

func printEventList(list *api.EventList, w io.Writer) error {
	for i := range list.Items {
		if err := printEvent(&list.Items[i], w); err != nil {
			return err
		}
	}
	return nil
}

// Print parses the data as JSON, then prints the parsed data in a human-friendly
// format according to the type of the data.
func (h *HumanReadablePrinter) Print(data []byte, output io.Writer) error {
	var mapObj map[string]runtime.Object
	if err := json.Unmarshal([]byte(data), &mapObj); err != nil {
		return err
	}

	if _, contains := mapObj["kind"]; !contains {
		return fmt.Errorf("unexpected object with no 'kind' field: %s", data)
	}

	obj, err := latest.Codec.Decode(data)
	if err != nil {
		return err
	}
	return h.PrintObj(obj, output)
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	w := tabwriter.NewWriter(output, 20, 5, 3, ' ', 0)
	defer w.Flush()
	if handler := h.handlerMap[reflect.TypeOf(obj)]; handler != nil {
		h.printHeader(handler.columns, w)
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(w)}
		resultValue := handler.printFunc.Call(args)[0]
		if resultValue.IsNil() {
			return nil
		} else {
			return resultValue.Interface().(error)
		}
	} else {
		return fmt.Errorf("unknown type %#v", obj)
	}
}

// TemplatePrinter is an implementation of ResourcePrinter which formats data with a Go Template.
type TemplatePrinter struct {
	rawTemplate string
	template    *template.Template
}

func NewTemplatePrinter(tmpl []byte) (*TemplatePrinter, error) {
	t, err := template.New("output").
		Funcs(template.FuncMap{"exists": exists}).
		Parse(string(tmpl))
	if err != nil {
		return nil, err
	}
	return &TemplatePrinter{string(tmpl), t}, nil
}

// Print parses the data as JSON, and re-formats it with the Go Template.
func (t *TemplatePrinter) Print(data []byte, w io.Writer) error {
	out := map[string]interface{}{}
	err := json.Unmarshal(data, &out)
	if err != nil {
		return err
	}
	if err := t.safeExecute(w, out); err != nil {
		// It is way easier to debug this stuff when it shows up in
		// stdout instead of just stdin. So in addition to returning
		// a nice error, also print useful stuff with the writer.
		fmt.Fprintf(w, "Error executing template: %v\n", err)
		fmt.Fprintf(w, "template was:\n%v\n", t.rawTemplate)
		fmt.Fprintf(w, "raw data was:\n%v\n", string(data))
		fmt.Fprintf(w, "object given to template engine was:\n%+v\n", out)
		return fmt.Errorf("error executing template '%v': '%v'\n----data----\n%#v\n", t.rawTemplate, err, out)
	}
	return nil
}

// PrintObj formats the obj with the Go Template.
func (t *TemplatePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	data, err := latest.Codec.Encode(obj)
	if err != nil {
		return err
	}
	return t.Print(data, w)
}

// safeExecute tries to execute the template, but catches panics and returns an error
// should the template engine panic.
func (p *TemplatePrinter) safeExecute(w io.Writer, obj interface{}) error {
	var panicErr error
	// Sorry for the double anonymous function. There's probably a clever way
	// to do this that has the defer'd func setting the value to be returned, but
	// that would be even less obvious.
	retErr := func() error {
		defer func() {
			if x := recover(); x != nil {
				panicErr = fmt.Errorf("caught panic: %+v", x)
			}
		}()
		return p.template.Execute(w, obj)
	}()
	if panicErr != nil {
		return panicErr
	}
	return retErr
}

// exists returns true if it would be possible to call the index function
// with these arguments.
//
// TODO: how to document this for users?
//
// index returns the result of indexing its first argument by the following
// arguments.  Thus "index x 1 2 3" is, in Go syntax, x[1][2][3]. Each
// indexed item must be a map, slice, or array.
func exists(item interface{}, indices ...interface{}) bool {
	v := reflect.ValueOf(item)
	for _, i := range indices {
		index := reflect.ValueOf(i)
		var isNil bool
		if v, isNil = indirect(v); isNil {
			return false
		}
		switch v.Kind() {
		case reflect.Array, reflect.Slice, reflect.String:
			var x int64
			switch index.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				x = index.Int()
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
				x = int64(index.Uint())
			default:
				return false
			}
			if x < 0 || x >= int64(v.Len()) {
				return false
			}
			v = v.Index(int(x))
		case reflect.Map:
			if !index.IsValid() {
				index = reflect.Zero(v.Type().Key())
			}
			if !index.Type().AssignableTo(v.Type().Key()) {
				return false
			}
			if x := v.MapIndex(index); x.IsValid() {
				v = x
			} else {
				v = reflect.Zero(v.Type().Elem())
			}
		default:
			return false
		}
	}
	if _, isNil := indirect(v); isNil {
		return false
	}
	return true
}

// stolen from text/template
// indirect returns the item at the end of indirection, and a bool to indicate if it's nil.
// We indirect through pointers and empty interfaces (only) because
// non-empty interfaces have methods we might need.
func indirect(v reflect.Value) (rv reflect.Value, isNil bool) {
	for ; v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface; v = v.Elem() {
		if v.IsNil() {
			return v, true
		}
		if v.Kind() == reflect.Interface && v.NumMethod() > 0 {
			break
		}
	}
	return v, false
}
