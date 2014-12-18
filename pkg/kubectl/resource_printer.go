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
	"io/ioutil"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"
	"text/template"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

// GetPrinter takes a format type, an optional format argument, a version and a convertor
// to be used if the underlying printer requires the object to be in a specific schema (
// any of the generic formatters), and the default printer to use for this object.
func GetPrinter(format, formatArgument, version string, convertor runtime.ObjectConvertor, defaultPrinter ResourcePrinter) (ResourcePrinter, error) {
	var printer ResourcePrinter
	switch format {
	case "json":
		printer = &JSONPrinter{version, convertor}
	case "yaml":
		printer = &YAMLPrinter{version, convertor}
	case "template":
		if len(formatArgument) == 0 {
			return nil, fmt.Errorf("template format specified but no template given")
		}
		var err error
		printer, err = NewTemplatePrinter([]byte(formatArgument), version, convertor)
		if err != nil {
			return nil, fmt.Errorf("error parsing template %s, %v\n", formatArgument, err)
		}
	case "templatefile":
		if len(formatArgument) == 0 {
			return nil, fmt.Errorf("templatefile format specified but no template file given")
		}
		data, err := ioutil.ReadFile(formatArgument)
		if err != nil {
			return nil, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		printer, err = NewTemplatePrinter(data, version, convertor)
		if err != nil {
			return nil, fmt.Errorf("error parsing template %s, %v\n", string(data), err)
		}
	case "":
		printer = defaultPrinter
	default:
		return nil, fmt.Errorf("output format %q not recognized", format)
	}
	return printer, nil
}

// ResourcePrinter is an interface that knows how to print runtime objects.
type ResourcePrinter interface {
	// Print receives an arbitrary object, formats it and prints it to a writer.
	PrintObj(runtime.Object, io.Writer) error
}

// JSONPrinter is an implementation of ResourcePrinter which outputs an object as JSON.
// The input object is assumed to be in the internal version of an API and is converted
// to the given version first.
type JSONPrinter struct {
	version   string
	convertor runtime.ObjectConvertor
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which simply writes the object to the Writer.
func (p *JSONPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	outObj, err := p.convertor.ConvertToVersion(obj, p.version)
	if err != nil {
		return err
	}
	data, err := json.Marshal(outObj)
	if err != nil {
		return err
	}
	dst := bytes.Buffer{}
	err = json.Indent(&dst, data, "", "    ")
	dst.WriteByte('\n')
	_, err = w.Write(dst.Bytes())
	return err
}

// YAMLPrinter is an implementation of ResourcePrinter which outputs an object as YAML.
// The input object is assumed to be in the internal version of an API and is converted
// to the given version first.
type YAMLPrinter struct {
	version   string
	convertor runtime.ObjectConvertor
}

// PrintObj prints the data as YAML.
func (p *YAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	outObj, err := p.convertor.ConvertToVersion(obj, p.version)
	if err != nil {
		return err
	}
	output, err := yaml.Marshal(outObj)
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

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// recieved from watches.
type HumanReadablePrinter struct {
	handlerMap map[reflect.Type]*handlerEntry
	noHeaders  bool
	lastType   reflect.Type
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter(noHeaders bool) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap: make(map[reflect.Type]*handlerEntry),
		noHeaders:  noHeaders,
	}
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

var podColumns = []string{"NAME", "IMAGE(S)", "HOST", "LABELS", "STATUS"}
var replicationControllerColumns = []string{"NAME", "IMAGE(S)", "SELECTOR", "REPLICAS"}
var serviceColumns = []string{"NAME", "LABELS", "SELECTOR", "IP", "PORT"}
var minionColumns = []string{"NAME", "LABELS"}
var statusColumns = []string{"STATUS"}
var eventColumns = []string{"TIME", "NAME", "KIND", "CONDITION", "REASON", "MESSAGE"}

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
	return nil
}

func podHostString(host, ip string) string {
	if host == "" && ip == "" {
		return "<unassigned>"
	}
	return host + "/" + ip
}

func printPod(pod *api.Pod, w io.Writer) error {
	// TODO: remove me when pods are converted
	spec := &api.PodSpec{}
	if err := api.Scheme.Convert(&pod.Spec, spec); err != nil {
		glog.Errorf("Unable to convert pod manifest: %v", err)
	}
	il := listOfImages(spec)
	// Be paranoid about the case where there is no image.
	var firstImage string
	if len(il) > 0 {
		firstImage, il = il[0], il[1:]
	}
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
		pod.Name, firstImage,
		podHostString(pod.Status.Host, pod.Status.HostIP),
		formatLabels(pod.Labels), pod.Status.Phase)
	if err != nil {
		return err
	}
	// Lay out all the other images on separate lines.
	for _, image := range il {
		_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", "", image, "", "", "")
		if err != nil {
			return err
		}
	}
	return nil
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
		controller.Name, makeImageList(&controller.Spec.Template.Spec),
		formatLabels(controller.Spec.Selector), controller.Spec.Replicas)
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
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\n", svc.Name, formatLabels(svc.Labels),
		formatLabels(svc.Spec.Selector), svc.Spec.PortalIP, svc.Spec.Port)
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
	_, err := fmt.Fprintf(w, "%s\t%s\n", minion.Name, formatLabels(minion.Labels))
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
		w, "%s\t%s\t%s\t%s\t%s\t%s\n",
		event.Timestamp.Time.Format(time.RFC1123Z),
		event.InvolvedObject.Name,
		event.InvolvedObject.Kind,
		event.Condition,
		event.Reason,
		event.Message,
	)
	return err
}

// Sorts and prints the EventList in a human-friendly format.
func printEventList(list *api.EventList, w io.Writer) error {
	sort.Sort(SortableEvents(list.Items))
	for i := range list.Items {
		if err := printEvent(&list.Items[i], w); err != nil {
			return err
		}
	}
	return nil
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	w := tabwriter.NewWriter(output, 20, 5, 3, ' ', 0)
	defer w.Flush()
	t := reflect.TypeOf(obj)
	if handler := h.handlerMap[t]; handler != nil {
		if !h.noHeaders && t != h.lastType {
			h.printHeader(handler.columns, w)
			h.lastType = t
		}
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(w)}
		resultValue := handler.printFunc.Call(args)[0]
		if resultValue.IsNil() {
			return nil
		}
		return resultValue.Interface().(error)
	}
	return fmt.Errorf("error: unknown type %#v", obj)
}

// TemplatePrinter is an implementation of ResourcePrinter which formats data with a Go Template.
type TemplatePrinter struct {
	rawTemplate string
	template    *template.Template
	version     string
	convertor   runtime.ObjectConvertor
}

func NewTemplatePrinter(tmpl []byte, asVersion string, convertor runtime.ObjectConvertor) (*TemplatePrinter, error) {
	t, err := template.New("output").Parse(string(tmpl))
	if err != nil {
		return nil, err
	}
	return &TemplatePrinter{
		rawTemplate: string(tmpl),
		template:    t,
		version:     asVersion,
		convertor:   convertor,
	}, nil
}

// PrintObj formats the obj with the Go Template.
func (p *TemplatePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	outObj, err := p.convertor.ConvertToVersion(obj, p.version)
	if err != nil {
		return err
	}
	data, err := json.Marshal(outObj)
	if err != nil {
		return err
	}
	out := map[string]interface{}{}
	if err := json.Unmarshal(data, &out); err != nil {
		return err
	}
	if err = p.template.Execute(w, out); err != nil {
		return fmt.Errorf("error executing template '%v': '%v'\n----data----\n%#v\n", p.rawTemplate, err, out)
	}
	return nil
}

func tabbedString(f func(io.Writer) error) (string, error) {
	out := new(tabwriter.Writer)
	buf := &bytes.Buffer{}
	out.Init(buf, 0, 8, 1, '\t', 0)

	err := f(out)
	if err != nil {
		return "", err
	}

	out.Flush()
	str := string(buf.String())
	return str, nil
}
