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
	"io/ioutil"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"
	"text/template"

	"github.com/ghodss/yaml"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// GetPrinter takes a format type, an optional format argument. It will return true
// if the format is generic (untyped), otherwise it will return false. The printer
// is agnostic to schema versions, so you must send arguments to PrintObj in the
// version you wish them to be shown using a VersionedPrinter (typically when
// generic is true).
func GetPrinter(format, formatArgument string) (ResourcePrinter, bool, error) {
	var printer ResourcePrinter
	switch format {
	case "json":
		printer = &JSONPrinter{}
	case "yaml":
		printer = &YAMLPrinter{}
	case "template":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("template format specified but no template given")
		}
		var err error
		printer, err = NewTemplatePrinter([]byte(formatArgument))
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", formatArgument, err)
		}
	case "templatefile":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("templatefile format specified but no template file given")
		}
		data, err := ioutil.ReadFile(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		printer, err = NewTemplatePrinter(data)
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", string(data), err)
		}
	case "wide":
		fallthrough
	case "":
		return nil, false, nil
	default:
		return nil, false, fmt.Errorf("output format %q not recognized", format)
	}
	return printer, true, nil
}

// ResourcePrinter is an interface that knows how to print runtime objects.
type ResourcePrinter interface {
	// Print receives a runtime object, formats it and prints it to a writer.
	PrintObj(runtime.Object, io.Writer) error

	// Adds an ObjectProcessor. All object processors will be executed
	// in the order they were added to the ResourcePrinter.
	AddObjectPreprocessor(ObjectPreprocessor)
}

// ResourcePrinterFunc is a function that can print objects
type ResourcePrinterFunc func(runtime.Object, io.Writer) error

// PrintObj implements ResourcePrinter
func (fn ResourcePrinterFunc) PrintObj(obj runtime.Object, w io.Writer) error {
	return fn(obj, w)
}

type ObjectPreprocessor interface {
	Process(runtime.Object) (runtime.Object, error)
}

type chainObjectPreprocessor struct {
	first ObjectPreprocessor
	next  ObjectPreprocessor
}

func (c chainObjectPreprocessor) Process(obj runtime.Object) (runtime.Object, error) {
	if c.first != nil {
		var err error
		obj, err = c.first.Process(obj)
		if err != nil {
			return nil, err
		}
	}
	if c.next != nil {
		return c.next.Process(obj)
	}
	return obj, nil
}

func addObjectPreprocessor(oldPreprocessor ObjectPreprocessor, newPreprocessor ObjectPreprocessor) ObjectPreprocessor {
	if oldPreprocessor == nil {
		return newPreprocessor
	}
	return chainObjectPreprocessor{
		first: oldPreprocessor,
		next:  newPreprocessor,
	}
}

// VersionConverter takes runtime objects and ensures they are converted to a given API version
type VersionConverter struct {
	convertor runtime.ObjectConvertor
	version   []string
}

func NewVersionConverter(convertor runtime.ObjectConvertor, version ...string) ObjectPreprocessor {
	return &VersionConverter{
		convertor: convertor,
		version:   version,
	}
}

func (p *VersionConverter) Process(obj runtime.Object) (runtime.Object, error) {
	if len(p.version) == 0 {
		return nil, fmt.Errorf("no version specified, object cannot be converted")
	}
	for _, version := range p.version {
		if len(version) == 0 {
			continue
		}
		converted, err := p.convertor.ConvertToVersion(obj, version)
		if conversion.IsNotRegisteredError(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		return converted, nil
	}
	return nil, fmt.Errorf("the object cannot be converted to any of the versions: %v", p.version)
}

// JSONPrinter is an implementation of ResourcePrinter which outputs an object as JSON.
type JSONPrinter struct {
	preprocessor ObjectPreprocessor
}

func (p *JSONPrinter) AddObjectPreprocessor(newPreprocessor ObjectPreprocessor) {
	p.preprocessor = addObjectPreprocessor(p.preprocessor, newPreprocessor)
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which simply writes the object to the Writer.
func (p *JSONPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if p.preprocessor != nil {
		var err error
		obj, err = p.preprocessor.Process(obj)
		if err != nil {
			return err
		}
	}
	data, err := json.Marshal(obj)
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
	preprocessor ObjectPreprocessor
}

func (p *YAMLPrinter) AddObjectPreprocessor(newPreprocessor ObjectPreprocessor) {
	p.preprocessor = addObjectPreprocessor(p.preprocessor, newPreprocessor)
}

// PrintObj prints the data as YAML.
func (p *YAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if p.preprocessor != nil {
		var err error
		obj, err = p.preprocessor.Process(obj)
		if err != nil {
			return err
		}
	}

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

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// received from watches.
type HumanReadablePrinter struct {
	preprocessor  ObjectPreprocessor
	noHeaders     bool
	withNamespace bool
	wide          bool
	columnLabels  []string
	extraColumns  []ColumnPrinter
	lastType      reflect.Type
}

func (h *HumanReadablePrinter) AddObjectPreprocessor(newPreprocessor ObjectPreprocessor) {
	h.preprocessor = addObjectPreprocessor(h.preprocessor, newPreprocessor)
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter(noHeaders, withNamespace bool, wide bool, columnLabels []string) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		noHeaders:     noHeaders,
		withNamespace: withNamespace,
		wide:          wide,
		columnLabels:  columnLabels,
	}
	return printer
}

func (h *HumanReadablePrinter) HandledResources() []string {
	return getSupportedTypeNames()
}

func printColumns(data [][]string, writer io.Writer) error {
	maxRows := 0
	for _, column := range data {
		if maxRows < len(column) {
			maxRows = len(column)
		}
	}

	for row := 0; row < maxRows; row++ {
		for i, column := range data {
			if i > 0 {
				_, err := fmt.Fprintf(writer, "\t")
				if err != nil {
					return err
				}
			}
			if row < len(column) {
				_, err := fmt.Fprintf(writer, "%s", column[row])
				if err != nil {
					return err
				}
			}
		}
		_, err := fmt.Fprintf(writer, "\n")
		if err != nil {
			return err
		}
	}
	return nil
}

func printHeaders(headers []string, writer io.Writer) error {
	_, err := fmt.Fprintln(writer, strings.Join(headers, "\t"))
	if err != nil {
		return err
	}
	return nil
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	return h.printObjWithExtraColumns(obj, output)
}

func (h *HumanReadablePrinter) printObjWithExtraColumns(obj runtime.Object, output io.Writer, extraColumns ...ColumnPrinter) error {
	if h.preprocessor != nil {
		var err error
		obj, err = h.preprocessor.Process(obj)
		if err != nil {
			return err
		}
	}
	writer := tabwriter.NewWriter(output, 10, 4, 3, ' ', 0)
	defer writer.Flush()

	toBePrinted := []runtime.Object{}

	// Sort events
	eventList, isEventList := obj.(*api.EventList)
	if isEventList {
		sort.Sort(SortableEvents(eventList.Items))
	}

	// Multiple items to be printed.
	if strings.HasSuffix(reflect.TypeOf(obj).String(), "List") {
		items := reflect.ValueOf(obj).Elem().FieldByName("Items")
		for i := 0; i < items.Len(); i++ {
			item := items.Index(i).Addr().Interface().(runtime.Object)
			toBePrinted = append(toBePrinted, item)
		}
	} else {
		toBePrinted = append(toBePrinted, obj)
	}

	c := printContext{
		withNamespace: h.withNamespace,
		wide:          h.wide,
		columnLabels:  h.columnLabels,
	}

	// Print all.
	for _, item := range toBePrinted {
		t := reflect.TypeOf(item)
		columns, err := getColumnPrintersFor(t, c)
		columns = append(extraColumns, columns...)

		if err != nil {
			return err
		}
		if !h.noHeaders && t != h.lastType {
			err := printHeaders(buildHeaders(columns), writer)
			if err != nil {
				return err
			}
			h.lastType = t
		}

		columnsData, err := populateColumns(item, columns)
		if err != nil {
			return err
		}
		printColumns(columnsData, writer)
	}
	return nil
}

// PrinterWithEvent is a wrapper around HumanReadablePrinter that prints
// the type of update that happened to the runtime.Object along with the object
// itself.
type printerWithEvent struct {
	printer   *HumanReadablePrinter
	eventType *watch.EventType
}

func NewPrinterWithEvent(h *HumanReadablePrinter, event *watch.EventType) *printerWithEvent {
	return &printerWithEvent{
		printer:   h,
		eventType: event,
	}
}

func (h *printerWithEvent) AddObjectPreprocessor(preprocessor ObjectPreprocessor) {
	h.printer.AddObjectPreprocessor(preprocessor)
}

func (h *printerWithEvent) PrintObj(obj runtime.Object, output io.Writer) error {
	return h.printer.printObjWithExtraColumns(obj, output, makeUpdateColumn(h.eventType))
}

// TemplatePrinter is an implementation of ResourcePrinter which formats data with a Go Template.
type TemplatePrinter struct {
	rawTemplate  string
	template     *template.Template
	preprocessor ObjectPreprocessor
}

func (p *TemplatePrinter) AddObjectPreprocessor(newPreprocessor ObjectPreprocessor) {
	p.preprocessor = addObjectPreprocessor(p.preprocessor, newPreprocessor)
}

func NewTemplatePrinter(tmpl []byte) (*TemplatePrinter, error) {
	t, err := template.New("output").
		Funcs(template.FuncMap{"exists": exists}).
		Parse(string(tmpl))
	if err != nil {
		return nil, err
	}
	return &TemplatePrinter{
		rawTemplate: string(tmpl),
		template:    t,
	}, nil
}

// PrintObj formats the obj with the Go Template.
func (p *TemplatePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if p.preprocessor != nil {
		var err error
		obj, err = p.preprocessor.Process(obj)
		if err != nil {
			return err
		}
	}
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	out := map[string]interface{}{}
	if err := json.Unmarshal(data, &out); err != nil {
		return err
	}
	if err = p.safeExecute(w, out); err != nil {
		// It is way easier to debug this stuff when it shows up in
		// stdout instead of just stdin. So in addition to returning
		// a nice error, also print useful stuff with the writer.
		fmt.Fprintf(w, "Error executing template: %v\n", err)
		fmt.Fprintf(w, "template was:\n\t%v\n", p.rawTemplate)
		fmt.Fprintf(w, "raw data was:\n\t%v\n", string(data))
		fmt.Fprintf(w, "object given to template engine was:\n\t%+v\n", out)
		return fmt.Errorf("error executing template '%v': '%v'\n----data----\n%+v\n", p.rawTemplate, err, out)
	}
	return nil
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
