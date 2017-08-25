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

package cmd

import (
	"fmt"
	"io"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	kapierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

// cmdArgsOpts used as function argument to simplify parameter list.
type cmdArgsOpts struct {
	Cmd  *cobra.Command
	Args []string
	Opts *GetOptions
}

// GetOptions holds the command options.
type GetOptions struct {
	resource.FilenameOptions

	AllNamespaces  bool
	Export         bool
	IgnoreNotFound bool
	IsWatch        bool
	IsWatchOnly    bool
	LabelColumns   []string
	Raw            string
	Selector       string
	ShowKind       bool
}

type errorSet struct {
	errs []error
	set  sets.String
}

func newErrorSet() *errorSet {
	s := sets.NewString()

	return &errorSet{
		errs: []error{},
		set:  s,
	}
}

func (es *errorSet) add(err error) {
	s := err.Error()
	if !es.set.Has(s) {
		es.set.Insert(s)
		es.errs = append(es.errs, err)
	}
}

func (es *errorSet) errors() []error {
	return es.errs
}

var (
	getLong = templates.LongDesc(`
		Display one or many resources.

		` + validResources + `

		This command will hide resources that have completed, such as pods that are
		in the Succeeded or Failed phases. You can see the full results for any
		resource by providing the '--show-all' flag.

		By specifying the output as 'template' and providing a Go template as the value
		of the --template flag, you can filter the attributes of the fetched resources.`)

	getExample = templates.Examples(i18n.T(`
		# List all pods in ps output format.
		kubectl get pods

		# List all pods in ps output format with more information (such as node name).
		kubectl get pods -o wide

		# List a single replication controller with specified NAME in ps output format.
		kubectl get replicationcontroller web

		# List a single pod in JSON output format.
		kubectl get -o json pod web-pod-13je7

		# List a pod identified by type and name specified in "pod.yaml" in JSON output format.
		kubectl get -f pod.yaml -o json

		# Return only the phase value of the specified pod.
		kubectl get -o template pod/web-pod-13je7 --template={{.status.phase}}

		# List all replication controllers and services together in ps output format.
		kubectl get rc,services

		# List one or more resources by their type and names.
		kubectl get rc/web service/frontend pods/web-pod-13je7

		# List all resources with different types.
		kubectl get all`))
)

const (
	useOpenAPIPrintColumnFlagLabel = "experimental-use-openapi-print-columns"
)

// NewCmdGet creates a command object for the generic "get" action, which
// retrieves one or more resources from a server.
func NewCmdGet(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &GetOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, printers.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	use := "get [(-o|--output=)json|yaml|wide|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=...] (TYPE [NAME | -l label] | TYPE/NAME ...) [flags]"

	cmd := &cobra.Command{
		Use:     use,
		Short:   i18n.T("Display one or many resources"),
		Long:    getLong,
		Example: getExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGet(f, out, errOut, cmdArgsOpts{
				Cmd:  cmd,
				Args: args,
				Opts: options},
			)
			cmdutil.CheckErr(err)
		},
		SuggestFor: []string{"list", "ps"},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}

	// Flags stored in options structure.

	use = "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, use)

	use = "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace."
	cmd.Flags().BoolVar(&options.AllNamespaces, "all-namespaces", false, use)

	use = "If true, use 'export' for the resources.  Exported resources are stripped of cluster-specific information."
	cmd.Flags().BoolVar(&options.Export, "export", false, use)

	use = "Treat \"resource not found\" as a successful retrieval."
	cmd.Flags().BoolVar(&options.IgnoreNotFound, "ignore-not-found", false, use)

	use = "After listing/getting the requested object, watch for changes."
	cmd.Flags().BoolVarP(&options.IsWatch, "watch", "w", false, use)

	use = "Watch for changes to the requested object(s), without listing/getting first."
	cmd.Flags().BoolVar(&options.IsWatchOnly, "watch-only", false, use)

	use = "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag options like -L label1 -L label2..."
	cmd.Flags().StringSliceVarP(&options.LabelColumns, "label-columns", "L", []string{}, use)

	use = "Raw URI to request from the server.  Uses the transport specified by the kubeconfig file."
	cmd.Flags().StringVar(&options.Raw, "raw", options.Raw, use)

	use = "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)"
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", use)

	use = "If present, list the resource type for the requested object(s)."
	cmd.Flags().BoolVar(&options.ShowKind, "show-kind", false, use)

	// Access these flags using cmd.Flags()

	addOpenAPIPrintColumnFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmdutil.AddOpenAPIFlags(cmd)

	return cmd
}

// RunGet implements the generic Get command
func RunGet(f cmdutil.Factory, out, errOut io.Writer, cao cmdArgsOpts) error {
	var (
		cmd     = cao.Cmd
		args    = cao.Args
		options = cao.Opts
	)

	if len(options.Raw) > 0 {
		return runGetForRawURI(f, out, options)
	}

	if err := validateArgumentsForGet(errOut, cao); err != nil {
		return err
	}

	// Handle watch separately since we cannot watch multiple resource types.
	if options.IsWatch || options.IsWatchOnly {
		return runGetForWatch(f, out, cao)
	}

	builder, err := f.NewUnstructuredBuilder(true)
	if err != nil {
		return err
	}
	cmdNamespace, enforceNamespace, err := modifiedDefaultNamespace(f, options.AllNamespaces)
	if err != nil {
		return err
	}
	filterFuncs := f.DefaultResourceFilterFunc()
	filterOpts := f.DefaultResourceFilterOptions(cmd, options.AllNamespaces)

	r := builder.
		NamespaceParam(cmdNamespace).
		DefaultNamespace().
		AllNamespaces(options.AllNamespaces).
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		SelectorParam(options.Selector).
		ExportParam(options.Export).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	if r.TargetsSingleItems() {
		filterFuncs = nil
	}
	if options.IgnoreNotFound {
		r.IgnoreErrors(kapierrors.IsNotFound)
	}

	printer, err := f.PrinterForCommand(cmd, false, nil, printers.PrintOptions{})
	if err != nil {
		return err
	}
	if printer.IsGeneric() {
		return printIsGenericWithFilters(r, printer, out, options, filterFuncs, filterOpts)
	}

	errSet := newErrorSet()

	infos, err := r.Infos()
	if err != nil {
		errSet.add(err)
	}
	objs := allObjectsFromInfos(infos)
	sorting, err := cmd.Flags().GetString("sort-by")
	if err != nil {
		return err
	}
	var sorter *kubectl.RuntimeSort
	if len(sorting) > 0 && len(objs) > 1 {
		// TODO: questionable
		if sorter, err = kubectl.SortObjects(f.Decoder(true), objs, sorting); err != nil {
			return err
		}
	}

	// Use the default printer for each object.
	printer = nil
	var lastMapping *meta.RESTMapping
	w := printers.GetNewTabWriter(out)

	useOpenAPIPrintColumns := cmdutil.GetFlagBool(cmd, useOpenAPIPrintColumnFlagLabel)

	if resource.MultipleTypesRequested(args) || cmdutil.MustPrintWithKinds(objs, infos, sorter) {
		options.ShowKind = true
	}

	filteredResourceCount := 0
	for ix := range objs {
		mapping, original := mappingAndOriginalObject(infos, ix, sorter)
		if shouldGetNewPrinterForMapping(printer, lastMapping, mapping) {
			if printer != nil {
				w.Flush()
			}

			var outputOpts *printers.OutputOptions
			// If cmd does not specify output format and useOpenAPIPrintColumnFlagLabel
			// flag is true, then get the default output options for this mapping from
			// OpenAPI schema.
			if !cmdSpecifiesOutputFmt(cmd) && useOpenAPIPrintColumns {
				outputOpts, _ = outputOptsForMappingFromOpenAPI(f, cmdutil.GetOpenAPICacheDir(cmd), mapping)
			}

			printer, err = f.PrinterForMapping(cmd, false, outputOpts, mapping, options.AllNamespaces)
			if err != nil {
				errSet.add(err)
				continue
			}

			addLineBreakIfNecessary(cmd, lastMapping, errOut)
			lastMapping = mapping
		}

		// Try to convert before apply filter func.
		decodedObj, _ := kubectl.DecodeUnknownObject(original)

		// Filter objects if filter has been defined for current object.
		if isFiltered, err := filterFuncs.Filter(decodedObj, filterOpts); isFiltered {
			if err == nil {
				filteredResourceCount++
				continue
			}
			errSet.add(err)
		}

		if resourcePrinter, found := printer.(*printers.HumanReadablePrinter); found {
			resourceName := humanReadableResourceName(resourcePrinter, mapping)

			if options.ShowKind {
				resourcePrinter.EnsurePrintWithKind(resourceName)
			}

			if err := printer.PrintObj(decodedObj, w); err != nil {
				errSet.add(err)
			}
			continue
		}
		objToPrint := decodedObj
		if printer.IsGeneric() {
			// Use raw object as recieved from the builder when using generic
			// printer instead of decodedObj.
			objToPrint = original
		}
		if err := printer.PrintObj(objToPrint, w); err != nil {
			errSet.add(err)
			continue
		}
	}

	w.Flush()
	errs := errSet.errors()
	cmdutil.PrintFilterCount(errOut, len(objs), filteredResourceCount, len(errs), filterOpts, options.IgnoreNotFound)
	return utilerrors.NewAggregate(errs)
}

func addOpenAPIPrintColumnFlags(cmd *cobra.Command) {
	cmd.Flags().Bool(useOpenAPIPrintColumnFlagLabel, false, "If true, use x-kubernetes-print-column metadata (if present) from openapi schema for displaying a resource.")
	// Marking it deprecated so that it is hidden from usage/help text.
	cmd.Flags().MarkDeprecated(useOpenAPIPrintColumnFlagLabel, "It's an experimental feature.")
}

// runGetForRawURI request raw URI from server.
func runGetForRawURI(f cmdutil.Factory, out io.Writer, options *GetOptions) error {
	restClient, err := f.RESTClient()
	if err != nil {
		return err
	}

	stream, err := restClient.Get().RequestURI(options.Raw).Stream()
	if err != nil {
		return err
	}
	defer stream.Close()

	_, err = io.Copy(out, stream)
	if err != nil && err != io.EOF {
		return err
	}
	return nil
}

// validateArgumentsForGet validates arguments for the 'get' command.
func validateArgumentsForGet(errOut io.Writer, cao cmdArgsOpts) error {
	var (
		cmd     = cao.Cmd
		args    = cao.Args
		options = cao.Opts
	)

	if len(args) == 0 && len(options.Filenames) == 0 {
		fmt.Fprint(errOut, "You must specify the type of resource to get. ", validResources)

		path := cmd.Parent().CommandPath()
		use := "Required resource not specified."

		if len(path) > 0 && cmdutil.IsSiblingCommandExists(cmd, "explain") {
			use = fmt.Sprintf("%s\nUse \"%s explain <resource>\" for a detailed description of that resource (e.g. %[2]s explain pods).", use, path)
		}

		return cmdutil.UsageErrorf(cmd, use)
	}

	return nil
}

func runGetForWatch(f cmdutil.Factory, out io.Writer, cao cmdArgsOpts) error {
	var (
		cmd     = cao.Cmd
		args    = cao.Args
		options = cao.Opts
	)

	builder, err := f.NewUnstructuredBuilder(true)
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := modifiedDefaultNamespace(f, options.AllNamespaces)
	if err != nil {
		return err
	}

	r := builder.
		NamespaceParam(cmdNamespace).
		DefaultNamespace().
		AllNamespaces(options.AllNamespaces).
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		SelectorParam(options.Selector).
		ExportParam(options.Export).
		ResourceTypeOrNameArgs(true, args...).
		SingleResourceType().
		Latest().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	mapping, err := mappingForFirstInfoFromResult(r)
	if err != nil {
		return err
	}
	printer, err := f.PrinterForMapping(cmd, false, nil, mapping, options.AllNamespaces)
	if err != nil {
		return err
	}
	obj, err := r.Object()
	if err != nil {
		return err
	}

	// Watching from resourceVersion 0, starts the watch at ~now and
	// will return an initial watch event.  Starting form ~now, rather
	// the rv of the object will insure that we start the watch from
	// inside the watch window, which the rv of the object might not be.
	rv := "0"
	isList := meta.IsListType(obj)
	if isList {
		// the resourceVersion of list objects is ~now but won't return
		// an initial watch event
		rv, err = mapping.MetadataAccessor.ResourceVersion(obj)
		if err != nil {
			return err
		}
	}

	if !options.IsWatchOnly {
		if err := printCurrentObject(printer, out, obj, isList); err != nil {
			return nil
		}
	}

	// Print watched changes.
	w, err := r.Watch(rv)
	if err != nil {
		return err
	}

	first := true
	intr := interrupt.New(nil, w.Stop)
	intr.Run(func() error {
		_, err := watch.Until(0, w, func(e watch.Event) (bool, error) {
			if !isList && first {
				// Drop the initial watch event in the single resource case.
				first = false
				return false, nil
			}
			return false, printer.PrintObj(e.Object, out)
		})
		return err
	})
	return nil
}

func modifiedDefaultNamespace(f cmdutil.Factory, allNamespaces bool) (string, bool, error) {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return "", false, err
	}

	if allNamespaces {
		enforceNamespace = false
	}
	return cmdNamespace, enforceNamespace, nil
}

func mappingForFirstInfoFromResult(r *resource.Result) (*meta.RESTMapping, error) {
	infos, err := r.Infos()
	if err != nil {
		return nil, err
	}
	if len(infos) != 1 {
		return nil, fmt.Errorf("watch is only supported on individual resources and resource collections - %d resources were found", len(infos))
	}
	info := infos[0]

	return info.ResourceMapping(), nil
}

func printCurrentObject(printer printers.ResourcePrinter, out io.Writer, obj runtime.Object, isList bool) error {
	var objsToPrint []runtime.Object
	writer := printers.GetNewTabWriter(out)

	if isList {
		objsToPrint, _ = meta.ExtractList(obj)
	} else {
		objsToPrint = append(objsToPrint, obj)
	}
	for _, objToPrint := range objsToPrint {
		if err := printer.PrintObj(objToPrint, writer); err != nil {
			return fmt.Errorf("unable to output the provided object: %v", err)
		}
	}
	writer.Flush()
	return nil
}

// We flattened the data from the builder, so we have individual items, but now we'd like to either:
//  1. if there is more than one item, combine them all into a single list
//  2. if there is a single item and that item is a list, leave it as its specific list
//  3. if there is a single item and it is not a a list, leave it as a single item
func printIsGenericWithFilters(r *resource.Result, printer printers.ResourcePrinter, out io.Writer, options *GetOptions, filterFuncs kubectl.Filters, filterOpts *printers.PrintOptions) error {
	var errs []error
	singleItemImplied := false
	infos, err := r.IntoSingleItemImplied(&singleItemImplied).Infos()
	if err != nil {
		if singleItemImplied {
			return err
		}
		errs = append(errs, err)
	}

	if len(infos) == 0 && options.IgnoreNotFound {
		return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
	}

	obj := runtimeObjectFromInfos(infos, singleItemImplied)
	if isList := meta.IsListType(obj); isList {
		return printFilteredItems(printer, out, obj, filterFuncs, filterOpts, errs)
	}

	if isFiltered, err := filterFuncs.Filter(obj, filterOpts); !isFiltered {
		if err != nil {
			glog.V(2).Infof("Unable to filter resource: %v", err)
		} else if err := printer.PrintObj(obj, out); err != nil {
			errs = append(errs, err)
		}
	}

	return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
}

func runtimeObjectFromInfos(infos []*resource.Info, singleItemImplied bool) runtime.Object {

	if singleItemImplied || len(infos) == 1 {
		return infos[0].Object
	}

	// We have more than one item, so coerce all items into a list.
	list := &unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"kind":       "List",
			"apiVersion": "v1",
			"metadata":   map[string]interface{}{},
		},
	}
	for _, info := range infos {
		list.Items = append(list.Items, *info.Object.(*unstructured.Unstructured))
	}
	return list
}

func printFilteredItems(printer printers.ResourcePrinter, out io.Writer, obj runtime.Object, filterFuncs kubectl.Filters, filterOpts *printers.PrintOptions, errs []error) error {
	_, items, err := cmdutil.FilterResourceList(obj, filterFuncs, filterOpts)
	if err != nil {
		return err
	}

	// Take the filtered items and create a new list for display.
	list := &unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"kind":       "List",
			"apiVersion": "v1",
			"metadata":   map[string]interface{}{},
		},
	}
	if listMeta, err := meta.ListAccessor(obj); err == nil {
		list.Object["metadata"] = map[string]interface{}{
			"selfLink":        listMeta.GetSelfLink(),
			"resourceVersion": listMeta.GetResourceVersion(),
		}
	}

	for _, item := range items {
		list.Items = append(list.Items, *item.(*unstructured.Unstructured))
	}
	if err := printer.PrintObj(list, out); err != nil {
		errs = append(errs, err)
	}
	return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
}

func allObjectsFromInfos(infos []*resource.Info) []runtime.Object {
	objs := make([]runtime.Object, len(infos))
	for ix := range infos {
		objs[ix] = infos[ix].Object
	}
	return objs
}

func mappingAndOriginalObject(infos []*resource.Info, ix int, sorter *kubectl.RuntimeSort) (*meta.RESTMapping, runtime.Object) {
	var mapping *meta.RESTMapping
	var original runtime.Object

	if sorter == nil {
		mapping = infos[ix].Mapping
		original = infos[ix].Object
		return mapping, original
	}

	mapping = infos[sorter.OriginalPosition(ix)].Mapping
	original = infos[sorter.OriginalPosition(ix)].Object
	return mapping, original
}

func shouldGetNewPrinterForMapping(printer printers.ResourcePrinter, lastMapping, mapping *meta.RESTMapping) bool {
	return printer == nil || lastMapping == nil || mapping == nil || mapping.Resource != lastMapping.Resource
}

func cmdSpecifiesOutputFmt(cmd *cobra.Command) bool {
	return cmdutil.GetFlagString(cmd, "output") != ""
}

// outputOptsForMappingFromOpenAPI looks for the output format metatadata in the
// openapi schema and returns the output options for the mapping if found.
func outputOptsForMappingFromOpenAPI(f cmdutil.Factory, openAPIcacheDir string, mapping *meta.RESTMapping) (*printers.OutputOptions, bool) {

	// User has not specified any output format, check if OpenAPI has
	// default specification to print this resource type.
	api, err := f.OpenAPISchema(openAPIcacheDir)
	if err != nil {
		// Error getting schema.
		return nil, false
	}
	// Found openapi metadata for this resource.
	schema := api.LookupResource(mapping.GroupVersionKind)
	if schema == nil {
		// Schema not found, return empty columns.
		return nil, false
	}

	columns, found := openapi.GetPrintColumns(schema.GetExtensions())
	if !found {
		// Extension not found, return empty columns.
		return nil, false
	}

	return outputOptsFromStr(columns)
}

// outputOptsFromStr parses the print-column metadata and generates printer.OutputOptions object.
func outputOptsFromStr(columnStr string) (*printers.OutputOptions, bool) {
	if columnStr == "" {
		return nil, false
	}
	parts := strings.SplitN(columnStr, "=", 2)
	if len(parts) < 2 {
		return nil, false
	}
	return &printers.OutputOptions{
		FmtType:          parts[0],
		FmtArg:           parts[1],
		AllowMissingKeys: true,
	}, true
}

// addLineBreakIfNecessary adds a linebreak between resource groups (if there is
// more than one) skip linebreak above first resource group.
func addLineBreakIfNecessary(cmd *cobra.Command, mapping *meta.RESTMapping, errOut io.Writer) {
	noHeaders := cmdutil.GetFlagBool(cmd, "no-headers")
	if mapping != nil && !noHeaders {
		fmt.Fprintf(errOut, "%s\n", "")
	}

}

func humanReadableResourceName(printer *printers.HumanReadablePrinter, mapping *meta.RESTMapping) string {
	resourceName := printer.GetResourceKind()
	if mapping != nil {
		if resourceName == "" {
			resourceName = mapping.Resource
		}
		if alias, ok := kubectl.ResourceShortFormFor(mapping.Resource); ok {
			resourceName = alias
		} else if resourceName == "" {
			resourceName = "none"
		}
	} else {
		resourceName = "none"
	}
	return resourceName
}
