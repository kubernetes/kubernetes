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

package resource

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"net/url"

	kapierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

// GetOptions contains the input to the get command.
type GetOptions struct {
	Out, ErrOut io.Writer

	resource.FilenameOptions

	Raw       string
	Watch     bool
	WatchOnly bool
	ChunkSize int64

	LabelSelector     string
	FieldSelector     string
	AllNamespaces     bool
	Namespace         string
	ExplicitNamespace bool

	IgnoreNotFound bool
	ShowKind       bool
	LabelColumns   []string
	Export         bool

	IncludeUninitialized bool
}

var (
	getLong = templates.LongDesc(`
		Display one or many resources

		Prints a table of the most important information about the specified resources.
		You can filter the list using a label selector and the --selector flag. If the
		desired resource type is namespaced you will only see results in your current
		namespace unless you pass --all-namespaces.

		This command will hide resources that have completed, such as pods that are
		in the Succeeded or Failed phases. You can see the full results for any
		resource by providing the --show-all flag. Uninitialized objects are not
		shown unless --include-uninitialized is passed.

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
	useOpenAPIPrintColumnFlagLabel = "use-openapi-print-columns"
)

// NewCmdGet creates a command object for the generic "get" action, which
// retrieves one or more resources from a server.
func NewCmdGet(f cmdutil.Factory, out io.Writer, errOut io.Writer) *cobra.Command {
	options := &GetOptions{
		Out:    out,
		ErrOut: errOut,
	}
	validArgs := cmdutil.ValidArgList(f)

	cmd := &cobra.Command{
		Use: "get [(-o|--output=)json|yaml|wide|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=...] (TYPE [NAME | -l label] | TYPE/NAME ...) [flags]",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Display one or many resources"),
		Long:    getLong + "\n\n" + cmdutil.ValidResourceTypeList(f),
		Example: getExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate(cmd))
			cmdutil.CheckErr(options.Run(f, cmd, args))
		},
		SuggestFor: []string{"list", "ps"},
		ValidArgs:  validArgs,
		ArgAliases: kubectl.ResourceAliases(validArgs),
	}

	cmd.Flags().StringVar(&options.Raw, "raw", options.Raw, "Raw URI to request from the server.  Uses the transport specified by the kubeconfig file.")
	cmd.Flags().BoolVarP(&options.Watch, "watch", "w", options.Watch, "After listing/getting the requested object, watch for changes. Uninitialized objects are excluded if no object name is provided.")
	cmd.Flags().BoolVar(&options.WatchOnly, "watch-only", options.WatchOnly, "Watch for changes to the requested object(s), without listing/getting first.")
	cmd.Flags().Int64Var(&options.ChunkSize, "chunk-size", 500, "Return large lists in chunks rather than all at once. Pass 0 to disable. This flag is beta and may change in the future.")
	cmd.Flags().BoolVar(&options.IgnoreNotFound, "ignore-not-found", options.IgnoreNotFound, "If the requested object does not exist the command will return exit code 0.")
	cmd.Flags().StringVarP(&options.LabelSelector, "selector", "l", options.LabelSelector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().StringVar(&options.FieldSelector, "field-selector", options.FieldSelector, "Selector (field query) to filter on, supports '=', '==', and '!='.(e.g. --field-selector key1=value1,key2=value2). The server only supports a limited number of field queries per type.")
	cmd.Flags().BoolVar(&options.AllNamespaces, "all-namespaces", options.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmdutil.AddIncludeUninitializedFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)
	addOpenAPIPrintColumnFlags(cmd)
	cmd.Flags().BoolVar(&options.ShowKind, "show-kind", options.ShowKind, "If present, list the resource type for the requested object(s).")
	cmd.Flags().StringSliceVarP(&options.LabelColumns, "label-columns", "L", options.LabelColumns, "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag options like -L label1 -L label2...")
	cmd.Flags().BoolVar(&options.Export, "export", options.Export, "If true, use 'export' for the resources.  Exported resources are stripped of cluster-specific information.")
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, "identifying the resource to get from a server.")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

// Complete takes the command arguments and factory and infers any remaining options.
func (options *GetOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(options.Raw) > 0 {
		if len(args) > 0 {
			return fmt.Errorf("arguments may not be passed when --raw is specified")
		}
		return nil
	}

	var err error
	options.Namespace, options.ExplicitNamespace, err = f.DefaultNamespace()
	if err != nil {
		return err
	}
	if options.AllNamespaces {
		options.ExplicitNamespace = false
	}

	options.IncludeUninitialized = cmdutil.ShouldIncludeUninitialized(cmd, false)

	switch {
	case options.Watch || options.WatchOnly:
		// include uninitialized objects when watching on a single object
		// unless explicitly set --include-uninitialized=false
		options.IncludeUninitialized = cmdutil.ShouldIncludeUninitialized(cmd, len(args) == 2)
	default:
		if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(options.Filenames) {
			fmt.Fprint(options.ErrOut, "You must specify the type of resource to get. ", cmdutil.ValidResourceTypeList(f))
			fullCmdName := cmd.Parent().CommandPath()
			usageString := "Required resource not specified."
			if len(fullCmdName) > 0 && cmdutil.IsSiblingCommandExists(cmd, "explain") {
				usageString = fmt.Sprintf("%s\nUse \"%s explain <resource>\" for a detailed description of that resource (e.g. %[2]s explain pods).", usageString, fullCmdName)
			}

			return cmdutil.UsageErrorf(cmd, usageString)
		}
	}
	return nil
}

// Validate checks the set of flags provided by the user.
func (options *GetOptions) Validate(cmd *cobra.Command) error {
	if len(options.Raw) > 0 {
		if options.Watch || options.WatchOnly || len(options.LabelSelector) > 0 || options.Export {
			return fmt.Errorf("--raw may not be specified with other flags that filter the server request or alter the output")
		}
		if len(cmdutil.GetFlagString(cmd, "output")) > 0 {
			return cmdutil.UsageErrorf(cmd, "--raw and --output are mutually exclusive")
		}
		if _, err := url.ParseRequestURI(options.Raw); err != nil {
			return cmdutil.UsageErrorf(cmd, "--raw must be a valid URL path: %v", err)
		}
	}
	if cmdutil.GetFlagBool(cmd, "show-labels") {
		outputOption := cmd.Flags().Lookup("output").Value.String()
		if outputOption != "" && outputOption != "wide" {
			return fmt.Errorf("--show-labels option cannot be used with %s printer", outputOption)
		}
	}
	return nil
}

// Run performs the get operation.
// TODO: remove the need to pass these arguments, like other commands.
func (options *GetOptions) Run(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(options.Raw) > 0 {
		return options.raw(f)
	}
	if options.Watch || options.WatchOnly {
		return options.watch(f, cmd, args)
	}

	r := f.NewBuilder().
		Unstructured().
		NamespaceParam(options.Namespace).DefaultNamespace().AllNamespaces(options.AllNamespaces).
		FilenameParam(options.ExplicitNamespace, &options.FilenameOptions).
		LabelSelectorParam(options.LabelSelector).
		FieldSelectorParam(options.FieldSelector).
		ExportParam(options.Export).
		RequestChunksOf(options.ChunkSize).
		IncludeUninitialized(options.IncludeUninitialized).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()

	if options.IgnoreNotFound {
		r.IgnoreErrors(kapierrors.IsNotFound)
	}
	if err := r.Err(); err != nil {
		return err
	}

	printOpts := cmdutil.ExtractCmdPrintOptions(cmd, options.AllNamespaces)
	printer, err := f.PrinterForOptions(printOpts)
	if err != nil {
		return err
	}

	filterOpts := cmdutil.ExtractCmdPrintOptions(cmd, options.AllNamespaces)
	filterFuncs := f.DefaultResourceFilterFunc()
	if r.TargetsSingleItems() {
		filterFuncs = nil
	}

	if printer.IsGeneric() {
		return options.printGeneric(printer, r, filterFuncs, filterOpts)
	}

	allErrs := []error{}
	errs := sets.NewString()
	infos, err := r.Infos()
	if err != nil {
		allErrs = append(allErrs, err)
	}

	objs := make([]runtime.Object, len(infos))
	for ix := range infos {
		objs[ix] = infos[ix].Object
	}

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

	// use the default printer for each object
	printer = nil
	var lastMapping *meta.RESTMapping
	w := printers.GetNewTabWriter(options.Out)

	useOpenAPIPrintColumns := cmdutil.GetFlagBool(cmd, useOpenAPIPrintColumnFlagLabel)

	showKind := options.ShowKind || resource.MultipleTypesRequested(args) || cmdutil.MustPrintWithKinds(objs, infos, sorter)

	filteredResourceCount := 0
	noHeaders := cmdutil.GetFlagBool(cmd, "no-headers")
	for ix := range objs {
		var mapping *meta.RESTMapping
		var original runtime.Object
		var info *resource.Info
		if sorter != nil {
			info = infos[sorter.OriginalPosition(ix)]
			mapping = info.Mapping
			original = info.Object
		} else {
			info = infos[ix]
			mapping = info.Mapping
			original = info.Object
		}
		if shouldGetNewPrinterForMapping(printer, lastMapping, mapping) {
			if printer != nil {
				w.Flush()
			}

			printWithNamespace := options.AllNamespaces
			if mapping != nil && mapping.Scope.Name() == meta.RESTScopeNameRoot {
				printWithNamespace = false
			}

			printOpts := cmdutil.ExtractCmdPrintOptions(cmd, printWithNamespace)
			// if cmd does not specify output format and useOpenAPIPrintColumnFlagLabel flag is true,
			// then get the default output options for this mapping from OpenAPI schema.
			if !cmdSpecifiesOutputFmt(cmd) && useOpenAPIPrintColumns {
				updatePrintOptionsForOpenAPI(f, mapping, printOpts)
			}

			printer, err = f.PrinterForMapping(printOpts, mapping)
			if err != nil {
				if !errs.Has(err.Error()) {
					errs.Insert(err.Error())
					allErrs = append(allErrs, err)
				}
				continue
			}

			// TODO: this doesn't belong here
			// add linebreak between resource groups (if there is more than one)
			// skip linebreak above first resource group
			if lastMapping != nil && !noHeaders {
				fmt.Fprintf(options.ErrOut, "%s\n", "")
			}

			lastMapping = mapping
		}

		typedObj := info.AsInternal()

		// filter objects if filter has been defined for current object
		if isFiltered, err := filterFuncs.Filter(typedObj, filterOpts); isFiltered {
			if err == nil {
				filteredResourceCount++
				continue
			}
			if !errs.Has(err.Error()) {
				errs.Insert(err.Error())
				allErrs = append(allErrs, err)
			}
		}

		if resourcePrinter, found := printer.(*printers.HumanReadablePrinter); found {
			resourceName := resourcePrinter.GetResourceKind()
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

			if showKind {
				resourcePrinter.EnsurePrintWithKind(resourceName)
			}

			if err := printer.PrintObj(typedObj, w); err != nil {
				if !errs.Has(err.Error()) {
					errs.Insert(err.Error())
					allErrs = append(allErrs, err)
				}
			}
			continue
		}
		objToPrint := typedObj
		if printer.IsGeneric() {
			// use raw object as recieved from the builder when using generic
			// printer instead of decodedObj
			objToPrint = original
		}
		if err := printer.PrintObj(objToPrint, w); err != nil {
			if !errs.Has(err.Error()) {
				errs.Insert(err.Error())
				allErrs = append(allErrs, err)
			}
			continue
		}
	}
	w.Flush()
	cmdutil.PrintFilterCount(options.ErrOut, len(objs), filteredResourceCount, len(allErrs), filterOpts, options.IgnoreNotFound)
	return utilerrors.NewAggregate(allErrs)
}

// raw makes a simple HTTP request to the provided path on the server using the default
// credentials.
func (options *GetOptions) raw(f cmdutil.Factory) error {
	restClient, err := f.RESTClient()
	if err != nil {
		return err
	}

	stream, err := restClient.Get().RequestURI(options.Raw).Stream()
	if err != nil {
		return err
	}
	defer stream.Close()

	_, err = io.Copy(options.Out, stream)
	if err != nil && err != io.EOF {
		return err
	}
	return nil
}

// watch starts a client-side watch of one or more resources.
// TODO: remove the need for arguments here.
func (options *GetOptions) watch(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	r := f.NewBuilder().
		Unstructured().
		NamespaceParam(options.Namespace).DefaultNamespace().AllNamespaces(options.AllNamespaces).
		FilenameParam(options.ExplicitNamespace, &options.FilenameOptions).
		LabelSelectorParam(options.LabelSelector).
		FieldSelectorParam(options.FieldSelector).
		ExportParam(options.Export).
		RequestChunksOf(options.ChunkSize).
		IncludeUninitialized(options.IncludeUninitialized).
		ResourceTypeOrNameArgs(true, args...).
		SingleResourceType().
		Latest().
		Do()
	if err := r.Err(); err != nil {
		return err
	}
	infos, err := r.Infos()
	if err != nil {
		return err
	}
	if len(infos) != 1 {
		return i18n.Errorf("watch is only supported on individual resources and resource collections - %d resources were found", len(infos))
	}

	filterOpts := cmdutil.ExtractCmdPrintOptions(cmd, options.AllNamespaces)
	filterFuncs := f.DefaultResourceFilterFunc()
	if r.TargetsSingleItems() {
		filterFuncs = nil
	}

	info := infos[0]
	mapping := info.ResourceMapping()
	printOpts := cmdutil.ExtractCmdPrintOptions(cmd, options.AllNamespaces)
	printer, err := f.PrinterForMapping(printOpts, mapping)
	if err != nil {
		return err
	}
	obj, err := r.Object()
	if err != nil {
		return err
	}

	// watching from resourceVersion 0, starts the watch at ~now and
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

	// print the current object
	if !options.WatchOnly {
		var objsToPrint []runtime.Object
		writer := printers.GetNewTabWriter(options.Out)

		if isList {
			objsToPrint, _ = meta.ExtractList(obj)
		} else {
			objsToPrint = append(objsToPrint, obj)
		}
		for _, objToPrint := range objsToPrint {
			if isFiltered, err := filterFuncs.Filter(objToPrint, filterOpts); !isFiltered {
				if err != nil {
					glog.V(2).Infof("Unable to filter resource: %v", err)
				} else if err := printer.PrintObj(objToPrint, writer); err != nil {
					return fmt.Errorf("unable to output the provided object: %v", err)
				}
			}
		}
		writer.Flush()
	}

	// print watched changes
	w, err := r.Watch(rv)
	if err != nil {
		return err
	}

	first := true
	intr := interrupt.New(nil, w.Stop)
	intr.Run(func() error {
		_, err := watch.Until(0, w, func(e watch.Event) (bool, error) {
			if !isList && first {
				// drop the initial watch event in the single resource case
				first = false
				return false, nil
			}

			if isFiltered, err := filterFuncs.Filter(e.Object, filterOpts); !isFiltered {
				if err != nil {
					glog.V(2).Infof("Unable to filter resource: %v", err)
				} else if err := printer.PrintObj(e.Object, options.Out); err != nil {
					return false, err
				}
			}
			return false, nil
		})
		return err
	})
	return nil
}

func (options *GetOptions) printGeneric(printer printers.ResourcePrinter, r *resource.Result, filterFuncs kubectl.Filters, filterOpts *printers.PrintOptions) error {
	// we flattened the data from the builder, so we have individual items, but now we'd like to either:
	// 1. if there is more than one item, combine them all into a single list
	// 2. if there is a single item and that item is a list, leave it as its specific list
	// 3. if there is a single item and it is not a list, leave it as a single item
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

	var obj runtime.Object
	if !singleItemImplied || len(infos) > 1 {
		// we have more than one item, so coerce all items into a list
		// we have more than one item, so coerce all items into a list.
		// we don't want an *unstructured.Unstructured list yet, as we
		// may be dealing with non-unstructured objects. Compose all items
		// into an api.List, and then decode using an unstructured scheme.
		list := api.List{
			TypeMeta: metav1.TypeMeta{
				Kind:       "List",
				APIVersion: "v1",
			},
			ListMeta: metav1.ListMeta{},
		}
		for _, info := range infos {
			list.Items = append(list.Items, info.Object)
		}

		listData, err := json.Marshal(list)
		if err != nil {
			return err
		}

		converted, err := runtime.Decode(unstructured.UnstructuredJSONScheme, listData)
		if err != nil {
			return err
		}

		obj = converted
	} else {
		obj = infos[0].Object
	}

	isList := meta.IsListType(obj)
	if isList {
		_, items, err := cmdutil.FilterResourceList(obj, filterFuncs, filterOpts)
		if err != nil {
			return err
		}

		// take the filtered items and create a new list for display
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
		if err := printer.PrintObj(list, options.Out); err != nil {
			errs = append(errs, err)
		}
		return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
	}

	if isFiltered, err := filterFuncs.Filter(obj, filterOpts); !isFiltered {
		if err != nil {
			glog.V(2).Infof("Unable to filter resource: %v", err)
		} else if err := printer.PrintObj(obj, options.Out); err != nil {
			errs = append(errs, err)
		}
	}

	return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
}

func addOpenAPIPrintColumnFlags(cmd *cobra.Command) {
	cmd.Flags().Bool(useOpenAPIPrintColumnFlagLabel, true, "If true, use x-kubernetes-print-column metadata (if present) from the OpenAPI schema for displaying a resource.")
}

func shouldGetNewPrinterForMapping(printer printers.ResourcePrinter, lastMapping, mapping *meta.RESTMapping) bool {
	return printer == nil || lastMapping == nil || mapping == nil || mapping.Resource != lastMapping.Resource
}

func cmdSpecifiesOutputFmt(cmd *cobra.Command) bool {
	return cmdutil.GetFlagString(cmd, "output") != ""
}

// outputOptsForMappingFromOpenAPI looks for the output format metatadata in the
// openapi schema and modifies the passed print options for the mapping if found.
func updatePrintOptionsForOpenAPI(f cmdutil.Factory, mapping *meta.RESTMapping, printOpts *printers.PrintOptions) bool {

	// user has not specified any output format, check if OpenAPI has
	// default specification to print this resource type
	api, err := f.OpenAPISchema()
	if err != nil {
		// Error getting schema
		return false
	}
	// Found openapi metadata for this resource
	schema := api.LookupResource(mapping.GroupVersionKind)
	if schema == nil {
		// Schema not found, return empty columns
		return false
	}

	columns, found := openapi.GetPrintColumns(schema.GetExtensions())
	if !found {
		// Extension not found, return empty columns
		return false
	}

	return outputOptsFromStr(columns, printOpts)
}

// outputOptsFromStr parses the print-column metadata and generates printer.OutputOptions object.
func outputOptsFromStr(columnStr string, printOpts *printers.PrintOptions) bool {
	if columnStr == "" {
		return false
	}
	parts := strings.SplitN(columnStr, "=", 2)
	if len(parts) < 2 {
		return false
	}

	printOpts.OutputFormatType = parts[0]
	printOpts.OutputFormatArgument = parts[1]
	printOpts.AllowMissingKeys = true

	return true
}
