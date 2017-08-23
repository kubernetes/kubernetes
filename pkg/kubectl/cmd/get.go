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

// GetOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type GetOptions struct {
	resource.FilenameOptions

	IgnoreNotFound bool
	Raw            string
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

	cmd := &cobra.Command{
		Use:     "get [(-o|--output=)json|yaml|wide|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=...] (TYPE [NAME | -l label] | TYPE/NAME ...) [flags]",
		Short:   i18n.T("Display one or many resources"),
		Long:    getLong,
		Example: getExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGet(f, out, errOut, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		SuggestFor: []string{"list", "ps"},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolP("watch", "w", false, "After listing/getting the requested object, watch for changes.")
	cmd.Flags().Bool("watch-only", false, "Watch for changes to the requested object(s), without listing/getting first.")
	cmd.Flags().Bool("show-kind", false, "If present, list the resource type for the requested object(s).")
	cmd.Flags().Bool("all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&options.IgnoreNotFound, "ignore-not-found", false, "Treat \"resource not found\" as a successful retrieval.")
	cmd.Flags().StringSliceP("label-columns", "L", []string{}, "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag options like -L label1 -L label2...")
	cmd.Flags().Bool("export", false, "If true, use 'export' for the resources.  Exported resources are stripped of cluster-specific information.")
	addOpenAPIPrintColumnFlags(cmd)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmdutil.AddOpenAPIFlags(cmd)
	cmd.Flags().StringVar(&options.Raw, "raw", options.Raw, "Raw URI to request from the server.  Uses the transport specified by the kubeconfig file.")
	return cmd
}

// RunGet implements the generic Get command
// TODO: convert all direct flag accessors to a struct and pass that instead of cmd
func RunGet(f cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, args []string, options *GetOptions) error {
	if len(options.Raw) > 0 {
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

	selector := cmdutil.GetFlagString(cmd, "selector")
	allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")
	showKind := cmdutil.GetFlagBool(cmd, "show-kind")
	builder, err := f.NewUnstructuredBuilder(true)
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if allNamespaces {
		enforceNamespace = false
	}

	if len(args) == 0 && cmdutil.IsFilenameSliceEmpty(options.Filenames) {
		fmt.Fprint(errOut, "You must specify the type of resource to get. ", validResources)

		fullCmdName := cmd.Parent().CommandPath()
		usageString := "Required resource not specified."
		if len(fullCmdName) > 0 && cmdutil.IsSiblingCommandExists(cmd, "explain") {
			usageString = fmt.Sprintf("%s\nUse \"%s explain <resource>\" for a detailed description of that resource (e.g. %[2]s explain pods).", usageString, fullCmdName)
		}

		return cmdutil.UsageErrorf(cmd, usageString)
	}

	export := cmdutil.GetFlagBool(cmd, "export")

	filterFuncs := f.DefaultResourceFilterFunc()
	filterOpts := f.DefaultResourceFilterOptions(cmd, allNamespaces)

	// handle watch separately since we cannot watch multiple resource types
	isWatch, isWatchOnly := cmdutil.GetFlagBool(cmd, "watch"), cmdutil.GetFlagBool(cmd, "watch-only")
	if isWatch || isWatchOnly {
		builder, err := f.NewUnstructuredBuilder(true)
		if err != nil {
			return err
		}

		r := builder.
			NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
			FilenameParam(enforceNamespace, &options.FilenameOptions).
			SelectorParam(selector).
			ExportParam(export).
			ResourceTypeOrNameArgs(true, args...).
			SingleResourceType().
			Latest().
			Do()
		err = r.Err()
		if err != nil {
			return err
		}
		infos, err := r.Infos()
		if err != nil {
			return err
		}
		if len(infos) != 1 {
			return i18n.Errorf("watch is only supported on individual resources and resource collections - %d resources were found", len(infos))
		}
		if r.TargetsSingleItems() {
			filterFuncs = nil
		}
		info := infos[0]
		mapping := info.ResourceMapping()
		printer, err := f.PrinterForMapping(cmd, false, nil, mapping, allNamespaces)
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
		if !isWatchOnly {
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
				return false, printer.PrintObj(e.Object, out)
			})
			return err
		})
		return nil
	}

	r := builder.
		NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
		FilenameParam(enforceNamespace, &options.FilenameOptions).
		SelectorParam(selector).
		ExportParam(export).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	printer, err := f.PrinterForCommand(cmd, false, nil, printers.PrintOptions{})
	if err != nil {
		return err
	}
	if r.TargetsSingleItems() {
		filterFuncs = nil
	}
	if options.IgnoreNotFound {
		r.IgnoreErrors(kapierrors.IsNotFound)
	}

	if printer.IsGeneric() {
		// we flattened the data from the builder, so we have individual items, but now we'd like to either:
		// 1. if there is more than one item, combine them all into a single list
		// 2. if there is a single item and that item is a list, leave it as its specific list
		// 3. if there is a single item and it is not a a list, leave it as a single item
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
			obj = list
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
			if err := printer.PrintObj(list, out); err != nil {
				errs = append(errs, err)
			}
			return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
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
	w := printers.GetNewTabWriter(out)

	useOpenAPIPrintColumns := cmdutil.GetFlagBool(cmd, useOpenAPIPrintColumnFlagLabel)

	if resource.MultipleTypesRequested(args) || cmdutil.MustPrintWithKinds(objs, infos, sorter) {
		showKind = true
	}

	filteredResourceCount := 0
	for ix := range objs {
		var mapping *meta.RESTMapping
		var original runtime.Object

		if sorter != nil {
			mapping = infos[sorter.OriginalPosition(ix)].Mapping
			original = infos[sorter.OriginalPosition(ix)].Object
		} else {
			mapping = infos[ix].Mapping
			original = infos[ix].Object
		}
		if shouldGetNewPrinterForMapping(printer, lastMapping, mapping) {
			if printer != nil {
				w.Flush()
			}

			var outputOpts *printers.OutputOptions
			// if cmd does not specify output format and useOpenAPIPrintColumnFlagLabel flag is true,
			// then get the default output options for this mapping from OpenAPI schema.
			if !cmdSpecifiesOutputFmt(cmd) && useOpenAPIPrintColumns {
				outputOpts, _ = outputOptsForMappingFromOpenAPI(f, cmdutil.GetOpenAPICacheDir(cmd), mapping)
			}

			printer, err = f.PrinterForMapping(cmd, false, outputOpts, mapping, allNamespaces)
			if err != nil {
				if !errs.Has(err.Error()) {
					errs.Insert(err.Error())
					allErrs = append(allErrs, err)
				}
				continue
			}

			// add linebreak between resource groups (if there is more than one)
			// skip linebreak above first resource group
			noHeaders := cmdutil.GetFlagBool(cmd, "no-headers")
			if lastMapping != nil && !noHeaders {
				fmt.Fprintf(errOut, "%s\n", "")
			}

			lastMapping = mapping
		}

		// try to convert before apply filter func
		decodedObj, _ := kubectl.DecodeUnknownObject(original)

		// filter objects if filter has been defined for current object
		if isFiltered, err := filterFuncs.Filter(decodedObj, filterOpts); isFiltered {
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

			if err := printer.PrintObj(decodedObj, w); err != nil {
				if !errs.Has(err.Error()) {
					errs.Insert(err.Error())
					allErrs = append(allErrs, err)
				}
			}
			continue
		}
		objToPrint := decodedObj
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
	cmdutil.PrintFilterCount(errOut, len(objs), filteredResourceCount, len(allErrs), filterOpts, options.IgnoreNotFound)
	return utilerrors.NewAggregate(allErrs)
}

func addOpenAPIPrintColumnFlags(cmd *cobra.Command) {
	cmd.Flags().Bool(useOpenAPIPrintColumnFlagLabel, false, "If true, use x-kubernetes-print-column metadata (if present) from openapi schema for displaying a resource.")
	// marking it deprecated so that it is hidden from usage/help text.
	cmd.Flags().MarkDeprecated(useOpenAPIPrintColumnFlagLabel, "It's an experimental feature.")
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

	// user has not specified any output format, check if OpenAPI has
	// default specification to print this resource type
	api, err := f.OpenAPISchema(openAPIcacheDir)
	if err != nil {
		// Error getting schema
		return nil, false
	}
	// Found openapi metadata for this resource
	schema := api.LookupResource(mapping.GroupVersionKind)
	if schema == nil {
		// Schema not found, return empty columns
		return nil, false
	}

	columns, found := openapi.GetPrintColumns(schema.GetExtensions())
	if !found {
		// Extension not found, return empty columns
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
