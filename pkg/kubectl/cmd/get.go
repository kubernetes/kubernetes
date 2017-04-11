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

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	kapierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/util/i18n"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

// GetOptions contains all the options for running get cli command. As new fields are added, add them here
// instead of referencing the cmd.Flags()
type GetOptions struct {
	resource.FilenameOptions

	IgnoreNotFound bool
	Raw            string

	Selector     string
	LabelColumns []string

	ShowKind      bool
	AllNamespaces bool
	ShowAll       bool
	Export        bool
	Watch         bool
	IsWatchOnly   bool

	Include3rdParty bool

	cmdutil.PrinterOptions

	FilterOptions *printers.PrintOptions
	FilterFunc    kubectl.Filters

	// Out/ErrOut is the place to write user-facing output.
	Out    io.Writer
	ErrOut io.Writer

	Client  *restclient.RESTClient
	Builder *resource.Builder

	Decoder               runtime.Decoder
	GetPrinterWithMapping func(o cmdutil.PrinterOptions, mapping *meta.RESTMapping, isWatch, withNamespace bool) (printers.ResourcePrinter, error)
	GetPrinter            func(o cmdutil.PrinterOptions) (printers.ResourcePrinter, bool, error)
}

var (
	get_long = templates.LongDesc(`
		Display one or many resources.

		` + valid_resources + `

		This command will hide resources that have completed, such as pods that are
		in the Succeeded or Failed phases. You can see the full results for any
		resource by providing the '--show-all' flag.

		By specifying the output as 'template' and providing a Go template as the value
		of the --template flag, you can filter the attributes of the fetched resources.`)

	get_example = templates.Examples(i18n.T(`
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
		Long:    get_long,
		Example: get_example,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, out, errOut, cmd, args); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.RunGet(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		SuggestFor: []string{"list", "ps"},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmdutil.AddPrinterVarFlags(cmd, &options.PrinterOptions)
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	cmd.Flags().BoolVarP(&options.Watch, "watch", "w", false, "After listing/getting the requested object, watch for changes.")
	cmd.Flags().BoolVar(&options.IsWatchOnly, "watch-only", false, "Watch for changes to the requested object(s), without listing/getting first.")
	cmd.Flags().BoolVar(&options.ShowKind, "show-kind", false, "If present, list the resource type for the requested object(s).")
	cmd.Flags().BoolVar(&options.AllNamespaces, "all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&options.IgnoreNotFound, "ignore-not-found", false, "Treat \"resource not found\" as a successful retrieval.")
	cmd.Flags().StringSliceVarP(&options.LabelColumns, "label-columns", "L", []string{}, "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag options like -L label1 -L label2...")
	cmd.Flags().BoolVar(&options.Export, "export", false, "If true, use 'export' for the resources.  Exported resources are stripped of cluster-specific information.")
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddInclude3rdPartyVarFlags(cmd, &options.Include3rdParty)
	cmd.Flags().StringVar(&options.Raw, "raw", options.Raw, "Raw URI to request from the server.  Uses the transport specified by the kubeconfig file.")
	return cmd
}

func (o *GetOptions) Complete(f cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, args []string) error {
	// Setup writer
	o.Out = out
	o.ErrOut = errOut

	if len(o.Raw) > 0 {
		var err error
		o.Client, err = f.RESTClient()
		return err
	}

	if len(args) == 0 && cmdutil.IsFilenameEmpty(o.Filenames) {
		fmt.Fprint(o.ErrOut, "You must specify the type of resource to get. ", valid_resources)

		fullCmdName := cmd.Parent().CommandPath()
		usageString := "Required resource not specified."
		if len(fullCmdName) > 0 && cmdutil.IsSiblingCommandExists(cmd, "explain") {
			usageString = fmt.Sprintf("%s\nUse \"%s explain <resource>\" for a detailed description of that resource (e.g. %[2]s explain pods).", usageString, fullCmdName)
		}

		return cmdutil.UsageError(cmd, usageString)
	}

	// always show resources when getting by name or filename
	argsHasNames, err := resource.HasNames(args)
	if err != nil {
		return err
	}
	if len(o.Filenames) > 0 || argsHasNames {
		o.ShowAll = true
	}

	if resource.MultipleTypesRequested(args) {
		o.ShowKind = true
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if o.AllNamespaces {
		enforceNamespace = false
	}

	// Set up client based support.
	mapper, typer, err := f.UnstructuredObject()
	if err != nil {
		return err
	}
	o.Builder = resource.NewBuilder(mapper, f.CategoryExpander(), typer, resource.ClientMapperFunc(f.UnstructuredClientForMapping), unstructured.UnstructuredJSONScheme).
		NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(o.AllNamespaces).
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		SelectorParam(o.Selector).
		ExportParam(o.Export).
		ResourceTypeOrNameArgs(true, args...).
		Latest()

	// Set up filters
	o.FilterFunc = f.DefaultResourceFilterFunc()
	o.FilterOptions = f.DefaultResourceFilterOptions(cmd, o.AllNamespaces)

	o.Decoder = f.Decoder(true)
	o.GetPrinterWithMapping = func(po cmdutil.PrinterOptions, mapping *meta.RESTMapping, isWatch, withNamespace bool) (printers.ResourcePrinter, error) {
		return f.PrinterForMappingWithOption(po, mapping, isWatch, withNamespace)
	}
	o.GetPrinter = func(po cmdutil.PrinterOptions) (printers.ResourcePrinter, bool, error) {
		return f.PrinterWithOption(po)
	}

	return nil
}

// TODO: add some validation here
func (o *GetOptions) Validate() error {
	return nil
}

// RunGet implements the generic Get command
func (o *GetOptions) RunGet() error {
	if len(o.Raw) > 0 {
		return o.getRaw()
	}
	if o.Watch || o.IsWatchOnly {
		return o.watch()
	}
	return o.simpleGet()
}

func (o *GetOptions) getRaw() error {
	stream, err := o.Client.Get().RequestURI(o.Raw).Stream()
	if err != nil {
		return err
	}
	defer stream.Close()

	_, err = io.Copy(o.Out, stream)
	if err != nil && err != io.EOF {
		return err
	}
	return nil
}

func (o *GetOptions) watch() error {
	builder := o.Builder.SingleResourceType()
	r := builder.Do()
	err := r.Err()
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
		o.FilterFunc = nil
	}
	info := infos[0]
	mapping := info.ResourceMapping()
	printer, err := o.GetPrinterWithMapping(o.PrinterOptions, mapping, true, o.AllNamespaces)
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
	if !o.IsWatchOnly {
		var objsToPrint []runtime.Object
		if isList {
			objsToPrint, _ = meta.ExtractList(obj)
		} else {
			objsToPrint = append(objsToPrint, obj)
		}
		for _, objToPrint := range objsToPrint {
			if err := printer.PrintObj(objToPrint, o.Out); err != nil {
				return fmt.Errorf("unable to output the provided object: %v", err)
			}
		}
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
			err := printer.PrintObj(e.Object, o.Out)
			if err != nil {
				return false, err
			}
			return false, nil
		})
		return err
	})
	return nil

}

func (o *GetOptions) getWithGenericPrinter(r *resource.Result, printer printers.ResourcePrinter) error {
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

	if len(infos) == 0 && o.IgnoreNotFound {
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
			list.Items = append(list.Items, info.Object.(*unstructured.Unstructured))
		}
		obj = list
	} else {
		obj = infos[0].Object
	}

	isList := meta.IsListType(obj)
	if isList {
		_, items, err := cmdutil.FilterResourceList(obj, o.FilterFunc, o.FilterOptions)
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
			list.Object["selfLink"] = listMeta.GetSelfLink()
			list.Object["resourceVersion"] = listMeta.GetResourceVersion()
		}

		for _, item := range items {
			list.Items = append(list.Items, item.(*unstructured.Unstructured))
		}
		if err := printer.PrintObj(list, o.Out); err != nil {
			errs = append(errs, err)
		}

		return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
	}

	filteredResourceCount := 0
	if isFiltered, err := o.FilterFunc.Filter(obj, o.FilterOptions); !isFiltered {
		if err != nil {
			glog.V(2).Infof("Unable to filter resource: %v", err)
		} else if err := printer.PrintObj(obj, o.Out); err != nil {
			errs = append(errs, err)
		}
	} else if isFiltered {
		filteredResourceCount++
	}

	return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
}

func (o *GetOptions) simpleGet() error {
	builder := o.Builder.ContinueOnError().Flatten()
	r := builder.Do()
	err := r.Err()
	if err != nil {
		return err
	}

	// get the printer to decide whether we are using generic printer
	printer, generic, err := o.GetPrinter(o.PrinterOptions)
	if err != nil {
		return err
	}

	if r.TargetsSingleItems() {
		o.FilterFunc = nil
	}
	if o.IgnoreNotFound {
		r.IgnoreErrors(kapierrors.IsNotFound)
	}

	if generic {
		return o.getWithGenericPrinter(r, printer)
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
	var sorter *kubectl.RuntimeSort
	if len(o.SortBy) > 0 && len(objs) > 1 {
		// TODO: questionable
		if sorter, err = kubectl.SortObjects(o.Decoder, objs, o.SortBy); err != nil {
			return err
		}
	}

	// use the default printer for each object
	printer = nil
	var lastMapping *meta.RESTMapping
	w := printers.GetNewTabWriter(o.Out)

	showKind := o.ShowKind
	if cmdutil.MustPrintWithKinds(objs, infos, sorter) {
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
		if printer == nil || lastMapping == nil || mapping == nil || mapping.Resource != lastMapping.Resource {
			if printer != nil {
				w.Flush()
			}
			printer, err = o.GetPrinterWithMapping(o.PrinterOptions, mapping, true, o.AllNamespaces)
			if err != nil {
				if !errs.Has(err.Error()) {
					errs.Insert(err.Error())
					allErrs = append(allErrs, err)
				}
				continue
			}

			// add linebreak between resource groups (if there is more than one)
			// skip linebreak above first resource group
			if lastMapping != nil && !o.NoHeaders {
				fmt.Fprintf(o.ErrOut, "%s\n", "")
			}

			lastMapping = mapping
		}

		// try to convert before apply filter func
		decodedObj, _ := kubectl.DecodeUnknownObject(original)

		// filter objects if filter has been defined for current object
		if isFiltered, err := o.FilterFunc.Filter(decodedObj, o.FilterOptions); isFiltered {
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
		if err := printer.PrintObj(decodedObj, w); err != nil {
			if !errs.Has(err.Error()) {
				errs.Insert(err.Error())
				allErrs = append(allErrs, err)
			}
			continue
		}
	}
	w.Flush()
	cmdutil.PrintFilterCount(o.ErrOut, len(objs), filteredResourceCount, len(allErrs), "", o.FilterOptions, o.IgnoreNotFound)
	return utilerrors.NewAggregate(allErrs)
}
