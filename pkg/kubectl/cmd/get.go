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
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/watch"
)

// GetOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type GetOptions struct {
	Filenames []string
	Recursive bool

	Raw string
}

var (
	get_long = dedent.Dedent(`
		Display one or many resources.

		`) + valid_resources + dedent.Dedent(`

		This command will hide resources that have completed. For instance, pods that are in the Succeeded or Failed phases.
		You can see the full results for any resource by providing the '--show-all' flag.

		By specifying the output as 'template' and providing a Go template as the value
		of the --template flag, you can filter the attributes of the fetched resource(s).`)
	get_example = dedent.Dedent(`
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
		kubectl get rc/web service/frontend pods/web-pod-13je7`)
)

// NewCmdGet creates a command object for the generic "get" action, which
// retrieves one or more resources from a server.
func NewCmdGet(f *cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &GetOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, kubectl.PrintOptions{
		ColumnLabels: []string{},
	})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     "get [(-o|--output=)json|yaml|wide|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=...] (TYPE [NAME | -l label] | TYPE/NAME ...) [flags]",
		Short:   "Display one or many resources",
		Long:    get_long,
		Example: get_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGet(f, out, errOut, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		SuggestFor: []string{"list", "ps"},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolP("watch", "w", false, "After listing/getting the requested object, watch for changes.")
	cmd.Flags().Bool("watch-only", false, "Watch for changes to the requested object(s), without listing/getting first.")
	cmd.Flags().Bool("show-kind", false, "If present, list the resource type for the requested object(s).")
	cmd.Flags().Bool("all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().StringSliceP("label-columns", "L", []string{}, "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag options like -L label1 -L label2...")
	cmd.Flags().Bool("export", false, "If true, use 'export' for the resources.  Exported resources are stripped of cluster-specific information.")
	usage := "Filename, directory, or URL to a file identifying the resource to get from a server."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	cmd.Flags().StringVar(&options.Raw, "raw", options.Raw, "Raw URI to request from the server.  Uses the transport specified by the kubeconfig file.")
	return cmd
}

// RunGet implements the generic Get command
// TODO: convert all direct flag accessors to a struct and pass that instead of cmd
func RunGet(f *cmdutil.Factory, out, errOut io.Writer, cmd *cobra.Command, args []string, options *GetOptions) error {
	if len(options.Raw) > 0 {
		client, err := f.Client()
		if err != nil {
			return err
		}

		stream, err := client.RESTClient.Get().RequestURI(options.Raw).Stream()
		if err != nil {
			return err
		}
		defer stream.Close()

		for {
			buffer := make([]byte, 1024, 1024)
			bytesRead, err := stream.Read(buffer)
			if bytesRead > 0 {
				fmt.Printf("%s", string(buffer[:bytesRead]))
			}
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return err
			}
		}
	}

	selector := cmdutil.GetFlagString(cmd, "selector")
	allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")
	showKind := cmdutil.GetFlagBool(cmd, "show-kind")
	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	filterFuncs := f.DefaultResourceFilterFunc()
	filterOpts := f.DefaultResourceFilterOptions(cmd, allNamespaces)
	printAll := false

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if allNamespaces {
		enforceNamespace = false
	}

	if len(args) == 0 && len(options.Filenames) == 0 {
		fmt.Fprint(errOut, "You must specify the type of resource to get. ", valid_resources)

		fullCmdName := cmd.Parent().CommandPath()
		usageString := "Required resource not specified."
		if len(fullCmdName) > 0 && cmdutil.IsSiblingCommandExists(cmd, "explain") {
			usageString = fmt.Sprintf("%s\nUse \"%s explain <resource>\" for a detailed description of that resource (e.g. %[2]s explain pods).", usageString, fullCmdName)
		}

		return cmdutil.UsageError(cmd, usageString)
	}

	// determine if args contains "all"
	for _, a := range args {
		if a == "all" {
			printAll = true
			break
		}
	}

	// always show resources when getting by name or filename
	argsHasNames, err := resource.HasNames(args)
	if err != nil {
		return err
	}
	if len(options.Filenames) > 0 || argsHasNames {
		cmd.Flag("show-all").Value.Set("true")
	}
	export := cmdutil.GetFlagBool(cmd, "export")

	// handle watch separately since we cannot watch multiple resource types
	isWatch, isWatchOnly := cmdutil.GetFlagBool(cmd, "watch"), cmdutil.GetFlagBool(cmd, "watch-only")
	if isWatch || isWatchOnly {
		r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
			NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
			FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
			SelectorParam(selector).
			ExportParam(export).
			ResourceTypeOrNameArgs(true, args...).
			SingleResourceType().
			Latest().
			Do()
		err := r.Err()
		if err != nil {
			return err
		}
		infos, err := r.Infos()
		if err != nil {
			return err
		}
		if len(infos) != 1 {
			return fmt.Errorf("watch is only supported on individual resources and resource collections - %d resources were found", len(infos))
		}
		info := infos[0]
		mapping := info.ResourceMapping()
		printer, err := f.PrinterForMapping(cmd, mapping, allNamespaces)
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
		filteredResourceCount := 0
		if !isWatchOnly {
			if err := printer.PrintObj(obj, out); err != nil {
				return fmt.Errorf("unable to output the provided object: %v", err)
			}
			filteredResourceCount++
			cmdutil.PrintFilterCount(filteredResourceCount, mapping.Resource, errOut, filterOpts)
		}

		// print watched changes
		w, err := r.Watch(rv)
		if err != nil {
			return err
		}

		first := true
		filteredResourceCount = 0
		kubectl.WatchLoop(w, func(e watch.Event) error {
			if !isList && first {
				// drop the initial watch event in the single resource case
				first = false
				return nil
			}
			err := printer.PrintObj(e.Object, out)
			if err == nil {
				filteredResourceCount++
				cmdutil.PrintFilterCount(filteredResourceCount, mapping.Resource, errOut, filterOpts)
			}
			return err
		})
		return nil
	}

	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
		FilenameParam(enforceNamespace, options.Recursive, options.Filenames...).
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

	printer, generic, err := cmdutil.PrinterForCommand(cmd)
	if err != nil {
		return err
	}

	if generic {
		clientConfig, err := f.ClientConfig()
		if err != nil {
			return err
		}

		// the outermost object will be converted to the output-version, but inner
		// objects can use their mappings
		version, err := cmdutil.OutputVersion(cmd, clientConfig.GroupVersion)
		if err != nil {
			return err
		}

		var errs []error
		singular := false
		infos, err := r.IntoSingular(&singular).Infos()
		if err != nil {
			if singular {
				return err
			}
			errs = append(errs, err)
		}
		if len(infos) == 0 {
			outputEmptyListWarning(errOut)
		}

		res := ""
		if len(infos) > 0 {
			res = infos[0].ResourceMapping().Resource
		}

		obj, err := resource.AsVersionedObject(infos, !singular, version, f.JSONEncoder())
		if err != nil {
			return err
		}

		isList := meta.IsListType(obj)
		if isList {
			filteredResourceCount, items, err := cmdutil.FilterResourceList(obj, filterFuncs, filterOpts)
			if err != nil {
				return err
			}
			filteredObj, err := cmdutil.ObjectListToVersionedObject(items, version)
			if err != nil {
				return err
			}
			if err := printer.PrintObj(filteredObj, out); err != nil {
				errs = append(errs, err)
			}

			cmdutil.PrintFilterCount(filteredResourceCount, res, errOut, filterOpts)
			// we can receive multiple layers of aggregates, so attempt to flatten them as much
			// as is possible before returning.
			return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
		}

		filteredResourceCount := 0
		if isFiltered, err := filterFuncs.Filter(obj, filterOpts); !isFiltered {
			if err != nil {
				glog.V(2).Infof("Unable to filter resource: %v", err)
			} else if err := printer.PrintObj(obj, out); err != nil {
				errs = append(errs, err)
			}
		} else if isFiltered {
			filteredResourceCount++
		}
		cmdutil.PrintFilterCount(filteredResourceCount, res, errOut, filterOpts)
		// we can receive multiple layers of aggregates, so attempt to flatten them as much
		// as is possible before returning.
		return utilerrors.Reduce(utilerrors.Flatten(utilerrors.NewAggregate(errs)))
	}

	allErrs := []error{}
	infos, err := r.Infos()
	if err != nil {
		allErrs = append(allErrs, err)
	}
	if len(infos) == 0 {
		outputEmptyListWarning(errOut)
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
		clientConfig, err := f.ClientConfig()
		if err != nil {
			return err
		}

		version, err := cmdutil.OutputVersion(cmd, clientConfig.GroupVersion)
		if err != nil {
			return err
		}

		for ix := range infos {
			objs[ix], err = infos[ix].Mapping.ConvertToVersion(infos[ix].Object, version)
			if err != nil {
				allErrs = append(allErrs, err)
				continue
			}
		}

		// TODO: questionable
		if sorter, err = kubectl.SortObjects(f.Decoder(true), objs, sorting); err != nil {
			return err
		}
	}

	// use the default printer for each object
	printer = nil
	var lastMapping *meta.RESTMapping
	w := kubectl.GetNewTabWriter(out)
	filteredResourceCount := 0

	if cmdutil.MustPrintWithKinds(objs, infos, sorter, printAll) {
		showKind = true
	}

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
				cmdutil.PrintFilterCount(filteredResourceCount, lastMapping.Resource, errOut, filterOpts)
			}
			printer, err = f.PrinterForMapping(cmd, mapping, allNamespaces)
			if err != nil {
				allErrs = append(allErrs, err)
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

		// filter objects if filter has been defined for current object
		if isFiltered, err := filterFuncs.Filter(original, filterOpts); isFiltered {
			if err == nil {
				filteredResourceCount++
				continue
			}
			allErrs = append(allErrs, err)
		}

		if resourcePrinter, found := printer.(*kubectl.HumanReadablePrinter); found {
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

			if err := printer.PrintObj(original, w); err != nil {
				allErrs = append(allErrs, err)
			}
			continue
		}
		if err := printer.PrintObj(original, w); err != nil {
			allErrs = append(allErrs, err)
			continue
		}
	}
	w.Flush()
	if printer != nil && lastMapping != nil {
		cmdutil.PrintFilterCount(filteredResourceCount, lastMapping.Resource, errOut, filterOpts)
	}
	return utilerrors.NewAggregate(allErrs)
}

// outputEmptyListWarning outputs a warning indicating that no items are available to display
func outputEmptyListWarning(out io.Writer) error {
	_, err := fmt.Fprintf(out, "%s\n", "No resources found.")
	return err
}
