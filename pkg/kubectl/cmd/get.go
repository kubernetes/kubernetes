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

package cmd

import (
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/spf13/cobra"
)

const (
	get_long = `Display one or many resources.

Possible resource types include (case insensitive): pods (po), services (svc),
replicationcontrollers (rc), nodes (no), events (ev), componentstatuses (cs),
limitranges (limits), persistentvolumes (pv), persistentvolumeclaims (pvc),
resourcequotas (quota), namespaces (ns), endpoints (ep) or secrets.

By specifying the output as 'template' and providing a Go template as the value
of the --template flag, you can filter the attributes of the fetched resource(s).`
	get_example = `// List all pods in ps output format.
$ kubectl get pods

// List all pods in ps output format with more information (such as node name).
$ kubectl get pods -o wide

// List a single replication controller with specified NAME in ps output format.
$ kubectl get replicationcontroller web

// List a single pod in JSON output format.
$ kubectl get -o json pod web-pod-13je7

// Return only the phase value of the specified pod.
$ kubectl get -o template web-pod-13je7 --template={{.status.phase}} --api-version=v1

// List all replication controllers and services together in ps output format.
$ kubectl get rc,services

// List one or more resources by their type and names.
$ kubectl get rc/web service/frontend pods/web-pod-13je7`
)

// NewCmdGet creates a command object for the generic "get" action, which
// retrieves one or more resources from a server.
func NewCmdGet(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	p := kubectl.NewHumanReadablePrinter(false, false, false, []string{})
	validArgs := p.HandledResources()

	cmd := &cobra.Command{
		Use:     "get [(-o|--output=)json|yaml|template|wide|...] (TYPE [(NAME | -l label] | TYPE/NAME ...)",
		Short:   "Display one or many resources",
		Long:    get_long,
		Example: get_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGet(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
		ValidArgs: validArgs,
	}
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolP("watch", "w", false, "After listing/getting the requested object, watch for changes.")
	cmd.Flags().Bool("watch-only", false, "Watch for changes to the requested object(s), without listing/getting first.")
	cmd.Flags().Bool("all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().StringSliceP("label-columns", "L", []string{}, "Accepts a comma separated list of labels that are going to be presented as columns. Names are case-sensitive. You can also use multiple flag statements like -L label1 -L label2...")
	return cmd
}

// RunGet implements the generic Get command
// TODO: convert all direct flag accessors to a struct and pass that instead of cmd
func RunGet(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	selector := cmdutil.GetFlagString(cmd, "selector")
	allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")
	mapper, typer := f.Object()

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to get. ", valid_resources, `   * componentstatuses (aka 'cs')
   * endpoints (aka 'ep')
`)
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	// handle watch separately since we cannot watch multiple resource types
	isWatch, isWatchOnly := cmdutil.GetFlagBool(cmd, "watch"), cmdutil.GetFlagBool(cmd, "watch-only")
	if isWatch || isWatchOnly {
		r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
			NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
			SelectorParam(selector).
			ResourceTypeOrNameArgs(true, args...).
			SingleResourceType().
			Do()
		if err != nil {
			return err
		}

		mapping, err := r.ResourceMapping()
		if err != nil {
			return err
		}

		printer, err := f.PrinterForMapping(cmd, mapping, allNamespaces)
		if err != nil {
			return err
		}

		obj, err := r.Object()
		if err != nil {
			return err
		}

		rv, err := mapping.MetadataAccessor.ResourceVersion(obj)
		if err != nil {
			return err
		}

		// print the current object
		if !isWatchOnly {
			if err := printer.PrintObj(obj, out); err != nil {
				return fmt.Errorf("unable to output the provided object: %v", err)
			}
		}

		// print watched changes
		w, err := r.Watch(rv)
		if err != nil {
			return err
		}

		kubectl.WatchLoop(w, func(e watch.Event) error {
			return printer.PrintObj(e.Object, out)
		})
		return nil
	}

	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand()).
		NamespaceParam(cmdNamespace).DefaultNamespace().AllNamespaces(allNamespaces).
		SelectorParam(selector).
		ResourceTypeOrNameArgs(true, args...).
		ContinueOnError().
		Latest()
	printer, generic, err := cmdutil.PrinterForCommand(cmd)
	if err != nil {
		return err
	}

	if generic {
		clientConfig, err := f.ClientConfig()
		if err != nil {
			return err
		}
		defaultVersion := clientConfig.Version

		singular := false
		r := b.Flatten().Do()
		infos, err := r.IntoSingular(&singular).Infos()
		if err != nil {
			return err
		}

		// the outermost object will be converted to the output-version, but inner
		// objects can use their mappings
		version := cmdutil.OutputVersion(cmd, defaultVersion)
		obj, err := resource.AsVersionedObject(infos, !singular, version)
		if err != nil {
			return err
		}

		return printer.PrintObj(obj, out)
	}

	// use the default printer for each object
	return b.Do().Visit(func(r *resource.Info) error {
		printer, err := f.PrinterForMapping(cmd, r.Mapping, allNamespaces)
		if err != nil {
			return err
		}
		return printer.PrintObj(r.Object, out)
	})
}
