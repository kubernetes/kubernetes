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

package cmd

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/spf13/cobra"
)

// NewCmdGet creates a command object for the generic "get" action, which
// retrieves one or more resources from a server.
func (f *Factory) NewCmdGet(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get [(-o|--output=)json|yaml|...] <resource> [<id>]",
		Short: "Display one or many resources",
		Long: `Display one or many resources.

Possible resources include pods (po), replication controllers (rc), services
(se), minions (mi), or events (ev).

If you specify a Go template, you can use any fields defined for the API version
you are connecting to the server with.

Examples:
  $ kubectl get pods
  <list all pods in ps output format>

  $ kubectl get replicationController 1234-56-7890-234234-456456
  <list single replication controller in ps output format>

  $ kubectl get -o json pod 1234-56-7890-234234-456456
  <list single pod in json output format>

  $ kubectl get rc,services
  <list replication controllers and services together in ps output format>`,
		Run: func(cmd *cobra.Command, args []string) {
			RunGet(f, out, cmd, args)
		},
	}
	util.AddPrinterFlags(cmd)
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolP("watch", "w", false, "After listing/getting the requested object, watch for changes.")
	cmd.Flags().Bool("watch-only", false, "Watch for changes to the requested object(s), without listing/getting first.")
	return cmd
}

// RunGet implements the generic Get command
// TODO: convert all direct flag accessors to a struct and pass that instead of cmd
// TODO: return an error instead of using glog.Fatal and checkErr
func RunGet(f *Factory, out io.Writer, cmd *cobra.Command, args []string) {
	selector := util.GetFlagString(cmd, "selector")
	mapper, typer := f.Object(cmd)

	cmdNamespace, err := f.DefaultNamespace(cmd)
	checkErr(err)

	// handle watch separately since we cannot watch multiple resource types
	isWatch, isWatchOnly := util.GetFlagBool(cmd, "watch"), util.GetFlagBool(cmd, "watch-only")
	if isWatch || isWatchOnly {
		r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
			NamespaceParam(cmdNamespace).DefaultNamespace().
			SelectorParam(selector).
			ResourceTypeOrNameArgs(args...).
			SingleResourceType().
			Do()

		mapping, err := r.ResourceMapping()
		checkErr(err)

		printer, err := f.PrinterForMapping(cmd, mapping)
		checkErr(err)

		obj, err := r.Object()
		checkErr(err)

		rv, err := mapping.MetadataAccessor.ResourceVersion(obj)
		checkErr(err)

		// print the current object
		if !isWatchOnly {
			if err := printer.PrintObj(obj, out); err != nil {
				checkErr(fmt.Errorf("unable to output the provided object: %v", err))
			}
		}

		// print watched changes
		w, err := r.Watch(rv)
		checkErr(err)

		kubectl.WatchLoop(w, func(e watch.Event) error {
			return printer.PrintObj(e.Object, out)
		})
		return
	}

	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		SelectorParam(selector).
		ResourceTypeOrNameArgs(args...).
		Latest()
	printer, generic, err := util.PrinterForCommand(cmd)
	checkErr(err)

	if generic {
		clientConfig, err := f.ClientConfig(cmd)
		checkErr(err)
		defaultVersion := clientConfig.Version

		// the outermost object will be converted to the output-version
		version := util.OutputVersion(cmd, defaultVersion)
		if len(version) == 0 {
			// TODO: add a new ResourceBuilder mode for Object() that attempts to ensure the objects
			// are in the appropriate version if one exists (and if not, use the best effort).
			// TODO: ensure api-version is set with the default preferred api version by the client
			// builder on initialization
			version = latest.Version
		}
		printer = kubectl.NewVersionedPrinter(printer, api.Scheme, version)

		obj, err := b.Flatten().Do().Object()
		checkErr(err)

		err = printer.PrintObj(obj, out)
		checkErr(err)
		return
	}

	// use the default printer for each object
	err = b.Do().Visit(func(r *resource.Info) error {
		printer, err := f.PrinterForMapping(cmd, r.Mapping)
		if err != nil {
			return err
		}
		return printer.PrintObj(r.Object, out)
	})
	checkErr(err)
}
