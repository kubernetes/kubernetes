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

const (
	get_long = `Display one or many resources.

Possible resources include pods (po), replication controllers (rc), services
(se), minions (mi), or events (ev).

By specifying the output as 'template' and providing a Go template as the value
of the --template flag, you can filter the attributes of the fetched resource(s).`
	get_example = `// List all pods in ps output format.
$ kubectl get pods

// List a single replication controller with specified ID in ps output format.
$ kubectl get replicationController 1234-56-7890-234234-456456

// List a single pod in JSON output format.
$ kubectl get -o json pod 1234-56-7890-234234-456456

// Return only the status value of the specified pod.
$ kubectl get -o template pod 1234-56-7890-234234-456456 --template={{.currentState.status}}

// List all replication controllers and services together in ps output format.
$ kubectl get rc,services`
)

// NewCmdGet creates a command object for the generic "get" action, which
// retrieves one or more resources from a server.
func (f *Factory) NewCmdGet(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "get [(-o|--output=)json|yaml|template|...] RESOURCE [ID]",
		Short:   "Display one or many resources",
		Long:    get_long,
		Example: get_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGet(f, out, cmd, args)
			util.CheckErr(err)
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
func RunGet(f *Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	selector := util.GetFlagString(cmd, "selector")
	mapper, typer := f.Object()

	cmdNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	// handle watch separately since we cannot watch multiple resource types
	isWatch, isWatchOnly := util.GetFlagBool(cmd, "watch"), util.GetFlagBool(cmd, "watch-only")
	if isWatch || isWatchOnly {
		r := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
			NamespaceParam(cmdNamespace).DefaultNamespace().
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

		printer, err := f.PrinterForMapping(cmd, mapping)
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

	b := resource.NewBuilder(mapper, typer, f.ClientMapperForCommand(cmd)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		SelectorParam(selector).
		ResourceTypeOrNameArgs(true, args...).
		Latest()
	printer, generic, err := util.PrinterForCommand(cmd)
	if err != nil {
		return err
	}

	if generic {
		clientConfig, err := f.ClientConfig()
		if err != nil {
			return err
		}
		defaultVersion := clientConfig.Version

		// the outermost object will be converted to the output-version
		version := util.OutputVersion(cmd, defaultVersion)

		r := b.Flatten().Do()
		obj, err := r.Object()
		if err != nil {
			return err
		}

		// try conversion to all the possible versions
		// TODO: simplify by adding a ResourceBuilder mode
		versions := []string{version, latest.Version}
		infos, _ := r.Infos()
		for _, info := range infos {
			versions = append(versions, info.Mapping.APIVersion)
		}

		// TODO: add a new ResourceBuilder mode for Object() that attempts to ensure the objects
		// are in the appropriate version if one exists (and if not, use the best effort).
		// TODO: ensure api-version is set with the default preferred api version by the client
		// builder on initialization
		printer := kubectl.NewVersionedPrinter(printer, api.Scheme, versions...)

		return printer.PrintObj(obj, out)
	}

	// use the default printer for each object
	return b.Do().Visit(func(r *resource.Info) error {
		printer, err := f.PrinterForMapping(cmd, r.Mapping)
		if err != nil {
			return err
		}
		return printer.PrintObj(r.Object, out)
	})
}
