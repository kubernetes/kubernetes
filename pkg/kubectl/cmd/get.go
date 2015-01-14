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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
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
	cmd.Flags().StringP("output", "o", "", "Output format: json|yaml|template|templatefile")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given version (default api-version)")
	cmd.Flags().Bool("no-headers", false, "When using the default output, don't print headers")
	cmd.Flags().StringP("template", "t", "", "Template string or path to template file to use when -o=template or -o=templatefile.")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolP("watch", "w", false, "After listing/getting the requested object, watch for changes.")
	cmd.Flags().Bool("watch-only", false, "Watch for changes to the requseted object(s), without listing/getting first.")
	return cmd
}

// RunGet implements the generic Get command
// TODO: convert all direct flag accessors to a struct and pass that instead of cmd
// TODO: return an error instead of using glog.Fatal and checkErr
func RunGet(f *Factory, out io.Writer, cmd *cobra.Command, args []string) {
	selector := GetFlagString(cmd, "selector")
	mapper, typer := f.Object(cmd)

	// handle watch separately since we cannot watch multiple resource types
	isWatch, isWatchOnly := GetFlagBool(cmd, "watch"), GetFlagBool(cmd, "watch-only")
	if isWatch || isWatchOnly {
		r := resource.NewBuilder(mapper, typer, ClientMapperForCommand(cmd, f)).
			NamespaceParam(GetKubeNamespace(cmd)).DefaultNamespace().
			SelectorParam(selector).
			ResourceTypeOrNameArgs(args...).
			SingleResourceType().
			Do()

		mapping, err := r.ResourceMapping()
		checkErr(err)

		printer, err := printerForMapping(f, cmd, mapping)
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

	printer, generic, err := printerForCommand(cmd)
	checkErr(err)

	b := resource.NewBuilder(mapper, typer, ClientMapperForCommand(cmd, f)).
		NamespaceParam(GetKubeNamespace(cmd)).DefaultNamespace().
		SelectorParam(selector).
		ResourceTypeOrNameArgs(args...).
		Latest()

	if generic {
		// the outermost object will be converted to the output-version
		version := outputVersion(cmd)
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
		printer, err := printerForMapping(f, cmd, r.Mapping)
		if err != nil {
			return err
		}
		return printer.PrintObj(r.Object, out)
	})
	checkErr(err)
}

// outputVersion returns the preferred output version for generic content (JSON, YAML, or templates)
func outputVersion(cmd *cobra.Command) string {
	outputVersion := GetFlagString(cmd, "output-version")
	if len(outputVersion) == 0 {
		outputVersion = GetFlagString(cmd, "api-version")
	}
	return outputVersion
}

// printerForCommand returns the default printer for this command.
func printerForCommand(cmd *cobra.Command) (kubectl.ResourcePrinter, bool, error) {
	outputFormat := GetFlagString(cmd, "output")
	templateFile := GetFlagString(cmd, "template")
	if len(outputFormat) == 0 && len(templateFile) != 0 {
		outputFormat = "template"
	}

	return kubectl.GetPrinter(outputFormat, templateFile)
}

// printerForMapping returns a printer suitable for displaying the provided resource type.
func printerForMapping(f *Factory, cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.ResourcePrinter, error) {
	printer, ok, err := printerForCommand(cmd)
	if err != nil {
		return nil, err
	}
	if ok {
		version := outputVersion(cmd)
		if len(version) == 0 {
			version = mapping.APIVersion
		}
		if len(version) == 0 {
			return nil, fmt.Errorf("you must specify an output-version when using this output format")
		}
		printer = kubectl.NewVersionedPrinter(printer, mapping.ObjectConvertor, version)
	} else {
		printer, err = f.Printer(cmd, mapping, GetFlagBool(cmd, "no-headers"))
		if err != nil {
			return nil, err
		}
	}
	return printer, nil
}
