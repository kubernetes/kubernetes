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
	"log"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"

	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdEvents(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "events [(-o|--output=)json|yaml|...] [--watch] [--namespace=<namespace>] [--name=<name>] [--kind=<kind>] [--uid=<uid>]",
		Short: "List or watch events",
		Long: `List or watch events.

Provide as many or as few modifiers as you wish.

If you specify a Go template, you can use any fields defined for the API version
you are connecting to the server with.

Examples:
  $ kubectl events
  <list all events for the entire cluster in ps output format>

  $ kubectl events -uid=1234-56-7890-234234-456456
  <list events for any object matching the given uid>

  $ kubectl events --watch --namespace=mynamespace --kind=pod
  <list all events for pods in the namespace 'mynamespace', then stream new events for same>`,
		Run: func(cmd *cobra.Command, args []string) {
			mapping, namespace, _ := ResourceOrTypeFromArgs(cmd, []string{"events"}, f.Mapper)

			fields := labels.Set{}
			if f := cmd.Flags().Lookup("namespace"); f != nil && f.Changed {
				log.Printf("got %v", f.Value.String())
				namespace = f.Value.String()
				fields["involvedObject.namespace"] = namespace
			}
			if f := cmd.Flags().Lookup("name"); f != nil && f.Changed {
				log.Printf("got %v", f.Value.String())
				fields["involvedObject.name"] = f.Value.String()
			}
			if f := cmd.Flags().Lookup("kind"); f != nil && f.Changed {
				log.Printf("got %v", f.Value.String())
				fields["involvedObject.kind"] = f.Value.String()
			}
			if f := cmd.Flags().Lookup("uid"); f != nil && f.Changed {
				log.Printf("got %v", f.Value.String())
				fields["involvedObject.uid"] = f.Value.String()
			}

			client, err := f.Client(cmd, mapping)
			checkErr(err)

			outputFormat := GetFlagString(cmd, "output")
			templateFile := GetFlagString(cmd, "template")
			defaultPrinter, err := f.Printer(cmd, mapping, GetFlagBool(cmd, "no-headers"))
			checkErr(err)

			outputVersion := GetFlagString(cmd, "output-version")
			if len(outputVersion) == 0 {
				outputVersion = mapping.APIVersion
			}
			printer, err := kubectl.GetPrinter(outputVersion, outputFormat, templateFile, defaultPrinter)
			checkErr(err)

			// TODO: support watch
			log.Printf("field selector = %s", fields.AsSelector())

			restHelper := kubectl.NewRESTHelper(client, mapping)
			obj, err := restHelper.List(namespace, labels.Set{}.AsSelector(), fields.AsSelector())
			checkErr(err)

			if err := printer.PrintObj(obj, out); err != nil {
				checkErr(fmt.Errorf("Unable to output the provided object: %v", err))
			}

			if GetFlagBool(cmd, "watch") {
				vi, err := latest.InterfacesFor(outputVersion)
				checkErr(err)

				rv, err := vi.MetadataAccessor.ResourceVersion(obj)
				checkErr(err)

				w, err := restHelper.Watch(namespace, rv, labels.Set{}.AsSelector(), fields.AsSelector())
				checkErr(err)

				kubectl.WatchLoop(w, printer, out)
			}
		},
	}
	cmd.Flags().StringP("output", "o", "", "Output format: json|yaml|template|templatefile")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given version (default api-version)")
	cmd.Flags().Bool("no-headers", false, "When using the default output, don't print headers")
	cmd.Flags().StringP("template", "t", "", "Template string or path to template file to use when --output=template or --output=templatefile")
	cmd.Flags().Bool("watch", false, "List and then stream events instead of just listing them.")
	cmd.Flags().String("namespace", "", "Find only events about objects in this namespace.")
	cmd.Flags().StringP("name", "n", "", "Find only events about objects with this name.")
	cmd.Flags().String("uid", "", "Find only events about the object with this uid.")
	cmd.Flags().StringP("kind", "k", "", "Find only events about objects with this kind.")
	return cmd
}
