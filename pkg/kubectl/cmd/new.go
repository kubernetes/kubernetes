/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"github.com/spf13/cobra"
	"io"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

const (
	new_long = `Create a new object of the given type using a template.

The new command pulls down an example for the given object type, allows
you to edit the filled in example, and then submits it for creation.  The
editing functionality functions similarly to the 'edit' command -- you
can control the editor used, as well as the format and version, using
the same flags and environment variables.

By default, before submission your new object will be validated.  If an
invalid object is encountered, you will be returned to the editor with the
invalid part indicated with comments.  Use the '--validate=false' in order
to skip local validation.

You can specify namespace, name, and labels (in 'key1=value1,key2=value2' form)
from the command line using the '--namespace', '--name', and '--labels' flags.

You can use the '--dry-run' flag to just output the example to standard out
without launching an editor or submitting the object for creation.
`

	new_example = `# Create a new pod named 'website':
  $ kubectl new pod --name website --labels end=front,app=site

  # Don't validate the object before creation
  $ kubectl new namespace --validate=false --name my-ns

  # Don't edit or create a new object -- just output the example
  $ kubectl new namespace --name my-ns --dry-run`
)

func NewCmdNew(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "new TYPE [--name NAME] [--labels key1=val,key2=val]",
		Short:   "Create a new object of the given type by filling in a template",
		Long:    new_long,
		Example: new_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunNew(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	cmd.Flags().StringP("output", "o", "yaml", "Output format. One of: 'yaml' or 'json'.")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given version (default api-version).")
	cmd.PersistentFlags().String("name", "", "The name of the new object")
	cmd.PersistentFlags().String("labels", "", "A comma-separated list of labels in key=value form to apply to the created object")
	cmd.Flags().Bool("dry-run", false, "If true, only print the example object, without editing or creating it.")

	cmdutil.AddValidateFlags(cmd)

	return cmd
}

func initExample(cmdNamespace string, resource string, f *cmdutil.Factory, cmd *cobra.Command) (runtime.Object, error) {
	mapper, _ := f.Object()
	version, kind, err := mapper.VersionAndKindForResource(resource)
	if err != nil {
		return nil, err
	}

	obj, err := api.Scheme.New(version, kind)
	if err != nil {
		return nil, err
	}

	mapping, err := mapper.RESTMapping(kind, version)
	if err != nil {
		return nil, err
	}

	err = mapping.SetAPIVersion(obj, version)
	if err != nil {
		return nil, err
	}
	err = mapping.SetKind(obj, kind)
	if err != nil {
		return nil, err
	}
	err = mapping.SetNamespace(obj, cmdNamespace)
	if err != nil {
		return nil, err
	}

	nameFlag := cmdutil.GetFlagString(cmd, "name")
	if nameFlag != "" {
		err = mapping.SetName(obj, nameFlag)
	} else {
		err = mapping.SetName(obj, " ")
	}
	if err != nil {
		return nil, err
	}

	labelsFlag := cmdutil.GetFlagString(cmd, "labels")
	if labelsFlag != "" {
		parsedLabels, err := kubectl.ParseLabels(labelsFlag)
		if err != nil {
			return nil, err
		}

		err = mapping.SetLabels(obj, parsedLabels)
		if err != nil {
			return nil, err
		}
	}

	return obj, nil
}

func RunNew(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to create. ", valid_resources)
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}
	targetResource := args[0]
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	obj, err := initExample(cmdNamespace, targetResource, f, cmd)
	if err != nil {
		return err
	}

	schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
	if err != nil {
		return err
	}

	// if we are using dry run, just output directly
	if cmdutil.GetFlagBool(cmd, "dry-run") {
		printer, _, err := getEditPrinter(cmd)
		if err != nil {
			return err
		}
		return printer.PrintObj(obj, out)
	}

	addSource := func(b *resource.Builder, enforceNamespace bool, printer kubectl.ResourcePrinter) *resource.Builder {
		initialOutput := &bytes.Buffer{}
		printer.PrintObj(obj, initialOutput)
		b = b.Stream(initialOutput, "[new]")

		if enforceNamespace {
			b = b.RequireNamespace()
		}

		return b
	}

	process := func(visitor resource.Visitor, original, edited []byte, obj runtime.Object, updates *resource.Info, file string, results *editResults, mapper meta.RESTMapper, defaultVersion string) error {

		// validate if we asked for it
		if err := resource.ValidateSchema(edited, schema); err != nil {
			// add in our own validation error
			results.edit = append(results.edit, updates)
			reason := editReason{
				head:  fmt.Sprintf("failed to validate the %s %s", updates.Mapping.Kind, updates.Name),
				other: []string{err.Error()},
			}
			results.header.reasons = append(results.header.reasons, reason)
			fmt.Fprintln(out, "Error: failed to validate the %s %s: %v", updates.Mapping.Kind, updates.Name, err)
			return nil
		}

		return visitor.Visit(func(info *resource.Info, err error) error {
			if err != nil {
				return err
			}

			data, err := info.Mapping.Codec.Encode(info.Object)
			if err != nil {
				fmt.Fprintln(out, results.addError(err, info))
				return nil
				//return cmdutil.AddSourceToErr("creating", info.Source, err)
			}
			obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, data)
			if err != nil {
				//return cmdutil.AddSourceToErr("creating", info.Source, err)
				fmt.Fprintln(out, results.addError(err, info))
				return nil
			}
			info.Refresh(obj, true)
			printObjectSpecificMessage(info.Object, out)
			cmdutil.PrintSuccess(mapper, false, out, info.Mapping.Resource, info.Name, "created")
			return nil
		})
	}

	return doEdit(f, out, cmd, "kubectl-new-", false, addSource, process)
}
