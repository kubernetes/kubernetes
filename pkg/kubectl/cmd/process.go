/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// ProcessOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ProcessOptions struct {
	Output    string
	Filenames []string
}

// ProcessSubcommandOptions wraps the flags passed on the command line as well
// as the StructuredGenerator used to create the request
type ProcessSubcommandOptions struct {
	Name                string
	Output              string
	StructuredGenerator kubectl.StructuredGenerator
}

const (
	process_long = `Process a Template by substituting parametrized fields and printing the resulting Config as yaml or json.

Objects are returned without being created, and must be passed to a subsequent create operation if this is the desired result. `
	process_short   = `Expand a Template by substitution parametrized fields and printing the resulting Config.`
	process_example = `# Process the Template whose Name matches the Name in TemplateParameters.
kubectl process -f params.yaml

# Output the results as json
kubectl process -o json -f params.yaml

# Process a Template and Create the resulting Objects
kubectl process -f params.yaml | kubectl create -f -
`
	process_usage = `process -f FILENAME`
)

func NewCmdProcess(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ProcessOptions{}

	// retrieve a list of handled resources from printer as valid args
	validArgs, argAliases := []string{}, []string{}
	p, err := f.Printer(nil, false, false, false, false, false, false, []string{})
	cmdutil.CheckErr(err)
	if p != nil {
		validArgs = p.HandledResources()
		argAliases = kubectl.ResourceAliases(validArgs)
	}

	cmd := &cobra.Command{
		Use:     process_usage,
		Short:   process_short,
		Long:    process_long,
		Example: process_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunProcess(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	usage := "Filename, directory, or URL to a file containing the template parameters."
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, usage)
	cmd.Flags().StringVarP(&options.Output, "output", "o", "yaml", "Output format. One of: json|yaml.")

	// TemplateParameters Generator Command
	cmd.AddCommand(NewCmdProcessTemplate(f, out))
	return cmd
}

func RunProcess(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *ProcessOptions) error {
	outputMode := cmdutil.GetFlagString(cmd, "output")
	if outputMode != "" && outputMode != "json" && outputMode != "yaml" {
		return cmdutil.UsageError(cmd, "Unexpected -o output mode: %v. We only support '-o yaml and -o json'.", outputMode)
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, false, options.Filenames...).
		ResourceTypeOrNameArgs(false, args...).RequireObject(false).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	return ProcessResult(f, options, r, out, cmd, mapper)
}

// RunProcessSubcommand executes a process generator subcommand using the specified options (command line flags).
func RunProcessSubcommand(f *cmdutil.Factory, cmd *cobra.Command, out io.Writer, options *ProcessSubcommandOptions) error {
	// Get the namespace
	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	// Generate the object using the generator
	obj, err := options.StructuredGenerator.StructuredGenerate()
	if err != nil {
		return err
	}

	// Create the Processor used to talk to the Api Server
	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	gvk, err := typer.ObjectKind(obj)
	mapping, err := mapper.RESTMapping(unversioned.GroupKind{Group: gvk.Group, Kind: gvk.Kind}, gvk.Version)
	proc, err := f.Processor(mapping)
	if err != nil {
		return err
	}

	// Execute the Process Request
	result, err := proc.Process(namespace, obj)
	if err != nil {
		return err
	}

	// Format and print the result
	printer, _, err := kubectl.GetPrinter(options.Output, "")
	return printer.PrintObj(result, out)
}

// ProcessResult executes a process command using the specified resources (uris passedusing '-f').
func ProcessResult(f *cmdutil.Factory, options *ProcessOptions, r *resource.Result, out io.Writer, cmd *cobra.Command, mapper meta.RESTMapper) error {
	printer, _, err := kubectl.GetPrinter(options.Output, "")
	if err != nil {
		return err
	}

	addResultSeparator := false
	err = r.Visit(func(info *resource.Info, err error) error {
		// Make sure results are separated so the output can be piped to `kubectl -f -`
		if addResultSeparator {
			out.Write([]byte("-------------------------\n"))
		} else {
			addResultSeparator = true
		}

		// Create a Processor for talking to the Api Server
		proc, err := f.Processor(info.ResourceMapping())
		if err != nil {
			return err
		}

		// Send the Process request
		result, err := proc.Process(info.Namespace, info.Object)
		if err != nil {
			return err
		}

		// Format and print the result
		return printer.PrintObj(result, out)
	})
	return err
}
