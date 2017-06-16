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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/util/i18n"

	"github.com/spf13/cobra"
)

var (
	convert_long = templates.LongDesc(i18n.T(`
		Convert config files between different API versions. Both YAML
		and JSON formats are accepted.

		The command takes filename, directory, or URL as input, and convert it into format
		of version specified by --output-version flag. If target version is not specified or
		not supported, convert to latest version.

		The default output will be printed to stdout in YAML format. One can use -o option
		to change to output destination.`))

	convert_example = templates.Examples(i18n.T(`
		# Convert 'pod.yaml' to latest version and print to stdout.
		kubectl convert -f pod.yaml

		# Convert the live state of the resource specified by 'pod.yaml' to the latest version
		# and print to stdout in json format.
		kubectl convert -f pod.yaml --local -o json

		# Convert all files under current directory to latest version and create them all.
		kubectl convert -f . | kubectl create -f -`))
)

// NewCmdConvert creates a command object for the generic "convert" action, which
// translates the config file into a given version.
func NewCmdConvert(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ConvertOptions{}

	cmd := &cobra.Command{
		Use:     "convert -f FILENAME",
		Short:   i18n.T("Convert config files between different API versions"),
		Long:    convert_long,
		Example: convert_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := options.Complete(f, out, cmd)
			cmdutil.CheckErr(err)
			err = options.RunConvert()
			cmdutil.CheckErr(err)
		},
	}

	usage := "to need to get converted."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.MarkFlagRequired("filename")
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddNonDeprecatedPrinterFlags(cmd)
	cmd.Flags().BoolVar(&options.local, "local", true, "If true, convert will NOT try to contact api-server but run locally.")
	cmd.Flags().String("output-version", "", i18n.T("Output the formatted object with the given group version (for ex: 'extensions/v1beta1').)"))
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

// ConvertOptions have the data required to perform the convert operation
type ConvertOptions struct {
	resource.FilenameOptions

	builder *resource.Builder
	local   bool

	encoder runtime.Encoder
	out     io.Writer
	printer printers.ResourcePrinter

	outputVersion schema.GroupVersion
}

// outputVersion returns the preferred output version for generic content (JSON, YAML, or templates)
// defaultVersion is never mutated.  Nil simply allows clean passing in common usage from client.Config
func outputVersion(cmd *cobra.Command, defaultVersion *schema.GroupVersion) (schema.GroupVersion, error) {
	outputVersionString := cmdutil.GetFlagString(cmd, "output-version")
	if len(outputVersionString) == 0 {
		if defaultVersion == nil {
			return schema.GroupVersion{}, nil
		}

		return *defaultVersion, nil
	}

	return schema.ParseGroupVersion(outputVersionString)
}

// Complete collects information required to run Convert command from command line.
func (o *ConvertOptions) Complete(f cmdutil.Factory, out io.Writer, cmd *cobra.Command) (err error) {
	o.outputVersion, err = outputVersion(cmd, &api.Registry.EnabledVersionsForGroup(api.GroupName)[0])
	if err != nil {
		return err
	}
	if !api.Registry.IsEnabledVersion(o.outputVersion) {
		cmdutil.UsageError(cmd, "'%s' is not a registered version.", o.outputVersion)
	}

	// build the builder
	o.builder = f.NewBuilder(!o.local)
	if !o.local {
		schema, err := f.Validator(cmdutil.GetFlagBool(cmd, "validate"), cmdutil.GetFlagString(cmd, "schema-cache-dir"))
		if err != nil {
			return err
		}
		o.builder = o.builder.Schema(schema)
	}
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	o.builder = o.builder.NamespaceParam(cmdNamespace).
		ContinueOnError().
		FilenameParam(false, &o.FilenameOptions).
		Flatten()

	// build the printer
	o.out = out
	outputFormat := cmdutil.GetFlagString(cmd, "output")
	templateFile := cmdutil.GetFlagString(cmd, "template")
	if len(outputFormat) == 0 {
		if len(templateFile) == 0 {
			outputFormat = "yaml"
		} else {
			outputFormat = "template"
		}
		// TODO: once printing is abstracted, this should be handled at flag declaration time
		cmd.Flags().Set("output", outputFormat)
	}
	o.encoder = f.JSONEncoder()
	o.printer, err = f.PrinterForCommand(cmd, o.local, nil, printers.PrintOptions{})
	if err != nil {
		return err
	}

	return nil
}

// RunConvert implements the generic Convert command
func (o *ConvertOptions) RunConvert() error {
	r := o.builder.Do()
	err := r.Err()
	if err != nil {
		return err
	}

	singleItemImplied := false
	infos, err := r.IntoSingleItemImplied(&singleItemImplied).Infos()
	if err != nil {
		return err
	}

	if len(infos) == 0 {
		return fmt.Errorf("no objects passed to convert")
	}

	objects, err := resource.AsVersionedObject(infos, !singleItemImplied, o.outputVersion, o.encoder)
	if err != nil {
		return err
	}

	if meta.IsListType(objects) {
		_, items, err := cmdutil.FilterResourceList(objects, nil, nil)
		if err != nil {
			return err
		}
		filteredObj, err := cmdutil.ObjectListToVersionedObject(items, o.outputVersion)
		if err != nil {
			return err
		}
		return o.printer.PrintObj(filteredObj, o.out)
	}

	return o.printer.PrintObj(objects, o.out)
}
