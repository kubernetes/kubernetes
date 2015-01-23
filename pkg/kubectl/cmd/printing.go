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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/spf13/cobra"
)

func AddPrinterFlags(cmd *cobra.Command) {
	cmd.Flags().StringP("output", "o", "", "Output format: json|yaml|template|templatefile")
	cmd.Flags().String("output-version", "", "Output the formatted object with the given version (default api-version)")
	cmd.Flags().Bool("no-headers", false, "When using the default output, don't print headers")
	cmd.Flags().StringP("template", "t", "", "Template string or path to template file to use when -o=template or -o=templatefile.")
}

// PrintObject prints an api object given command line flags to modify the output format
func PrintObject(cmd *cobra.Command, obj runtime.Object, f *Factory, out io.Writer) error {
	mapper, _ := f.Object(cmd)
	_, kind, err := api.Scheme.ObjectVersionAndKind(obj)
	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(kind)
	if err != nil {
		return err
	}

	printer, err := PrinterForMapping(f, cmd, mapping)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

// outputVersion returns the preferred output version for generic content (JSON, YAML, or templates)
func outputVersion(cmd *cobra.Command, defaultVersion string) string {
	outputVersion := GetFlagString(cmd, "output-version")
	if len(outputVersion) == 0 {
		outputVersion = defaultVersion
	}
	return outputVersion
}

// PrinterForCommand returns the default printer for this command.
// Requires that printer flags have been added to cmd (see AddPrinterFlags).
func PrinterForCommand(cmd *cobra.Command) (kubectl.ResourcePrinter, bool, error) {
	outputFormat := GetFlagString(cmd, "output")
	templateFile := GetFlagString(cmd, "template")
	if len(outputFormat) == 0 && len(templateFile) != 0 {
		outputFormat = "template"
	}

	return kubectl.GetPrinter(outputFormat, templateFile)
}

// PrinterForMapping returns a printer suitable for displaying the provided resource type.
// Requires that printer flags have been added to cmd (see AddPrinterFlags).
func PrinterForMapping(f *Factory, cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.ResourcePrinter, error) {
	printer, ok, err := PrinterForCommand(cmd)
	if err != nil {
		return nil, err
	}
	if ok {
		clientConfig, err := f.ClientConfig(cmd)
		checkErr(err)
		defaultVersion := clientConfig.Version

		version := outputVersion(cmd, defaultVersion)
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
