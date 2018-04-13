/*
Copyright 2017 The Kubernetes Authors.

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

package printers

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
)

// GetStandardPrinter takes a format type, an optional format argument. It will return
// a printer or an error. The printer is agnostic to schema versions, so you must
// send arguments to PrintObj in the version you wish them to be shown using a
// VersionedPrinter (typically when generic is true).
func GetStandardPrinter(typer runtime.ObjectTyper, encoder runtime.Encoder, decoders []runtime.Decoder, options PrintOptions) (ResourcePrinter, error) {
	format, formatArgument, allowMissingTemplateKeys := options.OutputFormatType, options.OutputFormatArgument, options.AllowMissingKeys

	var printer ResourcePrinter
	switch format {

	case "json", "yaml":
		jsonYamlFlags := NewJSONYamlPrintFlags()
		p, err := jsonYamlFlags.ToPrinter(format)
		if err != nil {
			return nil, err
		}

		printer = p

	case "name":
		nameFlags := NewNamePrintFlags("")
		namePrinter, err := nameFlags.ToPrinter(format)
		if err != nil {
			return nil, err
		}

		printer = namePrinter

	case "template", "go-template", "jsonpath", "templatefile", "go-template-file", "jsonpath-file":
		// TODO: construct and bind this separately (at the command level)
		kubeTemplateFlags := KubeTemplatePrintFlags{
			GoTemplatePrintFlags: &GoTemplatePrintFlags{
				AllowMissingKeys: &allowMissingTemplateKeys,
				TemplateArgument: &formatArgument,
			},
			JSONPathPrintFlags: &JSONPathPrintFlags{
				AllowMissingKeys: &allowMissingTemplateKeys,
				TemplateArgument: &formatArgument,
			},
		}

		kubeTemplatePrinter, err := kubeTemplateFlags.ToPrinter(format)
		if err != nil {
			return nil, err
		}

		printer = kubeTemplatePrinter

	case "custom-columns", "custom-columns-file":
		customColumnsFlags := &CustomColumnsPrintFlags{
			NoHeaders:        options.NoHeaders,
			TemplateArgument: formatArgument,
		}
		customColumnsPrinter, matched, err := customColumnsFlags.ToPrinter(format)
		if !matched {
			return nil, fmt.Errorf("unable to match a name printer to handle current print options")
		}
		if err != nil {
			return nil, err
		}

		printer = customColumnsPrinter

	case "wide":
		fallthrough
	case "":
		humanPrintFlags := NewHumanPrintFlags(options.Kind, options.NoHeaders, options.WithNamespace, options.AbsoluteTimestamps)

		// TODO: these should be bound through a call to humanPrintFlags#AddFlags(cmd) once we instantiate PrintFlags at the command level
		humanPrintFlags.ShowKind = &options.WithKind
		humanPrintFlags.ShowLabels = &options.ShowLabels
		humanPrintFlags.ColumnLabels = &options.ColumnLabels
		humanPrintFlags.SortBy = &options.SortBy

		humanPrinter, matches, err := humanPrintFlags.ToPrinter(format)
		if !matches {
			return nil, fmt.Errorf("unable to match a printer to handle current print options")
		}
		if err != nil {
			return nil, err
		}

		printer = humanPrinter

	default:
		return nil, fmt.Errorf("output format %q not recognized", format)
	}
	return printer, nil
}
