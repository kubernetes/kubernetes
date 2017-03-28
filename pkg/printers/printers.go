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
	"io/ioutil"
	"os"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// GetStandardPrinter takes a format type, an optional format argument. It will return true
// if the format is generic (untyped), otherwise it will return false. The printer
// is agnostic to schema versions, so you must send arguments to PrintObj in the
// version you wish them to be shown using a VersionedPrinter (typically when
// generic is true).
func GetStandardPrinter(format, formatArgument string, noHeaders, allowMissingTemplateKeys bool, mapper meta.RESTMapper, typer runtime.ObjectTyper, decoders []runtime.Decoder) (ResourcePrinter, bool, error) {
	var printer ResourcePrinter
	switch format {
	case "json":
		printer = &JSONPrinter{}
	case "yaml":
		printer = &YAMLPrinter{}
	case "name":
		printer = &NamePrinter{
			Typer:    typer,
			Decoders: decoders,
			Mapper:   mapper,
		}
	case "template", "go-template":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("template format specified but no template given")
		}
		templatePrinter, err := NewTemplatePrinter([]byte(formatArgument))
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", formatArgument, err)
		}
		templatePrinter.AllowMissingKeys(allowMissingTemplateKeys)
		printer = templatePrinter
	case "templatefile", "go-template-file":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("templatefile format specified but no template file given")
		}
		data, err := ioutil.ReadFile(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		templatePrinter, err := NewTemplatePrinter(data)
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", string(data), err)
		}
		templatePrinter.AllowMissingKeys(allowMissingTemplateKeys)
		printer = templatePrinter
	case "jsonpath":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("jsonpath template format specified but no template given")
		}
		jsonpathPrinter, err := NewJSONPathPrinter(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error parsing jsonpath %s, %v\n", formatArgument, err)
		}
		jsonpathPrinter.AllowMissingKeys(allowMissingTemplateKeys)
		printer = jsonpathPrinter
	case "jsonpath-file":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("jsonpath file format specified but no template file file given")
		}
		data, err := ioutil.ReadFile(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		jsonpathPrinter, err := NewJSONPathPrinter(string(data))
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", string(data), err)
		}
		jsonpathPrinter.AllowMissingKeys(allowMissingTemplateKeys)
		printer = jsonpathPrinter
	case "custom-columns":
		var err error
		if printer, err = NewCustomColumnsPrinterFromSpec(formatArgument, decoders[0], noHeaders); err != nil {
			return nil, false, err
		}
	case "custom-columns-file":
		file, err := os.Open(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		defer file.Close()
		if printer, err = NewCustomColumnsPrinterFromTemplate(file, decoders[0]); err != nil {
			return nil, false, err
		}
	case "wide":
		fallthrough
	case "":
		return nil, false, nil
	default:
		return nil, false, fmt.Errorf("output format %q not recognized", format)
	}
	return printer, true, nil
}
