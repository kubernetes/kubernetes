/*
Copyright 2016 The Kubernetes Authors.

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

// this file contains factories with no other dependencies

package util

import (
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	kubectlscheme "k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/printers"
)

type ring2Factory struct {
	clientAccessFactory  ClientAccessFactory
	objectMappingFactory ObjectMappingFactory
}

func NewBuilderFactory(clientAccessFactory ClientAccessFactory, objectMappingFactory ObjectMappingFactory) BuilderFactory {
	f := &ring2Factory{
		clientAccessFactory:  clientAccessFactory,
		objectMappingFactory: objectMappingFactory,
	}

	return f
}

func (f *ring2Factory) PrinterForOptions(options *printers.PrintOptions) (printers.ResourcePrinter, error) {
	return printerForOptions(options)
}

func (f *ring2Factory) PrinterForMapping(options *printers.PrintOptions) (printers.ResourcePrinter, error) {
	printer, err := f.PrinterForOptions(options)
	if err != nil {
		return nil, err
	}

	// wrap the printer in a versioning printer that understands when to convert and when not to convert
	printer = printers.NewVersionedPrinter(printer, legacyscheme.Scheme, legacyscheme.Scheme, kubectlscheme.Versions...)
	return printer, nil
}

func (f *ring2Factory) PrintSuccess(shortOutput bool, out io.Writer, resource, name string, dryRun bool, operation string) {
	dryRunMsg := ""
	if dryRun {
		dryRunMsg = " (dry run)"
	}
	if shortOutput {
		// -o name: prints resource/name
		if len(resource) > 0 {
			fmt.Fprintf(out, "%s/%s\n", resource, name)
		} else {
			fmt.Fprintf(out, "%s\n", name)
		}
	} else {
		// understandable output by default
		if len(resource) > 0 {
			fmt.Fprintf(out, "%s \"%s\" %s%s\n", resource, name, operation, dryRunMsg)
		} else {
			fmt.Fprintf(out, "\"%s\" %s%s\n", name, operation, dryRunMsg)
		}
	}
}

func (f *ring2Factory) PrintObject(cmd *cobra.Command, obj runtime.Object, out io.Writer) error {
	printer, err := f.PrinterForMapping(ExtractCmdPrintOptions(cmd, false))
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

func (f *ring2Factory) PrintResourceInfoForCommand(cmd *cobra.Command, info *resource.Info, out io.Writer) error {
	printOpts := ExtractCmdPrintOptions(cmd, false)
	printer, err := f.PrinterForOptions(printOpts)
	if err != nil {
		return err
	}
	if !printer.IsGeneric() {
		printer, err = f.PrinterForMapping(printOpts)
		if err != nil {
			return err
		}
	}
	return printer.PrintObj(info.Object, out)
}

// NewBuilder returns a new resource builder for structured api objects.
func (f *ring2Factory) NewBuilder() *resource.Builder {
	clientMapperFunc := resource.ClientMapperFunc(f.objectMappingFactory.ClientForMapping)
	mapper, typer := f.objectMappingFactory.Object()

	unstructuredClientMapperFunc := resource.ClientMapperFunc(f.objectMappingFactory.UnstructuredClientForMapping)

	categoryExpander := f.objectMappingFactory.CategoryExpander()

	return resource.NewBuilder(
		&resource.Mapper{
			RESTMapper:   mapper,
			ObjectTyper:  typer,
			ClientMapper: clientMapperFunc,
			Decoder:      f.clientAccessFactory.Decoder(true),
		},
		&resource.Mapper{
			RESTMapper:   mapper,
			ObjectTyper:  typer,
			ClientMapper: unstructuredClientMapperFunc,
			Decoder:      unstructured.UnstructuredJSONScheme,
		},
		categoryExpander,
	)
}

// PluginLoader loads plugins from a path set by the KUBECTL_PLUGINS_PATH env var.
// If this env var is not set, it defaults to
//   "~/.kube/plugins", plus
//  "./kubectl/plugins" directory under the "data dir" directory specified by the XDG
// system directory structure spec for the given platform.
func (f *ring2Factory) PluginLoader() plugins.PluginLoader {
	if len(os.Getenv("KUBECTL_PLUGINS_PATH")) > 0 {
		return plugins.KubectlPluginsPathPluginLoader()
	}
	return plugins.TolerantMultiPluginLoader{
		plugins.XDGDataDirsPluginLoader(),
		plugins.UserDirPluginLoader(),
	}
}

func (f *ring2Factory) PluginRunner() plugins.PluginRunner {
	return &plugins.ExecPluginRunner{}
}
