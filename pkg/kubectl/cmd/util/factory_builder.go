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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
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

func (f *ring2Factory) PrinterForCommand(cmd *cobra.Command, isLocal bool, outputOpts *printers.OutputOptions, options printers.PrintOptions) (printers.ResourcePrinter, error) {
	var mapper meta.RESTMapper
	var typer runtime.ObjectTyper
	var err error

	if isLocal {
		mapper = api.Registry.RESTMapper()
		typer = api.Scheme
	} else {
		mapper, typer, err = f.objectMappingFactory.UnstructuredObject()
		if err != nil {
			return nil, err
		}
	}
	// TODO: used by the custom column implementation and the name implementation, break this dependency
	decoders := []runtime.Decoder{f.clientAccessFactory.Decoder(true), unstructured.UnstructuredJSONScheme}
	encoder := f.clientAccessFactory.JSONEncoder()
	return PrinterForCommand(cmd, outputOpts, mapper, typer, encoder, decoders, options)
}

func (f *ring2Factory) PrinterForMapping(cmd *cobra.Command, isLocal bool, outputOpts *printers.OutputOptions, mapping *meta.RESTMapping, withNamespace bool) (printers.ResourcePrinter, error) {
	// Some callers do not have "label-columns" so we can't use the GetFlagStringSlice() helper
	columnLabel, err := cmd.Flags().GetStringSlice("label-columns")
	if err != nil {
		columnLabel = []string{}
	}

	options := printers.PrintOptions{
		NoHeaders:          GetFlagBool(cmd, "no-headers"),
		WithNamespace:      withNamespace,
		Wide:               GetWideFlag(cmd),
		ShowAll:            GetFlagBool(cmd, "show-all"),
		ShowLabels:         GetFlagBool(cmd, "show-labels"),
		AbsoluteTimestamps: isWatch(cmd),
		ColumnLabels:       columnLabel,
	}

	printer, err := f.PrinterForCommand(cmd, isLocal, outputOpts, options)
	if err != nil {
		return nil, err
	}

	// Make sure we output versioned data for generic printers
	if printer.IsGeneric() {
		if mapping == nil {
			return nil, fmt.Errorf("no serialization format found")
		}
		version := mapping.GroupVersionKind.GroupVersion()
		if version.Empty() {
			return nil, fmt.Errorf("no serialization format found")
		}

		printer = printers.NewVersionedPrinter(printer, mapping.ObjectConvertor, version, mapping.GroupVersionKind.GroupVersion())

	} else {
		// We add handlers to the printer in case it is printers.HumanReadablePrinter.
		// printers.AddHandlers expects concrete type of printers.HumanReadablePrinter
		// as its parameter because of this we have to do a type check on printer and
		// extract out concrete HumanReadablePrinter from it. We are then able to attach
		// handlers on it.
		if humanReadablePrinter, ok := printer.(*printers.HumanReadablePrinter); ok {
			printersinternal.AddHandlers(humanReadablePrinter)
			printer = humanReadablePrinter
		}
	}

	return printer, nil
}

func (f *ring2Factory) PrintObject(cmd *cobra.Command, isLocal bool, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
	// try to get a typed object
	_, typer := f.objectMappingFactory.Object()
	gvks, _, err := typer.ObjectKinds(obj)

	// fall back to an unstructured object if we get something unregistered
	if runtime.IsNotRegisteredError(err) {
		_, typer, unstructuredErr := f.objectMappingFactory.UnstructuredObject()
		if unstructuredErr != nil {
			// if we can't get an unstructured typer, return the original error
			return err
		}
		gvks, _, err = typer.ObjectKinds(obj)
	}

	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(gvks[0].GroupKind())
	if err != nil {
		return err
	}

	printer, err := f.PrinterForMapping(cmd, isLocal, nil, mapping, false)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

// NewBuilder returns a new resource builder.
// Receives a bool flag and avoids remote calls if set to false
func (f *ring2Factory) NewBuilder(allowRemoteCalls bool) *resource.Builder {
	var clientMapper resource.ClientMapper
	clientMapperFunc := resource.ClientMapperFunc(f.objectMappingFactory.ClientForMapping)

	mapper, typer := f.objectMappingFactory.Object()
	categoryExpander := f.objectMappingFactory.CategoryExpander()

	if allowRemoteCalls {
		clientMapper = clientMapperFunc
	} else {
		clientMapper = resource.DisabledClientForMapping{ClientMapper: clientMapperFunc}
	}

	return resource.NewBuilder(mapper, categoryExpander, typer, clientMapper, f.clientAccessFactory.Decoder(true))
}

func (f *ring2Factory) NewUnstructuredBuilder(allowRemoteCalls bool) (*resource.Builder, error) {
	if !allowRemoteCalls {
		return f.NewBuilder(allowRemoteCalls), nil
	}

	clientMapperFunc := resource.ClientMapperFunc(f.objectMappingFactory.UnstructuredClientForMapping)

	mapper, typer, err := f.objectMappingFactory.UnstructuredObject()
	if err != nil {
		return nil, err
	}

	categoryExpander := f.objectMappingFactory.CategoryExpander()
	return resource.NewBuilder(mapper, categoryExpander, typer, clientMapperFunc, unstructured.UnstructuredJSONScheme), nil

}

// PluginLoader loads plugins from a path set by the KUBECTL_PLUGINS_PATH env var.
// If this env var is not set, it defaults to
//   "~/.kube/plugins", plus
//  "./kubectl/plugins" directory under the "data dir" directory specified by the XDG
// system directory structure spec for the given platform.
func (f *ring2Factory) PluginLoader() plugins.PluginLoader {
	if len(os.Getenv("KUBECTL_PLUGINS_PATH")) > 0 {
		return plugins.PluginsEnvVarPluginLoader()
	}
	return plugins.TolerantMultiPluginLoader{
		plugins.XDGDataPluginLoader(),
		plugins.UserDirPluginLoader(),
	}
}

func (f *ring2Factory) PluginRunner() plugins.PluginRunner {
	return &plugins.ExecPluginRunner{}
}
