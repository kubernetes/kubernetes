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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/resource"
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

func (f *ring2Factory) PrintObject(cmd *cobra.Command, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error {
	gvks, _, err := api.Scheme.ObjectKinds(obj)
	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(gvks[0].GroupKind())
	if err != nil {
		return err
	}

	printer, err := f.objectMappingFactory.PrinterForMapping(cmd, mapping, false)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

func (f *ring2Factory) NewBuilder() *resource.Builder {
	mapper, typer := f.objectMappingFactory.Object()

	return resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.objectMappingFactory.ClientForMapping), f.clientAccessFactory.Decoder(true))
}
