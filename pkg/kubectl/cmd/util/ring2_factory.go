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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
)

type ring2Factory struct {
	parent Ring1Factory
}

func NewRing2Factory(parent Ring1Factory) Ring2Factory {
	f := &ring2Factory{
		parent: parent,
	}

	return f
}

func (f *ring2Factory) ring1Factory() Ring1Factory {
	return f.parent
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

	printer, err := f.ring1Factory().PrinterForMapping(cmd, mapping, false)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

func (f *ring2Factory) NewBuilder() *resource.Builder {
	mapper, typer := f.ring1Factory().Object()

	return resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ring1Factory().ClientForMapping), f.ring1Factory().ring0Factory().Decoder(true))
}
