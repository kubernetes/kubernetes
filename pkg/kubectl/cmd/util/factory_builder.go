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
	"os"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
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
			Decoder:      InternalVersionDecoder(),
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
