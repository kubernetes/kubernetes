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
	"k8s.io/client-go/dynamic"
	scaleclient "k8s.io/client-go/scale"
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

func (f *ring2Factory) ScaleClient() (scaleclient.ScalesGetter, error) {
	discoClient, err := f.clientAccessFactory.ToDiscoveryClient()
	if err != nil {
		return nil, err
	}
	restClient, err := f.clientAccessFactory.RESTClient()
	if err != nil {
		return nil, err
	}
	resolver := scaleclient.NewDiscoveryScaleKindResolver(discoClient)
	mapper, err := f.clientAccessFactory.ToRESTMapper()
	if err != nil {
		return nil, err
	}

	return scaleclient.New(restClient, mapper, dynamic.LegacyAPIPathResolverFunc, resolver), nil
}
