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

package apiserver

import (
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/version"

	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1"
	apiservicestorage "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/registry/apiservice"
)

// TODO move to genericapiserver or something like that
// RESTOptionsGetter is used to construct storage for a particular resource
type RESTOptionsGetter interface {
	NewFor(resource schema.GroupResource) generic.RESTOptions
}

type Config struct {
	GenericConfig *genericapiserver.Config

	// RESTOptionsGetter is used to construct storage for a particular resource
	RESTOptionsGetter RESTOptionsGetter
}

// APIDiscoveryServer contains state for a Kubernetes cluster master/api server.
type APIDiscoveryServer struct {
	GenericAPIServer *genericapiserver.GenericAPIServer
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	c.GenericConfig.Complete()

	version := version.Get()
	c.GenericConfig.Version = &version

	return completedConfig{c}
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// New returns a new instance of APIDiscoveryServer from the given config.
func (c completedConfig) New() (*APIDiscoveryServer, error) {
	genericServer, err := c.Config.GenericConfig.SkipComplete().New() // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	s := &APIDiscoveryServer{
		GenericAPIServer: genericServer,
	}

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiregistration.GroupName)
	apiGroupInfo.GroupMeta.GroupVersion = v1alpha1.SchemeGroupVersion

	v1alpha1storage := map[string]rest.Storage{}
	v1alpha1storage["apiservices"] = apiservicestorage.NewREST(c.RESTOptionsGetter.NewFor(apiregistration.Resource("apiservices")))

	apiGroupInfo.VersionedResourcesStorageMap["v1alpha1"] = v1alpha1storage

	if err := s.GenericAPIServer.InstallAPIGroup(&apiGroupInfo); err != nil {
		return nil, err
	}

	return s, nil
}
