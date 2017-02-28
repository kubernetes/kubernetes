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

package app

import (
	"github.com/golang/glog"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	restStorage "k8s.io/kubernetes/pkg/registry/extensions/rest"
)

func installExtensionsAPIs(g *genericapiserver.GenericAPIServer, optsGetter generic.RESTOptionsGetter, apiResourceConfigSource storage.APIResourceConfigSource) {
	groupName := extensions.GroupName
	if !apiResourceConfigSource.AnyResourcesForGroupEnabled(groupName) {
		glog.V(1).Infof("Skipping disabled API group %q", groupName)
		return
	}
	resources := map[string]rest.Storage{}
	restStorage.RESTStorageProvider{}.AddReplicaSetsStorage(apiResourceConfigSource, optsGetter, resources)
	restStorage.RESTStorageProvider{}.AddIngressesStorage(apiResourceConfigSource, optsGetter, resources)
	restStorage.RESTStorageProvider{}.AddDaemonSetsStorage(apiResourceConfigSource, optsGetter, resources)
	restStorage.RESTStorageProvider{}.AddDeploymentsStorage(apiResourceConfigSource, optsGetter, resources)
	if len(resources) == 0 {
		glog.V(1).Infof("Skipping API group %q since there is no enabled resource", groupName)
		return
	}

	extensionsGroupMeta := api.Registry.GroupOrDie(extensions.GroupName)
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *extensionsGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			"v1beta1": resources,
		},
		OptionsExternalVersion: &api.Registry.GroupOrDie(api.GroupName).GroupVersion,
		Scheme:                 api.Scheme,
		ParameterCodec:         api.ParameterCodec,
		NegotiatedSerializer:   api.Codecs,
	}
	if err := g.InstallAPIGroup(&apiGroupInfo); err != nil {
		glog.Fatalf("Error in registering group versions: %v", err)
	}
}
