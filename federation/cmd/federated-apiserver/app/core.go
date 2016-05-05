/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/genericapiserver"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/v1"
	serviceetcd "k8s.io/kubernetes/pkg/registry/service/etcd"
)

func installCoreAPIs(s *genericapiserver.ServerRunOptions, g *genericapiserver.GenericAPIServer, f genericapiserver.StorageFactory) {
	serviceStore, serviceStatusStorage := serviceetcd.NewREST(createRESTOptionsOrDie(s, g, f, api.Resource("service")))
	coreResources := map[string]rest.Storage{
		"services":        serviceStore,
		"services/status": serviceStatusStorage,
	}
	coreGroupMeta := registered.GroupOrDie(api.GroupName)
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *coreGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			v1.SchemeGroupVersion.Version: coreResources,
		},
		OptionsExternalVersion: &registered.GroupOrDie(api.GroupName).GroupVersion,
		IsLegacyGroup:          true,
		Scheme:                 api.Scheme,
		ParameterCodec:         api.ParameterCodec,
		NegotiatedSerializer:   api.Codecs,
	}
	if err := g.InstallAPIGroup(&apiGroupInfo); err != nil {
		glog.Fatalf("Error in registering group versions: %v", err)
	}
}
