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
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	jobstorage "k8s.io/kubernetes/pkg/registry/batch/job/storage"
)

func installBatchAPIs(g *genericapiserver.GenericAPIServer, optsGetter generic.RESTOptionsGetter, apiResourceConfigSource storage.APIResourceConfigSource) {
	jobsStorageFn := func() map[string]rest.Storage {
		jobStorage := jobstorage.NewStorage(optsGetter)
		return map[string]rest.Storage{
			"jobs":        jobStorage.Job,
			"jobs/status": jobStorage.Status,
		}
	}
	resourcesStorageMap := map[string]getResourcesStorageFunc{
		"jobs": jobsStorageFn,
	}
	shouldInstallGroup, resources := enabledResources(batchv1.SchemeGroupVersion, resourcesStorageMap, apiResourceConfigSource)
	if !shouldInstallGroup {
		return
	}
	batchGroupMeta := api.Registry.GroupOrDie(batch.GroupName)
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *batchGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			"v1": resources,
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
