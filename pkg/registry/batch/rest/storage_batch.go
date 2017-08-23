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

package rest

import (
	batchapiv1 "k8s.io/api/batch/v1"
	batchapiv1beta1 "k8s.io/api/batch/v1beta1"
	batchapiv2alpha1 "k8s.io/api/batch/v2alpha1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	cronjobstore "k8s.io/kubernetes/pkg/registry/batch/cronjob/storage"
	jobstore "k8s.io/kubernetes/pkg/registry/batch/job/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(batch.GroupName, api.Registry, api.Scheme, api.ParameterCodec, api.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv1.SchemeGroupVersion.Version] = p.v1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv1.SchemeGroupVersion
	}
	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv1beta1.SchemeGroupVersion
	}
	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv2alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv2alpha1.SchemeGroupVersion.Version] = p.v2alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv2alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := batchapiv1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("jobs")) {
		jobsStorage, jobsStatusStorage := jobstore.NewREST(restOptionsGetter)
		storage["jobs"] = jobsStorage
		storage["jobs/status"] = jobsStatusStorage
	}
	return storage
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := batchapiv1beta1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("cronjobs")) {
		cronJobsStorage, cronJobsStatusStorage := cronjobstore.NewREST(restOptionsGetter)
		storage["cronjobs"] = cronJobsStorage
		storage["cronjobs/status"] = cronJobsStatusStorage
	}
	return storage
}

func (p RESTStorageProvider) v2alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	version := batchapiv2alpha1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("cronjobs")) {
		cronJobsStorage, cronJobsStatusStorage := cronjobstore.NewREST(restOptionsGetter)
		storage["cronjobs"] = cronJobsStorage
		storage["cronjobs/status"] = cronJobsStatusStorage
	}
	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return batch.GroupName
}
