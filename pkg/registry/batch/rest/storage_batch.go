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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchapiv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/genericapiserver"
	cronjobetcd "k8s.io/kubernetes/pkg/registry/batch/cronjob/etcd"
	jobetcd "k8s.io/kubernetes/pkg/registry/batch/job/etcd"
)

type RESTStorageProvider struct{}

var _ genericapiserver.RESTStorageProvider = &RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(batch.GroupName)

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv2alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv2alpha1.SchemeGroupVersion.Version] = p.v2alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv2alpha1.SchemeGroupVersion
		apiGroupInfo.SubresourceGroupVersionKind = map[string]unversioned.GroupVersionKind{
			"scheduledjobs":        batchapiv2alpha1.SchemeGroupVersion.WithKind("ScheduledJob"),
			"scheduledjobs/status": batchapiv2alpha1.SchemeGroupVersion.WithKind("ScheduledJob"),
		}
	}
	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv1.SchemeGroupVersion.Version] = p.v1Storage(apiResourceConfigSource, restOptionsGetter)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) map[string]rest.Storage {
	version := batchapiv1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("jobs")) {
		jobsStorage, jobsStatusStorage := jobetcd.NewREST(restOptionsGetter(batch.Resource("jobs")))
		storage["jobs"] = jobsStorage
		storage["jobs/status"] = jobsStatusStorage
	}
	return storage
}

func (p RESTStorageProvider) v2alpha1Storage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter) map[string]rest.Storage {
	version := batchapiv2alpha1.SchemeGroupVersion

	storage := map[string]rest.Storage{}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("jobs")) {
		jobsStorage, jobsStatusStorage := jobetcd.NewREST(restOptionsGetter(batch.Resource("jobs")))
		storage["jobs"] = jobsStorage
		storage["jobs/status"] = jobsStatusStorage
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("cronjobs")) {
		cronJobsStorage, cronJobsStatusStorage := cronjobetcd.NewREST(restOptionsGetter(batch.Resource("cronjobs")))
		storage["cronjobs"] = cronJobsStorage
		storage["cronjobs/status"] = cronJobsStatusStorage
		storage["scheduledjobs"] = cronJobsStorage
		storage["scheduledjobs/status"] = cronJobsStatusStorage
	}
	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return batch.GroupName
}
