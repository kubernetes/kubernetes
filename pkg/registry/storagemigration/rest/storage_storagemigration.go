/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/storagemigration"
	"k8s.io/kubernetes/pkg/features"

	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	storagemigrationstore "k8s.io/kubernetes/pkg/registry/storagemigration/storagemigration/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(storagemigration.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if storageMap, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[svmv1alpha1.SchemeGroupVersion.Version] = storageMap
	}

	return apiGroupInfo, nil
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	if resource := "storageversionmigrations"; apiResourceConfigSource.ResourceEnabled(svmv1alpha1.SchemeGroupVersion.WithResource(resource)) {
		if utilfeature.DefaultFeatureGate.Enabled(features.StorageVersionMigrator) {
			svm, svmStatus, err := storagemigrationstore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = svm
			storage[resource+"/status"] = svmStatus
		} else {
			klog.Warning("StorageVersionMigrator storage is disabled because the StorageVersionMigrator feature gate is disabled")
		}
	}

	return storage, nil
}

func (p RESTStorageProvider) GroupName() string {
	return storagemigration.GroupName
}
