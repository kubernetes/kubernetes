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
	storageapiv1 "k8s.io/api/storage/v1"
	storageapiv1alpha1 "k8s.io/api/storage/v1alpha1"
	storageapiv1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
	csidriverstore "k8s.io/kubernetes/pkg/registry/storage/csidriver/storage"
	csinodestore "k8s.io/kubernetes/pkg/registry/storage/csinode/storage"
	storageclassstore "k8s.io/kubernetes/pkg/registry/storage/storageclass/storage"
	volumeattachmentstore "k8s.io/kubernetes/pkg/registry/storage/volumeattachment/storage"
)

type RESTStorageProvider struct {
}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(storageapi.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(storageapiv1alpha1.SchemeGroupVersion) {
		if storageMap, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[storageapiv1alpha1.SchemeGroupVersion.Version] = storageMap
		}
	}
	if apiResourceConfigSource.VersionEnabled(storageapiv1beta1.SchemeGroupVersion) {
		if storageMap, err := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[storageapiv1beta1.SchemeGroupVersion.Version] = storageMap
		}
	}
	if apiResourceConfigSource.VersionEnabled(storageapiv1.SchemeGroupVersion) {
		if storageMap, err := p.v1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[storageapiv1.SchemeGroupVersion.Version] = storageMap
		}
	}

	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// volumeattachments
	volumeAttachmentStorage, err := volumeattachmentstore.NewStorage(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["volumeattachments"] = volumeAttachmentStorage.VolumeAttachment

	return storage, nil
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}
	// storageclasses
	storageClassStorage, err := storageclassstore.NewREST(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["storageclasses"] = storageClassStorage

	// volumeattachments
	volumeAttachmentStorage, err := volumeattachmentstore.NewStorage(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["volumeattachments"] = volumeAttachmentStorage.VolumeAttachment

	// register csinodes if CSINodeInfo feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) {
		csiNodeStorage, err := csinodestore.NewStorage(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["csinodes"] = csiNodeStorage.CSINode
	}

	// register csidrivers
	csiDriverStorage, err := csidriverstore.NewStorage(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["csidrivers"] = csiDriverStorage.CSIDriver

	return storage, nil
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storageClassStorage, err := storageclassstore.NewREST(restOptionsGetter)
	if err != nil {
		return nil, err
	}
	volumeAttachmentStorage, err := volumeattachmentstore.NewStorage(restOptionsGetter)
	if err != nil {
		return nil, err
	}

	storage := map[string]rest.Storage{
		// storageclasses
		"storageclasses": storageClassStorage,

		// volumeattachments
		"volumeattachments":        volumeAttachmentStorage.VolumeAttachment,
		"volumeattachments/status": volumeAttachmentStorage.Status,
	}

	// register csinodes if CSINodeInfo feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) {
		csiNodeStorage, err := csinodestore.NewStorage(restOptionsGetter)
		if err != nil {
			return nil, err
		}
		storage["csinodes"] = csiNodeStorage.CSINode
	}

	// register csidrivers
	csiDriverStorage, err := csidriverstore.NewStorage(restOptionsGetter)
	if err != nil {
		return storage, err
	}
	storage["csidrivers"] = csiDriverStorage.CSIDriver

	return storage, nil
}

func (p RESTStorageProvider) GroupName() string {
	return storageapi.GroupName
}
