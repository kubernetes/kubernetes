/*
Copyright 2022 The Kubernetes Authors.

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
	resourcev1alpha3 "k8s.io/api/resource/v1alpha3"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
	deviceclassstore "k8s.io/kubernetes/pkg/registry/resource/deviceclass/storage"
	resourceclaimstore "k8s.io/kubernetes/pkg/registry/resource/resourceclaim/storage"
	resourceclaimtemplatestore "k8s.io/kubernetes/pkg/registry/resource/resourceclaimtemplate/storage"
	resourceslicestore "k8s.io/kubernetes/pkg/registry/resource/resourceslice/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(resource.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if storageMap, err := p.v1alpha3Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[resourcev1alpha3.SchemeGroupVersion.Version] = storageMap
	}

	if storageMap, err := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[resourcev1beta1.SchemeGroupVersion.Version] = storageMap
	}

	return apiGroupInfo, nil
}

func (p RESTStorageProvider) v1alpha3Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	if resource, gvr, feature := "deviceclasses", resourcev1alpha3.SchemeGroupVersion.WithResource("deviceClass"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			deviceclassStorage, err := deviceclassstore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = deviceclassStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	if resource, gvr, feature := "resourceclaims", resourcev1alpha3.SchemeGroupVersion.WithResource("resourceclaims"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			resourceClaimStorage, resourceClaimStatusStorage, err := resourceclaimstore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = resourceClaimStorage
			storage[resource+"/status"] = resourceClaimStatusStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	if resource, gvr, feature := "resourceclaimtemplates", resourcev1alpha3.SchemeGroupVersion.WithResource("resourceclaimtemplates"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			resourceClaimTemplateStorage, err := resourceclaimtemplatestore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = resourceClaimTemplateStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	if resource, gvr, feature := "resourceslices", resourcev1alpha3.SchemeGroupVersion.WithResource("resourceslices"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			resourceSliceStorage, err := resourceslicestore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = resourceSliceStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	return storage, nil
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	if resource, gvr, feature := "deviceclasses", resourcev1beta1.SchemeGroupVersion.WithResource("deviceclasses"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			deviceclassStorage, err := deviceclassstore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = deviceclassStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	if resource, gvr, feature := "resourceclaims", resourcev1beta1.SchemeGroupVersion.WithResource("resourceclaims"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			resourceClaimStorage, resourceClaimStatusStorage, err := resourceclaimstore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = resourceClaimStorage
			storage[resource+"/status"] = resourceClaimStatusStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	if resource, gvr, feature := "resourceclaimtemplates", resourcev1beta1.SchemeGroupVersion.WithResource("resourceclaimtemplates"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			resourceClaimTemplateStorage, err := resourceclaimtemplatestore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = resourceClaimTemplateStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	if resource, gvr, feature := "resourceslices", resourcev1beta1.SchemeGroupVersion.WithResource("resourceslices"), features.DynamicResourceAllocation; apiResourceConfigSource.ResourceEnabled(gvr) {
		if utilfeature.DefaultFeatureGate.Enabled(feature) {
			resourceSliceStorage, err := resourceslicestore.NewREST(restOptionsGetter)
			if err != nil {
				return nil, err
			}
			storage[resource] = resourceSliceStorage
		} else {
			klog.Warningf("%s is disabled because the %s feature is disabled", gvr, feature)
		}
	}

	return storage, nil
}

func (p RESTStorageProvider) GroupName() string {
	return resource.GroupName
}
