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
	extensionsapiv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/extensions"
	daemonstore "k8s.io/kubernetes/pkg/registry/apps/daemonset/storage"
	deploymentstore "k8s.io/kubernetes/pkg/registry/apps/deployment/storage"
	replicasetstore "k8s.io/kubernetes/pkg/registry/apps/replicaset/storage"
	expcontrollerstore "k8s.io/kubernetes/pkg/registry/extensions/controller/storage"
	ingressstore "k8s.io/kubernetes/pkg/registry/networking/ingress/storage"
	networkpolicystore "k8s.io/kubernetes/pkg/registry/networking/networkpolicy/storage"
	pspstore "k8s.io/kubernetes/pkg/registry/policy/podsecuritypolicy/storage"
)

type RESTStorageProvider struct{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(extensions.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(extensionsapiv1beta1.SchemeGroupVersion) {
		if storageMap, err := p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		} else {
			apiGroupInfo.VersionedResourcesStorageMap[extensionsapiv1beta1.SchemeGroupVersion.Version] = storageMap
		}
	}

	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	// This is a dummy replication controller for scale subresource purposes.
	// TODO: figure out how to enable this only if needed as a part of scale subresource GA.
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicationcontrollers")) {
		controllerStorage, err := expcontrollerstore.NewStorage(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["replicationcontrollers"] = controllerStorage.ReplicationController
		storage["replicationcontrollers/scale"] = controllerStorage.Scale
	}

	// daemonsets
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("daemonsets")) {
		daemonSetStorage, daemonSetStatusStorage, err := daemonstore.NewREST(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["daemonsets"] = daemonSetStorage.WithCategories(nil)
		storage["daemonsets/status"] = daemonSetStatusStorage
	}

	//deployments
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("deployments")) {
		deploymentStorage, err := deploymentstore.NewStorage(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["deployments"] = deploymentStorage.Deployment.WithCategories(nil)
		storage["deployments/status"] = deploymentStorage.Status
		storage["deployments/rollback"] = deploymentStorage.Rollback
		storage["deployments/scale"] = deploymentStorage.Scale
	}
	// ingresses
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses")) {
		ingressStorage, ingressStatusStorage, err := ingressstore.NewREST(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["ingresses"] = ingressStorage
		storage["ingresses/status"] = ingressStatusStorage
	}

	// podsecuritypolicy
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("podsecuritypolicies")) {
		podSecurityPolicyStorage, err := pspstore.NewREST(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["podSecurityPolicies"] = podSecurityPolicyStorage
	}

	// replicasets
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicasets")) {
		replicaSetStorage, err := replicasetstore.NewStorage(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["replicasets"] = replicaSetStorage.ReplicaSet.WithCategories(nil)
		storage["replicasets/status"] = replicaSetStorage.Status
		storage["replicasets/scale"] = replicaSetStorage.Scale
	}

	// networkpolicies
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("networkpolicies")) {
		networkExtensionsStorage, err := networkpolicystore.NewREST(restOptionsGetter)
		if err != nil {
			return storage, err
		}
		storage["networkpolicies"] = networkExtensionsStorage
	}

	return storage, nil
}

func (p RESTStorageProvider) GroupName() string {
	return extensions.GroupName
}
