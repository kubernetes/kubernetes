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

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(extensions.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
	// If you add a version here, be sure to add an entry in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go with specific priorities.
	// TODO refactor the plumbing to provide the information in the APIGroupInfo

	if apiResourceConfigSource.VersionEnabled(extensionsapiv1beta1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[extensionsapiv1beta1.SchemeGroupVersion.Version] = p.v1beta1Storage(apiResourceConfigSource, restOptionsGetter)
	}

	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1beta1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	storage := map[string]rest.Storage{}

	// This is a dummy replication controller for scale subresource purposes.
	// TODO: figure out how to enable this only if needed as a part of scale subresource GA.
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicationcontrollers")) {
		controllerStorage := expcontrollerstore.NewStorage(restOptionsGetter)
		storage["replicationcontrollers"] = controllerStorage.ReplicationController
		storage["replicationcontrollers/scale"] = controllerStorage.Scale
	}

	// daemonsets
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("daemonsets")) {
		daemonSetStorage, daemonSetStatusStorage := daemonstore.NewREST(restOptionsGetter)
		storage["daemonsets"] = daemonSetStorage.WithCategories(nil)
		storage["daemonsets/status"] = daemonSetStatusStorage
	}

	//deployments
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("deployments")) {
		deploymentStorage := deploymentstore.NewStorage(restOptionsGetter)
		storage["deployments"] = deploymentStorage.Deployment.WithCategories(nil)
		storage["deployments/status"] = deploymentStorage.Status
		storage["deployments/rollback"] = deploymentStorage.Rollback
		storage["deployments/scale"] = deploymentStorage.Scale
	}
	// ingresses
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses")) {
		ingressStorage, ingressStatusStorage := ingressstore.NewREST(restOptionsGetter)
		storage["ingresses"] = ingressStorage
		storage["ingresses/status"] = ingressStatusStorage
	}

	// podsecuritypolicy
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("podsecuritypolicies")) {
		podSecurityPolicyStorage := pspstore.NewREST(restOptionsGetter)
		storage["podSecurityPolicies"] = podSecurityPolicyStorage
	}

	// replicasets
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicasets")) {
		replicaSetStorage := replicasetstore.NewStorage(restOptionsGetter)
		storage["replicasets"] = replicaSetStorage.ReplicaSet.WithCategories(nil)
		storage["replicasets/status"] = replicaSetStorage.Status
		storage["replicasets/scale"] = replicaSetStorage.Scale
	}

	// networkpolicies
	if apiResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("networkpolicies")) {
		networkExtensionsStorage := networkpolicystore.NewREST(restOptionsGetter)
		storage["networkpolicies"] = networkExtensionsStorage
	}

	return storage
}

func (p RESTStorageProvider) GroupName() string {
	return extensions.GroupName
}
