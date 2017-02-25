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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	daemonsetstore "k8s.io/kubernetes/pkg/registry/extensions/daemonset/storage"
	deploymentstore "k8s.io/kubernetes/pkg/registry/extensions/deployment/storage"
	ingressstore "k8s.io/kubernetes/pkg/registry/extensions/ingress/storage"
	replicasetstore "k8s.io/kubernetes/pkg/registry/extensions/replicaset/storage"
)

func installExtensionsAPIs(g *genericapiserver.GenericAPIServer, optsGetter generic.RESTOptionsGetter) {
	replicaSetStorage := replicasetstore.NewStorage(optsGetter)
	deploymentStorage := deploymentstore.NewStorage(optsGetter)
	ingressStorage, ingressStatusStorage := ingressstore.NewREST(optsGetter)
	daemonSetStorage, daemonSetStatusStorage := daemonsetstore.NewREST(optsGetter)

	extensionsResources := map[string]rest.Storage{
		"replicasets":          replicaSetStorage.ReplicaSet,
		"replicasets/status":   replicaSetStorage.Status,
		"replicasets/scale":    replicaSetStorage.Scale,
		"ingresses":            ingressStorage,
		"ingresses/status":     ingressStatusStorage,
		"daemonsets":           daemonSetStorage,
		"daemonsets/status":    daemonSetStatusStorage,
		"deployments":          deploymentStorage.Deployment,
		"deployments/status":   deploymentStorage.Status,
		"deployments/scale":    deploymentStorage.Scale,
		"deployments/rollback": deploymentStorage.Rollback,
	}
	extensionsGroupMeta := api.Registry.GroupOrDie(extensions.GroupName)
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *extensionsGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			"v1beta1": extensionsResources,
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
