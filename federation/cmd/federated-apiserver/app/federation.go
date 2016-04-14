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

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/cmd/federated-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/registry/generic"

	_ "k8s.io/kubernetes/federation/apis/federation/install"
	clusteretcd "k8s.io/kubernetes/federation/registry/cluster/etcd"
)

func addFederationAPIGroup(d genericapiserver.StorageDestinations, s *options.APIServer) {
	glog.Infof("Configuring federation/v1alpha1 storage destination")
	storageVersions := s.StorageGroupsToGroupVersions()
	federationGroup, err := registered.Group(federation.GroupName)
	if err != nil {
		glog.Fatalf("Federation API is enabled in runtime config, but not enabled in the environment variable KUBE_API_VERSIONS. Error: %v", err)
	}
	// Figure out what storage group/version we should use.
	storageGroupVersion, found := storageVersions[federationGroup.GroupVersion.Group]
	if !found {
		glog.Fatalf("Couldn't find the storage version for group: %q in storageVersions: %v", federationGroup.GroupVersion.Group, storageVersions)
	}

	if storageGroupVersion != v1alpha1.SchemeGroupVersion.String() {
		glog.Fatalf("The storage version for federation must be '%v'", v1alpha1.SchemeGroupVersion.String())
	}
	glog.Infof("Using %v for federation group storage version", storageGroupVersion)
	federationEtcdStorage, err := newEtcd(api.Codecs, storageGroupVersion, federation.SchemeGroupVersion.String(), s.EtcdConfig)
	if err != nil {
		glog.Fatalf("Invalid federation storage version or misconfigured etcd: %v", err)
	}
	d.AddAPIGroup(federation.GroupName, federationEtcdStorage)
}

func installFederationAPI(m *master.Master, d genericapiserver.StorageDestinations) {
	clusterStorage, clusterStatusStorage := clusteretcd.NewREST(generic.RESTOptions{
		Storage:   d.Get(federation.GroupName, "clusters"),
		Decorator: m.StorageDecorator(),
	})
	clusterResources := map[string]rest.Storage{
		"clusters":        clusterStorage,
		"clusters/status": clusterStatusStorage,
	}
	clusterGroupMeta := registered.GroupOrDie(federation.GroupName)
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *clusterGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			"v1alpha1": clusterResources,
		},
		OptionsExternalVersion: &registered.GroupOrDie(api.GroupName).GroupVersion,
		Scheme:                 api.Scheme,
		ParameterCodec:         api.ParameterCodec,
		NegotiatedSerializer:   api.Codecs,
	}
	if err := m.InstallAPIGroup(&apiGroupInfo); err != nil {
		glog.Fatalf("Error in registering group versions: %v", err)
	}
}
