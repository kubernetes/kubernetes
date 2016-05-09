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

package apiserver

import (
	"fmt"
	"net"

	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io/v1"
	testgroupetcd "k8s.io/kubernetes/examples/apiserver/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/storage/storagebackend"

	// Install the testgroup API
	_ "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io/install"
)

const (
	// Ports on which to run the server.
	// Explicitly setting these to a different value than the default values, to prevent this from clashing with a local cluster.
	InsecurePort = 8081
	SecurePort   = 6444
)

func newStorageFactory() genericapiserver.StorageFactory {
	config := storagebackend.Config{
		Prefix:     genericapiserver.DefaultEtcdPathPrefix,
		ServerList: []string{"http://127.0.0.1:4001"},
	}
	storageFactory := genericapiserver.NewDefaultStorageFactory(config, "application/json", api.Codecs, genericapiserver.NewDefaultResourceEncodingConfig(), genericapiserver.NewResourceConfig())

	return storageFactory
}

func NewServerRunOptions() *genericapiserver.ServerRunOptions {
	serverOptions := genericapiserver.NewServerRunOptions()
	serverOptions.InsecurePort = InsecurePort
	serverOptions.SecurePort = SecurePort
	return serverOptions
}

func Run(serverOptions *genericapiserver.ServerRunOptions) error {
	// Set ServiceClusterIPRange
	_, serviceClusterIPRange, _ := net.ParseCIDR("10.0.0.0/24")
	serverOptions.ServiceClusterIPRange = *serviceClusterIPRange
	serverOptions.StorageConfig.ServerList = []string{"http://127.0.0.1:4001"}
	genericapiserver.ValidateRunOptions(serverOptions)
	config := genericapiserver.NewConfig(serverOptions)
	config.Serializer = api.Codecs
	s, err := genericapiserver.New(config)
	if err != nil {
		return fmt.Errorf("Error in bringing up the server: %v", err)
	}

	groupVersion := v1.SchemeGroupVersion
	groupName := groupVersion.Group
	groupMeta, err := registered.Group(groupName)
	if err != nil {
		return fmt.Errorf("%v", err)
	}
	storageFactory := newStorageFactory()
	storage, err := storageFactory.New(unversioned.GroupResource{Group: groupName, Resource: "testtype"})
	if err != nil {
		return fmt.Errorf("Unable to get storage: %v", err)
	}

	restStorageMap := map[string]rest.Storage{
		"testtypes": testgroupetcd.NewREST(storage, s.StorageDecorator()),
	}
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *groupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			groupVersion.Version: restStorageMap,
		},
		Scheme:               api.Scheme,
		NegotiatedSerializer: api.Codecs,
	}
	if err := s.InstallAPIGroups([]genericapiserver.APIGroupInfo{apiGroupInfo}); err != nil {
		return fmt.Errorf("Error in installing API: %v", err)
	}
	s.Run(serverOptions)
	return nil
}
