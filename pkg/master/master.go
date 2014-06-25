/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"math/rand"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
)

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	podRegistry        registry.PodRegistry
	controllerRegistry registry.ControllerRegistry
	serviceRegistry    registry.ServiceRegistry
	minionRegistry     registry.MinionRegistry

	random  *rand.Rand
	storage map[string]apiserver.RESTStorage
}

// Returns a memory (not etcd) backed apiserver.
func NewMemoryServer(minions []string, cloud cloudprovider.Interface) *Master {
	m := &Master{
		podRegistry:        registry.MakeMemoryRegistry(),
		controllerRegistry: registry.MakeMemoryRegistry(),
		serviceRegistry:    registry.MakeMemoryRegistry(),
		minionRegistry:     registry.MakeMinionRegistry(minions),
	}
	m.init(cloud)
	return m
}

// Returns a new apiserver.
func New(etcdServers, minions []string, cloud cloudprovider.Interface) *Master {
	etcdClient := etcd.NewClient(etcdServers)
	minionRegistry := registry.MakeMinionRegistry(minions)
	m := &Master{
		podRegistry:        registry.MakeEtcdRegistry(etcdClient, minionRegistry),
		controllerRegistry: registry.MakeEtcdRegistry(etcdClient, minionRegistry),
		serviceRegistry:    registry.MakeEtcdRegistry(etcdClient, minionRegistry),
		minionRegistry:     minionRegistry,
	}
	m.init(cloud)
	return m
}

func (m *Master) init(cloud cloudprovider.Interface) {
	containerInfo := &client.HTTPContainerInfo{
		Client: http.DefaultClient,
		Port:   10250,
	}

	m.random = rand.New(rand.NewSource(int64(time.Now().Nanosecond())))
	podCache := NewPodCache(containerInfo, m.podRegistry, time.Second*30)
	go podCache.Loop()
	m.storage = map[string]apiserver.RESTStorage{
		"pods": registry.MakePodRegistryStorage(m.podRegistry, containerInfo, registry.MakeFirstFitScheduler(m.minionRegistry, m.podRegistry, m.random), cloud, podCache),
		"replicationControllers": registry.MakeControllerRegistryStorage(m.controllerRegistry),
		"services":               registry.MakeServiceRegistryStorage(m.serviceRegistry, cloud, m.minionRegistry),
		"minions":                registry.MakeMinionRegistryStorage(m.minionRegistry),
	}

}

// Runs master. Never returns.
func (m *Master) Run(myAddress, apiPrefix string) error {
	endpoints := registry.MakeEndpointController(m.serviceRegistry, m.podRegistry)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	s := &http.Server{
		Addr:           myAddress,
		Handler:        apiserver.New(m.storage, apiPrefix),
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
	return s.ListenAndServe()
}

// Instead of calling Run, call ConstructHandler to get a handler for your own
// server. Intended for testing. Only call once.
func (m *Master) ConstructHandler(apiPrefix string) http.Handler {
	endpoints := registry.MakeEndpointController(m.serviceRegistry, m.podRegistry)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	return apiserver.New(m.storage, apiPrefix)
}
