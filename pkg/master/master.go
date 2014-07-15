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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	podRegistry        registry.PodRegistry
	controllerRegistry registry.ControllerRegistry
	serviceRegistry    registry.ServiceRegistry
	minionRegistry     registry.MinionRegistry

	// TODO: don't reuse non-threadsafe objects.
	random  *rand.Rand
	storage map[string]apiserver.RESTStorage
}

// NewMemoryServer returns a new instance of Master backed with memory (not etcd).
func NewMemoryServer(minions []string, podInfoGetter client.PodInfoGetter, cloud cloudprovider.Interface) *Master {
	m := &Master{
		podRegistry:        registry.MakeMemoryRegistry(),
		controllerRegistry: registry.MakeMemoryRegistry(),
		serviceRegistry:    registry.MakeMemoryRegistry(),
		minionRegistry:     registry.MakeMinionRegistry(minions),
	}
	m.init(cloud, podInfoGetter)
	return m
}

// New returns a new instance of Master connected to the given etcdServer.
func New(etcdServers, minions []string, podInfoGetter client.PodInfoGetter, cloud cloudprovider.Interface, minionRegexp string) *Master {
	etcdClient := etcd.NewClient(etcdServers)
	minionRegistry := minionRegistryMaker(minions, cloud, minionRegexp)
	m := &Master{
		podRegistry:        registry.MakeEtcdRegistry(etcdClient, minionRegistry),
		controllerRegistry: registry.MakeEtcdRegistry(etcdClient, minionRegistry),
		serviceRegistry:    registry.MakeEtcdRegistry(etcdClient, minionRegistry),
		minionRegistry:     minionRegistry,
	}
	m.init(cloud, podInfoGetter)
	return m
}

func minionRegistryMaker(minions []string, cloud cloudprovider.Interface, minionRegexp string) registry.MinionRegistry {
	if cloud != nil && len(minionRegexp) > 0 {
		minionRegistry, err := registry.MakeCloudMinionRegistry(cloud, minionRegexp)
		if err != nil {
			glog.Errorf("Failed to initalize cloud minion registry reverting to static registry (%#v)", err)
		}
		return minionRegistry
	}
	return registry.MakeMinionRegistry(minions)
}

func (m *Master) init(cloud cloudprovider.Interface, podInfoGetter client.PodInfoGetter) {
	m.random = rand.New(rand.NewSource(int64(time.Now().Nanosecond())))
	podCache := NewPodCache(podInfoGetter, m.podRegistry, time.Second*30)
	go podCache.Loop()
	s := scheduler.NewRandomFitScheduler(m.podRegistry, m.random)
	m.storage = map[string]apiserver.RESTStorage{
		"pods": registry.MakePodRegistryStorage(m.podRegistry, podInfoGetter, s, m.minionRegistry, cloud, podCache),
		"replicationControllers": registry.NewControllerRegistryStorage(m.controllerRegistry, m.podRegistry),
		"services":               registry.MakeServiceRegistryStorage(m.serviceRegistry, cloud, m.minionRegistry),
		"minions":                registry.MakeMinionRegistryStorage(m.minionRegistry),
	}

}

// Run begins serving the Kubernetes API. It never returns.
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

// ConstructHandler returns an http.Handler which serves the Kubernetes API.
// Instead of calling Run, you can call this function to get a handler for your own server.
// It is intended for testing. Only call once.
func (m *Master) ConstructHandler(apiPrefix string) http.Handler {
	endpoints := registry.MakeEndpointController(m.serviceRegistry, m.podRegistry)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	return apiserver.New(m.storage, apiPrefix)
}
