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
	"net"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	cloudcontroller "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/binding"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/event"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Config is a structure used to configure a Master.
type Config struct {
	Client             *client.Client
	Cloud              cloudprovider.Interface
	EtcdHelper         tools.EtcdHelper
	HealthCheckMinions bool
	Minions            []string
	MinionCacheTTL     time.Duration
	EventTTL           time.Duration
	MinionRegexp       string
	PodInfoGetter      client.PodInfoGetter
	NodeResources      api.NodeResources
	PortalNet          *net.IPNet
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	podRegistry        pod.Registry
	controllerRegistry controller.Registry
	serviceRegistry    service.Registry
	endpointRegistry   endpoint.Registry
	minionRegistry     minion.Registry
	bindingRegistry    binding.Registry
	eventRegistry      generic.Registry
	storage            map[string]apiserver.RESTStorage
	client             *client.Client
	portalNet          *net.IPNet
}

// NewEtcdHelper returns an EtcdHelper for the provided arguments or an error if the version
// is incorrect.
func NewEtcdHelper(client tools.EtcdGetSet, version string) (helper tools.EtcdHelper, err error) {
	if version == "" {
		version = latest.Version
	}
	versionInterfaces, err := latest.InterfacesFor(version)
	if err != nil {
		return helper, err
	}
	return tools.EtcdHelper{client, versionInterfaces.Codec, tools.RuntimeVersionAdapter{versionInterfaces.ResourceVersioner}}, nil
}

// New returns a new instance of Master connected to the given etcd server.
func New(c *Config) *Master {
	minionRegistry := makeMinionRegistry(c)
	serviceRegistry := etcd.NewRegistry(c.EtcdHelper, nil)
	boundPodFactory := &pod.BasicBoundPodFactory{
		ServiceRegistry: serviceRegistry,
	}
	m := &Master{
		podRegistry:        etcd.NewRegistry(c.EtcdHelper, boundPodFactory),
		controllerRegistry: etcd.NewRegistry(c.EtcdHelper, nil),
		serviceRegistry:    serviceRegistry,
		endpointRegistry:   etcd.NewRegistry(c.EtcdHelper, nil),
		bindingRegistry:    etcd.NewRegistry(c.EtcdHelper, boundPodFactory),
		eventRegistry:      event.NewEtcdRegistry(c.EtcdHelper, uint64(c.EventTTL.Seconds())),
		minionRegistry:     minionRegistry,
		client:             c.Client,
		portalNet:          c.PortalNet,
	}
	m.init(c)
	return m
}

func makeMinionRegistry(c *Config) minion.Registry {
	var minionRegistry minion.Registry = etcd.NewRegistry(c.EtcdHelper, nil)
	if c.HealthCheckMinions {
		minionRegistry = minion.NewHealthyRegistry(minionRegistry, &http.Client{})
	}
	return minionRegistry
}

// init initializes master.
func (m *Master) init(c *Config) {
	podCache := NewPodCache(c.PodInfoGetter, m.podRegistry)
	go util.Forever(func() { podCache.UpdateAllContainers() }, time.Second*30)

	if c.Cloud != nil && len(c.MinionRegexp) > 0 {
		// TODO: Move minion controller to its own code.
		cloudcontroller.NewMinionController(c.Cloud, c.MinionRegexp, &c.NodeResources, m.minionRegistry, c.MinionCacheTTL).Run()
	} else {
		for _, minionID := range c.Minions {
			m.minionRegistry.CreateMinion(nil, &api.Minion{
				TypeMeta:      api.TypeMeta{Name: minionID},
				NodeResources: c.NodeResources,
			})
		}
	}

	m.storage = map[string]apiserver.RESTStorage{
		"pods": pod.NewREST(&pod.RESTConfig{
			CloudProvider: c.Cloud,
			PodCache:      podCache,
			PodInfoGetter: c.PodInfoGetter,
			Registry:      m.podRegistry,
			Minions:       m.client,
		}),
		"replicationControllers": controller.NewREST(m.controllerRegistry, m.podRegistry),
		"services":               service.NewREST(m.serviceRegistry, c.Cloud, m.minionRegistry, m.portalNet),
		"endpoints":              endpoint.NewREST(m.endpointRegistry),
		"minions":                minion.NewREST(m.minionRegistry),
		"events":                 event.NewREST(m.eventRegistry),

		// TODO: should appear only in scheduler API group.
		"bindings": binding.NewREST(m.bindingRegistry),
	}
}

// API_v1beta1 returns the resources and codec for API version v1beta1.
func (m *Master) API_v1beta1() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		storage[k] = v
	}
	return storage, v1beta1.Codec, "/api/v1beta1", latest.SelfLinker
}

// API_v1beta2 returns the resources and codec for API version v1beta2.
func (m *Master) API_v1beta2() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		storage[k] = v
	}
	return storage, v1beta2.Codec, "/api/v1beta2", latest.SelfLinker
}
