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

// Watches etcd and gets the full configuration on preset intervals.
// It expects the list of exposed services to live under:
// registry/services
// which in etcd is exposed like so:
// http://<etcd server>/v2/keys/registry/services
//
// The port that proxy needs to listen in for each service is a value in:
// registry/services/<service>
//
// The endpoints for each of the services found is a json string
// representing that service at:
// /registry/services/<service>/endpoint
// and the format is:
// '[ { "machine": <host>, "name": <name", "port": <port> },
//    { "machine": <host2>, "name": <name2", "port": <port2> }
//  ]',

package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

// registryRoot is the key prefix for service configs in etcd.
const registryRoot = "registry/services"

// ConfigSourceEtcd communicates with a etcd via the client, and sends the change notification of services and endpoints to the specified channels.
type ConfigSourceEtcd struct {
	client           *etcd.Client
	serviceChannel   chan ServiceUpdate
	endpointsChannel chan EndpointsUpdate
	interval         time.Duration
}

// NewConfigSourceEtcd creates a new ConfigSourceEtcd and immediately runs the created ConfigSourceEtcd in a goroutine.
func NewConfigSourceEtcd(client *etcd.Client, serviceChannel chan ServiceUpdate, endpointsChannel chan EndpointsUpdate) ConfigSourceEtcd {
	config := ConfigSourceEtcd{
		client:           client,
		serviceChannel:   serviceChannel,
		endpointsChannel: endpointsChannel,
		interval:         2 * time.Second,
	}
	go config.Run()
	return config
}

// Run begins watching for new services and their endpoints on etcd.
func (s ConfigSourceEtcd) Run() {
	// Initially, just wait for the etcd to come up before doing anything more complicated.
	var services []api.Service
	var endpoints []api.Endpoints
	var err error
	for {
		services, endpoints, err = s.GetServices()
		if err == nil {
			break
		}
		glog.V(1).Infof("Failed to get any services: %v", err)
		time.Sleep(s.interval)
	}

	if len(services) > 0 {
		serviceUpdate := ServiceUpdate{Op: SET, Services: services}
		s.serviceChannel <- serviceUpdate
	}
	if len(endpoints) > 0 {
		endpointsUpdate := EndpointsUpdate{Op: SET, Endpoints: endpoints}
		s.endpointsChannel <- endpointsUpdate
	}

	// Ok, so we got something back from etcd. Let's set up a watch for new services, and
	// their endpoints
	go util.Forever(s.WatchForChanges, 1*time.Second)

	for {
		services, endpoints, err = s.GetServices()
		if err != nil {
			glog.Errorf("ConfigSourceEtcd: Failed to get services: %v", err)
		} else {
			if len(services) > 0 {
				serviceUpdate := ServiceUpdate{Op: SET, Services: services}
				s.serviceChannel <- serviceUpdate
			}
			if len(endpoints) > 0 {
				endpointsUpdate := EndpointsUpdate{Op: SET, Endpoints: endpoints}
				s.endpointsChannel <- endpointsUpdate
			}
		}
		time.Sleep(30 * time.Second)
	}
}

// GetServices finds the list of services and their endpoints from etcd.
// This operation is akin to a set a known good at regular intervals.
func (s ConfigSourceEtcd) GetServices() ([]api.Service, []api.Endpoints, error) {
	response, err := s.client.Get(registryRoot+"/specs", true, false)
	if err != nil {
		glog.V(1).Infof("Failed to get the key %s: %v", registryRoot, err)
		if tools.IsEtcdNotFound(err) {
			return []api.Service{}, []api.Endpoints{}, err
		}
	}
	if response.Node.Dir == true {
		retServices := make([]api.Service, len(response.Node.Nodes))
		retEndpoints := make([]api.Endpoints, len(response.Node.Nodes))
		// Ok, so we have directories, this list should be the list
		// of services. Find the local port to listen on and remote endpoints
		// and create a Service entry for it.
		for i, node := range response.Node.Nodes {
			var svc api.Service
			err = api.DecodeInto([]byte(node.Value), &svc)
			if err != nil {
				glog.Errorf("Failed to load Service: %s (%#v)", node.Value, err)
				continue
			}
			retServices[i] = svc
			endpoints, err := s.GetEndpoints(svc.ID)
			if err != nil {
				if tools.IsEtcdNotFound(err) {
					glog.V(1).Infof("Unable to get endpoints for %s : %v", svc.ID, err)
				}
				glog.Errorf("Couldn't get endpoints for %s : %v skipping", svc.ID, err)
				endpoints = api.Endpoints{}
			} else {
				glog.Infof("Got service: %s on localport %d mapping to: %s", svc.ID, svc.Port, endpoints)
			}
			retEndpoints[i] = endpoints
		}
		return retServices, retEndpoints, err
	}
	return nil, nil, fmt.Errorf("did not get the root of the registry %s", registryRoot)
}

// GetEndpoints finds the list of endpoints of the service from etcd.
func (s ConfigSourceEtcd) GetEndpoints(service string) (api.Endpoints, error) {
	key := fmt.Sprintf(registryRoot + "/endpoints/" + service)
	response, err := s.client.Get(key, true, false)
	if err != nil {
		glog.Errorf("Failed to get the key: %s %v", key, err)
		return api.Endpoints{}, err
	}
	// Parse all the endpoint specifications in this value.
	var e api.Endpoints
	err = api.DecodeInto([]byte(response.Node.Value), &e)
	return e, err
}

// etcdResponseToService takes an etcd response and pulls it apart to find service.
func etcdResponseToService(response *etcd.Response) (*api.Service, error) {
	if response.Node == nil {
		return nil, fmt.Errorf("invalid response from etcd: %#v", response)
	}
	var svc api.Service
	err := api.DecodeInto([]byte(response.Node.Value), &svc)
	if err != nil {
		return nil, err
	}
	return &svc, err
}

func (s ConfigSourceEtcd) WatchForChanges() {
	glog.Info("Setting up a watch for new services")
	watchChannel := make(chan *etcd.Response)
	go s.client.Watch("/registry/services/", 0, true, watchChannel, nil)
	for {
		watchResponse, ok := <-watchChannel
		if !ok {
			break
		}
		s.ProcessChange(watchResponse)
	}
}

func (s ConfigSourceEtcd) ProcessChange(response *etcd.Response) {
	glog.Infof("Processing a change in service configuration... %s", *response)

	// If it's a new service being added (signified by a localport being added)
	// then process it as such
	if strings.Contains(response.Node.Key, "/endpoints/") {
		s.ProcessEndpointResponse(response)
	} else if response.Action == "set" {
		service, err := etcdResponseToService(response)
		if err != nil {
			glog.Errorf("Failed to parse %s Port: %s", response, err)
			return
		}

		glog.Infof("New service added/updated: %#v", service)
		serviceUpdate := ServiceUpdate{Op: ADD, Services: []api.Service{*service}}
		s.serviceChannel <- serviceUpdate
		return
	}
	if response.Action == "delete" {
		parts := strings.Split(response.Node.Key[1:], "/")
		if len(parts) == 4 {
			glog.Infof("Deleting service: %s", parts[3])
			serviceUpdate := ServiceUpdate{Op: REMOVE, Services: []api.Service{{JSONBase: api.JSONBase{ID: parts[3]}}}}
			s.serviceChannel <- serviceUpdate
			return
		}
		glog.Infof("Unknown service delete: %#v", parts)
	}
}

func (s ConfigSourceEtcd) ProcessEndpointResponse(response *etcd.Response) {
	glog.Infof("Processing a change in endpoint configuration... %s", *response)
	var endpoints api.Endpoints
	err := api.DecodeInto([]byte(response.Node.Value), &endpoints)
	if err != nil {
		glog.Errorf("Failed to parse service out of etcd key: %v : %+v", response.Node.Value, err)
		return
	}
	endpointsUpdate := EndpointsUpdate{Op: ADD, Endpoints: []api.Endpoints{endpoints}}
	s.endpointsChannel <- endpointsUpdate
}
