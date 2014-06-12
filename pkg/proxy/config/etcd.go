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
// Expects the list of exposed services to live under:
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
//
package config

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/coreos/go-etcd/etcd"
)

const RegistryRoot = "registry/services"

type ConfigSourceEtcd struct {
	client           *etcd.Client
	serviceChannel   chan ServiceUpdate
	endpointsChannel chan EndpointsUpdate
}

func NewConfigSourceEtcd(client *etcd.Client, serviceChannel chan ServiceUpdate, endpointsChannel chan EndpointsUpdate) ConfigSourceEtcd {
	config := ConfigSourceEtcd{
		client:           client,
		serviceChannel:   serviceChannel,
		endpointsChannel: endpointsChannel,
	}
	go config.Run()
	return config
}

func (impl ConfigSourceEtcd) Run() {
	// Initially, just wait for the etcd to come up before doing anything more complicated.
	var services []api.Service
	var endpoints []api.Endpoints
	var err error
	for {
		services, endpoints, err = impl.GetServices()
		if err == nil {
			break
		}
		log.Printf("Failed to get any services: %v", err)
		time.Sleep(2 * time.Second)
	}

	if len(services) > 0 {
		serviceUpdate := ServiceUpdate{Op: SET, Services: services}
		impl.serviceChannel <- serviceUpdate
	}
	if len(endpoints) > 0 {
		endpointsUpdate := EndpointsUpdate{Op: SET, Endpoints: endpoints}
		impl.endpointsChannel <- endpointsUpdate
	}

	// Ok, so we got something back from etcd. Let's set up a watch for new services, and
	// their endpoints
	go impl.WatchForChanges()

	for {
		services, endpoints, err = impl.GetServices()
		if err != nil {
			log.Printf("ConfigSourceEtcd: Failed to get services: %v", err)
		} else {
			if len(services) > 0 {
				serviceUpdate := ServiceUpdate{Op: SET, Services: services}
				impl.serviceChannel <- serviceUpdate
			}
			if len(endpoints) > 0 {
				endpointsUpdate := EndpointsUpdate{Op: SET, Endpoints: endpoints}
				impl.endpointsChannel <- endpointsUpdate
			}
		}
		time.Sleep(30 * time.Second)
	}
}

// Finds the list of services and their endpoints from etcd.
// This operation is akin to a set a known good at regular intervals.
func (impl ConfigSourceEtcd) GetServices() ([]api.Service, []api.Endpoints, error) {
	response, err := impl.client.Get(RegistryRoot+"/specs", true, false)
	if err != nil {
		log.Printf("Failed to get the key %s: %v", RegistryRoot, err)
		return make([]api.Service, 0), make([]api.Endpoints, 0), err
	}
	if response.Node.Dir == true {
		retServices := make([]api.Service, len(response.Node.Nodes))
		retEndpoints := make([]api.Endpoints, len(response.Node.Nodes))
		// Ok, so we have directories, this list should be the list
		// of services. Find the local port to listen on and remote endpoints
		// and create a Service entry for it.
		for i, node := range response.Node.Nodes {
			var svc api.Service
			err = json.Unmarshal([]byte(node.Value), &svc)
			if err != nil {
				log.Printf("Failed to load Service: %s (%#v)", node.Value, err)
				continue
			}
			retServices[i] = svc
			endpoints, err := impl.GetEndpoints(svc.ID)
			if err != nil {
				log.Printf("Couldn't get endpoints for %s : %v skipping", svc.ID, err)
			}
			log.Printf("Got service: %s on localport %d mapping to: %s", svc.ID, svc.Port, endpoints)
			retEndpoints[i] = endpoints
		}
		return retServices, retEndpoints, err
	}
	return nil, nil, fmt.Errorf("did not get the root of the registry %s", RegistryRoot)
}

func (impl ConfigSourceEtcd) GetEndpoints(service string) (api.Endpoints, error) {
	key := fmt.Sprintf(RegistryRoot + "/endpoints/" + service)
	response, err := impl.client.Get(key, true, false)
	if err != nil {
		log.Printf("Failed to get the key: %s %v", key, err)
		return api.Endpoints{}, err
	}
	// Parse all the endpoint specifications in this value.
	return ParseEndpoints(response.Node.Value)
}

// EtcdResponseToServiceAndLocalport takes an etcd response and pulls it apart to find
// service
func EtcdResponseToService(response *etcd.Response) (*api.Service, error) {
	if response.Node == nil {
		return nil, fmt.Errorf("invalid response from etcd: %#v", response)
	}
	var svc api.Service
	err := json.Unmarshal([]byte(response.Node.Value), &svc)
	if err != nil {
		return nil, err
	}
	return &svc, err
}

func ParseEndpoints(jsonString string) (api.Endpoints, error) {
	var e api.Endpoints
	err := json.Unmarshal([]byte(jsonString), &e)
	return e, err
}

func (impl ConfigSourceEtcd) WatchForChanges() {
	log.Print("Setting up a watch for new services")
	watchChannel := make(chan *etcd.Response)
	go impl.client.Watch("/registry/services/", 0, true, watchChannel, nil)
	for {
		watchResponse := <-watchChannel
		impl.ProcessChange(watchResponse)
	}
}

func (impl ConfigSourceEtcd) ProcessChange(response *etcd.Response) {
	log.Printf("Processing a change in service configuration... %s", *response)

	// If it's a new service being added (signified by a localport being added)
	// then process it as such
	if strings.Contains(response.Node.Key, "/endpoints/") {
		impl.ProcessEndpointResponse(response)
	} else if response.Action == "set" {
		service, err := EtcdResponseToService(response)
		if err != nil {
			log.Printf("Failed to parse %s Port: %s", response, err)
			return
		}

		log.Printf("New service added/updated: %#v", service)
		serviceUpdate := ServiceUpdate{Op: ADD, Services: []api.Service{*service}}
		impl.serviceChannel <- serviceUpdate
		return
	}
	if response.Action == "delete" {
		parts := strings.Split(response.Node.Key[1:], "/")
		if len(parts) == 4 {
			log.Printf("Deleting service: %s", parts[3])
			serviceUpdate := ServiceUpdate{Op: REMOVE, Services: []api.Service{{JSONBase: api.JSONBase{ID: parts[3]}}}}
			impl.serviceChannel <- serviceUpdate
			return
		} else {
			log.Printf("Unknown service delete: %#v", parts)
		}
	}
}

func (impl ConfigSourceEtcd) ProcessEndpointResponse(response *etcd.Response) {
	log.Printf("Processing a change in endpoint configuration... %s", *response)
	var endpoints api.Endpoints
	err := json.Unmarshal([]byte(response.Node.Value), &endpoints)
	if err != nil {
		log.Printf("Failed to parse service out of etcd key: %v : %+v", response.Node.Value, err)
		return
	}
	endpointsUpdate := EndpointsUpdate{Op: ADD, Endpoints: []api.Endpoints{endpoints}}
	impl.endpointsChannel <- endpointsUpdate
}
