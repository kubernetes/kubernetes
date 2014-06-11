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

// A basic integration test for the service.
// Assumes that there is a pre-existing etcd server running on localhost.
package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http/httptest"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
	"github.com/coreos/go-etcd/etcd"
)

func main() {

	// Setup
	servers := []string{"http://localhost:4001"}
	log.Printf("Creating etcd client pointing to %v", servers)
	etcdClient := etcd.NewClient(servers)
	machineList := []string{"machine"}

	reg := registry.MakeEtcdRegistry(etcdClient, machineList)

	apiserver := apiserver.New(map[string]apiserver.RESTStorage{
		"pods": registry.MakePodRegistryStorage(reg, &client.FakeContainerInfo{}, registry.MakeRoundRobinScheduler(machineList)),
		"replicationControllers": registry.MakeControllerRegistryStorage(reg),
	}, "/api/v1beta1")
	server := httptest.NewServer(apiserver)

	controllerManager := registry.MakeReplicationManager(etcd.NewClient(servers),
		client.Client{
			Host: server.URL,
		})

	go controllerManager.Synchronize()
	go controllerManager.WatchControllers()

	// Ok. we're good to go.
	log.Printf("API Server started on %s", server.URL)
	// Wait for the synchronization threads to come up.
	time.Sleep(time.Second * 10)

	kubeClient := client.Client{
		Host: server.URL,
	}
	data, err := ioutil.ReadFile("api/examples/controller.json")
	if err != nil {
		log.Fatalf("Unexpected error: %#v", err)
	}
	var controllerRequest api.ReplicationController
	if err = json.Unmarshal(data, &controllerRequest); err != nil {
		log.Fatalf("Unexpected error: %#v", err)
	}

	if _, err = kubeClient.CreateReplicationController(controllerRequest); err != nil {
		log.Fatalf("Unexpected error: %#v", err)
	}
	// Give the controllers some time to actually create the pods
	time.Sleep(time.Second * 10)

	// Validate that they're truly up.
	pods, err := kubeClient.ListPods(nil)
	if err != nil || len(pods.Items) != 2 {
		log.Fatal("FAILED")
	}
	log.Printf("OK")
}
