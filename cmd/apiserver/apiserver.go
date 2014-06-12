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
// apiserver is the main api server and master for the cluster.
// it is responsible for serving the cluster management API.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/coreos/go-etcd/etcd"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	kube_client "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var (
	port                        = flag.Uint("port", 8080, "The port to listen on.  Default 8080.")
	address                     = flag.String("address", "127.0.0.1", "The address on the local server to listen to. Default 127.0.0.1")
	apiPrefix                   = flag.String("api_prefix", "/api/v1beta1", "The prefix for API requests on the server. Default '/api/v1beta1'")
	etcdServerList, machineList util.StringList
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "Servers for the etcd (http://ip:port), comma separated")
	flag.Var(&machineList, "machines", "List of machines to schedule onto, comma separated.")
}

func main() {
	flag.Parse()

	if len(machineList) == 0 {
		log.Fatal("No machines specified!")
	}

	var (
		podRegistry       registry.PodRegistry
		controllerRegistry registry.ControllerRegistry
		serviceRegistry    registry.ServiceRegistry
	)

	if len(etcdServerList) > 0 {
		log.Printf("Creating etcd client pointing to %v", etcdServerList)
		etcdClient := etcd.NewClient(etcdServerList)
		podRegistry = registry.MakeEtcdRegistry(etcdClient, machineList)
		controllerRegistry = registry.MakeEtcdRegistry(etcdClient, machineList)
		serviceRegistry = registry.MakeEtcdRegistry(etcdClient, machineList)
	} else {
		podRegistry = registry.MakeMemoryRegistry()
		controllerRegistry = registry.MakeMemoryRegistry()
		serviceRegistry = registry.MakeMemoryRegistry()
	}

	containerInfo := &kube_client.HTTPContainerInfo{
		Client: http.DefaultClient,
		Port:   10250,
	}

	random := rand.New(rand.NewSource(int64(time.Now().Nanosecond())))
	storage := map[string]apiserver.RESTStorage{
		"pods":                  registry.MakePodRegistryStorage(podRegistry, containerInfo, registry.MakeFirstFitScheduler(machineList, podRegistry, random)),
		"replicationControllers": registry.MakeControllerRegistryStorage(controllerRegistry),
		"services":               registry.MakeServiceRegistryStorage(serviceRegistry),
	}

	endpoints := registry.MakeEndpointController(serviceRegistry, podRegistry)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	s := &http.Server{
		Addr:           fmt.Sprintf("%s:%d", *address, *port),
		Handler:        apiserver.New(storage, *apiPrefix),
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
	log.Fatal(s.ListenAndServe())
}
