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

// A binary that is capable of running a complete, standalone kubernetes cluster.
// Expects an etcd server is available, or on the path somewhere.
// Does *not* currently setup the Kubernetes network model, that must be done ahead of time.
// TODO: Setup the k8s network bridge as part of setup.
package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/standalone"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

var (
	addr           = flag.String("addr", "127.0.0.1", "The address to use for the apiserver.")
	port           = flag.Int("port", 8080, "The port for the apiserver to use.")
	dockerEndpoint = flag.String("docker_endpoint", "", "If non-empty, use this for the docker endpoint to communicate with")
	etcdServer     = flag.String("etcd_server", "http://localhost:4001", "If non-empty, path to the set of etcd server to use")
	// TODO: Discover these by pinging the host machines, and rip out these flags.
	nodeMilliCPU = flag.Int64("node_milli_cpu", 1000, "The amount of MilliCPU provisioned on each node")
	nodeMemory   = flag.Int64("node_memory", 3*1024*1024*1024, "The amount of memory (in bytes) provisioned on each node")
)

func startComponents(etcdClient tools.EtcdClient, cl *client.Client, addr string, port int) {
	machineList := []string{"localhost"}

	standalone.RunApiServer(cl, etcdClient, addr, port)
	standalone.RunScheduler(cl)
	standalone.RunControllerManager(machineList, cl, *nodeMilliCPU, *nodeMemory)

	dockerClient := util.ConnectToDockerOrDie(*dockerEndpoint)
	standalone.SimpleRunKubelet(etcdClient, cl, dockerClient, machineList[0], "/tmp/kubernetes", "", "127.0.0.1", 10250)
}

func newApiClient(addr string, port int) *client.Client {
	apiServerURL := fmt.Sprintf("http://%s:%d", addr, port)
	cl := client.NewOrDie(&client.Config{Host: apiServerURL, Version: testapi.Version()})
	cl.PollPeriod = time.Second * 1
	cl.Sync = true
	return cl
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	glog.Infof("Creating etcd client pointing to %v", *etcdServer)
	etcdClient, err := tools.NewEtcdClientStartServerIfNecessary(*etcdServer)
	if err != nil {
		glog.Fatalf("Failed to connect to etcd: %v", err)
	}
	startComponents(etcdClient, newApiClient(*addr, *port), *addr, *port)
	glog.Infof("Kubernetes API Server is up and running on http://%s:%d", *addr, *port)

	select {}
}
