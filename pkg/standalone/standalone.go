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

package standalone

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	minionControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/resources"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const testRootDir = "/tmp/kubelet"

type delegateHandler struct {
	delegate http.Handler
}

func (h *delegateHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if h.delegate != nil {
		h.delegate.ServeHTTP(w, req)
		return
	}
	w.WriteHeader(http.StatusNotFound)
}

// Get a docker endpoint, either from the string passed in, or $DOCKER_HOST environment variables
func GetDockerEndpoint(dockerEndpoint string) string {
	var endpoint string
	if len(dockerEndpoint) > 0 {
		endpoint = dockerEndpoint
	} else if len(os.Getenv("DOCKER_HOST")) > 0 {
		endpoint = os.Getenv("DOCKER_HOST")
	} else {
		endpoint = "unix:///var/run/docker.sock"
	}
	glog.Infof("Connecting to docker on %s", endpoint)

	return endpoint
}

// RunApiServer starts an API server in a go routine.
func RunApiServer(cl *client.Client, etcdClient tools.EtcdClient, addr string, port int) {
	handler := delegateHandler{}

	helper, err := master.NewEtcdHelper(etcdClient, "")
	if err != nil {
		glog.Fatalf("Unable to get etcd helper: %v", err)
	}

	// Create a master and install handlers into mux.
	m := master.New(&master.Config{
		Client:     cl,
		EtcdHelper: helper,
		KubeletClient: &client.HTTPKubeletClient{
			Client: http.DefaultClient,
			Port:   10250,
		},
		EnableLogsSupport: false,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysAllowAuthorizer(),

		ReadWritePort: port,
		ReadOnlyPort:  port,
		PublicAddress: addr,
	})
	mux := http.NewServeMux()
	apiserver.NewAPIGroup(m.API_v1beta1()).InstallREST(mux, "/api/v1beta1")
	apiserver.NewAPIGroup(m.API_v1beta2()).InstallREST(mux, "/api/v1beta2")
	apiserver.InstallSupport(mux)
	handler.delegate = mux

	go http.ListenAndServe(fmt.Sprintf("%s:%d", addr, port), &handler)
}

// RunScheduler starts up a scheduler in it's own goroutine
func RunScheduler(cl *client.Client) {
	// Scheduler
	schedulerConfigFactory := &factory.ConfigFactory{cl}
	schedulerConfig := schedulerConfigFactory.Create()
	scheduler.New(schedulerConfig).Run()
}

// RunControllerManager starts a controller
func RunControllerManager(machineList []string, cl *client.Client, nodeMilliCPU, nodeMemory int64) {
	if int64(int(nodeMilliCPU)) != nodeMilliCPU || int64(int(nodeMemory)) != nodeMemory {
		glog.Fatalf("Overflow, nodeCPU or nodeMemory too large for the platform")
	}
	nodeResources := &api.NodeResources{
		Capacity: api.ResourceList{
			resources.CPU:    util.NewIntOrStringFromInt(int(nodeMilliCPU)),
			resources.Memory: util.NewIntOrStringFromInt(int(nodeMemory)),
		},
	}
	minionController := minionControllerPkg.NewMinionController(nil, "", machineList, nodeResources, cl)
	minionController.Run(10 * time.Second)

	endpoints := service.NewEndpointController(cl)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	controllerManager := controller.NewReplicationManager(cl)
	controllerManager.Run(10 * time.Second)
}

// RunKubelet starts a Kubelet talking to dockerEndpoint
func RunKubelet(etcdClient tools.EtcdClient, hostname, dockerEndpoint string) {
	dockerClient, err := docker.NewClient(GetDockerEndpoint(dockerEndpoint))
	if err != nil {
		glog.Fatal("Couldn't connect to docker.")
	}

	// Kubelet (localhost)
	os.MkdirAll(testRootDir, 0750)
	cfg1 := config.NewPodConfig(config.PodConfigNotificationSnapshotAndUpdates)
	config.NewSourceEtcd(config.EtcdKeyForHost(hostname), etcdClient, cfg1.Channel("etcd"))
	myKubelet := kubelet.NewIntegrationTestKubelet(hostname, testRootDir, dockerClient)
	go util.Forever(func() { myKubelet.Run(cfg1.Updates()) }, 0)
	go util.Forever(func() {
		kubelet.ListenAndServeKubeletServer(myKubelet, cfg1.Channel("http"), net.ParseIP("127.0.0.1"), 10250, true)
	}, 0)
}
