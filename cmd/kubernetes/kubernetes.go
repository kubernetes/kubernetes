/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
// TODO: combine this with the hypercube thingy.
package main

import (
	"fmt"
	"net"
	"net/http"
	"runtime"
	"time"

	kubeletapp "github.com/GoogleCloudPlatform/kubernetes/cmd/kubelet/app"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/nodecontroller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/servicecontroller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/replication"
	explatest "github.com/GoogleCloudPlatform/kubernetes/pkg/expapi/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	etcdstorage "github.com/GoogleCloudPlatform/kubernetes/pkg/storage/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	addr                   = flag.String("addr", "127.0.0.1", "The address to use for the apiserver.")
	port                   = flag.Int("port", 8080, "The port for the apiserver to use.")
	dockerEndpoint         = flag.String("docker-endpoint", "", "If non-empty, use this for the docker endpoint to communicate with")
	etcdServer             = flag.String("etcd-server", "http://localhost:4001", "If non-empty, path to the set of etcd server to use")
	masterServiceNamespace = flag.String("master-service-namespace", api.NamespaceDefault, "The namespace from which the kubernetes master services should be injected into pods")
	enableProfiling        = flag.Bool("profiling", false, "Enable profiling via web interface host:port/debug/pprof/")
	deletingPodsQps        = flag.Float32("deleting-pods-qps", 0.1, "")
	deletingPodsBurst      = flag.Int("deleting-pods-burst", 10, "")
)

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

// RunApiServer starts an API server in a go routine.
func runApiServer(etcdClient tools.EtcdClient, addr net.IP, port int, masterServiceNamespace string) {
	handler := delegateHandler{}

	etcdStorage, err := master.NewEtcdStorage(etcdClient, latest.InterfacesFor, latest.Version, master.DefaultEtcdPathPrefix)
	if err != nil {
		glog.Fatalf("Unable to get etcd storage: %v", err)
	}
	expEtcdStorage, err := master.NewEtcdStorage(etcdClient, explatest.InterfacesFor, explatest.Version, master.DefaultEtcdPathPrefix)
	if err != nil {
		glog.Fatalf("Unable to get etcd storage for experimental: %v", err)
	}

	// Create a master and install handlers into mux.
	m := master.New(&master.Config{
		DatabaseStorage:    etcdStorage,
		ExpDatabaseStorage: expEtcdStorage,
		KubeletClient: &client.HTTPKubeletClient{
			Client: http.DefaultClient,
			Config: &client.KubeletConfig{Port: 10250},
		},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableSwaggerSupport:  true,
		EnableProfiling:       *enableProfiling,
		APIPrefix:             "/api",
		ExpAPIPrefix:          "/experimental",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),

		ReadWritePort:          port,
		PublicAddress:          addr,
		MasterServiceNamespace: masterServiceNamespace,
	})
	handler.delegate = m.InsecureHandler

	go http.ListenAndServe(fmt.Sprintf("%s:%d", addr, port), &handler)
}

// RunScheduler starts up a scheduler in it's own goroutine
func runScheduler(cl *client.Client) {
	// Scheduler
	schedulerConfigFactory := factory.NewConfigFactory(cl)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		glog.Fatalf("Couldn't create scheduler config: %v", err)
	}
	scheduler.New(schedulerConfig).Run()
}

// RunControllerManager starts a controller
func runControllerManager(cl *client.Client) {
	const serviceSyncPeriod = 5 * time.Minute
	const nodeSyncPeriod = 10 * time.Second
	nodeController := nodecontroller.NewNodeController(
		nil, cl, 10, 5*time.Minute, nodecontroller.NewPodEvictor(util.NewTokenBucketRateLimiter(*deletingPodsQps, *deletingPodsBurst)),
		40*time.Second, 60*time.Second, 5*time.Second, nil, false)
	nodeController.Run(nodeSyncPeriod)

	serviceController := servicecontroller.New(nil, cl, "kubernetes")
	if err := serviceController.Run(serviceSyncPeriod, nodeSyncPeriod); err != nil {
		glog.Warningf("Running without a service controller: %v", err)
	}

	endpoints := service.NewEndpointController(cl)
	go endpoints.Run(5, util.NeverStop)

	controllerManager := replication.NewReplicationManager(cl, replication.BurstReplicas)
	go controllerManager.Run(5, util.NeverStop)
}

func startComponents(etcdClient tools.EtcdClient, cl *client.Client, addr net.IP, port int) {
	runApiServer(etcdClient, addr, port, *masterServiceNamespace)
	runScheduler(cl)
	runControllerManager(cl)

	dockerClient := dockertools.ConnectToDockerOrDie(*dockerEndpoint)
	cadvisorInterface, err := cadvisor.New(0)
	if err != nil {
		glog.Fatalf("Failed to create cAdvisor: %v", err)
	}
	kcfg := kubeletapp.SimpleKubelet(cl, dockerClient, "localhost", "/tmp/kubernetes", "", "127.0.0.1", 10250, *masterServiceNamespace, kubeletapp.ProbeVolumePlugins(), nil, cadvisorInterface, "", nil, kubecontainer.RealOS{})
	kubeletapp.RunKubelet(kcfg, nil)

}

func newApiClient(addr net.IP, port int) *client.Client {
	apiServerURL := fmt.Sprintf("http://%s:%d", addr, port)
	cl := client.NewOrDie(&client.Config{Host: apiServerURL, Version: testapi.Version()})
	return cl
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	util.InitFlags()
	util.InitLogs()
	defer util.FlushLogs()

	glog.Infof("Creating etcd client pointing to %v", *etcdServer)
	etcdClient, err := etcdstorage.NewEtcdClientStartServerIfNecessary(*etcdServer)
	if err != nil {
		glog.Fatalf("Failed to connect to etcd: %v", err)
	}
	address := net.ParseIP(*addr)
	startComponents(etcdClient, newApiClient(address, *port), address, *port)
	glog.Infof("Kubernetes API Server is up and running on http://%s:%d", *addr, *port)

	select {}
}
