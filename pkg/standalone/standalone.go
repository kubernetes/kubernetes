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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	nodeControllerPkg "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/golang/glog"
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

// TODO: replace this with clientcmd
func GetAPIServerClient(authPath string, apiServerList util.StringList) (*client.Client, error) {
	authInfo, err := clientauth.LoadFromFile(authPath)
	if err != nil {
		return nil, err
	}
	clientConfig, err := authInfo.MergeWithConfig(client.Config{})
	if err != nil {
		return nil, err
	}
	if len(apiServerList) < 1 {
		return nil, fmt.Errorf("no api servers specified.")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(apiServerList) > 1 {
		glog.Infof("Multiple api servers specified.  Picking first one")
	}
	clientConfig.Host = apiServerList[0]
	c, err := client.New(&clientConfig)
	if err != nil {
		return nil, err
	}
	return c, nil
}

// RunApiServer starts an API server in a go routine.
func RunApiServer(cl *client.Client, etcdClient tools.EtcdClient, addr string, port int, masterServiceNamespace string) {
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
		EnableLogsSupport:    false,
		EnableSwaggerSupport: true,
		APIPrefix:            "/api",
		Authorizer:           apiserver.NewAlwaysAllowAuthorizer(),

		ReadWritePort:          port,
		ReadOnlyPort:           port,
		PublicAddress:          addr,
		MasterServiceNamespace: masterServiceNamespace,
	})
	handler.delegate = m.InsecureHandler

	go http.ListenAndServe(fmt.Sprintf("%s:%d", addr, port), &handler)
}

// RunScheduler starts up a scheduler in it's own goroutine
func RunScheduler(cl *client.Client) {
	// Scheduler
	schedulerConfigFactory := factory.NewConfigFactory(cl)
	schedulerConfig, err := schedulerConfigFactory.Create()
	if err != nil {
		glog.Fatalf("Couldn't create scheduler config: %v", err)
	}
	scheduler.New(schedulerConfig).Run()
}

// RunControllerManager starts a controller
func RunControllerManager(machineList []string, cl *client.Client, nodeMilliCPU, nodeMemory int64) {
	nodeResources := &api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(nodeMilliCPU, resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(nodeMemory, resource.BinarySI),
		},
	}
	nodeController := nodeControllerPkg.NewNodeController(nil, "", machineList, nodeResources, cl)
	nodeController.Run(10 * time.Second)

	endpoints := service.NewEndpointController(cl)
	go util.Forever(func() { endpoints.SyncServiceEndpoints() }, time.Second*10)

	controllerManager := controller.NewReplicationManager(cl)
	controllerManager.Run(10 * time.Second)
}

// SimpleRunKubelet is a simple way to start a Kubelet talking to dockerEndpoint, using an etcdClient.
// Under the hood it calls RunKubelet (below)
func SimpleRunKubelet(client *client.Client,
	etcdClient tools.EtcdClient,
	dockerClient dockertools.DockerInterface,
	hostname, rootDir, manifestURL, address string,
	port uint,
	masterServiceNamespace string,
	volumePlugins []volume.Plugin) {
	kcfg := KubeletConfig{
		KubeClient:            client,
		EtcdClient:            etcdClient,
		DockerClient:          dockerClient,
		HostnameOverride:      hostname,
		RootDirectory:         rootDir,
		ManifestURL:           manifestURL,
		NetworkContainerImage: kubelet.NetworkContainerImage,
		Port:                    port,
		Address:                 util.IP(net.ParseIP(address)),
		EnableServer:            true,
		EnableDebuggingHandlers: true,
		SyncFrequency:           3 * time.Second,
		MinimumGCAge:            10 * time.Second,
		MaxContainerCount:       5,
		MasterServiceNamespace:  masterServiceNamespace,
		VolumePlugins:           volumePlugins,
	}
	RunKubelet(&kcfg)
}

// RunKubelet is responsible for setting up and running a kubelet.  It is used in three different applications:
//   1 Integration tests
//   2 Kubelet binary
//   3 Standalone 'kubernetes' binary
// Eventually, #2 will be replaced with instances of #3
func RunKubelet(kcfg *KubeletConfig) {
	kcfg.Hostname = util.GetHostname(kcfg.HostnameOverride)
	if kcfg.KubeClient != nil {
		kubelet.SetupEventSending(kcfg.KubeClient, kcfg.Hostname)
	} else {
		glog.Infof("No api server defined - no events will be sent.")
	}
	kubelet.SetupLogging()
	kubelet.SetupCapabilities(kcfg.AllowPrivileged)

	cfg := makePodSourceConfig(kcfg)
	k, err := createAndInitKubelet(kcfg, cfg)
	if err != nil {
		glog.Errorf("Failed to create kubelet: %s", err)
		return
	}
	// process pods and exit.
	if kcfg.Runonce {
		if _, err := k.RunOnce(cfg.Updates()); err != nil {
			glog.Errorf("--runonce failed: %v", err)
		}
	} else {
		startKubelet(k, cfg, kcfg)
	}
}

func startKubelet(k *kubelet.Kubelet, cfg *config.PodConfig, kc *KubeletConfig) {
	// start the kubelet
	go util.Forever(func() { k.Run(cfg.Updates()) }, 0)

	// start the kubelet server
	if kc.EnableServer {
		go util.Forever(func() {
			kubelet.ListenAndServeKubeletServer(k, net.IP(kc.Address), kc.Port, kc.EnableDebuggingHandlers)
		}, 0)
	}
}

func makePodSourceConfig(kc *KubeletConfig) *config.PodConfig {
	// source of all configuration
	cfg := config.NewPodConfig(config.PodConfigNotificationSnapshotAndUpdates)

	// define file config source
	if kc.ConfigFile != "" {
		glog.Infof("Adding manifest file: %v", kc.ConfigFile)
		config.NewSourceFile(kc.ConfigFile, kc.FileCheckFrequency, cfg.Channel(kubelet.FileSource))
	}

	// define url config source
	if kc.ManifestURL != "" {
		glog.Infof("Adding manifest url: %v", kc.ManifestURL)
		config.NewSourceURL(kc.ManifestURL, kc.HttpCheckFrequency, cfg.Channel(kubelet.HTTPSource))
	}
	if kc.EtcdClient != nil {
		glog.Infof("Watching for etcd configs at %v", kc.EtcdClient.GetCluster())
		config.NewSourceEtcd(config.EtcdKeyForHost(kc.Hostname), kc.EtcdClient, cfg.Channel(kubelet.EtcdSource))
	}
	if kc.KubeClient != nil {
		glog.Infof("Watching apiserver")
		config.NewSourceApiserver(kc.KubeClient, kc.Hostname, cfg.Channel(kubelet.ApiserverSource))
	}
	return cfg
}

type KubeletConfig struct {
	EtcdClient              tools.EtcdClient
	KubeClient              *client.Client
	DockerClient            dockertools.DockerInterface
	CAdvisorPort            uint
	Address                 util.IP
	AllowPrivileged         bool
	HostnameOverride        string
	RootDirectory           string
	ConfigFile              string
	ManifestURL             string
	FileCheckFrequency      time.Duration
	HttpCheckFrequency      time.Duration
	Hostname                string
	NetworkContainerImage   string
	SyncFrequency           time.Duration
	RegistryPullQPS         float64
	RegistryBurst           int
	MinimumGCAge            time.Duration
	MaxContainerCount       int
	ClusterDomain           string
	ClusterDNS              util.IP
	EnableServer            bool
	EnableDebuggingHandlers bool
	Port                    uint
	Runonce                 bool
	MasterServiceNamespace  string
	VolumePlugins           []volume.Plugin
}

func createAndInitKubelet(kc *KubeletConfig, pc *config.PodConfig) (*kubelet.Kubelet, error) {
	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations

	k, err := kubelet.NewMainKubelet(
		kc.Hostname,
		kc.DockerClient,
		kc.EtcdClient,
		kc.KubeClient,
		kc.RootDirectory,
		kc.NetworkContainerImage,
		kc.SyncFrequency,
		float32(kc.RegistryPullQPS),
		kc.RegistryBurst,
		kc.MinimumGCAge,
		kc.MaxContainerCount,
		pc.IsSourceSeen,
		kc.ClusterDomain,
		net.IP(kc.ClusterDNS),
		kc.MasterServiceNamespace,
		kc.VolumePlugins)

	if err != nil {
		return nil, err
	}

	k.BirthCry()

	go k.GarbageCollectLoop()
	go kubelet.MonitorCAdvisor(k, kc.CAdvisorPort)
	kubelet.InitHealthChecking(k)

	return k, nil
}
