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

// The kubelet binary is responsible for maintaining a set of containers on a particular host VM.
// It syncs data from both configuration file(s) as well as from a quorum of etcd servers.
// It then queries Docker to see what is currently running.  It synchronizes the configuration data,
// with the running set of containers by starting or stopping Docker containers.
package main

import (
	"flag"
	"math/rand"
	"net"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/standalone"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version/verflag"
	"github.com/golang/glog"
)

const defaultRootDir = "/var/lib/kubelet"

var (
	config                  = flag.String("config", "", "Path to the config file or directory of files")
	syncFrequency           = flag.Duration("sync_frequency", 10*time.Second, "Max period between synchronizing running containers and config")
	fileCheckFrequency      = flag.Duration("file_check_frequency", 20*time.Second, "Duration between checking config files for new data")
	httpCheckFrequency      = flag.Duration("http_check_frequency", 20*time.Second, "Duration between checking http for new data")
	manifestURL             = flag.String("manifest_url", "", "URL for accessing the container manifest")
	enableServer            = flag.Bool("enable_server", true, "Enable the info server")
	address                 = util.IP(net.ParseIP("127.0.0.1"))
	port                    = flag.Uint("port", ports.KubeletPort, "The port for the info server to serve on")
	hostnameOverride        = flag.String("hostname_override", "", "If non-empty, will use this string as identification instead of the actual hostname.")
	networkContainerImage   = flag.String("network_container_image", kubelet.NetworkContainerImage, "The image that network containers in each pod will use.")
	dockerEndpoint          = flag.String("docker_endpoint", "", "If non-empty, use this for the docker endpoint to communicate with")
	etcdServerList          util.StringList
	etcdConfigFile          = flag.String("etcd_config", "", "The config file for the etcd client. Mutually exclusive with -etcd_servers")
	rootDirectory           = flag.String("root_dir", defaultRootDir, "Directory path for managing kubelet files (volume mounts,etc).")
	allowPrivileged         = flag.Bool("allow_privileged", false, "If true, allow containers to request privileged mode. [default=false]")
	registryPullQPS         = flag.Float64("registry_qps", 0.0, "If > 0, limit registry pull QPS to this value.  If 0, unlimited. [default=0.0]")
	registryBurst           = flag.Int("registry_burst", 10, "Maximum size of a bursty pulls, temporarily allows pulls to burst to this number, while still not exceeding registry_qps.  Only used if --registry_qps > 0")
	runonce                 = flag.Bool("runonce", false, "If true, exit after spawning pods from local manifests or remote urls. Exclusive with --etcd_servers, --api_servers, and --enable-server")
	enableDebuggingHandlers = flag.Bool("enable_debugging_handlers", true, "Enables server endpoints for log collection and local running of containers and commands")
	minimumGCAge            = flag.Duration("minimum_container_ttl_duration", 1*time.Minute, "Minimum age for a finished container before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'")
	maxContainerCount       = flag.Int("maximum_dead_containers_per_container", 5, "Maximum number of old instances of a container to retain per container.  Each container takes up some disk space.  Default: 5.")
	authPath                = flag.String("auth_path", "", "Path to .kubernetes_auth file, specifying how to authenticate to API server.")
	cAdvisorPort            = flag.Uint("cadvisor_port", 4194, "The port of the localhost cAdvisor endpoint")
	oomScoreAdj             = flag.Int("oom_score_adj", -900, "The oom_score_adj value for kubelet process. Values must be within the range [-1000, 1000]")
	apiServerList           util.StringList
	clusterDomain           = flag.String("cluster_domain", "", "Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains")
	masterServiceNamespace  = flag.String("master_service_namespace", api.NamespaceDefault, "The namespace from which the kubernetes master services should be injected into pods")
	clusterDNS              = util.IP(nil)
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated. Mutually exclusive with -etcd_config")
	flag.Var(&address, "address", "The IP address for the info server to serve on (set to 0.0.0.0 for all interfaces)")
	flag.Var(&apiServerList, "api_servers", "List of Kubernetes API servers for publishing events, and reading pods and services. (ip:port), comma separated.")
	flag.Var(&clusterDNS, "cluster_dns", "IP address for a cluster DNS server.  If set, kubelet will configure all containers to use this for DNS resolution in addition to the host's DNS servers")
}

func setupRunOnce() {
	if *runonce {
		// Don't use remote (etcd or apiserver) sources
		if len(etcdServerList) > 0 {
			glog.Fatalf("invalid option: --runonce and --etcd_servers are mutually exclusive")
		}
		if len(apiServerList) > 0 {
			glog.Fatalf("invalid option: --runonce and --api_servers are mutually exclusive")
		}
		if *enableServer {
			glog.Infof("--runonce is set, disabling server")
			*enableServer = false
		}
	}
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()
	rand.Seed(time.Now().UTC().UnixNano())

	verflag.PrintAndExitIfRequested()

	// Cluster creation scripts support both kubernetes versions that 1) support kublet watching
	// apiserver for pods, and 2) ones that don't. So they ca set both --etcd_servers and
	// --api_servers.  The current code will ignore the --etcd_servers flag, while older kubelet
	// code will use the --etd_servers flag for pods, and use --api_servers for event publising.
	//
	// TODO(erictune): convert all cloud provider scripts and Google Container Engine to
	// use only --api_servers, then delete --etcd_servers flag and the resulting dead code.
	if len(etcdServerList) > 0 && len(apiServerList) > 0 {
		glog.Infof("Both --etcd_servers and --api_servers are set.  Not using etcd source.")
		etcdServerList = util.StringList{}
	}

	setupRunOnce()

	if err := util.ApplyOomScoreAdj(*oomScoreAdj); err != nil {
		glog.Info(err)
	}

	client, err := standalone.GetAPIServerClient(*authPath, apiServerList)
	if err != nil && len(apiServerList) > 0 {
		glog.Warningf("No API client: %v", err)
	}

	kcfg := standalone.KubeletConfig{
		Address:                 address,
		AllowPrivileged:         *allowPrivileged,
		HostnameOverride:        *hostnameOverride,
		RootDirectory:           *rootDirectory,
		ConfigFile:              *config,
		ManifestURL:             *manifestURL,
		FileCheckFrequency:      *fileCheckFrequency,
		HttpCheckFrequency:      *httpCheckFrequency,
		NetworkContainerImage:   *networkContainerImage,
		SyncFrequency:           *syncFrequency,
		RegistryPullQPS:         *registryPullQPS,
		RegistryBurst:           *registryBurst,
		MinimumGCAge:            *minimumGCAge,
		MaxContainerCount:       *maxContainerCount,
		ClusterDomain:           *clusterDomain,
		ClusterDNS:              clusterDNS,
		Runonce:                 *runonce,
		Port:                    *port,
		CAdvisorPort:            *cAdvisorPort,
		EnableServer:            *enableServer,
		EnableDebuggingHandlers: *enableDebuggingHandlers,
		DockerClient:            util.ConnectToDockerOrDie(*dockerEndpoint),
		KubeClient:              client,
		EtcdClient:              kubelet.EtcdClientOrDie(etcdServerList, *etcdConfigFile),
		MasterServiceNamespace:  *masterServiceNamespace,
	}

	standalone.RunKubelet(&kcfg)
	// runs forever
	select {}
}
