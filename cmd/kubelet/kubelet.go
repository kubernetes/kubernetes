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
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	kconfig "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version/verflag"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/google/cadvisor/client"
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
	runonce                 = flag.Bool("runonce", false, "If true, exit after spawning pods from local manifests or remote urls. Exclusive with --etcd_servers and --enable-server")
	enableDebuggingHandlers = flag.Bool("enable_debugging_handlers", true, "Enables server endpoints for log collection and local running of containers and commands")
	minimumGCAge            = flag.Duration("minimum_container_ttl_duration", 0, "Minimum age for a finished container before it is garbage collected.  Examples: '300ms', '10s' or '2h45m'")
	maxContainerCount       = flag.Int("maximum_dead_containers_per_container", 5, "Maximum number of old instances of a container to retain per container.  Each container takes up some disk space.  Default: 5.")
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated. Mutually exclusive with -etcd_config")
	flag.Var(&address, "address", "The IP address for the info server to serve on (set to 0.0.0.0 for all interfaces)")
}

func getDockerEndpoint() string {
	var endpoint string
	if len(*dockerEndpoint) > 0 {
		endpoint = *dockerEndpoint
	} else if len(os.Getenv("DOCKER_HOST")) > 0 {
		endpoint = os.Getenv("DOCKER_HOST")
	} else {
		endpoint = "unix:///var/run/docker.sock"
	}
	glog.Infof("Connecting to docker on %s", endpoint)

	return endpoint
}

func getHostname() string {
	hostname := []byte(*hostnameOverride)
	if string(hostname) == "" {
		// Note: We use exec here instead of os.Hostname() because we
		// want the FQDN, and this is the easiest way to get it.
		fqdn, err := exec.Command("hostname", "-f").Output()
		if err != nil {
			glog.Fatalf("Couldn't determine hostname: %v", err)
		}
		hostname = fqdn
	}
	return strings.TrimSpace(string(hostname))
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()
	rand.Seed(time.Now().UTC().UnixNano())

	verflag.PrintAndExitIfRequested()

	if *runonce {
		exclusiveFlag := "invalid option: --runonce and %s are mutually exclusive"
		if len(etcdServerList) > 0 {
			glog.Fatalf(exclusiveFlag, "--etcd_servers")
		}
		if *enableServer {
			glog.Infof("--runonce is set, disabling server")
			*enableServer = false
		}
	}

	etcd.SetLogger(util.NewLogger("etcd "))

	// Log the events locally too.
	record.StartLogging(glog.Infof)

	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: *allowPrivileged,
	})

	dockerClient, err := docker.NewClient(getDockerEndpoint())
	if err != nil {
		glog.Fatal("Couldn't connect to docker.")
	}

	hostname := getHostname()

	if *rootDirectory == "" {
		glog.Fatal("Invalid root directory path.")
	}
	*rootDirectory = path.Clean(*rootDirectory)
	if err := os.MkdirAll(*rootDirectory, 0750); err != nil {
		glog.Fatalf("Error creating root directory: %v", err)
	}

	// source of all configuration
	cfg := kconfig.NewPodConfig(kconfig.PodConfigNotificationSnapshotAndUpdates)

	// define file config source
	if *config != "" {
		kconfig.NewSourceFile(*config, *fileCheckFrequency, cfg.Channel("file"))
	}

	// define url config source
	if *manifestURL != "" {
		kconfig.NewSourceURL(*manifestURL, *httpCheckFrequency, cfg.Channel("http"))
	}

	// define etcd config source and initialize etcd client
	var etcdClient *etcd.Client
	if len(etcdServerList) > 0 {
		etcdClient = etcd.NewClient(etcdServerList)
	} else if *etcdConfigFile != "" {
		var err error
		etcdClient, err = etcd.NewClientFromFile(*etcdConfigFile)
		if err != nil {
			glog.Fatalf("Error with etcd config file: %v", err)
		}
	}

	if etcdClient != nil {
		glog.Infof("Watching for etcd configs at %v", etcdClient.GetCluster())
		kconfig.NewSourceEtcd(kconfig.EtcdKeyForHost(hostname), etcdClient, cfg.Channel("etcd"))
	}

	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations

	k := kubelet.NewMainKubelet(
		getHostname(),
		dockerClient,
		etcdClient,
		*rootDirectory,
		*networkContainerImage,
		*syncFrequency,
		float32(*registryPullQPS),
		*registryBurst,
		*minimumGCAge,
		*maxContainerCount)

	k.BirthCry()

	go func() {
		util.Forever(func() {
			err := k.GarbageCollectContainers()
			if err != nil {
				glog.Errorf("Garbage collect failed: %v", err)
			}
		}, time.Minute*1)
	}()

	go func() {
		defer util.HandleCrash()
		// TODO: Monitor this connection, reconnect if needed?
		glog.V(1).Infof("Trying to create cadvisor client.")
		cadvisorClient, err := cadvisor.NewClient("http://127.0.0.1:4194")
		if err != nil {
			glog.Errorf("Error on creating cadvisor client: %v", err)
			return
		}
		glog.V(1).Infof("Successfully created cadvisor client.")
		k.SetCadvisorClient(cadvisorClient)
	}()

	// TODO: These should probably become more plugin-ish: register a factory func
	// in each checker's init(), iterate those here.
	health.AddHealthChecker(health.NewExecHealthChecker(k))
	health.AddHealthChecker(health.NewHTTPHealthChecker(&http.Client{}))
	health.AddHealthChecker(&health.TCPHealthChecker{})

	// process pods and exit.
	if *runonce {
		if _, err := k.RunOnce(cfg.Updates()); err != nil {
			glog.Fatalf("--runonce failed: %v", err)
		}
		return
	}

	// start the kubelet
	go util.Forever(func() { k.Run(cfg.Updates()) }, 0)

	// start the kubelet server
	if *enableServer {
		go util.Forever(func() {
			kubelet.ListenAndServeKubeletServer(k, cfg.Channel("http"), net.IP(address), *port, *enableDebuggingHandlers)
		}, 0)
	}

	// runs forever
	select {}
}
