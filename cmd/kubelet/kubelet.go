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
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	kconfig "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	verflag "github.com/GoogleCloudPlatform/kubernetes/pkg/version/flag"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/google/cadvisor/client"
)

const defaultRootDir = "/var/lib/kubelet"

var (
	config             = flag.String("config", "", "Path to the config file or directory of files")
	syncFrequency      = flag.Duration("sync_frequency", 10*time.Second, "Max period between synchronizing running containers and config")
	fileCheckFrequency = flag.Duration("file_check_frequency", 20*time.Second, "Duration between checking config files for new data")
	httpCheckFrequency = flag.Duration("http_check_frequency", 20*time.Second, "Duration between checking http for new data")
	manifestURL        = flag.String("manifest_url", "", "URL for accessing the container manifest")
	enableServer       = flag.Bool("enable_server", true, "Enable the info server")
	address            = flag.String("address", "127.0.0.1", "The address for the info server to serve on (set to 0.0.0.0 or \"\" for all interfaces)")
	port               = flag.Uint("port", 10250, "The port for the info server to serve on")
	hostnameOverride   = flag.String("hostname_override", "", "If non-empty, will use this string as identification instead of the actual hostname.")
	dockerEndpoint     = flag.String("docker_endpoint", "", "If non-empty, use this for the docker endpoint to communicate with")
	etcdServerList     util.StringList
	rootDirectory      = flag.String("root_dir", defaultRootDir, "Directory path for managing kubelet files (volume mounts,etc).")
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated")
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

	etcd.SetLogger(util.NewLogger("etcd "))

	dockerClient, err := docker.NewClient(getDockerEndpoint())
	if err != nil {
		glog.Fatal("Couldn't connect to docker.")
	}

	cadvisorClient, err := cadvisor.NewClient("http://127.0.0.1:4194")
	if err != nil {
		glog.Errorf("Error on creating cadvisor client: %v", err)
	}

	hostname := getHostname()

	if *rootDirectory == "" {
		glog.Fatal("Invalid root directory path.")
	}
	*rootDirectory = path.Clean(*rootDirectory)
	os.MkdirAll(*rootDirectory, 0750)

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
	var etcdClient tools.EtcdClient
	if len(etcdServerList) > 0 {
		glog.Infof("Watching for etcd configs at %v", etcdServerList)
		etcdClient = etcd.NewClient(etcdServerList)
		kconfig.NewSourceEtcd(kconfig.EtcdKeyForHost(hostname), etcdClient, cfg.Channel("etcd"))
	}

	// TODO: block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations

	k := kubelet.NewMainKubelet(
		getHostname(),
		dockerClient,
		cadvisorClient,
		etcdClient,
		*rootDirectory,
		*syncFrequency)

	health.AddHealthChecker("exec", health.NewExecHealthChecker(k))
	health.AddHealthChecker("http", health.NewHTTPHealthChecker(&http.Client{}))
	health.AddHealthChecker("tcp", &health.TCPHealthChecker{})

	// start the kubelet
	go util.Forever(func() { k.Run(cfg.Updates()) }, 0)

	// start the kubelet server
	if *enableServer {
		go util.Forever(func() {
			kubelet.ListenAndServeKubeletServer(k, cfg.Channel("http"), *address, *port)
		}, 0)
	}

	// runs forever
	select {}
}
