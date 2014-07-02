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
	"os"
	"os/exec"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

var (
	config             = flag.String("config", "", "Path to the config file or directory of files")
	etcdServers        = flag.String("etcd_servers", "", "Url of etcd servers in the cluster")
	syncFrequency      = flag.Duration("sync_frequency", 10*time.Second, "Max period between synchronizing running containers and config")
	fileCheckFrequency = flag.Duration("file_check_frequency", 20*time.Second, "Duration between checking config files for new data")
	httpCheckFrequency = flag.Duration("http_check_frequency", 20*time.Second, "Duration between checking http for new data")
	manifestUrl        = flag.String("manifest_url", "", "URL for accessing the container manifest")
	address            = flag.String("address", "127.0.0.1", "The address for the info server to serve on")
	port               = flag.Uint("port", 10250, "The port for the info server to serve on")
	hostnameOverride   = flag.String("hostname_override", "", "If non-empty, will use this string as identification instead of the actual hostname.")
	dockerEndpoint     = flag.String("docker_endpoint", "", "If non-empty, use this for the docker endpoint to communicate with")
)

const dockerBinary = "/usr/bin/docker"

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()
	rand.Seed(time.Now().UTC().UnixNano())

	// Set up logger for etcd client
	etcd.SetLogger(util.NewLogger("etcd "))

	var endpoint string
	if len(*dockerEndpoint) > 0 {
		endpoint = *dockerEndpoint
	} else if len(os.Getenv("DOCKER_HOST")) > 0 {
		endpoint = os.Getenv("DOCKER_HOST")
	} else {
		endpoint = "unix:///var/run/docker.sock"
	}
	glog.Infof("Connecting to docker on %s", endpoint)
	dockerClient, err := docker.NewClient(endpoint)
	if err != nil {
		glog.Fatal("Couldn't connnect to docker.")
	}

	hostname := []byte(*hostnameOverride)
	if string(hostname) == "" {
		// Note: We use exec here instead of os.Hostname() because we
		// want the FQDN, and this is the easiest way to get it.
		hostname, err = exec.Command("hostname", "-f").Output()
		if err != nil {
			glog.Fatalf("Couldn't determine hostname: %v", err)
		}
	}

	my_kubelet := kubelet.Kubelet{
		Hostname:           string(hostname),
		DockerClient:       dockerClient,
		FileCheckFrequency: *fileCheckFrequency,
		SyncFrequency:      *syncFrequency,
		HTTPCheckFrequency: *httpCheckFrequency,
	}
	my_kubelet.RunKubelet(*dockerEndpoint, *config, *manifestUrl, *etcdServers, *address, *port)
}
