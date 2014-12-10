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

package kubelet

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	cadvisor "github.com/google/cadvisor/client"
)

// TODO: move this into a pkg/util
func GetHostname(hostnameOverride string) string {
	hostname := []byte(hostnameOverride)
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

// TODO: move this into a pkg/util
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

// TODO: move this into pkg/util
func ConnectToDockerOrDie(dockerEndpoint string) *docker.Client {
	client, err := docker.NewClient(GetDockerEndpoint(dockerEndpoint))
	if err != nil {
		glog.Fatal("Couldn't connect to docker.")
	}
	return client
}

// TODO: move this into the kubelet itself
func GarbageCollectLoop(k *Kubelet) {
	func() {
		util.Forever(func() {
			err := k.GarbageCollectContainers()
			if err != nil {
				glog.Errorf("Garbage collect failed: %v", err)
			}
		}, time.Minute*1)
	}()
}

// TODO: move this into the kubelet itself
func MonitorCAdvisor(k *Kubelet, cp uint) {
	defer util.HandleCrash()
	// TODO: Monitor this connection, reconnect if needed?
	glog.V(1).Infof("Trying to create cadvisor client.")
	cadvisorClient, err := cadvisor.NewClient("http://127.0.0.1:" + strconv.Itoa(int(cp)))
	if err != nil {
		glog.Errorf("Error on creating cadvisor client: %v", err)
		return
	}
	glog.V(1).Infof("Successfully created cadvisor client.")
	k.SetCadvisorClient(cadvisorClient)
}

// TODO: move this into the kubelet itself
func InitHealthChecking(k *Kubelet) {
	// TODO: These should probably become more plugin-ish: register a factory func
	// in each checker's init(), iterate those here.
	health.AddHealthChecker(health.NewExecHealthChecker(k))
	health.AddHealthChecker(health.NewHTTPHealthChecker(&http.Client{}))
	health.AddHealthChecker(&health.TCPHealthChecker{})
}

// TODO: move this into a pkg/tools/etcd_tools
func EtcdClientOrDie(etcdServerList util.StringList, etcdConfigFile string) *etcd.Client {
	if len(etcdServerList) > 0 {
		return etcd.NewClient(etcdServerList)
	} else if etcdConfigFile != "" {
		etcdClient, err := etcd.NewClientFromFile(etcdConfigFile)
		if err != nil {
			glog.Fatalf("Error with etcd config file: %v", err)
		}
		return etcdClient
	}
	return nil
}

// TODO: move this into pkg/util
func SetupRootDirectoryOrDie(rootDirectory string) {
	if rootDirectory == "" {
		glog.Fatal("Invalid root directory path.")
	}
	rootDirectory = path.Clean(rootDirectory)
	if err := os.MkdirAll(rootDirectory, 0750); err != nil {
		glog.Fatalf("Error creating root directory: %v", err)
	}
}

// TODO: move this into pkg/capabilities
func SetupCapabilities(allowPrivileged bool) {
	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: allowPrivileged,
	})
}

// TODO: Split this up?
func SetupLogging() {
	etcd.SetLogger(util.NewLogger("etcd "))
	// Log the events locally too.
	record.StartLogging(glog.Infof)
}

// TODO: move this into pkg/client
func getApiserverClient(authPath string, apiServerList util.StringList) (*client.Client, error) {
	authInfo, err := clientauth.LoadFromFile(authPath)
	if err != nil {
		return nil, err
	}
	clientConfig, err := authInfo.MergeWithConfig(client.Config{})
	if err != nil {
		return nil, err
	}
	if len(apiServerList) < 1 {
		return nil, fmt.Errorf("no apiservers specified.")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(apiServerList) > 1 {
		glog.Infof("Mulitple api servers specified.  Picking first one")
	}
	clientConfig.Host = apiServerList[0]
	if c, err := client.New(&clientConfig); err != nil {
		return nil, err
	} else {
		return c, nil
	}
}

func SetupEventSending(authPath string, apiServerList util.StringList) {
	// Make an API client if possible.
	if len(apiServerList) < 1 {
		glog.Info("No api servers specified.")
	} else {
		if apiClient, err := getApiserverClient(authPath, apiServerList); err != nil {
			glog.Errorf("Unable to make apiserver client: %v", err)
		} else {
			// Send events to APIserver if there is a client.
			glog.Infof("Sending events to APIserver.")
			record.StartRecording(apiClient.Events(""), "kubelet")
		}
	}
}
