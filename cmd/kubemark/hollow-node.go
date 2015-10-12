/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"fmt"
	"runtime"
	"time"

	docker "github.com/fsouza/go-dockerclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubemark"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

type HollowNodeConfig struct {
	KubeconfigPath      string
	KubeletPort         int
	KubeletReadOnlyPort int
	NodeName            string
	ServerPort          int
}

func (c *HollowNodeConfig) addFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.KubeconfigPath, "kubeconfig", "/kubeconfig/kubeconfig", "Path to kubeconfig file.")
	fs.IntVar(&c.KubeletPort, "kubelet-port", 10250, "Port on which HollowKubelet should be listening.")
	fs.IntVar(&c.KubeletReadOnlyPort, "kubelet-read-only-port", 10255, "Read-only port on which Kubelet is listening.")
	fs.StringVar(&c.NodeName, "name", "fake-node", "Name of this Hollow Node.")
	fs.IntVar(&c.ServerPort, "api-server-port", 443, "Port on which API server is listening.")
}

func createClientFromFile(path string) (*client.Client, error) {
	c, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("error while loading kubeconfig from file %v: %v", path, err)
	}
	config, err := clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("error while creating kubeconfig: %v", err)
	}
	client, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error while creating client: %v", err)
	}
	if client.Timeout == 0 {
		client.Timeout = 30 * time.Second
	}
	return client, nil
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	config := HollowNodeConfig{}
	config.addFlags(pflag.CommandLine)
	util.InitFlags()

	// create a client for Kubelet to communicate with API server.
	cl, err := createClientFromFile(config.KubeconfigPath)
	if err != nil {
		glog.Fatal("Failed to create a Client. Exiting.")
	}
	cadvisorInterface := new(cadvisor.Fake)

	fakeDockerClient := &dockertools.FakeDockerClient{}
	fakeDockerClient.VersionInfo = docker.Env{"ApiVersion=1.18"}
	fakeDockerClient.ContainerMap = make(map[string]*docker.Container)
	fakeDockerClient.EnableSleep = true

	hollowKubelet := kubemark.NewHollowKubelet(
		config.NodeName,
		cl,
		cadvisorInterface,
		fakeDockerClient,
		config.KubeletPort,
		config.KubeletReadOnlyPort,
	)
	hollowKubelet.Run()
}
