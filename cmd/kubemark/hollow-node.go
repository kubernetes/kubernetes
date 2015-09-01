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
	"io/ioutil"
	"os"
	"runtime"
	"time"

	docker "github.com/fsouza/go-dockerclient"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/volume/empty_dir"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

var (
	fakeDockerClient dockertools.FakeDockerClient

	apiServer           string
	kubeconfigPath      string
	kubeletPort         int
	kubeletReadOnlyPort int
	nodeName            string
	serverPort          int
)

func addFlags(fs *pflag.FlagSet) {
	fs.StringVar(&apiServer, "server", "", "API server IP.")
	fs.StringVar(&kubeconfigPath, "kubeconfig", "/kubeconfig/kubeconfig", "Path to kubeconfig file.")
	fs.IntVar(&kubeletPort, "kubelet-port", 10250, "Port on which HollowKubelet should be listening.")
	fs.IntVar(&kubeletReadOnlyPort, "kubelet-read-only-port", 10255, "Read-only port on which Kubelet is listening.")
	fs.StringVar(&nodeName, "name", "fake-node", "Name of this Hollow Node.")
	fs.IntVar(&serverPort, "api-server-port", 443, "Port on which API server is listening.")
}

func makeTempDirOrDie(prefix string, baseDir string) string {
	if baseDir == "" {
		baseDir = "/tmp"
	}
	tempDir, err := ioutil.TempDir(baseDir, prefix)
	if err != nil {
		glog.Fatalf("Can't make a temp rootdir: %v", err)
	}
	if err = os.MkdirAll(tempDir, 0750); err != nil {
		glog.Fatalf("Can't mkdir(%q): %v", tempDir, err)
	}
	return tempDir
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
	addFlags(pflag.CommandLine)
	util.InitFlags()

	// create a client for Kubelet to communicate with API server.
	cl, err := createClientFromFile(kubeconfigPath)
	if err != nil {
		glog.Fatal("Failed to create a Client. Exiting.")
	}
	cadvisorInterface := new(cadvisor.Fake)

	testRootDir := makeTempDirOrDie("hollow-kubelet.", "")
	configFilePath := makeTempDirOrDie("config", testRootDir)
	glog.Infof("Using %s as root dir for hollow-kubelet", testRootDir)
	fakeDockerClient.VersionInfo = docker.Env{"ApiVersion=1.18"}
	kcfg := kubeletapp.SimpleKubelet(
		cl,
		&fakeDockerClient,
		nodeName,
		testRootDir,
		"",        /* manifest-url */
		"0.0.0.0", /* bind address */
		uint(kubeletPort),
		uint(kubeletReadOnlyPort),
		api.NamespaceDefault,
		empty_dir.ProbeVolumePlugins(),
		nil, /* tls-options */
		cadvisorInterface,
		configFilePath,
		nil, /* cloud-provider */
		kubecontainer.FakeOS{}, /* os-interface */
		20*time.Second,         /* FileCheckFrequency */
		20*time.Second,         /* HTTPCheckFrequency */
		1*time.Minute,          /* MinimumGCAge */
		10*time.Second,         /* NodeStatusUpdateFrequency */
		10*time.Second,         /* SyncFrequency */
		40,                     /* MaxPods */
	)
	kubeletapp.RunKubelet(kcfg, nil)

	select {}
}
