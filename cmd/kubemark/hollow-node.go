/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/flag"
	clientgoclientset "k8s.io/client-go/kubernetes"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	_ "k8s.io/kubernetes/pkg/client/metrics/prometheus" // for client metric registration
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubemark"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	fakeiptables "k8s.io/kubernetes/pkg/util/iptables/testing"
	_ "k8s.io/kubernetes/pkg/version/prometheus" // for version metric registration

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

type HollowNodeConfig struct {
	KubeconfigPath      string
	KubeletPort         int
	KubeletReadOnlyPort int
	Morph               string
	NodeName            string
	ServerPort          int
	ContentType         string
}

const (
	maxPods     = 110
	podsPerCore = 0
)

var knownMorphs = sets.NewString("kubelet", "proxy")

func (c *HollowNodeConfig) addFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.KubeconfigPath, "kubeconfig", "/kubeconfig/kubeconfig", "Path to kubeconfig file.")
	fs.IntVar(&c.KubeletPort, "kubelet-port", 10250, "Port on which HollowKubelet should be listening.")
	fs.IntVar(&c.KubeletReadOnlyPort, "kubelet-read-only-port", 10255, "Read-only port on which Kubelet is listening.")
	fs.StringVar(&c.NodeName, "name", "fake-node", "Name of this Hollow Node.")
	fs.IntVar(&c.ServerPort, "api-server-port", 443, "Port on which API server is listening.")
	fs.StringVar(&c.Morph, "morph", "", fmt.Sprintf("Specifies into which Hollow component this binary should morph. Allowed values: %v", knownMorphs.List()))
	fs.StringVar(&c.ContentType, "kube-api-content-type", "application/vnd.kubernetes.protobuf", "ContentType of requests sent to apiserver.")
}

func (c *HollowNodeConfig) createClientConfigFromFile() (*restclient.Config, error) {
	clientConfig, err := clientcmd.LoadFromFile(c.KubeconfigPath)
	if err != nil {
		return nil, fmt.Errorf("error while loading kubeconfig from file %v: %v", c.KubeconfigPath, err)
	}
	config, err := clientcmd.NewDefaultClientConfig(*clientConfig, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("error while creating kubeconfig: %v", err)
	}
	config.ContentType = c.ContentType
	config.QPS = 10
	config.Burst = 20
	return config, nil
}

func main() {
	config := HollowNodeConfig{}
	config.addFlags(pflag.CommandLine)
	flag.InitFlags()

	if !knownMorphs.Has(config.Morph) {
		glog.Fatalf("Unknown morph: %v. Allowed values: %v", config.Morph, knownMorphs.List())
	}

	// create a client to communicate with API server.
	clientConfig, err := config.createClientConfigFromFile()
	if err != nil {
		glog.Fatalf("Failed to create a ClientConfig: %v. Exiting.", err)
	}

	clientset, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		glog.Fatalf("Failed to create a ClientSet: %v. Exiting.", err)
	}
	internalClientset, err := internalclientset.NewForConfig(clientConfig)
	if err != nil {
		glog.Fatalf("Failed to create an internal ClientSet: %v. Exiting.", err)
	}

	if config.Morph == "kubelet" {
		cadvisorInterface := new(cadvisortest.Fake)
		containerManager := cm.NewStubContainerManager()

		fakeDockerClient := dockertools.NewFakeDockerClient().WithTraceDisabled()
		fakeDockerClient.EnableSleep = true

		hollowKubelet := kubemark.NewHollowKubelet(
			config.NodeName,
			clientset,
			cadvisorInterface,
			fakeDockerClient,
			config.KubeletPort,
			config.KubeletReadOnlyPort,
			containerManager,
			maxPods,
			podsPerCore,
		)
		hollowKubelet.Run()
	}

	if config.Morph == "proxy" {
		eventBroadcaster := record.NewBroadcaster()
		recorder := eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "kube-proxy", Host: config.NodeName})

		iptInterface := fakeiptables.NewFake()

		serviceConfig := proxyconfig.NewServiceConfig(internalClientset.Core().RESTClient(), 15*time.Minute)
		serviceConfig.RegisterHandler(&kubemark.FakeProxyHandler{})

		endpointsConfig := proxyconfig.NewEndpointsConfig(internalClientset.Core().RESTClient(), 15*time.Minute)
		endpointsConfig.RegisterHandler(&kubemark.FakeProxyHandler{})

		eventClient, err := clientgoclientset.NewForConfig(clientConfig)
		if err != nil {
			glog.Fatalf("Failed to create API Server client: %v", err)
		}

		hollowProxy := kubemark.NewHollowProxyOrDie(
			config.NodeName,
			internalClientset,
			eventClient,
			endpointsConfig,
			serviceConfig,
			iptInterface,
			eventBroadcaster,
			recorder,
		)
		hollowProxy.Run()
	}
}
