// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sources

import (
	"fmt"
	"io/ioutil"
	"net/url"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/heapster/extpoints"
	"k8s.io/heapster/sinks/cache"
	"k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/datasource"
	"k8s.io/heapster/sources/nodes"
	kube_client "k8s.io/kubernetes/pkg/client/unversioned"
	kubeClientCmd "k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	kubeClientCmdApi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

const (
	APIVersion = "v1"

	defaultKubeletPort        = 10255
	defaultKubeletHttps       = false
	defaultUseServiceAccount  = false
	defaultServiceAccountFile = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	defaultInClusterConfig    = true
)

func init() {
	extpoints.SourceFactories.Register(CreateKubeSources, "kubernetes")
}

func getConfigOverrides(uri *url.URL) (*kubeClientCmd.ConfigOverrides, error) {
	kubeConfigOverride := kubeClientCmd.ConfigOverrides{
		ClusterInfo: kubeClientCmdApi.Cluster{
			APIVersion: APIVersion,
		},
	}
	if len(uri.Scheme) != 0 && len(uri.Host) != 0 {
		kubeConfigOverride.ClusterInfo.Server = fmt.Sprintf("%s://%s", uri.Scheme, uri.Host)
	}

	opts := uri.Query()

	if len(opts["apiVersion"]) >= 1 {
		kubeConfigOverride.ClusterInfo.APIVersion = opts["apiVersion"][0]
	}

	if len(opts["insecure"]) > 0 {
		insecure, err := strconv.ParseBool(opts["insecure"][0])
		if err != nil {
			return nil, err
		}
		kubeConfigOverride.ClusterInfo.InsecureSkipTLSVerify = insecure
	}

	return &kubeConfigOverride, nil
}

func CreateKubeSources(uri *url.URL, c cache.Cache) ([]api.Source, error) {
	var (
		kubeConfig *kube_client.Config
		err        error
	)

	opts := uri.Query()
	configOverrides, err := getConfigOverrides(uri)
	if err != nil {
		return nil, err
	}

	inClusterConfig := defaultInClusterConfig
	if len(opts["inClusterConfig"]) > 0 {
		inClusterConfig, err = strconv.ParseBool(opts["inClusterConfig"][0])
		if err != nil {
			return nil, err
		}
	}

	if inClusterConfig {
		kubeConfig, err = kube_client.InClusterConfig()
		if err != nil {
			return nil, err
		}

		if configOverrides.ClusterInfo.Server != "" {
			kubeConfig.Host = configOverrides.ClusterInfo.Server
		}
		kubeConfig.Version = configOverrides.ClusterInfo.APIVersion
		kubeConfig.Insecure = configOverrides.ClusterInfo.InsecureSkipTLSVerify
		if configOverrides.ClusterInfo.InsecureSkipTLSVerify {
			kubeConfig.TLSClientConfig.CAFile = ""
		}
	} else {
		authFile := ""
		if len(opts["auth"]) > 0 {
			authFile = opts["auth"][0]
		}

		if authFile != "" {
			if kubeConfig, err = kubeClientCmd.NewNonInteractiveDeferredLoadingClientConfig(
				&kubeClientCmd.ClientConfigLoadingRules{ExplicitPath: authFile},
				configOverrides).ClientConfig(); err != nil {
				return nil, err
			}
		} else {
			kubeConfig = &kube_client.Config{
				Host:     configOverrides.ClusterInfo.Server,
				Version:  configOverrides.ClusterInfo.APIVersion,
				Insecure: configOverrides.ClusterInfo.InsecureSkipTLSVerify,
			}
		}
	}
	if len(kubeConfig.Host) == 0 {
		return nil, fmt.Errorf("invalid kubernetes master url specified")
	}
	if len(kubeConfig.Version) == 0 {
		return nil, fmt.Errorf("invalid kubernetes API version specified")
	}

	useServiceAccount := defaultUseServiceAccount
	if len(opts["useServiceAccount"]) >= 1 {
		useServiceAccount, err = strconv.ParseBool(opts["useServiceAccount"][0])
		if err != nil {
			return nil, err
		}
	}

	if useServiceAccount {
		// If a readable service account token exists, then use it
		if contents, err := ioutil.ReadFile(defaultServiceAccountFile); err == nil {
			kubeConfig.BearerToken = string(contents)
		}
	}

	kubeClient := kube_client.NewOrDie(kubeConfig)

	nodesApi, err := nodes.NewKubeNodes(kubeClient)
	if err != nil {
		return nil, err
	}
	kubeletPort := defaultKubeletPort
	if len(opts["kubeletPort"]) >= 1 {
		kubeletPort, err = strconv.Atoi(opts["kubeletPort"][0])
		if err != nil {
			return nil, err
		}
	}

	kubeletHttps := defaultKubeletHttps
	if len(opts["kubeletHttps"]) >= 1 {
		kubeletHttps, err = strconv.ParseBool(opts["kubeletHttps"][0])
		if err != nil {
			return nil, err
		}
	}
	glog.Infof("Using Kubernetes client with master %q and version %q\n", kubeConfig.Host, kubeConfig.Version)
	glog.Infof("Using kubelet port %d", kubeletPort)

	kubeletConfig := &kube_client.KubeletConfig{
		Port:            uint(kubeletPort),
		EnableHttps:     kubeletHttps,
		TLSClientConfig: kubeConfig.TLSClientConfig,
	}

	kubeletApi, err := datasource.NewKubelet(kubeletConfig)
	if err != nil {
		return nil, err
	}

	kubePodsSource := NewKubePodMetrics(kubeletPort, kubeletApi, nodesApi, newPodsApi(kubeClient))
	kubeNodeSource := NewKubeNodeMetrics(kubeletPort, kubeletApi, nodesApi)
	kubeEventsSource := NewKubeEvents(kubeClient, c)

	return []api.Source{kubePodsSource, kubeNodeSource, kubeEventsSource}, nil
}
