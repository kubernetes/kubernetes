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

package kubelet

import (
	"net/url"
	"strconv"

	"github.com/golang/glog"
	kube_config "k8s.io/heapster/common/kubernetes"
	kube_client "k8s.io/kubernetes/pkg/client/restclient"
	kubelet_client "k8s.io/kubernetes/pkg/kubelet/client"
)

const (
	APIVersion = "v1"

	defaultKubeletPort        = 10255
	defaultKubeletHttps       = false
	defaultUseServiceAccount  = false
	defaultServiceAccountFile = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	defaultInClusterConfig    = true
)

func GetKubeConfigs(uri *url.URL) (*kube_client.Config, *kubelet_client.KubeletClientConfig, error) {

	kubeConfig, err := kube_config.GetKubeClientConfig(uri)
	if err != nil {
		return nil, nil, err
	}
	opts := uri.Query()

	kubeletPort := defaultKubeletPort
	if len(opts["kubeletPort"]) >= 1 {
		kubeletPort, err = strconv.Atoi(opts["kubeletPort"][0])
		if err != nil {
			return nil, nil, err
		}
	}

	kubeletHttps := defaultKubeletHttps
	if len(opts["kubeletHttps"]) >= 1 {
		kubeletHttps, err = strconv.ParseBool(opts["kubeletHttps"][0])
		if err != nil {
			return nil, nil, err
		}
	}
	glog.Infof("Using Kubernetes client with master %q and version %+v\n", kubeConfig.Host, kubeConfig.GroupVersion)
	glog.Infof("Using kubelet port %d", kubeletPort)

	kubeletConfig := &kubelet_client.KubeletClientConfig{
		Port:            uint(kubeletPort),
		EnableHttps:     kubeletHttps,
		TLSClientConfig: kubeConfig.TLSClientConfig,
		BearerToken:     kubeConfig.BearerToken,
	}

	return kubeConfig, kubeletConfig, nil
}
