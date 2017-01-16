/*
Copyright 2016 The Kubernetes Authors.

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

package discovery

import (
	"fmt"
	"io/ioutil"
	"net/http"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubenode "k8s.io/kubernetes/cmd/kubeadm/app/node"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

// For identifies and executes the desired discovery mechanism.
func For(d kubeadmapi.Discovery) (*clientcmdapi.Config, error) {
	switch {
	case d.File != nil:
		return runFileDiscovery(d.File)
	case d.HTTPS != nil:
		return runHTTPSDiscovery(d.HTTPS)
	case d.Token != nil:
		return runTokenDiscovery(d.Token)
	default:
		return nil, fmt.Errorf("couldn't find a valid discovery configuration.")
	}
}

// runFileDiscovery executes file-based discovery.
func runFileDiscovery(fd *kubeadmapi.FileDiscovery) (*clientcmdapi.Config, error) {
	return clientcmd.LoadFromFile(fd.Path)
}

// runHTTPSDiscovery executes HTTPS-based discovery.
func runHTTPSDiscovery(hd *kubeadmapi.HTTPSDiscovery) (*clientcmdapi.Config, error) {
	response, err := http.Get(hd.URL)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	kubeconfig, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}

	return clientcmd.Load(kubeconfig)
}

// runTokenDiscovery executes token-based discovery.
func runTokenDiscovery(td *kubeadmapi.TokenDiscovery) (*clientcmdapi.Config, error) {
	if valid, err := kubeadmutil.ValidateToken(td); valid == false {
		return nil, err
	}

	clusterInfo, err := kubenode.RetrieveTrustedClusterInfo(td)
	if err != nil {
		return nil, err
	}

	cfg, err := kubenode.EstablishMasterConnection(td, clusterInfo)
	if err != nil {
		return nil, err
	}
	return cfg, nil
}
