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
	"net/url"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubenode "k8s.io/kubernetes/cmd/kubeadm/app/node"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
)

// For identifies and executes the desired discovery mechanism.
func For(d *kubeadmapi.NodeConfiguration) (*clientcmdapi.Config, error) {
	switch {
	case len(d.DiscoveryFile) != 0:
		if isHTTPSURL(d.DiscoveryFile) {
			return runHTTPSDiscovery(d.DiscoveryFile)
		}
		return runFileDiscovery(d.DiscoveryFile)
	case len(d.DiscoveryToken) != 0:
		return runTokenDiscovery(d.DiscoveryToken, d.DiscoveryTokenAPIServers)
	default:
		return nil, fmt.Errorf("couldn't find a valid discovery configuration.")
	}
}

// isHTTPSURL checks whether the string is parsable as an URL
func isHTTPSURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && u.Scheme == "https"
}

// runFileDiscovery executes file-based discovery.
func runFileDiscovery(fd string) (*clientcmdapi.Config, error) {
	return clientcmd.LoadFromFile(fd)
}

// runHTTPSDiscovery executes HTTPS-based discovery.
func runHTTPSDiscovery(hd string) (*clientcmdapi.Config, error) {
	response, err := http.Get(hd)
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
func runTokenDiscovery(td string, m []string) (*clientcmdapi.Config, error) {
	id, secret, err := tokenutil.ParseToken(td)
	if err != nil {
		return nil, err
	}
	t := &kubeadmapi.TokenDiscovery{ID: id, Secret: secret, Addresses: m}

	if valid, err := tokenutil.ValidateToken(t); valid == false {
		return nil, err
	}

	clusterInfo, err := kubenode.RetrieveTrustedClusterInfo(t)
	if err != nil {
		return nil, err
	}

	cfg, err := kubenode.EstablishMasterConnection(t, clusterInfo)
	if err != nil {
		return nil, err
	}
	return cfg, nil
}
