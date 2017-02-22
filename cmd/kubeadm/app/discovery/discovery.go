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
	"strings"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/file"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/https"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/token"
	kubenode "k8s.io/kubernetes/cmd/kubeadm/app/node"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
)

func Check(discURL, discFile, discToken string) (string, error) {
	count := 0
	disc := ""
	if discURL != "" {
		disc = discURL
		count++
	}
	if discFile != "" {
		disc = discFile
		count++
	}
	if discToken != "" {
		disc = discToken
		count++
	}
	if count > 1 {
		return "", fmt.Errorf("too many discovery options chosen: count %d", count)
	}

	return disc, nil
}

func Assign(d *kubeadmapi.Discovery, disc string) error {
	u, err := url.Parse(disc)
	if err != nil {
		return err
	}
	switch u.Scheme {
	case "https":
		https.Parse(u, d)
	case "file":
		file.Parse(u, d)
	case "token":
		// Make sure a valid RFC 3986 URL has been passed and parsed.
		// See https://github.com/kubernetes/kubeadm/issues/95#issuecomment-270431296 for more details.
		if !strings.Contains(disc, "@") {
			disc := disc + "@"
			u, err = url.Parse(disc)
			if err != nil {
				return err
			}
		}
		token.Parse(u, d)
	default:
		return fmt.Errorf("unknown discovery scheme")
	}
	return nil
}

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
	if valid, err := tokenutil.ValidateToken(td); valid == false {
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
