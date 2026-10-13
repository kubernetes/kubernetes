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

package https

import (
	"io"
	"net/http"
	"time"

	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/file"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// RetrieveValidatedConfigInfo downloads a discovery kubeconfig from the given
// HTTPS URL and hands it to file.ValidateConfigInfo for the cluster-info
// ConfigMap validation that completes discovery. The HTTPS connection itself
// is verified only against the host's default TLS trust store; kubeadm does
// not pin to a caller-supplied CA at this stage, so the kubeconfig payload is
// retrieved from an effectively arbitrary location and only becomes trusted
// after file.ValidateConfigInfo succeeds.
func RetrieveValidatedConfigInfo(httpsURL string, discoveryTimeout time.Duration) (*clientcmdapi.Config, error) {
	client := &http.Client{Transport: netutil.SetOldTransportDefaults(&http.Transport{})}
	response, err := client.Get(httpsURL)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		return nil, errors.Errorf("error trying to fetch discovery kubeconfig over HTTPS from %s, received status %d", httpsURL, response.StatusCode)
	}

	kubeconfig, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}

	config, err := clientcmd.Load(kubeconfig)
	if err != nil {
		return nil, err
	}
	return file.ValidateConfigInfo(config, discoveryTimeout)
}
