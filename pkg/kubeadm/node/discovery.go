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
package kubenode

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/square/go-jose"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
)

func RetrieveTrustedClusterInfo(params *kubeadmapi.BootstrapParams) (*clientcmdapi.Config, error) {
	apiServerURL, err := url.Parse(strings.Split(params.Discovery.ApiServerURLs, ",")[0])
	if err != nil {
		return nil, err
	}

	host, port := strings.Split(apiServerURL.Host, ":")[0], 9898

	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/cluster-info/v1/?token-id=%s", host, port, params.Discovery.TokenID), nil)
	if err != nil {
		return nil, err
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	buf := new(bytes.Buffer)
	io.Copy(buf, res.Body)
	res.Body.Close()

	object, err := jose.ParseSigned(buf.String())
	if err != nil {
		return nil, err
	}

	output, err := object.Verify(params.Discovery.Token)
	if err != nil {
		return nil, err
	}

	clusterInfo := kubeadmapi.ClusterInfo{}

	if err := json.Unmarshal(output, &clusterInfo); err != nil {
		return nil, err
	}

	fmt.Printf("ClusterInfo: %#v", clusterInfo)

	if len(clusterInfo.CertificateAuthorities) == 0 || len(clusterInfo.Endpoints) == 0 {
		return nil, fmt.Errorf("Cluster discovery object (ClusterInfo) is invalid")
	}

	// TODO figure out what we should do when there is a chain of certificates and more then one API endpoint
	apiServer := clusterInfo.Endpoints[0]
	caCert := []byte(clusterInfo.CertificateAuthorities[0])

	return PerformTLSBootstrap(params, apiServer, caCert)
}
