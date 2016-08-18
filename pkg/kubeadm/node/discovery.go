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
	firstURL := strings.Split(params.Discovery.ApiServerURLs, ",")[0] // TODO obviously we should do something better.. .
	apiServerURL, err := url.Parse(firstURL)
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to parse given API server URL (%q) [%s]", firstURL, err)
	}

	host, port := strings.Split(apiServerURL.Host, ":")[0], 9898 // TODO this is too naive
	requestURL := fmt.Sprintf("http://%s:%d/cluster-info/v1/?token-id=%s", host, port, params.Discovery.TokenID)
	req, err := http.NewRequest("GET", requestURL, nil)
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to consturct an HTTP request [%s]", err)
	}

	fmt.Println("<node/discovery> created cluster info discovery client, requesting info from %q", requestURL)

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to request cluster info [%s]", err)
	}
	buf := new(bytes.Buffer)
	io.Copy(buf, res.Body)
	res.Body.Close()

	object, err := jose.ParseSigned(buf.String())
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to parse response as JWS object [%s]", err)
	}

	fmt.Println("<node/discovery> cluster info object recieved, verifying signature using given token")

	output, err := object.Verify(params.Discovery.Token)
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to verify JWS signature of recieved cluster info object [%s]", err)
	}

	clusterInfo := kubeadmapi.ClusterInfo{}

	if err := json.Unmarshal(output, &clusterInfo); err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to unmarshal recieved cluster info object [%s]", err)
	}

	if len(clusterInfo.CertificateAuthorities) == 0 || len(clusterInfo.Endpoints) == 0 {
		return nil, fmt.Errorf("<node/discovery> cluster info object is invalid - no endpoint(s) and/or root CA certificate(s) found")
	}

	// TODO print checksum of the CA certificate
	fmt.Printf("<node/discovery> cluser info signature and contents are valid, will use API endpoints %v\n", clusterInfo.Endpoints)

	// TODO we need to configure the client to validate the server
	// if it is signed by any of the returned certificates
	apiServer := clusterInfo.Endpoints[0]
	caCert := []byte(clusterInfo.CertificateAuthorities[0])

	return PerformTLSBootstrap(params, apiServer, caCert)
}
