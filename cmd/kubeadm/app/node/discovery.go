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

package node

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	jose "github.com/square/go-jose"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/util/wait"
)

// the amount of time to wait between each request to the discovery API
const discoveryRetryTimeout = 5 * time.Second

func RetrieveTrustedClusterInfo(s *kubeadmapi.NodeConfiguration) (*kubeadmapi.ClusterInfo, error) {
	host, port := s.MasterAddresses[0], s.DiscoveryPort
	requestURL := fmt.Sprintf("http://%s:%d/cluster-info/v1/?token-id=%s", host, port, s.Secrets.TokenID)
	req, err := http.NewRequest("GET", requestURL, nil)
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to consturct an HTTP request [%v]", err)
	}

	fmt.Printf("<node/discovery> created cluster info discovery client, requesting info from %q\n", requestURL)

	var res *http.Response
	wait.PollInfinite(discoveryRetryTimeout, func() (bool, error) {
		res, err = http.DefaultClient.Do(req)
		if err != nil {
			fmt.Printf("<node/discovery> failed to request cluster info, will try again: [%s]\n", err)
			return false, nil
		}
		return true, nil
	})

	buf := new(bytes.Buffer)
	io.Copy(buf, res.Body)
	res.Body.Close()

	object, err := jose.ParseSigned(buf.String())
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to parse response as JWS object [%v]", err)
	}

	fmt.Println("<node/discovery> cluster info object received, verifying signature using given token")

	output, err := object.Verify(s.Secrets.Token)
	if err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to verify JWS signature of received cluster info object [%v]", err)
	}

	clusterInfo := kubeadmapi.ClusterInfo{}

	if err := json.Unmarshal(output, &clusterInfo); err != nil {
		return nil, fmt.Errorf("<node/discovery> failed to decode received cluster info object [%v]", err)
	}

	if len(clusterInfo.CertificateAuthorities) == 0 || len(clusterInfo.Endpoints) == 0 {
		return nil, fmt.Errorf("<node/discovery> cluster info object is invalid - no endpoint(s) and/or root CA certificate(s) found")
	}

	// TODO(phase1+) print summary info about the CA certificate, along with the the checksum signature
	// we also need an ability for the user to configure the client to validate received CA cert against a checksum
	fmt.Printf("<node/discovery> cluster info signature and contents are valid, will use API endpoints %v\n", clusterInfo.Endpoints)
	return &clusterInfo, nil
}
