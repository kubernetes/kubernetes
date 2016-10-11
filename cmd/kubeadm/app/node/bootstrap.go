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
	"fmt"
	"os"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	certclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/types"
)

// ConnectionDetails represents a master API endpoint connection
type ConnectionDetails struct {
	CertClient *certclient.CertificatesClient
	Endpoint   string
	CACert     []byte
	NodeName   types.NodeName
}

// EstablishMasterConnection establishes a connection with exactly one of the provided API endpoints or errors.
func EstablishMasterConnection(s *kubeadmapi.NodeConfiguration, clusterInfo *kubeadmapi.ClusterInfo) (*ConnectionDetails, error) {
	hostName, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("<node/bootstrap> failed to get node hostname [%v]", err)
	}
	// TODO(phase1+) https://github.com/kubernetes/kubernetes/issues/33641
	nodeName := types.NodeName(hostName)

	endpoints := clusterInfo.Endpoints
	caCert := []byte(clusterInfo.CertificateAuthorities[0])

	var establishedConnection *ConnectionDetails
	// TODO: add a wait mechanism for the API endpoints (retrying to connect to at least one)
	for _, endpoint := range endpoints {
		clientSet, err := createClients(caCert, endpoint, s.Secrets.BearerToken, nodeName)
		if err != nil {
			fmt.Printf("<node/bootstrap> warning: %s. Skipping endpoint %s\n", err, endpoint)
			continue
		}
		fmt.Printf("<node/bootstrap> trying to connect to endpoint %s\n", endpoint)

		// TODO: add a simple GET /version request to fail early if needed before attempting
		// to connect with a discovery client.
		if err := checkCertsAPI(clientSet.DiscoveryClient); err != nil {
			fmt.Printf("<node/bootstrap> warning: failed to connect to %s: %v\n", endpoint, err)
			continue
		}

		fmt.Printf("<node/bootstrap> successfully established connection with endpoint %s\n", endpoint)
		// connection established
		establishedConnection = &ConnectionDetails{
			CertClient: clientSet.CertificatesClient,
			Endpoint:   endpoint,
			CACert:     caCert,
			NodeName:   nodeName,
		}
		break
	}

	if establishedConnection == nil {
		return nil, fmt.Errorf("<node/bootstrap> failed to create bootstrap clients " +
			"for any of the provided API endpoints")
	}
	return establishedConnection, nil
}

// Creates a set of clients for this endpoint
func createClients(caCert []byte, endpoint, token string, nodeName types.NodeName) (*clientset.Clientset, error) {
	bareClientConfig := kubeadmutil.CreateBasicClientConfig("kubernetes", endpoint, caCert)
	bootstrapClientConfig, err := clientcmd.NewDefaultClientConfig(
		*kubeadmutil.MakeClientConfigWithToken(
			bareClientConfig, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName), token,
		),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create API client configuration [%v]", err)
	}
	clientSet, err := clientset.NewForConfig(bootstrapClientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create clients for the API endpoint %s [%v]", endpoint, err)
	}
	return clientSet, nil
}
