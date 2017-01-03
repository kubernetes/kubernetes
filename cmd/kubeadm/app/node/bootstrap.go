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
	"sync"
	"time"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/certificates"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	certclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1alpha1"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
)

// ConnectionDetails represents a master API endpoint connection
type ConnectionDetails struct {
	ClientSet  *clientset.Clientset
	CertClient *certclient.CertificatesV1alpha1Client
	Endpoint   string
	CACert     []byte
	NodeName   types.NodeName
}

// retryTimeout between the subsequent attempts to connect
// to an API endpoint
const retryTimeout = 5

// EstablishMasterConnection establishes a connection with exactly one of the provided API endpoints.
// The function builds a client for every endpoint and concurrently keeps trying to connect to any one
// of the provided endpoints. Blocks until at least one connection is established, then it stops the
// connection attempts for other endpoints.
func EstablishMasterConnection(c *kubeadmapi.TokenDiscovery, clusterInfo *kubeadmapi.ClusterInfo) (*ConnectionDetails, error) {
	hostName, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("failed to get node hostname [%v]", err)
	}
	// TODO(phase1+) https://github.com/kubernetes/kubernetes/issues/33641
	nodeName := types.NodeName(hostName)

	endpoints := clusterInfo.Endpoints
	caCert := []byte(clusterInfo.CertificateAuthorities[0])

	stopChan := make(chan struct{})
	result := make(chan *ConnectionDetails)
	var wg sync.WaitGroup
	for _, endpoint := range endpoints {
		clientSet, err := createClients(caCert, endpoint, kubeadmutil.BearerToken(c), nodeName)
		if err != nil {
			fmt.Printf("[bootstrap] Warning: %s. Skipping endpoint %s\n", err, endpoint)
			continue
		}
		wg.Add(1)
		go func(apiEndpoint string) {
			defer wg.Done()
			wait.Until(func() {
				fmt.Printf("[bootstrap] Trying to connect to endpoint %s\n", apiEndpoint)
				err := checkAPIEndpoint(clientSet, apiEndpoint)
				if err != nil {
					fmt.Printf("[bootstrap] Endpoint check failed [%v]\n", err)
					return
				}
				fmt.Printf("[bootstrap] Successfully established connection with endpoint %q\n", apiEndpoint)
				// connection established, stop all wait threads
				close(stopChan)
				result <- &ConnectionDetails{
					ClientSet:  clientSet,
					CertClient: clientSet.CertificatesV1alpha1Client,
					Endpoint:   apiEndpoint,
					CACert:     caCert,
					NodeName:   nodeName,
				}
			}, retryTimeout*time.Second, stopChan)
		}(endpoint)
	}

	go func() {
		wg.Wait()
		// all wait.Until() calls have finished now
		close(result)
	}()

	establishedConnection, ok := <-result
	if !ok {
		return nil, fmt.Errorf("failed to create bootstrap clients for any of the provided API endpoints")
	}
	return establishedConnection, nil
}

// creates a set of clients for this endpoint
func createClients(caCert []byte, endpoint, token string, nodeName types.NodeName) (*clientset.Clientset, error) {
	clientConfig := kubeconfigphase.MakeClientConfigWithToken(
		endpoint,
		"kubernetes",
		fmt.Sprintf("kubelet-%s", nodeName),
		caCert,
		token,
	)

	bootstrapClientConfig, err := clientcmd.NewDefaultClientConfig(*clientConfig, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create API client configuration [%v]", err)
	}
	clientSet, err := clientset.NewForConfig(bootstrapClientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create clients for the API endpoint %q: [%v]", endpoint, err)
	}
	return clientSet, nil
}

// CheckForNodeNameDuplicates checks whether there are other nodes in the cluster with identical node names.
func CheckForNodeNameDuplicates(connection *ConnectionDetails) error {
	hostName, err := os.Hostname()
	if err != nil {
		return fmt.Errorf("Failed to get node hostname [%v]", err)
	}
	nodeList, err := connection.ClientSet.Nodes().List(v1.ListOptions{})
	if err != nil {
		return fmt.Errorf("Failed to list the nodes in the cluster: [%v]\n", err)
	}
	for _, node := range nodeList.Items {
		if hostName == node.Name {
			return fmt.Errorf("Node with name [%q] already exists.", node.Name)
		}
	}
	return nil
}

// checks the connection requirements for a specific API endpoint
func checkAPIEndpoint(clientSet *clientset.Clientset, endpoint string) error {
	// check general connectivity
	version, err := clientSet.DiscoveryClient.ServerVersion()
	if err != nil {
		return fmt.Errorf("failed to connect to %q [%v]", endpoint, err)
	}
	fmt.Printf("[bootstrap] Detected server version: %s\n", version.String())

	// check certificates API
	serverGroups, err := clientSet.DiscoveryClient.ServerGroups()
	if err != nil {
		return fmt.Errorf("certificate API check failed: failed to retrieve a list of supported API objects [%v]", err)
	}
	for _, group := range serverGroups.Groups {
		if group.Name == certificates.GroupName {
			return nil
		}
	}
	return fmt.Errorf("certificate API check failed: API version %s does not support certificates API, use v1.4.0 or newer",
		version.String())
}
