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
	"time"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/apis/certificates"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/util/wait"
)

func EstablishMasterConnection(s *kubeadmapi.KubeadmConfig, clusterInfo *kubeadmapi.ClusterInfo) (*kubeadmapi.ConnectionDetails, error) {
	nodeName, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("<node/bootstrap-client> failed to get hostname for node: [%v]", err)
	}
	endpoints := clusterInfo.Endpoints
	caCerts := clusterInfo.CertificateAuthorities
	if len(endpoints) != len(caCerts) {
		return nil, fmt.Errorf("<node/bootstrap-client> number of provided API endpoints does not match " +
			"the number of root certificates.")
	}

	stopChan := make(chan struct{})
	result := make(chan *kubeadmapi.ConnectionDetails)

	for i, endpoint := range endpoints {
		caCert := []byte(caCerts[i])
		clientSet, err := createClients(caCert, endpoint, nodeName, s.Secrets.BearerToken)
		if err != nil {
			// TODO: replace this with a WARN when we have logging
			fmt.Printf("<node/bootstrap-client> Warning: %s. Skipping endpoint %s.", err, endpoint)
			continue
		}
		go wait.Until(func() {
			fmt.Printf("<node/bootstrap-client> Trying to connect to endpoint %s", endpoint)
			if err := checkCertsAPI(clientSet.DiscoveryClient); err != nil {
				// TODO: replace this with a WARN when we have logging
				fmt.Printf("<node/bootstrap-client> Failed to connect to %s: %v", endpoint, err)
				return
			}
			// connection established, stop all wait threads
			close(stopChan)
			result <- &kubeadmapi.ConnectionDetails{
				CertClient: clientSet.CertificatesClient,
				Endpoint:   endpoint,
				CACert:     caCert,
				NodeName:   nodeName,
			}
		}, s.RetryTimeout*time.Second, stopChan)
	}

	establishedConnection := <-result
	if establishedConnection == nil {
		return nil, fmt.Errorf("<node/bootstrap-client> failed to create bootstrap clients " +
			"for any of the provided API endpoints. ")
	}

	return establishedConnection, nil
}

func createClients(caCert []byte, endpoint, nodeName, token string) (*clientset.Clientset, error) {
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

func checkCertsAPI(discoveryClient *discovery.DiscoveryClient) error {
	serverGroups, err := discoveryClient.ServerGroups()
	if err != nil {
		return fmt.Errorf("failed to retrieve a list of supported API objects [%v]", err)
	}
	for _, group := range serverGroups.Groups {
		if group.Name == certificates.GroupName {
			return nil
		}
	}
	version, err := discoveryClient.ServerVersion()
	if err != nil {
		return fmt.Errorf("unable to obtain API version [%v]", err)
	}

	return fmt.Errorf("API version %s does not support certificates API, use v1.4.0 or newer", version.String())
}
