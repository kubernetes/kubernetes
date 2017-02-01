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

package master

import (
	"fmt"
	"time"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/clientcmd"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

func CreateClientFromFile(path string) (*clientset.Clientset, error) {
	adminKubeconfig, err := clientcmd.LoadFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load admin kubeconfig [%v]", err)
	}
	adminClientConfig, err := clientcmd.NewDefaultClientConfig(
		*adminKubeconfig,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create API client configuration [%v]", err)
	}

	client, err := clientset.NewForConfig(adminClientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create API client [%v]", err)
	}
	return client, nil
}

func CreateClientAndWaitForAPI(file string) (*clientset.Clientset, error) {
	client, err := CreateClientFromFile(file)
	if err != nil {
		return nil, err
	}

	fmt.Println("[apiclient] Created API client, waiting for the control plane to become ready")
	WaitForAPI(client)

	fmt.Println("[apiclient] Waiting for at least one node to register and become ready")
	start := time.Now()
	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		nodeList, err := client.Nodes().List(metav1.ListOptions{})
		if err != nil {
			fmt.Println("[apiclient] Temporarily unable to list nodes (will retry)")
			return false, nil
		}
		if len(nodeList.Items) < 1 {
			return false, nil
		}
		n := &nodeList.Items[0]
		if !v1.IsNodeReady(n) {
			fmt.Println("[apiclient] First node has registered, but is not ready yet")
			return false, nil
		}

		fmt.Printf("[apiclient] First node is ready after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})

	if err := createAndWaitForADummyDeployment(client); err != nil {
		return nil, err
	}

	return client, nil
}

func WaitForAPI(client *clientset.Clientset) {
	start := time.Now()
	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		// TODO: use /healthz API instead of this
		cs, err := client.ComponentStatuses().List(metav1.ListOptions{})
		if err != nil {
			if apierrs.IsForbidden(err) {
				fmt.Println("[apiclient] Waiting for API server authorization")
			}
			return false, nil
		}

		// TODO(phase2) must revisit this when we implement HA
		if len(cs.Items) < 3 {
			return false, nil
		}
		for _, item := range cs.Items {
			for _, condition := range item.Conditions {
				if condition.Type != v1.ComponentHealthy {
					fmt.Printf("[apiclient] Control plane component %q is still unhealthy: %#v\n", item.ObjectMeta.Name, item.Conditions)
					return false, nil
				}
			}
		}

		fmt.Printf("[apiclient] All control plane components are healthy after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})
}
