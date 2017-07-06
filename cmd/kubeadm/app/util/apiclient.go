/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"net/http"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// CreateClientAndWaitForAPI takes a path to a kubeconfig file, makes a client of it and waits for the API to be healthy
func CreateClientAndWaitForAPI(file string) (*clientset.Clientset, error) {
	client, err := kubeconfigutil.ClientSetFromFile(file)
	if err != nil {
		return nil, err
	}

	fmt.Println("[apiclient] Created API client, waiting for the control plane to become ready")
	WaitForAPI(client)

	return client, nil
}

// WaitForAPI waits for the API Server's /healthz endpoint to report "ok"
func WaitForAPI(client *clientset.Clientset) {
	start := time.Now()
	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		healthStatus := 0
		client.Discovery().RESTClient().Get().AbsPath("/healthz").Do().StatusCode(&healthStatus)
		if healthStatus != http.StatusOK {
			return false, nil
		}

		fmt.Printf("[apiclient] All control plane components are healthy after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})
}

// WaitForPodsWithLabel will lookup pods with the given label and wait until they are all
// reporting status as running.
func WaitForPodsWithLabel(client *clientset.Clientset, labelKeyValPair string) {
	// TODO: Implement a timeout
	// TODO: Implement a verbosity switch
	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		listOpts := metav1.ListOptions{LabelSelector: labelKeyValPair}
		apiPods, err := client.CoreV1().Pods(metav1.NamespaceSystem).List(listOpts)
		if err != nil {
			fmt.Printf("[apiclient] Error getting Pods with label selector %q [%v]\n", labelKeyValPair, err)
			return false, nil
		}

		if len(apiPods.Items) == 0 {
			return false, nil
		}
		for _, pod := range apiPods.Items {
			fmt.Printf("[apiclient] Pod %s status: %s\n", pod.Name, pod.Status.Phase)
			if pod.Status.Phase != v1.PodRunning {
				return false, nil
			}
		}

		return true, nil
	})
}
