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
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

func CreateClientAndWaitForAPI(file string) (*clientset.Clientset, error) {
	client, err := kubeconfigutil.ClientSetFromFile(file)
	if err != nil {
		return nil, err
	}

	fmt.Println("[apiclient] Created API client, waiting for the control plane to become ready")
	WaitForAPI(client)

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
