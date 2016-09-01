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

package kubemaster

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	//"k8s.io/kubernetes/pkg/client/unversioned" // TODO use a clientset
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/wait"
)

func CreateClientAndWaitForAPI(adminConfig *clientcmdapi.Config) (*clientset.Clientset, error) {
	adminClientConfig, err := clientcmd.NewDefaultClientConfig(
		*adminConfig,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, err
	}

	fmt.Println("configuring the client...")

	client, err := clientset.NewForConfig(adminClientConfig)
	if err != nil {
		return nil, err
	}

	fmt.Println("polling API server forever...")

	start := time.Now()
	wait.PollInfinite(500*time.Millisecond, func() (bool, error) {
		cs, err := client.ComponentStatuses().List(api.ListOptions{})
		if err != nil {
			return false, nil
		}
		if len(cs.Items) < 3 {
			fmt.Println("Not all control plane components had been deployed yet")
			return false, nil
		}
		for _, item := range cs.Items {
			for _, condition := range item.Conditions {
				if condition.Type != api.ComponentHealthy {
					fmt.Printf("Control plane component %q is still unhealthy: %#v\n", item.ObjectMeta.Name, item.Conditions)
					return false, nil
				}
			}
		}

		fmt.Printf("All control plane components are healthy now (took %s seconds)\n", time.Since(start).Seconds())
		return true, nil
	})

	// TODO may be also check node status
	return client, nil
}

func TaintMaster(*clientset.Clientset) error {
	// TODO
	annotations := make(map[string]string)
	annotations[api.TaintsAnnotationKey] = ""
	return nil
}
