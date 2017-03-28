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
	"runtime"
	"time"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	extensions "k8s.io/client-go/pkg/apis/extensions/v1beta1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

func CreateClientAndWaitForAPI(file string) (*clientset.Clientset, error) {
	client, err := kubeconfigutil.ClientSetFromFile(file)
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

func createAndWaitForADummyDeployment(client *clientset.Clientset) error {
	dummyDeploymentBytes, err := kubeadmutil.ParseTemplate(DummyDeployment, struct{ ImageRepository, Arch string }{
		ImageRepository: kubeadmapi.GlobalEnvParams.RepositoryPrefix,
		Arch:            runtime.GOARCH,
	})
	if err != nil {
		return fmt.Errorf("error when parsing dummy deployment template: %v", err)
	}

	dummyDeployment := &extensions.Deployment{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), dummyDeploymentBytes, dummyDeployment); err != nil {
		return fmt.Errorf("unable to decode dummy deployment %v", err)
	}

	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		// TODO: we should check the error, as some cases may be fatal
		if _, err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Create(dummyDeployment); err != nil {
			fmt.Printf("[apiclient] Failed to create test deployment [%v] (will retry)\n", err)
			return false, nil
		}
		return true, nil
	})

	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		d, err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Get("dummy", metav1.GetOptions{})
		if err != nil {
			fmt.Printf("[apiclient] Failed to get test deployment [%v] (will retry)\n", err)
			return false, nil
		}
		if d.Status.AvailableReplicas < 1 {
			return false, nil
		}
		return true, nil
	})

	fmt.Println("[apiclient] Test deployment succeeded")

	foreground := metav1.DeletePropagationForeground
	if err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Delete("dummy", &metav1.DeleteOptions{
		PropagationPolicy: &foreground,
	}); err != nil {
		fmt.Printf("[apiclient] Failed to delete test deployment [%v] (will ignore)\n", err)
	}
	return nil
}
