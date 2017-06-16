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

package validate

import (
	"fmt"
	"runtime"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	nodeutil "k8s.io/client-go/pkg/api/v1/node"
	extensions "k8s.io/client-go/pkg/apis/extensions/v1beta1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
)

func Validate(kubeconfigPath string) error {
	client, err := kubeconfigutil.ClientSetFromFile(kubeconfigPath)
	if err != nil {
		return err
	}
	fmt.Println("[validate] Waiting for at least one node to register and become ready")
	start := time.Now()
	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		nodeList, err := client.Nodes().List(metav1.ListOptions{})
		if err != nil {
			fmt.Println("[validate] Temporarily unable to list nodes (will retry)")
			return false, nil
		}
		if len(nodeList.Items) < 1 {
			return false, nil
		}
		n := &nodeList.Items[0]
		if !nodeutil.IsNodeReady(n) {
			fmt.Println("[validate] First node has registered, but is not ready yet")
			return false, nil
		}

		fmt.Printf("[validate] First node is ready after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})

	if err := createAndWaitForADummyDeployment(client); err != nil {
		return err
	}
	return nil
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
			fmt.Printf("[validate] Failed to create test deployment [%v] (will retry)\n", err)
			return false, nil
		}
		return true, nil
	})

	wait.PollInfinite(kubeadmconstants.APICallRetryInterval, func() (bool, error) {
		d, err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Get("dummy", metav1.GetOptions{})
		if err != nil {
			fmt.Printf("[validate] Failed to get test deployment [%v] (will retry)\n", err)
			return false, nil
		}
		if d.Status.AvailableReplicas < 1 {
			return false, nil
		}
		return true, nil
	})

	fmt.Println("[validate] Test deployment succeeded")

	foreground := metav1.DeletePropagationForeground
	if err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Delete("dummy", &metav1.DeleteOptions{
		PropagationPolicy: &foreground,
	}); err != nil {
		fmt.Printf("[validate] Failed to delete test deployment [%v] (will ignore)\n", err)
	}
	return nil
}

const (
	DummyDeployment = `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    app: dummy
  name: dummy
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dummy
  template:
    metadata:
      labels:
        app: dummy
    spec:
      containers:
      - image: {{ .ImageRepository }}/pause-{{ .Arch }}:3.0
        name: dummy
      hostNetwork: true
`
)
