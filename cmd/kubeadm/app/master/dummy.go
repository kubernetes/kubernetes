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

package master

import (
	"fmt"
	"runtime"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const DummyDeployment = `
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

	// TODO: In the future, make sure the ReplicaSet and Pod are garbage collected
	if err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Delete("dummy", &metav1.DeleteOptions{}); err != nil {
		fmt.Printf("[apiclient] Failed to delete test deployment [%v] (will ignore)\n", err)
	}
	return nil
}
