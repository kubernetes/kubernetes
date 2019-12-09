/*
Copyright 2019 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
)

// GetKubeletPort find the kubelet port from the given nodeName.
func GetKubeletPort(c clientset.Interface, nodeName string) (int, error) {
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return -1, fmt.Errorf("Failed to get node %s: %v", nodeName, err)
	}
	return int(node.Status.DaemonEndpoints.KubeletEndpoint.Port), nil
}

// GetKubeletPods retrieves the list of pods on the kubelet.
func GetKubeletPods(c clientset.Interface, node string) (*v1.PodList, error) {
	return getKubeletPods(c, node, "pods")
}

// GetKubeletRunningPods retrieves the list of running pods on the kubelet. The pods
// includes necessary information (e.g., UID, name, namespace for
// pods/containers), but do not contain the full spec.
func GetKubeletRunningPods(c clientset.Interface, node string) (*v1.PodList, error) {
	return getKubeletPods(c, node, "runningpods")
}

func getKubeletPods(c clientset.Interface, nodeName, resource string) (*v1.PodList, error) {
	kubeletPort, err := GetKubeletPort(c, nodeName)
	if err != nil {
		return nil, fmt.Errorf("Failed to get kubelet port, node %s:%v", nodeName, kubeletPort)
	}

	result := &v1.PodList{}
	client, err := ProxyRequest(c, nodeName, resource, kubeletPort)
	if err != nil {
		return &v1.PodList{}, err
	}
	if err = client.Into(result); err != nil {
		return &v1.PodList{}, err
	}
	return result, nil
}
