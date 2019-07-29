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
	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/master/ports"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
)

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

func getKubeletPods(c clientset.Interface, node, resource string) (*v1.PodList, error) {
	result := &v1.PodList{}
	client, err := e2enode.ProxyRequest(c, node, resource, ports.KubeletPort)
	if err != nil {
		return &v1.PodList{}, err
	}
	if err = client.Into(result); err != nil {
		return &v1.PodList{}, err
	}
	return result, nil
}

// PrintAllKubeletPods outputs status of all kubelet pods into log.
func PrintAllKubeletPods(c clientset.Interface, nodeName string) {
	podList, err := GetKubeletPods(c, nodeName)
	if err != nil {
		e2elog.Logf("Unable to retrieve kubelet pods for node %v: %v", nodeName, err)
		return
	}
	for _, p := range podList.Items {
		e2elog.Logf("%v from %v started at %v (%d container statuses recorded)", p.Name, p.Namespace, p.Status.StartTime, len(p.Status.ContainerStatuses))
		for _, c := range p.Status.ContainerStatuses {
			e2elog.Logf("\tContainer %v ready: %v, restart count %v",
				c.Name, c.Ready, c.RestartCount)
		}
	}
}
