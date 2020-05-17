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

package types

const (
	// KubernetesPodNameLabel specifies a pod name label
	KubernetesPodNameLabel = "io.kubernetes.pod.name"
	// KubernetesPodNamespaceLabel specifies a pod namespace label
	KubernetesPodNamespaceLabel = "io.kubernetes.pod.namespace"
	// KubernetesPodUIDLabel specifies a pod UID label
	KubernetesPodUIDLabel = "io.kubernetes.pod.uid"
	// KubernetesContainerNameLabel specifies a container name label
	KubernetesContainerNameLabel = "io.kubernetes.container.name"
)

// GetContainerName returns container name from labels
func GetContainerName(labels map[string]string) string {
	return labels[KubernetesContainerNameLabel]
}

// GetPodName returns pod name from labels
func GetPodName(labels map[string]string) string {
	return labels[KubernetesPodNameLabel]
}

// GetPodUID returns pod UID from labels
func GetPodUID(labels map[string]string) string {
	return labels[KubernetesPodUIDLabel]
}

// GetPodNamespace returns pod namespace from labels
func GetPodNamespace(labels map[string]string) string {
	return labels[KubernetesPodNamespaceLabel]
}
