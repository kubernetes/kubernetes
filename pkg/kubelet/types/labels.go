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
	// KubernetesPodNameLabel is the label of Kubernetes pod name
	KubernetesPodNameLabel = "io.kubernetes.pod.name"
	// KubernetesPodNamespaceLabel is the label of Kubernetes pod namespace
	KubernetesPodNamespaceLabel = "io.kubernetes.pod.namespace"
	// KubernetesPodUIDLabel is the label of Kubernetes pod UID
	KubernetesPodUIDLabel = "io.kubernetes.pod.uid"
	// KubernetesContainerNameLabel is the label of Kubernetes container name
	KubernetesContainerNameLabel = "io.kubernetes.container.name"
	// KubernetesContainerTypeLabel is the label of Kubernetes container type
	KubernetesContainerTypeLabel = "io.kubernetes.container.type"
)

// GetContainerName returns container name if label exists
func GetContainerName(labels map[string]string) string {
	return labels[KubernetesContainerNameLabel]
}

// GetPodName returns pod name if label exists
func GetPodName(labels map[string]string) string {
	return labels[KubernetesPodNameLabel]
}

// GetPodUID returns pod UID if label exists
func GetPodUID(labels map[string]string) string {
	return labels[KubernetesPodUIDLabel]
}

// GetPodNamespace returns pod namespace if label exists
func GetPodNamespace(labels map[string]string) string {
	return labels[KubernetesPodNamespaceLabel]
}
