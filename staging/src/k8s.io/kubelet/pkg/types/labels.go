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

// Label keys for labels used in this package.
const (
	KubernetesPodNameLabel       = "io.kubernetes.pod.name"
	KubernetesPodNamespaceLabel  = "io.kubernetes.pod.namespace"
	KubernetesPodUIDLabel        = "io.kubernetes.pod.uid"
	KubernetesContainerNameLabel = "io.kubernetes.container.name"
)

// Label value constants
const (
	// PodInfraContainerName is the KubernetesPodNameLabel value for infra
	// containers.
	PodInfraContainerName = "POD"
)

// GetContainerName returns the value of the KubernetesContainerNameLabel.
func GetContainerName(labels map[string]string) string {
	return labels[KubernetesContainerNameLabel]
}

// GetPodName returns the value of the KubernetesPodNameLabel.
func GetPodName(labels map[string]string) string {
	return labels[KubernetesPodNameLabel]
}

// GetPodUID returns the value of the KubernetesPodUIDLabel.
func GetPodUID(labels map[string]string) string {
	return labels[KubernetesPodUIDLabel]
}

// GetPodNamespace returns the value of the KubernetesPodNamespaceLabel.
func GetPodNamespace(labels map[string]string) string {
	return labels[KubernetesPodNamespaceLabel]
}
