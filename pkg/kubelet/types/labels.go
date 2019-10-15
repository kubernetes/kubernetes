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
	KubernetesPodNameLabel       = "k8s.io/pod-name"
	KubernetesPodNamespaceLabel  = "k8s.io/pod-namespace"
	KubernetesPodUIDLabel        = "k8s.io/pod-uid"
	KubernetesContainerNameLabel = "k8s.io/container-name"

	PodDeletionGracePeriodLabel    = "k8s.io/pod-deletionGracePeriod"
	PodTerminationGracePeriodLabel = "k8s.io/pod-terminationGracePeriod"

	ContainerHashLabel                     = "k8s.io/container-hash"
	ContainerRestartCountLabel             = "k8s.io/container-restartCount"
	ContainerTerminationMessagePathLabel   = "k8s.io/container-terminationMessagePath"
	ContainerTerminationMessagePolicyLabel = "k8s.io/container-terminationMessagePolicy"
	ContainerPreStopHandlerLabel           = "k8s.io/container-preStopHandler"
	ContainerPortsLabel                    = "k8s.io/container-ports"
)

func GetContainerName(labels map[string]string) string {
	return labels[KubernetesContainerNameLabel]
}

func GetPodName(labels map[string]string) string {
	return labels[KubernetesPodNameLabel]
}

func GetPodUID(labels map[string]string) string {
	return labels[KubernetesPodUIDLabel]
}

func GetPodNamespace(labels map[string]string) string {
	return labels[KubernetesPodNamespaceLabel]
}
