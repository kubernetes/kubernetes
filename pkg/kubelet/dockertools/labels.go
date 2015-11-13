/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

// This file contains all docker label related constants and functions, including:
//  * label setters and getters
//  * label filters (maybe in the future)

const (
	kubernetesPodNameLabel      = "io.kubernetes.pod.name"
	kubernetesPodNamespaceLabel = "io.kubernetes.pod.namespace"
	kubernetesPodUIDLabel       = "io.kubernetes.pod.uid"

	kubernetesContainerNameLabel                   = "io.kubernetes.container.name"
	kubernetesContainerHashLabel                   = "io.kubernetes.container.hash"
	kubernetesContainerRestartCountLabel           = "io.kubernetes.container.restartCount"
	kubernetesContainerTerminationMessagePathLabel = "io.kubernetes.container.terminationMessagePath"

	kubernetesPodLabel                    = "io.kubernetes.pod.data"
	kubernetesTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"
	kubernetesContainerLabel              = "io.kubernetes.container.name"
)

// Container information which has been labelled on each docker container
type labelledContainerInfo struct {
	PodName                string
	PodNamespace           string
	PodUID                 types.UID
	Name                   string
	Hash                   string
	RestartCount           int
	TerminationMessagePath string
}

func newLabels(container *api.Container, pod *api.Pod, restartCount int) map[string]string {
	// TODO (random-liu) Move more label initialization here
	labels := map[string]string{}
	labels[kubernetesPodNameLabel] = pod.Name
	labels[kubernetesPodNamespaceLabel] = pod.Namespace
	labels[kubernetesPodUIDLabel] = string(pod.UID)

	labels[kubernetesContainerNameLabel] = container.Name
	labels[kubernetesContainerHashLabel] = strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	labels[kubernetesContainerRestartCountLabel] = strconv.Itoa(restartCount)
	labels[kubernetesContainerTerminationMessagePathLabel] = container.TerminationMessagePath

	return labels
}

func getContainerInfoFromLabel(labels map[string]string) (*labelledContainerInfo, error) {
	var err error
	containerInfo := labelledContainerInfo{
		PodName:      getStringValueFromLabel(labels, kubernetesPodNameLabel),
		PodNamespace: getStringValueFromLabel(labels, kubernetesPodNamespaceLabel),
		PodUID:       types.UID(getStringValueFromLabel(labels, kubernetesPodUIDLabel)),
		Name:         getStringValueFromLabel(labels, kubernetesContainerNameLabel),
		Hash:         getStringValueFromLabel(labels, kubernetesContainerHashLabel),
		TerminationMessagePath: getStringValueFromLabel(labels, kubernetesContainerTerminationMessagePathLabel),
	}
	containerInfo.RestartCount, err = getIntValueFromLabel(labels, kubernetesContainerRestartCountLabel)
	return &containerInfo, err
}

func getStringValueFromLabel(labels map[string]string, label string) string {
	if value, found := labels[label]; found {
		return value
	}
	// Do not report error, because there should be many old containers without label now.
	glog.V(3).Infof("Container doesn't have label %s, it may be an old or invalid container", label)
	// Return empty string "" for these containers, the caller will get value by other ways.
	return ""
}

func getIntValueFromLabel(labels map[string]string, label string) (int, error) {
	if strValue, found := labels[label]; found {
		intValue, err := strconv.Atoi(strValue)
		if err != nil {
			// This really should not happen. Just set value to 0 to handle this abnormal case
			return 0, err
		}
		return intValue, nil
	}
	// Do not report error, because there should be many old containers without label now.
	glog.V(3).Infof("Container doesn't have label %s, it may be an old or invalid container", label)
	// Just set the value to 0
	return 0, nil
}
