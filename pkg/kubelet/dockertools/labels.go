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
)

// This file contains all docker label related constants and functions, including:
//  * label setters and getters
//  * label filters (maybe in the future)

const (
	kubernetesPodNameLabel      = "io.kubernetes.pod.name"
	kubernetesPodNamespaceLabel = "io.kubernetes.pod.namespace"
	kubernetesPodUID            = "io.kubernetes.pod.uid"

	kubernetesContainerRestartCountLabel      = "io.kubernetes.container.restartCount"
	kubernetesContainerTerminationMessagePath = "io.kubernetes.container.terminationMessagePath"

	kubernetesPodLabel                    = "io.kubernetes.pod.data"
	kubernetesTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"
	kubernetesContainerLabel              = "io.kubernetes.container.name"
)

func newLabels(container *api.Container, pod *api.Pod, restartCount int) map[string]string {
	// TODO (random-liu) Move more label initialization here
	labels := map[string]string{}
	labels[kubernetesPodNameLabel] = pod.Name
	labels[kubernetesPodNamespaceLabel] = pod.Namespace
	labels[kubernetesPodUID] = string(pod.UID)

	labels[kubernetesContainerRestartCountLabel] = strconv.Itoa(restartCount)
	labels[kubernetesContainerTerminationMessagePath] = container.TerminationMessagePath

	return labels
}

func getRestartCountFromLabel(labels map[string]string) (restartCount int, err error) {
	if restartCountString, found := labels[kubernetesContainerRestartCountLabel]; found {
		restartCount, err = strconv.Atoi(restartCountString)
		if err != nil {
			// This really should not happen. Just set restartCount to 0 to handle this abnormal case
			restartCount = 0
		}
	} else {
		// Get restartCount from docker label. If there is no restart count label in a container,
		// it should be an old container or an invalid container, we just set restart count to 0.
		// Do not report error, because there should be many old containers without this label now
		glog.V(3).Infof("Container doesn't have label %s, it may be an old or invalid container", kubernetesContainerRestartCountLabel)
	}
	return restartCount, err
}

func getTerminationMessagePathFromLabel(labels map[string]string) string {
	if terminationMessagePath, found := labels[kubernetesContainerTerminationMessagePath]; found {
		return terminationMessagePath
	} else {
		// Do not report error, because there should be many old containers without this label now.
		// Return empty string "" for these containers, the caller will get terminationMessagePath by other ways.
		glog.V(3).Infof("Container doesn't have label %s, it may be an old or invalid container", kubernetesContainerTerminationMessagePath)
		return ""
	}
}
