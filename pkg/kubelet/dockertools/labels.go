/*
Copyright 2015 The Kubernetes Authors.

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
	"encoding/json"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime"
	kubetypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/custommetrics"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// This file contains all docker label related constants and functions, including:
//  * label setters and getters
//  * label filters (maybe in the future)

const (
	kubernetesPodDeletionGracePeriodLabel    = "io.kubernetes.pod.deletionGracePeriod"
	kubernetesPodTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"

	kubernetesContainerHashLabel                     = "io.kubernetes.container.hash"
	kubernetesContainerRestartCountLabel             = "io.kubernetes.container.restartCount"
	kubernetesContainerTerminationMessagePathLabel   = "io.kubernetes.container.terminationMessagePath"
	kubernetesContainerTerminationMessagePolicyLabel = "io.kubernetes.container.terminationMessagePolicy"
	kubernetesContainerPreStopHandlerLabel           = "io.kubernetes.container.preStopHandler"
	kubernetesContainerPortsLabel                    = "io.kubernetes.container.ports" // Added in 1.4

	// TODO(random-liu): Keep this for old containers, remove this when we drop support for v1.1.
	kubernetesPodLabel = "io.kubernetes.pod.data"

	cadvisorPrometheusMetricsLabel = "io.cadvisor.metric.prometheus"
)

// Container information which has been labelled on each docker container
// TODO(random-liu): The type of Hash should be compliance with kubelet container status.
type labelledContainerInfo struct {
	PodName                   string
	PodNamespace              string
	PodUID                    kubetypes.UID
	PodDeletionGracePeriod    *int64
	PodTerminationGracePeriod *int64
	Name                      string
	Hash                      string
	RestartCount              int
	TerminationMessagePath    string
	TerminationMessagePolicy  v1.TerminationMessagePolicy
	PreStopHandler            *v1.Handler
	Ports                     []v1.ContainerPort
}

func newLabels(container *v1.Container, pod *v1.Pod, restartCount int, enableCustomMetrics bool) map[string]string {
	labels := map[string]string{}
	labels[types.KubernetesPodNameLabel] = pod.Name
	labels[types.KubernetesPodNamespaceLabel] = pod.Namespace
	labels[types.KubernetesPodUIDLabel] = string(pod.UID)
	if pod.DeletionGracePeriodSeconds != nil {
		labels[kubernetesPodDeletionGracePeriodLabel] = strconv.FormatInt(*pod.DeletionGracePeriodSeconds, 10)
	}
	if pod.Spec.TerminationGracePeriodSeconds != nil {
		labels[kubernetesPodTerminationGracePeriodLabel] = strconv.FormatInt(*pod.Spec.TerminationGracePeriodSeconds, 10)
	}

	labels[types.KubernetesContainerNameLabel] = container.Name
	labels[kubernetesContainerHashLabel] = strconv.FormatUint(kubecontainer.HashContainerLegacy(container), 16)
	labels[kubernetesContainerRestartCountLabel] = strconv.Itoa(restartCount)
	labels[kubernetesContainerTerminationMessagePathLabel] = container.TerminationMessagePath
	labels[kubernetesContainerTerminationMessagePolicyLabel] = string(container.TerminationMessagePolicy)
	if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		// Using json enconding so that the PreStop handler object is readable after writing as a label
		rawPreStop, err := json.Marshal(container.Lifecycle.PreStop)
		if err != nil {
			glog.Errorf("Unable to marshal lifecycle PreStop handler for container %q of pod %q: %v", container.Name, format.Pod(pod), err)
		} else {
			labels[kubernetesContainerPreStopHandlerLabel] = string(rawPreStop)
		}
	}
	if len(container.Ports) > 0 {
		rawContainerPorts, err := json.Marshal(container.Ports)
		if err != nil {
			glog.Errorf("Unable to marshal container ports for container %q for pod %q: %v", container.Name, format.Pod(pod), err)
		} else {
			labels[kubernetesContainerPortsLabel] = string(rawContainerPorts)
		}
	}
	if enableCustomMetrics {
		path, err := custommetrics.GetCAdvisorCustomMetricsDefinitionPath(container)
		if path != nil && err == nil {
			labels[cadvisorPrometheusMetricsLabel] = *path
		}
	}

	return labels
}

func getContainerInfoFromLabel(labels map[string]string) *labelledContainerInfo {
	var err error
	containerInfo := &labelledContainerInfo{
		PodName:      getStringValueFromLabel(labels, types.KubernetesPodNameLabel),
		PodNamespace: getStringValueFromLabel(labels, types.KubernetesPodNamespaceLabel),
		PodUID:       kubetypes.UID(getStringValueFromLabel(labels, types.KubernetesPodUIDLabel)),
		Name:         getStringValueFromLabel(labels, types.KubernetesContainerNameLabel),
		Hash:         getStringValueFromLabel(labels, kubernetesContainerHashLabel),
		TerminationMessagePath:   getStringValueFromLabel(labels, kubernetesContainerTerminationMessagePathLabel),
		TerminationMessagePolicy: v1.TerminationMessagePolicy(getStringValueFromLabel(labels, kubernetesContainerTerminationMessagePolicyLabel)),
	}
	if containerInfo.RestartCount, err = getIntValueFromLabel(labels, kubernetesContainerRestartCountLabel); err != nil {
		logError(containerInfo, kubernetesContainerRestartCountLabel, err)
	}
	if containerInfo.PodDeletionGracePeriod, err = getInt64PointerFromLabel(labels, kubernetesPodDeletionGracePeriodLabel); err != nil {
		logError(containerInfo, kubernetesPodDeletionGracePeriodLabel, err)
	}
	if containerInfo.PodTerminationGracePeriod, err = getInt64PointerFromLabel(labels, kubernetesPodTerminationGracePeriodLabel); err != nil {
		logError(containerInfo, kubernetesPodTerminationGracePeriodLabel, err)
	}
	preStopHandler := &v1.Handler{}
	if found, err := getJsonObjectFromLabel(labels, kubernetesContainerPreStopHandlerLabel, preStopHandler); err != nil {
		logError(containerInfo, kubernetesContainerPreStopHandlerLabel, err)
	} else if found {
		containerInfo.PreStopHandler = preStopHandler
	}
	containerPorts := []v1.ContainerPort{}
	if found, err := getJsonObjectFromLabel(labels, kubernetesContainerPortsLabel, &containerPorts); err != nil {
		logError(containerInfo, kubernetesContainerPortsLabel, err)
	} else if found {
		containerInfo.Ports = containerPorts
	}
	supplyContainerInfoWithOldLabel(labels, containerInfo)
	return containerInfo
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

func getInt64PointerFromLabel(labels map[string]string, label string) (*int64, error) {
	if strValue, found := labels[label]; found {
		int64Value, err := strconv.ParseInt(strValue, 10, 64)
		if err != nil {
			return nil, err
		}
		return &int64Value, nil
	}
	// Because it's normal that a container has no PodDeletionGracePeriod and PodTerminationGracePeriod label,
	// don't report any error here.
	return nil, nil
}

// getJsonObjectFromLabel returns a bool value indicating whether an object is found
func getJsonObjectFromLabel(labels map[string]string, label string, value interface{}) (bool, error) {
	if strValue, found := labels[label]; found {
		err := json.Unmarshal([]byte(strValue), value)
		return found, err
	}
	// Because it's normal that a container has no PreStopHandler label, don't report any error here.
	return false, nil
}

// The label kubernetesPodLabel is added a long time ago (#7421), it serialized the whole v1.Pod to a docker label.
// We want to remove this label because it serialized too much useless information. However kubelet may still work
// with old containers which only have this label for a long time until we completely deprecate the old label.
// Before that to ensure correctness we have to supply information with the old labels when newly added labels
// are not available.
// TODO(random-liu): Remove this function when we can completely remove label kubernetesPodLabel, probably after
// dropping support for v1.1.
func supplyContainerInfoWithOldLabel(labels map[string]string, containerInfo *labelledContainerInfo) {
	// Get v1.Pod from old label
	var pod *v1.Pod
	data, found := labels[kubernetesPodLabel]
	if !found {
		// Don't report any error here, because it's normal that a container has no pod label, especially
		// when we gradually deprecate the old label
		return
	}
	pod = &v1.Pod{}
	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), []byte(data), pod); err != nil {
		// If the pod label can't be parsed, we should report an error
		logError(containerInfo, kubernetesPodLabel, err)
		return
	}
	if containerInfo.PodDeletionGracePeriod == nil {
		containerInfo.PodDeletionGracePeriod = pod.DeletionGracePeriodSeconds
	}
	if containerInfo.PodTerminationGracePeriod == nil {
		containerInfo.PodTerminationGracePeriod = pod.Spec.TerminationGracePeriodSeconds
	}

	// Get v1.Container from v1.Pod
	var container *v1.Container
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == containerInfo.Name {
			container = &pod.Spec.Containers[i]
			break
		}
	}
	if container == nil {
		glog.Errorf("Unable to find container %q in pod %q", containerInfo.Name, format.Pod(pod))
		return
	}
	if containerInfo.PreStopHandler == nil && container.Lifecycle != nil {
		containerInfo.PreStopHandler = container.Lifecycle.PreStop
	}
}

func logError(containerInfo *labelledContainerInfo, label string, err error) {
	glog.Errorf("Unable to get %q for container %q of pod %q: %v", label, containerInfo.Name,
		kubecontainer.BuildPodFullName(containerInfo.PodName, containerInfo.PodNamespace), err)
}
