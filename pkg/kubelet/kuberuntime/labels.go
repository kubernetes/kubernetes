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

package kuberuntime

import (
	"encoding/json"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	kubetypes "k8s.io/kubernetes/pkg/types"
)

const (
	kubernetesPodDeletionGracePeriodLabel    = "io.kubernetes.pod.deletionGracePeriod"
	kubernetesPodTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"

	kubernetesContainerHashLabel                   = "io.kubernetes.container.hash"
	kubernetesContainerRestartCountLabel           = "io.kubernetes.container.restartCount"
	kubernetesContainerTerminationMessagePathLabel = "io.kubernetes.container.terminationMessagePath"
	kubernetesContainerPreStopHandlerLabel         = "io.kubernetes.container.preStopHandler"
)

type labeledPodSandboxInfo struct {
	// Labels from api.Pod
	Labels       map[string]string
	PodName      string
	PodNamespace string
	PodUID       kubetypes.UID
}

type annotatedPodSandboxInfo struct {
	// Annotations from api.Pod
	Annotations               map[string]string
	PodDeletionGracePeriod    *int64
	PodTerminationGracePeriod *int64
}

type labeledContainerInfo struct {
	Name         string
	PodName      string
	PodNamespace string
	PodUID       kubetypes.UID
}

type annotatedContainerInfo struct {
	PodDeletionGracePeriod    *int64
	PodTerminationGracePeriod *int64
	Hash                      uint64
	RestartCount              int
	TerminationMessagePath    string
	PreStopHandler            *api.Handler
}

// newPodLabels creates pod labels from api.Pod.
func newPodLabels(pod *api.Pod) map[string]string {
	labels := map[string]string{}
	for k, v := range pod.Labels {
		labels[k] = v
	}

	labels[types.KubernetesPodNameLabel] = pod.Name
	labels[types.KubernetesPodNamespaceLabel] = pod.Namespace
	labels[types.KubernetesPodUIDLabel] = string(pod.UID)

	return labels
}

// newPodAnnotations creates pod annotations from api.Pod.
func newPodAnnotations(pod *api.Pod) map[string]string {
	annotations := map[string]string{}
	for k, v := range pod.Annotations {
		annotations[k] = v
	}

	if pod.DeletionGracePeriodSeconds != nil {
		annotations[kubernetesPodDeletionGracePeriodLabel] = strconv.FormatInt(*pod.DeletionGracePeriodSeconds, 10)
	}
	if pod.Spec.TerminationGracePeriodSeconds != nil {
		annotations[kubernetesPodTerminationGracePeriodLabel] = strconv.FormatInt(*pod.Spec.TerminationGracePeriodSeconds, 10)
	}

	return annotations
}

// newContainerLabels creates container labels from api.Container and api.Pod.
func newContainerLabels(container *api.Container, pod *api.Pod) map[string]string {
	labels := map[string]string{}
	labels[types.KubernetesPodNameLabel] = pod.Name
	labels[types.KubernetesPodNamespaceLabel] = pod.Namespace
	labels[types.KubernetesPodUIDLabel] = string(pod.UID)
	labels[types.KubernetesContainerNameLabel] = container.Name

	return labels
}

// newContainerAnnotations creates container annotations from api.Container and api.Pod.
func newContainerAnnotations(container *api.Container, pod *api.Pod, restartCount int) map[string]string {
	annotations := map[string]string{}
	annotations[kubernetesContainerHashLabel] = strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	annotations[kubernetesContainerRestartCountLabel] = strconv.Itoa(restartCount)
	annotations[kubernetesContainerTerminationMessagePathLabel] = container.TerminationMessagePath

	if pod.DeletionGracePeriodSeconds != nil {
		annotations[kubernetesPodDeletionGracePeriodLabel] = strconv.FormatInt(*pod.DeletionGracePeriodSeconds, 10)
	}
	if pod.Spec.TerminationGracePeriodSeconds != nil {
		annotations[kubernetesPodTerminationGracePeriodLabel] = strconv.FormatInt(*pod.Spec.TerminationGracePeriodSeconds, 10)
	}

	if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		// Using json enconding so that the PreStop handler object is readable after writing as a label
		rawPreStop, err := json.Marshal(container.Lifecycle.PreStop)
		if err != nil {
			glog.Errorf("Unable to marshal lifecycle PreStop handler for container %q of pod %q: %v", container.Name, format.Pod(pod), err)
		} else {
			annotations[kubernetesContainerPreStopHandlerLabel] = string(rawPreStop)
		}
	}

	return annotations
}

// getPodSandboxInfoFromLabels gets labeledPodSandboxInfo from labels.
func getPodSandboxInfoFromLabels(labels map[string]string) *labeledPodSandboxInfo {
	podSandboxInfo := &labeledPodSandboxInfo{
		Labels:       make(map[string]string),
		PodName:      getStringValueFromLabel(labels, types.KubernetesPodNameLabel),
		PodNamespace: getStringValueFromLabel(labels, types.KubernetesPodNamespaceLabel),
		PodUID:       kubetypes.UID(getStringValueFromLabel(labels, types.KubernetesPodUIDLabel)),
	}

	// Remain only labels from api.Pod
	for k, v := range labels {
		if k != types.KubernetesPodNameLabel && k != types.KubernetesPodNamespaceLabel && k != types.KubernetesPodUIDLabel {
			podSandboxInfo.Labels[k] = v
		}
	}

	return podSandboxInfo
}

// getPodSandboxInfoFromAnnotations gets annotatedPodSandboxInfo from annotations.
func getPodSandboxInfoFromAnnotations(annotations map[string]string) *annotatedPodSandboxInfo {
	var err error

	podSandboxInfo := &annotatedPodSandboxInfo{
		Annotations: make(map[string]string),
	}

	if podSandboxInfo.PodDeletionGracePeriod, err = getInt64PointerFromLabel(annotations, kubernetesPodDeletionGracePeriodLabel); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesPodDeletionGracePeriodLabel, annotations, err)
	}
	if podSandboxInfo.PodTerminationGracePeriod, err = getInt64PointerFromLabel(annotations, kubernetesPodTerminationGracePeriodLabel); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesPodTerminationGracePeriodLabel, annotations, err)
	}

	// Remain only annotations from api.Pod
	for k, v := range annotations {
		if k != kubernetesPodTerminationGracePeriodLabel && k != kubernetesPodDeletionGracePeriodLabel {
			podSandboxInfo.Annotations[k] = v
		}
	}

	return podSandboxInfo
}

// getContainerInfoFromLabels gets labeledContainerInfo from labels.
func getContainerInfoFromLabels(labels map[string]string) *labeledContainerInfo {
	return &labeledContainerInfo{
		PodName:      getStringValueFromLabel(labels, types.KubernetesPodNameLabel),
		PodNamespace: getStringValueFromLabel(labels, types.KubernetesPodNamespaceLabel),
		PodUID:       kubetypes.UID(getStringValueFromLabel(labels, types.KubernetesPodUIDLabel)),
		Name:         getStringValueFromLabel(labels, types.KubernetesContainerNameLabel),
	}
}

// getContainerInfoFromAnnotations gets annotatedContainerInfo from annotations.
func getContainerInfoFromAnnotations(annotations map[string]string) *annotatedContainerInfo {
	var err error
	containerInfo := &annotatedContainerInfo{
		TerminationMessagePath: getStringValueFromLabel(annotations, kubernetesContainerTerminationMessagePathLabel),
	}

	if containerInfo.Hash, err = getUint64ValueFromLabel(annotations, kubernetesContainerHashLabel); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesContainerHashLabel, annotations, err)
	}
	if containerInfo.RestartCount, err = getIntValueFromLabel(annotations, kubernetesContainerRestartCountLabel); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesContainerRestartCountLabel, annotations, err)
	}
	if containerInfo.PodDeletionGracePeriod, err = getInt64PointerFromLabel(annotations, kubernetesPodDeletionGracePeriodLabel); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesPodDeletionGracePeriodLabel, annotations, err)
	}
	if containerInfo.PodTerminationGracePeriod, err = getInt64PointerFromLabel(annotations, kubernetesPodTerminationGracePeriodLabel); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesPodTerminationGracePeriodLabel, annotations, err)
	}

	preStopHandler := &api.Handler{}
	if found, err := getJSONObjectFromLabel(annotations, kubernetesContainerPreStopHandlerLabel, preStopHandler); err != nil {
		glog.Errorf("Unable to get %q from annotations %q: %v", kubernetesContainerPreStopHandlerLabel, annotations, err)
	} else if found {
		containerInfo.PreStopHandler = preStopHandler
	}

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

func getUint64ValueFromLabel(labels map[string]string, label string) (uint64, error) {
	if strValue, found := labels[label]; found {
		intValue, err := strconv.ParseUint(strValue, 16, 64)
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

// getJSONObjectFromLabel returns a bool value indicating whether an object is found.
func getJSONObjectFromLabel(labels map[string]string, label string, value interface{}) (bool, error) {
	if strValue, found := labels[label]; found {
		err := json.Unmarshal([]byte(strValue), value)
		return found, err
	}
	// Because it's normal that a container has no PreStopHandler label, don't report any error here.
	return false, nil
}
