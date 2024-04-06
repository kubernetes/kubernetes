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

	v1 "k8s.io/api/core/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubelet/pkg/types"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	// TODO: change those label names to follow kubernetes's format
	podDeletionGracePeriodLabel    = "io.kubernetes.pod.deletionGracePeriod"
	podTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"

	containerHashLabel                     = "io.kubernetes.container.hash"
	containerHashWithoutResourcesLabel     = "io.kubernetes.container.hashWithoutResources"
	containerRestartCountLabel             = "io.kubernetes.container.restartCount"
	containerTerminationMessagePathLabel   = "io.kubernetes.container.terminationMessagePath"
	containerTerminationMessagePolicyLabel = "io.kubernetes.container.terminationMessagePolicy"
	containerPreStopHandlerLabel           = "io.kubernetes.container.preStopHandler"
	containerPortsLabel                    = "io.kubernetes.container.ports"
)

type labeledPodSandboxInfo struct {
	// Labels from v1.Pod
	Labels       map[string]string
	PodName      string
	PodNamespace string
	PodUID       kubetypes.UID
}

type annotatedPodSandboxInfo struct {
	// Annotations from v1.Pod
	Annotations map[string]string
}

type labeledContainerInfo struct {
	ContainerName string
	PodName       string
	PodNamespace  string
	PodUID        kubetypes.UID
}

type annotatedContainerInfo struct {
	Hash                      uint64
	HashWithoutResources      uint64
	RestartCount              int
	PodDeletionGracePeriod    *int64
	PodTerminationGracePeriod *int64
	TerminationMessagePath    string
	TerminationMessagePolicy  v1.TerminationMessagePolicy
	PreStopHandler            *v1.LifecycleHandler
	ContainerPorts            []v1.ContainerPort
}

// newPodLabels creates pod labels from v1.Pod.
func newPodLabels(pod *v1.Pod) map[string]string {
	labels := map[string]string{}

	// Get labels from v1.Pod
	for k, v := range pod.Labels {
		labels[k] = v
	}

	labels[types.KubernetesPodNameLabel] = pod.Name
	labels[types.KubernetesPodNamespaceLabel] = pod.Namespace
	labels[types.KubernetesPodUIDLabel] = string(pod.UID)

	return labels
}

// newPodAnnotations creates pod annotations from v1.Pod.
func newPodAnnotations(pod *v1.Pod) map[string]string {
	return pod.Annotations
}

// newContainerLabels creates container labels from v1.Container and v1.Pod.
func newContainerLabels(container *v1.Container, pod *v1.Pod) map[string]string {
	labels := map[string]string{}
	labels[types.KubernetesPodNameLabel] = pod.Name
	labels[types.KubernetesPodNamespaceLabel] = pod.Namespace
	labels[types.KubernetesPodUIDLabel] = string(pod.UID)
	labels[types.KubernetesContainerNameLabel] = container.Name

	return labels
}

// newContainerAnnotations creates container annotations from v1.Container and v1.Pod.
func newContainerAnnotations(container *v1.Container, pod *v1.Pod, restartCount int, opts *kubecontainer.RunContainerOptions) map[string]string {
	annotations := map[string]string{}

	// Kubelet always overrides device plugin annotations if they are conflicting
	for _, a := range opts.Annotations {
		annotations[a.Name] = a.Value
	}

	annotations[containerHashLabel] = strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		annotations[containerHashWithoutResourcesLabel] = strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	}
	annotations[containerRestartCountLabel] = strconv.Itoa(restartCount)
	annotations[containerTerminationMessagePathLabel] = container.TerminationMessagePath
	annotations[containerTerminationMessagePolicyLabel] = string(container.TerminationMessagePolicy)

	if pod.DeletionGracePeriodSeconds != nil {
		annotations[podDeletionGracePeriodLabel] = strconv.FormatInt(*pod.DeletionGracePeriodSeconds, 10)
	}
	if pod.Spec.TerminationGracePeriodSeconds != nil {
		annotations[podTerminationGracePeriodLabel] = strconv.FormatInt(*pod.Spec.TerminationGracePeriodSeconds, 10)
	}

	if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		// Using json encoding so that the PreStop handler object is readable after writing as a label
		rawPreStop, err := json.Marshal(container.Lifecycle.PreStop)
		if err != nil {
			klog.ErrorS(err, "Unable to marshal lifecycle PreStop handler for container", "containerName", container.Name, "pod", klog.KObj(pod))
		} else {
			annotations[containerPreStopHandlerLabel] = string(rawPreStop)
		}
	}

	if len(container.Ports) > 0 {
		rawContainerPorts, err := json.Marshal(container.Ports)
		if err != nil {
			klog.ErrorS(err, "Unable to marshal container ports for container", "containerName", container.Name, "pod", klog.KObj(pod))
		} else {
			annotations[containerPortsLabel] = string(rawContainerPorts)
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

	// Remain only labels from v1.Pod
	for k, v := range labels {
		if k != types.KubernetesPodNameLabel && k != types.KubernetesPodNamespaceLabel && k != types.KubernetesPodUIDLabel {
			podSandboxInfo.Labels[k] = v
		}
	}

	return podSandboxInfo
}

// getPodSandboxInfoFromAnnotations gets annotatedPodSandboxInfo from annotations.
func getPodSandboxInfoFromAnnotations(annotations map[string]string) *annotatedPodSandboxInfo {
	return &annotatedPodSandboxInfo{
		Annotations: annotations,
	}
}

// getContainerInfoFromLabels gets labeledContainerInfo from labels.
func getContainerInfoFromLabels(labels map[string]string) *labeledContainerInfo {
	return &labeledContainerInfo{
		PodName:       getStringValueFromLabel(labels, types.KubernetesPodNameLabel),
		PodNamespace:  getStringValueFromLabel(labels, types.KubernetesPodNamespaceLabel),
		PodUID:        kubetypes.UID(getStringValueFromLabel(labels, types.KubernetesPodUIDLabel)),
		ContainerName: getStringValueFromLabel(labels, types.KubernetesContainerNameLabel),
	}
}

// getContainerInfoFromAnnotations gets annotatedContainerInfo from annotations.
func getContainerInfoFromAnnotations(annotations map[string]string) *annotatedContainerInfo {
	var err error
	containerInfo := &annotatedContainerInfo{
		TerminationMessagePath:   getStringValueFromLabel(annotations, containerTerminationMessagePathLabel),
		TerminationMessagePolicy: v1.TerminationMessagePolicy(getStringValueFromLabel(annotations, containerTerminationMessagePolicyLabel)),
	}

	if containerInfo.Hash, err = getUint64ValueFromLabel(annotations, containerHashLabel); err != nil {
		klog.ErrorS(err, "Unable to get label value from annotations", "label", containerHashLabel, "annotations", annotations)
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		if containerInfo.HashWithoutResources, err = getUint64ValueFromLabel(annotations, containerHashWithoutResourcesLabel); err != nil {
			klog.ErrorS(err, "Unable to get label value from annotations", "label", containerHashWithoutResourcesLabel, "annotations", annotations)
		}
	}
	if containerInfo.RestartCount, err = getIntValueFromLabel(annotations, containerRestartCountLabel); err != nil {
		klog.ErrorS(err, "Unable to get label value from annotations", "label", containerRestartCountLabel, "annotations", annotations)
	}
	if containerInfo.PodDeletionGracePeriod, err = getInt64PointerFromLabel(annotations, podDeletionGracePeriodLabel); err != nil {
		klog.ErrorS(err, "Unable to get label value from annotations", "label", podDeletionGracePeriodLabel, "annotations", annotations)
	}
	if containerInfo.PodTerminationGracePeriod, err = getInt64PointerFromLabel(annotations, podTerminationGracePeriodLabel); err != nil {
		klog.ErrorS(err, "Unable to get label value from annotations", "label", podTerminationGracePeriodLabel, "annotations", annotations)
	}

	preStopHandler := &v1.LifecycleHandler{}
	if found, err := getJSONObjectFromLabel(annotations, containerPreStopHandlerLabel, preStopHandler); err != nil {
		klog.ErrorS(err, "Unable to get label value from annotations", "label", containerPreStopHandlerLabel, "annotations", annotations)
	} else if found {
		containerInfo.PreStopHandler = preStopHandler
	}

	containerPorts := []v1.ContainerPort{}
	if found, err := getJSONObjectFromLabel(annotations, containerPortsLabel, &containerPorts); err != nil {
		klog.ErrorS(err, "Unable to get label value from annotations", "label", containerPortsLabel, "annotations", annotations)
	} else if found {
		containerInfo.ContainerPorts = containerPorts
	}

	return containerInfo
}

func getStringValueFromLabel(labels map[string]string, label string) string {
	if value, found := labels[label]; found {
		return value
	}
	// Do not report error, because there should be many old containers without label now.
	klog.V(3).InfoS("Container doesn't have requested label, it may be an old or invalid container", "label", label)
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
	klog.V(3).InfoS("Container doesn't have requested label, it may be an old or invalid container", "label", label)
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
	klog.V(3).InfoS("Container doesn't have requested label, it may be an old or invalid container", "label", label)
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
	// If the label is not found, return pointer nil.
	return nil, nil
}

// getJSONObjectFromLabel returns a bool value indicating whether an object is found.
func getJSONObjectFromLabel(labels map[string]string, label string, value interface{}) (bool, error) {
	if strValue, found := labels[label]; found {
		err := json.Unmarshal([]byte(strValue), value)
		return found, err
	}
	// If the label is not found, return not found.
	return false, nil
}
