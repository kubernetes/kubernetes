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
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/custommetrics"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/runtime"
	kubetypes "k8s.io/kubernetes/pkg/types"
)

// This file contains all docker label related constants and functions, including:
//  * label setters and getters
//  * label filters (maybe in the future)

// Version label indicates the version of labels on a container. Whenever we make incompatible changes
// to the labels, we should bump up the current label version.
const (
	versionLabel        = "io.kubernetes.label-version"
	currentLabelVersion = "v1"
)

// Labels supported now
const (
	// Labels added from kubernetes v1.1. Invluding types.Kubernetes.KubernetesPodNameLabel,
	// types.KubernetesContainerNameLabel
	// Notice that the podNameLabel contains namespaced pod name in v1.1. However because we don't
	// really rely on the pod name now, it should be fine.
	// TODO(random-liu): Remove the comment above when we drop support for v1.1

	// Labels added from kubernetes v1.2. Including types.KubernetesPodNamespaceLabel, types.KubernetesPodUIDLabel
	containerHashLabel             = "io.kubernetes.container.hash"
	cadvisorPrometheusMetricsLabel = "io.cadvisor.metric.prometheus"

	// Labels added from kubernetes v1.3
	containerInfoLabel = "io.kubernetes.container.info"
)

// containerMeta is the metadata we need to labelled on each docker container. Each field will be one
// label on the container. We could also use the label to select specific containers in the future.
// Notice that the container meta should seldom or never change.
type containerMeta struct {
	PodName      string
	PodNamespace string
	PodUID       kubetypes.UID
	Name         string
	Hash         string
}

// containerSpec is the pod and container spec we need to checkpoint for each container, it is part
// of containerInfo.
type containerSpec struct {
	Ports                     []api.ContainerPort `json:"ports"`
	TerminationMessagePath    string              `json:"terminationMessagePath"`
	PodDeletionGracePeriod    *int64              `json:"podDeletionGracePeriod,omitempty"`
	PodTerminationGracePeriod *int64              `json:"podTerminationGracePeriod,omitempty"`
	PreStopHandler            *api.Handler        `json:"preStopHandler,omitempty"`
}

// containerInfo is all the information we need to checkpoint as the docker label for each docker
// container. The whole ContainerInfo will be serialized and checkpointed as one label.
type containerInfo struct {
	containerMeta `json:"-"`
	containerSpec
	// LastStatus is the status of the previous container. If the current container is the
	// first one, this field should be nil. With this, we can remove historical containers as
	// soon as the new one is created.
	// However, because kubelet may work with old containers without containerInfo, we still
	// can't rely on this field. We only use it to calculate restart count for now.
	// TODO(random-liu): Start to rely on this after we drop support for v1.2
	LastStatus *kubecontainer.ContainerStatus `json:"lastStatus, omitempty"`
}

func newLabels(container *api.Container, pod *api.Pod, lastStatus *kubecontainer.ContainerStatus, enableCustomMetrics bool) map[string]string {
	labels := map[string]string{}
	// Apply container meta
	labels[versionLabel] = currentLabelVersion
	labels[types.KubernetesPodNameLabel] = pod.Name
	labels[types.KubernetesPodNamespaceLabel] = pod.Namespace
	labels[types.KubernetesPodUIDLabel] = string(pod.UID)
	labels[types.KubernetesContainerNameLabel] = container.Name
	labels[containerHashLabel] = strconv.FormatUint(kubecontainer.HashContainer(container), 16)

	// Apply container info
	info := &containerInfo{
		containerSpec: containerSpec{
			Ports: container.Ports,
			TerminationMessagePath:    container.TerminationMessagePath,
			PodDeletionGracePeriod:    pod.DeletionGracePeriodSeconds,
			PodTerminationGracePeriod: pod.Spec.TerminationGracePeriodSeconds,
		},
		LastStatus: lastStatus,
	}
	if container.Lifecycle != nil {
		info.PreStopHandler = container.Lifecycle.PreStop
	}
	raw, err := json.Marshal(info)
	if err != nil {
		glog.Errorf("Unable to marshal container info %+v for container %q of pod %q: %v", *info, container.Name, format.Pod(pod), err)
	}
	labels[containerInfoLabel] = string(raw)

	// Special label used by cadvisor
	if enableCustomMetrics {
		path, err := custommetrics.GetCAdvisorCustomMetricsDefinitionPath(container)
		if path != nil && err == nil {
			labels[cadvisorPrometheusMetricsLabel] = *path
		}
	}
	return labels
}

func getContainerInfoFromLabel(labels map[string]string) containerInfo {
	meta := containerMeta{
		PodName:      types.GetPodName(labels),
		PodNamespace: types.GetPodNamespace(labels),
		PodUID:       kubetypes.UID(types.GetPodUID(labels)),
		Name:         types.GetContainerName(labels),
		Hash:         getStringFromLabel(labels, containerHashLabel),
	}
	info := containerInfo{containerMeta: meta}
	version := getStringFromLabel(labels, versionLabel)
	if version != currentLabelVersion {
		updateContainerInfoWithOldLabels(labels, &info)
		return info
	}
	err := getJsonObjectFromLabel(labels, containerInfoLabel, &info)
	if err != nil {
		logLabelError(meta, containerInfoLabel, err)
	}
	return info
}

func getStringFromLabel(labels map[string]string, label string) string {
	value, found := labels[label]
	if !found {
		// Do not report error for now, because there should be many old containers without label now.
		glog.V(4).Infof("Container doesn't have label %s, it may be an old or invalid container", label)
		// Return empty string "" for these containers, the caller will get value by other means.
		return ""
	}
	return value
}

// getJsonObjectFromLabel returns a bool value indicating whether an object is found
func getJsonObjectFromLabel(labels map[string]string, label string, value interface{}) error {
	strValue, found := labels[label]
	if !found {
		return labelNotFoundError(label)
	}
	err := json.Unmarshal([]byte(strValue), value)
	if err != nil {
		return parseLabelError(label, strValue, err)
	}
	return nil
}

func logLabelError(meta containerMeta, label string, err error) {
	glog.Errorf("Unable to get %q for container %q of pod %q: %v", label, meta.Name,
		kubecontainer.BuildPodFullName(meta.PodName, meta.PodNamespace), err)
}

func parseLabelError(label string, value string, err error) error {
	return fmt.Errorf("unable to parse label %q: value=%q, error=%v", label, value, err)
}

func labelNotFoundError(label string) error {
	return fmt.Errorf("label %q not found", label)
}

// TODO(random-liu): Remove the following code when we drop support for v1.2.
// Backward compatible labels for kubernetes v1.1 and before.
const (
	// podLabel is only set when the container has PreStop handler
	podLabel                       = "io.kubernetes.pod.data"
	podTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"
)

// The label podLabel is added a long time ago (#7421), it serialized the whole api.Pod to a docker label.
// We want to remove this label because it serialized too much useless information. However kubelet may still work
// with old containers which only have this label for a long time until we completely deprecate the old label.
// Before that to ensure correctness we have to supply information with the old labels when newly added labels
// are not available.
func updateContainerInfoWithOldLabels(labels map[string]string, info *containerInfo) {
	if s, found := labels[podTerminationGracePeriodLabel]; found {
		terminationGracePeriod, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			logLabelError(info.containerMeta, podTerminationGracePeriodLabel, err)
		} else {
			info.PodTerminationGracePeriod = &terminationGracePeriod
		}
	}
	// Get api.Pod from old label
	var pod *api.Pod
	data, found := labels[podLabel]
	if !found {
		return
	}
	pod = &api.Pod{}
	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), []byte(data), pod); err != nil {
		logLabelError(info.containerMeta, podLabel, err)
		return
	}
	info.PodDeletionGracePeriod = pod.DeletionGracePeriodSeconds
	info.PodTerminationGracePeriod = pod.Spec.TerminationGracePeriodSeconds

	// Get api.Container from api.Pod
	var container *api.Container
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == info.Name {
			container = &pod.Spec.Containers[i]
			break
		}
	}
	if container == nil {
		glog.Errorf("Unable to find container %q in pod %q", info.Name, format.Pod(pod))
		return
	}
	info.Ports = container.Ports
	info.TerminationMessagePath = container.TerminationMessagePath
	info.PreStopHandler = container.Lifecycle.PreStop
}
