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
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
)

// This file contains all docker label related constants and functions, including:
//  * label setters and getters
//  * label filters (maybe in the future)

// Version label indicates the version of labels on a container. We don't fully use this now,
// but if we add/remove/change labels in the future, we could bump up the currentLabelVersion,
// and rely on the versionLabel to support backward compatibility.
const (
	versionLabel        = "io.kubernetes.label-version"
	currentLabelVersion = "v1"
)

// Labels supported in kubernetes v1.3
const (
	podNameLabel                   = "io.kubernetes.pod.name"
	podNamespaceLabel              = "io.kubernetes.pod.namespace"
	podUIDLabel                    = "io.kubernetes.pod.uid"
	podDeletionGracePeriodLabel    = "io.kubernetes.pod.deletion-grace-period"
	podTerminationGracePeriodLabel = "io.kubernetes.pod.termination-grace-period"

	containerNameLabel                   = "io.kubernetes.container.name"
	containerHashLabel                   = "io.kubernetes.container.hash"
	containerRestartCountLabel           = "io.kubernetes.container.restart-count"
	containerTerminationMessagePathLabel = "io.kubernetes.container.termination-message-path"
	containerPreStopHandlerLabel         = "io.kubernetes.container.prestop-handler"

	cadvisorPrometheusMetricsLabel = "io.cadvisor.metric.prometheus"
)

// Container information which has been labelled on each docker container
// TODO(random-liu): The type of Hash should be compliance with kubelet container status.
type labelledContainerInfo struct {
	PodName                   string
	PodNamespace              string
	PodUID                    types.UID
	PodDeletionGracePeriod    *int64
	PodTerminationGracePeriod *int64
	Name                      string
	Hash                      string
	RestartCount              int
	TerminationMessagePath    string
	PreStopHandler            *api.Handler
}

func GetContainerName(labels map[string]string) string {
	return labels[containerNameLabel]
}

func GetPodName(labels map[string]string) string {
	return labels[podNameLabel]
}

func GetPodUID(labels map[string]string) string {
	return labels[podUIDLabel]
}

func GetPodNamespace(labels map[string]string) string {
	return labels[podNamespaceLabel]
}

func newLabels(container *api.Container, pod *api.Pod, restartCount int, enableCustomMetrics bool) map[string]string {
	labels := map[string]string{}
	labels[versionLabel] = currentLabelVersion
	labels[podNameLabel] = pod.Name
	labels[podNamespaceLabel] = pod.Namespace
	labels[podUIDLabel] = string(pod.UID)
	if pod.DeletionGracePeriodSeconds != nil {
		labels[podDeletionGracePeriodLabel] = strconv.FormatInt(*pod.DeletionGracePeriodSeconds, 10)
	}
	if pod.Spec.TerminationGracePeriodSeconds != nil {
		labels[podTerminationGracePeriodLabel] = strconv.FormatInt(*pod.Spec.TerminationGracePeriodSeconds, 10)
	}

	labels[containerNameLabel] = container.Name
	labels[containerHashLabel] = strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	labels[containerRestartCountLabel] = strconv.Itoa(restartCount)
	labels[containerTerminationMessagePathLabel] = container.TerminationMessagePath
	if container.Lifecycle != nil && container.Lifecycle.PreStop != nil {
		// Using json enconding so that the PreStop handler object is readable after writing as a label
		rawPreStop, err := json.Marshal(container.Lifecycle.PreStop)
		if err != nil {
			glog.Errorf("Unable to marshal lifecycle PreStop handler for container %q of pod %q: %v", container.Name, format.Pod(pod), err)
		} else {
			labels[containerPreStopHandlerLabel] = string(rawPreStop)
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

func getContainerInfoFromLabel(labels map[string]string) labelledContainerInfo {
	var err error
	containerInfo := labelledContainerInfo{
		PodName:      getStringFromLabel(labels, podNameLabel),
		PodNamespace: getStringFromLabel(labels, podNamespaceLabel),
		PodUID:       types.UID(getStringFromLabel(labels, podUIDLabel)),
		Name:         getStringFromLabel(labels, containerNameLabel),
		Hash:         getStringFromLabel(labels, containerHashLabel),
		TerminationMessagePath: getStringFromLabel(labels, containerTerminationMessagePathLabel),
	}
	containerInfo.RestartCount, err = getIntegerFromLabel(labels, containerRestartCountLabel)
	if err != nil {
		logLabelError(containerInfo, containerRestartCountLabel, err)
	}
	containerInfo.PodDeletionGracePeriod, err = getInt64PointerFromLabel(labels, podDeletionGracePeriodLabel)
	if err != nil {
		logLabelError(containerInfo, podDeletionGracePeriodLabel, err)
	}
	containerInfo.PodTerminationGracePeriod, err = getInt64PointerFromLabel(labels, podTerminationGracePeriodLabel)
	if err != nil {
		logLabelError(containerInfo, podTerminationGracePeriodLabel, err)
	}
	preStopHandler := &api.Handler{}
	found, err := getJsonObjectFromLabel(labels, containerPreStopHandlerLabel, preStopHandler)
	if err != nil {
		logLabelError(containerInfo, containerPreStopHandlerLabel, err)
	} else if found {
		containerInfo.PreStopHandler = preStopHandler
	}

	version := getStringFromLabel(labels, versionLabel)
	if version != currentLabelVersion {
		supplyContainerInfoWithV12Label(labels, &containerInfo)
		supplyContainerInfoWithV11Label(labels, &containerInfo)
	}
	return containerInfo
}

func getStringFromLabel(labels map[string]string, label string) string {
	if value, found := labels[label]; found {
		return value
	}
	// Do not report error, because there should be many old containers without label now.
	glog.V(3).Infof("Container doesn't have label %s, it may be an old or invalid container", label)
	// Return empty string "" for these containers, the caller will get value by other ways.
	return ""
}

func getIntegerFromLabel(labels map[string]string, label string) (int, error) {
	if strValue, found := labels[label]; found {
		intValue, err := strconv.Atoi(strValue)
		if err != nil {
			// This really should not happen. Just set value to 0 to handle this abnormal case
			return 0, parseLabelError(label, strValue, err)
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
			return nil, parseLabelError(label, strValue, err)
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
		if err != nil {
			err = parseLabelError(label, strValue, err)
		}
		return found, err
	}
	// Because it's normal that a container has no PreStopHandler label, don't report any error here.
	return false, nil
}

func logLabelError(containerInfo labelledContainerInfo, label string, err error) {
	glog.Errorf("Unable to get %q for container %q of pod %q: %v", label, containerInfo.Name,
		kubecontainer.BuildPodFullName(containerInfo.PodName, containerInfo.PodNamespace), err)
}

func parseLabelError(label string, value string, err error) error {
	return fmt.Errorf("unable to parse label %q: value=%q, error=%v", label, value, err)
}

// Backward compatible labels for kubernetes v1.2.
// TODO(random-liu): These labels don't follow the docker label naming convention, deprecate this
// when we drop support for v1.2.
const (
	oldPodDeletionGracePeriodLabel          = "io.kubernetes.pod.deletionGracePeriod"
	oldPodTerminationGracePeriodLabel       = "io.kubernetes.pod.terminationGracePeriod"
	oldContainerRestartCountLabel           = "io.kubernetes.container.restartCount"
	oldContainerTerminationMessagePathLabel = "io.kubernetes.container.terminationMessagePath"
	oldContainerPreStopHandlerLabel         = "io.kubernetes.container.preStopHandler"
)

// Both kubernetes and docker recommend to only use [a-z0-9-.] in label keys, so we decide to replace the old labels.
// The old labels are added before v1.2, and should be completely removed after we drop support for v1.2.
// TODO(random-liu): Remove this function when we drop support for v1.2.
func supplyContainerInfoWithV12Label(labels map[string]string, containerInfo *labelledContainerInfo) {
	var err error
	// Only try to get information when there are no corresponding new labels.
	if _, found := labels[podDeletionGracePeriodLabel]; !found {
		containerInfo.PodDeletionGracePeriod, err = getInt64PointerFromLabel(labels, oldPodDeletionGracePeriodLabel)
		if err != nil {
			logLabelError(*containerInfo, oldPodDeletionGracePeriodLabel, err)
		}
	}
	if _, found := labels[podTerminationGracePeriodLabel]; !found {
		containerInfo.PodTerminationGracePeriod, err = getInt64PointerFromLabel(labels, oldPodTerminationGracePeriodLabel)
		if err != nil {
			logLabelError(*containerInfo, oldPodTerminationGracePeriodLabel, err)
		}
	}
	if _, found := labels[containerRestartCountLabel]; !found {
		containerInfo.RestartCount, err = getIntegerFromLabel(labels, oldContainerRestartCountLabel)
		if err != nil {
			logLabelError(*containerInfo, oldPodTerminationGracePeriodLabel, err)
		}
	}
	if _, found := labels[containerTerminationMessagePathLabel]; !found {
		containerInfo.TerminationMessagePath = getStringFromLabel(labels, oldContainerTerminationMessagePathLabel)
	}
	if _, found := labels[containerPreStopHandlerLabel]; !found {
		preStopHandler := &api.Handler{}
		found, err = getJsonObjectFromLabel(labels, oldContainerPreStopHandlerLabel, preStopHandler)
		if err != nil {
			logLabelError(*containerInfo, oldContainerPreStopHandlerLabel, err)
		} else if found {
			containerInfo.PreStopHandler = preStopHandler
		}
	}
}

// Backward compatible labels for kubernetes v1.1 and before.
// TODO(random-liu): Keep this for old containers, remove this when we drop support for v1.1.
const (
	podLabel = "io.kubernetes.pod.data"
)

// The label podLabel is added a long time ago (#7421), it serialized the whole api.Pod to a docker label.
// We want to remove this label because it serialized too much useless information. However kubelet may still work
// with old containers which only have this label for a long time until we completely deprecate the old label.
// Before that to ensure correctness we have to supply information with the old labels when newly added labels
// are not available.
// TODO(random-liu): Remove this function when we can completely remove label podLabel, probably after
// dropping support for v1.1.
func supplyContainerInfoWithV11Label(labels map[string]string, containerInfo *labelledContainerInfo) {
	// Get api.Pod from old label
	var pod *api.Pod
	data, found := labels[podLabel]
	if !found {
		// Don't report any error here, because it's normal that a container has no pod label, especially
		// when we gradually deprecate the old label
		return
	}
	pod = &api.Pod{}
	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), []byte(data), pod); err != nil {
		// If the pod label can't be parsed, we should report an error
		logLabelError(*containerInfo, podLabel, err)
		return
	}
	if containerInfo.PodDeletionGracePeriod == nil {
		containerInfo.PodDeletionGracePeriod = pod.DeletionGracePeriodSeconds
	}
	if containerInfo.PodTerminationGracePeriod == nil {
		containerInfo.PodTerminationGracePeriod = pod.Spec.TerminationGracePeriodSeconds
	}

	// Get api.Container from api.Pod
	var container *api.Container
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
