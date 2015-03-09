/*
Copyright 2015 Google Inc. All rights reserved.

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

package kubelet

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// Kubelet discover pod updates from 3 sources: file, http, and apiserver.
// Pods from non-apiserver sources are called static pods, and API server is
// not aware of the existence of static pods. In order to monitor the status of
// such pods, kubelet create a mirror pod for each static pod via the API
// server.
//
// A mirror pod has the same pod full name (name and namespace) as its static
// counterpart (albeit different metadata such as UID, etc). By leveraging the
// fact that kubelet reports the pod status using the pod full name, the status
// of the mirror pod always reflects the acutal status of the static pod.
// When a static pod gets deleted, the associated orphaned mirror pods will
// also be removed.
//
// This file includes functions to manage the mirror pods.

type mirrorManager interface {
	CreateMirrorPod(api.Pod, string) error
	DeleteMirrorPod(string) error
}

type basicMirrorManager struct {
	// mirror pods are stored in the kubelet directly because they need to be
	// in sync with the internal pods.
	apiserverClient client.Interface
}

func newBasicMirrorManager(apiserverClient client.Interface) *basicMirrorManager {
	return &basicMirrorManager{apiserverClient: apiserverClient}
}

// Creates a mirror pod.
func (self *basicMirrorManager) CreateMirrorPod(pod api.Pod, hostname string) error {
	if self.apiserverClient == nil {
		return nil
	}
	// Indicate that the pod should be scheduled to the current node.
	pod.Spec.Host = hostname
	pod.Annotations[ConfigMirrorAnnotationKey] = MirrorType

	_, err := self.apiserverClient.Pods(NamespaceDefault).Create(&pod)
	return err
}

// Deletes a mirror pod.
func (self *basicMirrorManager) DeleteMirrorPod(podFullName string) error {
	if self.apiserverClient == nil {
		return nil
	}
	name, namespace, err := ParsePodFullName(podFullName)
	if err != nil {
		glog.Errorf("Failed to parse a pod full name %q", podFullName)
		return err
	}
	glog.V(4).Infof("Deleting a mirror pod %q", podFullName)
	if err := self.apiserverClient.Pods(namespace).Delete(name); err != nil {
		glog.Errorf("Failed deleting a mirror pod %q: %v", podFullName, err)
	}
	return nil
}

// Delete all orphaned mirror pods.
func deleteOrphanedMirrorPods(pods []api.Pod, mirrorPods util.StringSet, manager mirrorManager) {
	existingPods := util.NewStringSet()
	for _, pod := range pods {
		existingPods.Insert(GetPodFullName(&pod))
	}
	for podFullName := range mirrorPods {
		if !existingPods.Has(podFullName) {
			manager.DeleteMirrorPod(podFullName)
		}
	}
}

// Helper functions.
func getPodSource(pod *api.Pod) (string, error) {
	if pod.Annotations != nil {
		if source, ok := pod.Annotations[ConfigSourceAnnotationKey]; ok {
			return source, nil
		}
	}
	return "", fmt.Errorf("cannot get source of pod %q", pod.UID)
}

func isStaticPod(pod *api.Pod) bool {
	source, err := getPodSource(pod)
	return err == nil && source != ApiserverSource
}

func isMirrorPod(pod *api.Pod) bool {
	if value, ok := pod.Annotations[ConfigMirrorAnnotationKey]; !ok {
		return false
	} else {
		return value == MirrorType
	}
}

// This function separate the mirror pods from regular pods to
// facilitate pods syncing and mirror pod creation/deletion.
func filterAndCategorizePods(pods []api.Pod) ([]api.Pod, util.StringSet) {
	filteredPods := []api.Pod{}
	mirrorPods := util.NewStringSet()
	for _, pod := range pods {
		name := GetPodFullName(&pod)
		if isMirrorPod(&pod) {
			mirrorPods.Insert(name)
		} else {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods, mirrorPods
}
