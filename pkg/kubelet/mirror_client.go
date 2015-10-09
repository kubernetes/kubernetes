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

package kubelet

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// Mirror client is used to create/delete a mirror pod.

type mirrorClient interface {
	CreateMirrorPod(*api.Pod) error
	DeleteMirrorPod(string) error
}

type basicMirrorClient struct {
	// mirror pods are stored in the kubelet directly because they need to be
	// in sync with the internal pods.
	apiserverClient client.Interface
}

func newBasicMirrorClient(apiserverClient client.Interface) *basicMirrorClient {
	return &basicMirrorClient{apiserverClient: apiserverClient}
}

// Creates a mirror pod.
func (mc *basicMirrorClient) CreateMirrorPod(pod *api.Pod) error {
	if mc.apiserverClient == nil {
		return nil
	}
	// Make a copy of the pod.
	copyPod := *pod
	copyPod.Annotations = make(map[string]string)

	for k, v := range pod.Annotations {
		copyPod.Annotations[k] = v
	}
	copyPod.Annotations[kubetypes.ConfigMirrorAnnotationKey] = kubetypes.MirrorType

	_, err := mc.apiserverClient.Pods(copyPod.Namespace).Create(&copyPod)
	return err
}

// Deletes a mirror pod.
func (mc *basicMirrorClient) DeleteMirrorPod(podFullName string) error {
	if mc.apiserverClient == nil {
		return nil
	}
	name, namespace, err := kubecontainer.ParsePodFullName(podFullName)
	if err != nil {
		glog.Errorf("Failed to parse a pod full name %q", podFullName)
		return err
	}
	glog.V(4).Infof("Deleting a mirror pod %q", podFullName)
	if err := mc.apiserverClient.Pods(namespace).Delete(name, api.NewDeleteOptions(0)); err != nil {
		glog.Errorf("Failed deleting a mirror pod %q: %v", podFullName, err)
	}
	return nil
}

// Helper functions.
func getPodSource(pod *api.Pod) (string, error) {
	if pod.Annotations != nil {
		if source, ok := pod.Annotations[kubetypes.ConfigSourceAnnotationKey]; ok {
			return source, nil
		}
	}
	return "", fmt.Errorf("cannot get source of pod %q", pod.UID)
}

func isStaticPod(pod *api.Pod) bool {
	source, err := getPodSource(pod)
	return err == nil && source != kubetypes.ApiserverSource
}

func isMirrorPod(pod *api.Pod) bool {
	if value, ok := pod.Annotations[kubetypes.ConfigMirrorAnnotationKey]; !ok {
		return false
	} else {
		return value == kubetypes.MirrorType
	}
}
