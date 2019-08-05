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

package pod

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// MirrorClient knows how to create/delete a mirror pod in the API server.
type MirrorClient interface {
	// CreateMirrorPod creates a mirror pod in the API server for the given
	// pod or returns an error.  The mirror pod will have the same annotations
	// as the given pod as well as an extra annotation containing the hash of
	// the static pod.
	CreateMirrorPod(pod *v1.Pod) error
	// DeleteMirrorPod deletes the mirror pod with the given full name from
	// the API server or returns an error.
	DeleteMirrorPod(podFullName string, uid *types.UID) (bool, error)
}

// basicMirrorClient is a functional MirrorClient.  Mirror pods are stored in
// the kubelet directly because they need to be in sync with the internal
// pods.
type basicMirrorClient struct {
	apiserverClient clientset.Interface
}

// NewBasicMirrorClient returns a new MirrorClient.
func NewBasicMirrorClient(apiserverClient clientset.Interface) MirrorClient {
	return &basicMirrorClient{apiserverClient: apiserverClient}
}

func (mc *basicMirrorClient) CreateMirrorPod(pod *v1.Pod) error {
	if mc.apiserverClient == nil {
		return nil
	}
	// Make a copy of the pod.
	copyPod := *pod
	copyPod.Annotations = make(map[string]string)

	for k, v := range pod.Annotations {
		copyPod.Annotations[k] = v
	}
	hash := getPodHash(pod)
	copyPod.Annotations[kubetypes.ConfigMirrorAnnotationKey] = hash
	apiPod, err := mc.apiserverClient.CoreV1().Pods(copyPod.Namespace).Create(&copyPod)
	if err != nil && errors.IsAlreadyExists(err) {
		// Check if the existing pod is the same as the pod we want to create.
		if h, ok := apiPod.Annotations[kubetypes.ConfigMirrorAnnotationKey]; ok && h == hash {
			return nil
		}
	}
	return err
}

// DeleteMirrorPod deletes a mirror pod.
// It takes the full name of the pod and optionally a UID.  If the UID
// is non-nil, the pod is deleted only if its UID matches the supplied UID.
// It returns whether the pod was actually deleted, and any error returned
// while parsing the name of the pod.
// Non-existence of the pod or UID mismatch is not treated as an error; the
// routine simply returns false in that case.
func (mc *basicMirrorClient) DeleteMirrorPod(podFullName string, uid *types.UID) (bool, error) {
	if mc.apiserverClient == nil {
		return false, nil
	}
	name, namespace, err := kubecontainer.ParsePodFullName(podFullName)
	if err != nil {
		klog.Errorf("Failed to parse a pod full name %q", podFullName)
		return false, err
	}
	klog.V(2).Infof("Deleting a mirror pod %q (uid %#v)", podFullName, uid)
	var GracePeriodSeconds int64
	GracePeriodSeconds = 0
	if err := mc.apiserverClient.CoreV1().Pods(namespace).Delete(name, &metav1.DeleteOptions{GracePeriodSeconds: &GracePeriodSeconds, Preconditions: &metav1.Preconditions{UID: uid}}); err != nil {
		// Unfortunately, there's no generic error for failing a precondition
		if !(errors.IsNotFound(err) || errors.IsConflict(err)) {
			// We should return the error here, but historically this routine does
			// not return an error unless it can't parse the pod name
			klog.Errorf("Failed deleting a mirror pod %q: %v", podFullName, err)
		}
		return false, nil
	}
	return true, nil
}

// IsStaticPod returns true if the passed Pod is static.
func IsStaticPod(pod *v1.Pod) bool {
	source, err := kubetypes.GetPodSource(pod)
	return err == nil && source != kubetypes.ApiserverSource
}

// IsMirrorPod returns true if the passed Pod is a Mirror Pod.
func IsMirrorPod(pod *v1.Pod) bool {
	_, ok := pod.Annotations[kubetypes.ConfigMirrorAnnotationKey]
	return ok
}

func getHashFromMirrorPod(pod *v1.Pod) (string, bool) {
	hash, ok := pod.Annotations[kubetypes.ConfigMirrorAnnotationKey]
	return hash, ok
}

func getPodHash(pod *v1.Pod) string {
	// The annotation exists for all static pods.
	return pod.Annotations[kubetypes.ConfigHashAnnotationKey]
}
