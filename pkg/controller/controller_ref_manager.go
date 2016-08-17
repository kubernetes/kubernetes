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

package controller

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
)

type PodControllerRefManager struct {
	podControl         PodControlInterface
	controllerObject   api.ObjectMeta
	controllerSelector labels.Selector
	controllerKind     unversioned.GroupVersionKind
}

// NewPodControllerRefManager returns a PodControllerRefManager that exposes
// methods to manage the controllerRef of pods.
func NewPodControllerRefManager(
	podControl PodControlInterface,
	controllerObject api.ObjectMeta,
	controllerSelector labels.Selector,
	controllerKind unversioned.GroupVersionKind,
) *PodControllerRefManager {
	return &PodControllerRefManager{podControl, controllerObject, controllerSelector, controllerKind}
}

// Classify first filters out inactive pods, then it classify the remaining pods
// into three categories: 1. matchesAndControlled are the pods whose labels
// match the selector of the RC, and have a controllerRef pointing to the
// controller 2. matchesNeedsController are the pods whose labels match the RC,
// but don't have a controllerRef. (Pods with matching labels but with a
// controllerRef pointing to other object are ignored) 3. controlledDoesNotMatch
// are the pods that have a controllerRef pointing to the controller, but their
// labels no longer match the selector.
func (m *PodControllerRefManager) Classify(pods []*api.Pod) (
	matchesAndControlled []*api.Pod,
	matchesNeedsController []*api.Pod,
	controlledDoesNotMatch []*api.Pod) {
	for i := range pods {
		pod := pods[i]
		if !IsPodActive(pod) {
			glog.V(4).Infof("Ignoring inactive pod %v/%v in state %v, deletion time %v",
				pod.Namespace, pod.Name, pod.Status.Phase, pod.DeletionTimestamp)
			continue
		}
		controllerRef := getControllerOf(pod.ObjectMeta)
		if controllerRef != nil {
			if controllerRef.UID == m.controllerObject.UID {
				// already controlled
				if m.controllerSelector.Matches(labels.Set(pod.Labels)) {
					matchesAndControlled = append(matchesAndControlled, pod)
				} else {
					controlledDoesNotMatch = append(controlledDoesNotMatch, pod)
				}
			} else {
				// ignoring the pod controlled by other controller
				glog.V(4).Infof("Ignoring pod %v/%v, it's owned by [%s/%s, name: %s, uid: %s]",
					pod.Namespace, pod.Name, controllerRef.APIVersion, controllerRef.Kind, controllerRef.Name, controllerRef.UID)
				continue
			}
		} else {
			if !m.controllerSelector.Matches(labels.Set(pod.Labels)) {
				continue
			}
			matchesNeedsController = append(matchesNeedsController, pod)
		}
	}
	return matchesAndControlled, matchesNeedsController, controlledDoesNotMatch
}

// getControllerOf returns the controllerRef if controllee has a controller,
// otherwise returns nil.
func getControllerOf(controllee api.ObjectMeta) *api.OwnerReference {
	for _, owner := range controllee.OwnerReferences {
		// controlled by other controller
		if owner.Controller != nil && *owner.Controller == true {
			return &owner
		}
	}
	return nil
}

// AdoptPod sends a patch to take control of the pod. It returns the error if
// the patching fails.
func (m *PodControllerRefManager) AdoptPod(pod *api.Pod) error {
	// we should not adopt any pods if the controller is about to be deleted
	if m.controllerObject.DeletionTimestamp != nil {
		return fmt.Errorf("cancel the adopt attempt for pod %s because the controlller is being deleted",
			strings.Join([]string{pod.Namespace, pod.Name, string(pod.UID)}, "_"))
	}
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controllerObject.Name, m.controllerObject.UID, pod.UID)
	return m.podControl.PatchPod(pod.Namespace, pod.Name, []byte(addControllerPatch))
}

// ReleasePod sends a patch to free the pod from the control of the controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *PodControllerRefManager) ReleasePod(pod *api.Pod) error {
	glog.V(2).Infof("patching pod %s_%s to remove its controllerRef to %s/%s:%s",
		pod.Namespace, pod.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controllerObject.Name)
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controllerObject.UID, pod.UID)
	err := m.podControl.PatchPod(pod.Namespace, pod.Name, []byte(deleteOwnerRefPatch))
	if err != nil {
		if errors.IsNotFound(err) {
			// If the pod no longer exists, ignore it.
			return nil
		}
		if errors.IsInvalid(err) {
			// Invalid error will be returned in two cases: 1. the pod
			// has no owner reference, 2. the uid of the pod doesn't
			// match, which means the pod is deleted and then recreated.
			// In both cases, the error can be ignored.

			// TODO: If the pod has owner references, but none of them
			// has the owner.UID, server will silently ignore the patch.
			// Investigate why.
			return nil
		}
	}
	return err
}
