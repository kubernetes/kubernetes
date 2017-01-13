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
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

type PodControllerRefManager struct {
	podControl         PodControlInterface
	controllerObject   v1.ObjectMeta
	controllerSelector labels.Selector
	controllerKind     schema.GroupVersionKind
}

// NewPodControllerRefManager returns a PodControllerRefManager that exposes
// methods to manage the controllerRef of pods.
func NewPodControllerRefManager(
	podControl PodControlInterface,
	controllerObject v1.ObjectMeta,
	controllerSelector labels.Selector,
	controllerKind schema.GroupVersionKind,
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
func (m *PodControllerRefManager) Classify(pods []*v1.Pod) (
	matchesAndControlled []*v1.Pod,
	matchesNeedsController []*v1.Pod,
	controlledDoesNotMatch []*v1.Pod) {
	for i := range pods {
		pod := pods[i]
		if !IsPodActive(pod) {
			glog.V(4).Infof("Ignoring inactive pod %v/%v in state %v, deletion time %v",
				pod.Namespace, pod.Name, pod.Status.Phase, pod.DeletionTimestamp)
			continue
		}
		controllerRef := GetControllerOf(&pod.ObjectMeta)
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

// GetControllerOf returns the controllerRef if controllee has a controller,
// otherwise returns nil.
func GetControllerOf(controllee *v1.ObjectMeta) *metav1.OwnerReference {
	for i := range controllee.OwnerReferences {
		owner := &controllee.OwnerReferences[i]
		// controlled by other controller
		if owner.Controller != nil && *owner.Controller == true {
			return owner
		}
	}
	return nil
}

// AdoptPod sends a patch to take control of the pod. It returns the error if
// the patching fails.
func (m *PodControllerRefManager) AdoptPod(pod *v1.Pod) error {
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
func (m *PodControllerRefManager) ReleasePod(pod *v1.Pod) error {
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

// ReplicaSetControllerRefManager is used to manage controllerRef of ReplicaSets.
// Three methods are defined on this object 1: Classify 2: AdoptReplicaSet and
// 3: ReleaseReplicaSet which are used to classify the ReplicaSets into appropriate
// categories and accordingly adopt or release them. See comments on these functions
// for more details.
type ReplicaSetControllerRefManager struct {
	rsControl          RSControlInterface
	controllerObject   v1.ObjectMeta
	controllerSelector labels.Selector
	controllerKind     schema.GroupVersionKind
}

// NewReplicaSetControllerRefManager returns a ReplicaSetControllerRefManager that exposes
// methods to manage the controllerRef of ReplicaSets.
func NewReplicaSetControllerRefManager(
	rsControl RSControlInterface,
	controllerObject v1.ObjectMeta,
	controllerSelector labels.Selector,
	controllerKind schema.GroupVersionKind,
) *ReplicaSetControllerRefManager {
	return &ReplicaSetControllerRefManager{rsControl, controllerObject, controllerSelector, controllerKind}
}

// Classify, classifies the ReplicaSets into three categories:
// 1. matchesAndControlled are the ReplicaSets whose labels
// match the selector of the Deployment, and have a controllerRef pointing to the
// Deployment.
// 2. matchesNeedsController are ReplicaSets ,whose labels match the Deployment,
// but don't have a controllerRef. (ReplicaSets with matching labels but with a
// controllerRef pointing to other object are ignored)
// 3. controlledDoesNotMatch are the ReplicaSets that have a controllerRef pointing
// to the Deployment, but their labels no longer match the selector.
func (m *ReplicaSetControllerRefManager) Classify(replicaSets []*extensions.ReplicaSet) (
	matchesAndControlled []*extensions.ReplicaSet,
	matchesNeedsController []*extensions.ReplicaSet,
	controlledDoesNotMatch []*extensions.ReplicaSet) {
	for i := range replicaSets {
		replicaSet := replicaSets[i]
		controllerRef := GetControllerOf(&replicaSet.ObjectMeta)
		if controllerRef != nil {
			if controllerRef.UID != m.controllerObject.UID {
				// ignoring the ReplicaSet controlled by other Deployment
				glog.V(4).Infof("Ignoring ReplicaSet %v/%v, it's owned by [%s/%s, name: %s, uid: %s]",
					replicaSet.Namespace, replicaSet.Name, controllerRef.APIVersion, controllerRef.Kind, controllerRef.Name, controllerRef.UID)
				continue
			}
			// already controlled by this Deployment
			if m.controllerSelector.Matches(labels.Set(replicaSet.Labels)) {
				matchesAndControlled = append(matchesAndControlled, replicaSet)
			} else {
				controlledDoesNotMatch = append(controlledDoesNotMatch, replicaSet)
			}
		} else {
			if !m.controllerSelector.Matches(labels.Set(replicaSet.Labels)) {
				continue
			}
			matchesNeedsController = append(matchesNeedsController, replicaSet)
		}
	}
	return matchesAndControlled, matchesNeedsController, controlledDoesNotMatch
}

// AdoptReplicaSet sends a patch to take control of the ReplicaSet. It returns the error if
// the patching fails.
func (m *ReplicaSetControllerRefManager) AdoptReplicaSet(replicaSet *extensions.ReplicaSet) error {
	// we should not adopt any ReplicaSets if the Deployment is about to be deleted
	if m.controllerObject.DeletionTimestamp != nil {
		return fmt.Errorf("cancel the adopt attempt for RS %s because the controller %v is being deleted",
			strings.Join([]string{replicaSet.Namespace, replicaSet.Name, string(replicaSet.UID)}, "_"), m.controllerObject.Name)
	}
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controllerObject.Name, m.controllerObject.UID, replicaSet.UID)
	return m.rsControl.PatchReplicaSet(replicaSet.Namespace, replicaSet.Name, []byte(addControllerPatch))
}

// ReleaseReplicaSet sends a patch to free the ReplicaSet from the control of the Deployment controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *ReplicaSetControllerRefManager) ReleaseReplicaSet(replicaSet *extensions.ReplicaSet) error {
	glog.V(2).Infof("patching ReplicaSet %s_%s to remove its controllerRef to %s/%s:%s",
		replicaSet.Namespace, replicaSet.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controllerObject.Name)
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controllerObject.UID, replicaSet.UID)
	err := m.rsControl.PatchReplicaSet(replicaSet.Namespace, replicaSet.Name, []byte(deleteOwnerRefPatch))
	if err != nil {
		if errors.IsNotFound(err) {
			// If the ReplicaSet no longer exists, ignore it.
			return nil
		}
		if errors.IsInvalid(err) {
			// Invalid error will be returned in two cases: 1. the ReplicaSet
			// has no owner reference, 2. the uid of the ReplicaSet doesn't
			// match, which means the ReplicaSet is deleted and then recreated.
			// In both cases, the error can be ignored.
			return nil
		}
	}
	return err
}
