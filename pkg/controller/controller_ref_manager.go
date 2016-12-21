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
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime/schema"
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
		controllerRef := GetControllerOf(pod.ObjectMeta)
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
func GetControllerOf(controllee v1.ObjectMeta) *metav1.OwnerReference {
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

// RoleBindingControllerRefManager is used to manage controllerRef of RoleBindings.
// Three methods are defined on this object 1: Classify 2: AdoptRole and
// 3: ReleaseRole which are used to classify the Roles into appropriate
// categories and accordingly adopt or release them. See comments on these functions
// for more details.
type RoleBindingControllerRefManager struct {
	roleBindingControl RoleBindingControlInterface
	controllerObject   api.ObjectMeta
	controllerKind     schema.GroupVersionKind
}

// NewRoleBindingControllerRefManager returns a RoleBindingControllerRefManager that exposes
// methods to manage the controllerRef of Roles.
func NewRoleBindingControllerRefManager(
	roleBindingControl RoleBindingControlInterface,
	controllerObject api.ObjectMeta,
	controllerKind schema.GroupVersionKind,
) *RoleBindingControllerRefManager {
	return &RoleBindingControllerRefManager{roleBindingControl, controllerObject, controllerKind}
}

// Classify, classifies the RoleBindings into three categories:
// 1. matchesAndControlled are the Roles match the roleref of the Rolebinding,
// and have a controllerRef pointing to the Role.
// 2. matchesNeedsController are RoleBindings ,whose roleref match the Cluster,
// but don't have a controllerRef. (RoleBindings with matching roleref but with a
// controllerRef pointing to other object are ignored)
// 3. controlledDoesNotMatch are the RoleBindings that have a controllerRef pointing
// to the Cluster, but the roleref no longer match the Role.
func (m *RoleBindingControllerRefManager) Classify(bindings []rbac.RoleBinding) (
	matchesAndControlled []*rbac.RoleBinding,
	matchesNeedsController []*rbac.RoleBinding,
	controlledDoesNotMatch []*rbac.RoleBinding) {
	for i := range bindings {
		binding := bindings[i]
		out := v1.ObjectMeta{}
		if err := v1.Convert_api_ObjectMeta_To_v1_ObjectMeta(&binding.ObjectMeta, &out, nil); err != nil {
			return nil, nil, nil
		}
		controllerRef := GetControllerOf(out)
		if controllerRef != nil {
			if controllerRef.UID != m.controllerObject.UID {
				// ignoring the RoleBinding controlled by other Role
				glog.V(4).Infof("Ignoring RoleBinding %v/%v, it's not owned by [%s/%s, name: %s, uid: %s]",
					binding.Namespace, binding.Name, controllerRef.APIVersion, controllerRef.Kind, controllerRef.Name, controllerRef.UID)
				continue
			}
			// already controlled by this Role
			if m.controllerObject.Name == binding.RoleRef.Name && m.controllerKind.Kind == binding.RoleRef.Kind {
				matchesAndControlled = append(matchesAndControlled, &binding)
			} else {
				controlledDoesNotMatch = append(controlledDoesNotMatch, &binding)
			}
		} else {
			if m.controllerObject.Name != binding.RoleRef.Name || m.controllerKind.Kind != binding.RoleRef.Kind {
				continue
			}
			matchesNeedsController = append(matchesNeedsController, &binding)
		}
	}
	return matchesAndControlled, matchesNeedsController, controlledDoesNotMatch
}

// AdoptRole sends a patch to take control of the RoleBinding. It returns the error if
// the patching fails.
func (m *RoleBindingControllerRefManager) AdoptRoleBinding(binding *rbac.RoleBinding) error {
	// we should not adopt any Roles if the RoleBinding is about to be deleted
	if m.controllerObject.DeletionTimestamp != nil {
		return fmt.Errorf("cancel the adopt attempt for RoleBinding %s because the controller %v is being deleted",
			strings.Join([]string{binding.Namespace, binding.Name, string(binding.UID)}, "_"), m.controllerObject.Name)
	}
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controllerObject.Name, m.controllerObject.UID, binding.UID)
	return m.roleBindingControl.PatchRoleBinding(binding.Namespace, binding.Name, []byte(addControllerPatch))
}

// ReleaseRole sends a patch to free the RoleBinding from the control of the Role controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *RoleBindingControllerRefManager) ReleaseRoleBinding(binding *rbac.RoleBinding) error {
	glog.V(2).Infof("patching RoleBinding %s_%s to remove its controllerRef to %s/%s:%s",
		binding.Namespace, binding.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controllerObject.Name)
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controllerObject.UID, binding.UID)
	err := m.roleBindingControl.PatchRoleBinding(binding.Namespace, binding.Name, []byte(deleteOwnerRefPatch))
	if err != nil {
		if errors.IsNotFound(err) {
			// If the RoleBinding no longer exists, ignore it.
			return nil
		}
		if errors.IsInvalid(err) {
			// Invalid error will be returned in two cases: 1. the RoleBinding
			// has no owner reference, 2. the uid of the RoleBinding doesn't
			// match, which means the RoleBinding is deleted and then recreated.
			// In both cases, the error can be ignored.
			return nil
		}
	}
	return err
}

// ClusterRoleBindingControllerRefManager is used to manage controllerRef of ClusterRoleBindings.
// Three methods are defined on this object 1: Classify 2: AdoptRole and
// 3: ReleaseRole which are used to classify the ClusterRoles into appropriate
// categories and accordingly adopt or release them. See comments on these functions
// for more details.
type ClusterRoleBindingControllerRefManager struct {
	clusterRoleBindingControl ClusterRoleBindingControlInterface
	controllerObject          api.ObjectMeta
	controllerKind            schema.GroupVersionKind
}

// NewClusterRoleBindingControllerRefManager returns a ClusterRoleBindingControllerRefManager that exposes
// methods to manage the controllerRef of ClusterRoles.
func NewClusterRoleBindingControllerRefManager(
	clusterRoleBindingControl ClusterRoleBindingControlInterface,
	controllerObject api.ObjectMeta,
	controllerKind schema.GroupVersionKind,
) *ClusterRoleBindingControllerRefManager {
	return &ClusterRoleBindingControllerRefManager{clusterRoleBindingControl, controllerObject, controllerKind}
}

// Classify, classifies the ClusterRoleBindings into three categories:
// 1. matchesAndControlled are the ClusterRoles match the roleref of the ClusterRolebinding,
// and have a controllerRef pointing to the ClusterRole.
// 2. matchesNeedsController are ClusterRoleBindings ,whose roleref match the Cluster,
// but don't have a controllerRef. (ClusterRoleBindings with matching roleref but with a
// controllerRef pointing to other object are ignored)
// 3. controlledDoesNotMatch are the ClusterRoleBindings that have a controllerRef pointing
// to the Cluster, but the roleref no longer match the ClusterRole.
func (m *ClusterRoleBindingControllerRefManager) Classify(bindings []rbac.ClusterRoleBinding) (
	matchesAndControlled []*rbac.ClusterRoleBinding,
	matchesNeedsController []*rbac.ClusterRoleBinding,
	controlledDoesNotMatch []*rbac.ClusterRoleBinding) {
	for i := range bindings {
		binding := bindings[i]
		out := v1.ObjectMeta{}
		if err := v1.Convert_api_ObjectMeta_To_v1_ObjectMeta(&binding.ObjectMeta, &out, nil); err != nil {
			return nil, nil, nil
		}
		controllerRef := GetControllerOf(out)
		if controllerRef != nil {
			if controllerRef.UID != m.controllerObject.UID {
				// ignoring the RoleBinding controlled by other Role
				glog.V(4).Infof("Ignoring RoleBinding %v/%v, it's not owned by [%s/%s, name: %s, uid: %s]",
					binding.Namespace, binding.Name, controllerRef.APIVersion, controllerRef.Kind, controllerRef.Name, controllerRef.UID)
				continue
			}
			// already controlled by this Role
			if m.controllerObject.Name == binding.RoleRef.Name {
				matchesAndControlled = append(matchesAndControlled, &binding)
			} else {
				controlledDoesNotMatch = append(controlledDoesNotMatch, &binding)
			}
		} else {
			if m.controllerObject.Name != binding.RoleRef.Name {
				continue
			}
			matchesNeedsController = append(matchesNeedsController, &binding)
		}
	}
	return matchesAndControlled, matchesNeedsController, controlledDoesNotMatch
}

// AdoptClusterRoleBinding sends a patch to take control of the ClusterRoleBinding. It returns the error if
// the patching fails.
func (m *ClusterRoleBindingControllerRefManager) AdoptClusterRoleBinding(binding *rbac.ClusterRoleBinding) error {
	// we should not adopt any Roles if the ClusterRoleBinding is about to be deleted
	if m.controllerObject.DeletionTimestamp != nil {
		return fmt.Errorf("cancel the adopt attempt for ClusterRoleBinding %s because the controller %v is being deleted",
			strings.Join([]string{binding.Name, string(binding.UID)}, "_"), m.controllerObject.Name)
	}
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controllerObject.Name, m.controllerObject.UID, binding.UID)
	return m.clusterRoleBindingControl.PatchClusterRoleBinding(binding.Name, []byte(addControllerPatch))
}

// ReleaseClusterRoleBinding sends a patch to free the ClusterRoleBinding from the control of the ClusterRole controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *ClusterRoleBindingControllerRefManager) ReleaseClusterRoleBinding(binding *rbac.ClusterRoleBinding) error {
	glog.V(2).Infof("patching ClusterRoleBinding %s_%s to remove its controllerRef to %s/%s:%s",
		binding.Namespace, binding.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controllerObject.Name)
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controllerObject.UID, binding.UID)
	err := m.clusterRoleBindingControl.PatchClusterRoleBinding(binding.Name, []byte(deleteOwnerRefPatch))
	if err != nil {
		if errors.IsNotFound(err) {
			// If the ClusterRoleBinding no longer exists, ignore it.
			return nil
		}
		if errors.IsInvalid(err) {
			// Invalid error will be returned in two cases: 1. the ClusterRoleBinding
			// has no owner reference, 2. the uid of the ClusterRoleBinding doesn't
			// match, which means the ClusterRoleBinding is deleted and then recreated.
			// In both cases, the error can be ignored.
			return nil
		}
	}
	return err
}
