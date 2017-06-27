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
	"sync"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// GetControllerOf returns the controllerRef if controllee has a controller,
// otherwise returns nil.
func GetControllerOf(controllee metav1.Object) *metav1.OwnerReference {
	ownerRefs := controllee.GetOwnerReferences()
	for i := range ownerRefs {
		owner := &ownerRefs[i]
		if owner.Controller != nil && *owner.Controller == true {
			return owner
		}
	}
	return nil
}

type baseControllerRefManager struct {
	controller metav1.Object
	selector   labels.Selector

	canAdoptErr  error
	canAdoptOnce sync.Once
	canAdoptFunc func() error
}

func (m *baseControllerRefManager) canAdopt() error {
	m.canAdoptOnce.Do(func() {
		if m.canAdoptFunc != nil {
			m.canAdoptErr = m.canAdoptFunc()
		}
	})
	return m.canAdoptErr
}

// claimObject tries to take ownership of an object for this controller.
//
// It will reconcile the following:
//   * Adopt orphans if the match function returns true.
//   * Release owned objects if the match function returns false.
//
// A non-nil error is returned if some form of reconciliation was attemped and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The returned boolean indicates whether you now
// own the object.
//
// No reconciliation will be attempted if the controller is being deleted.
func (m *baseControllerRefManager) claimObject(obj metav1.Object, match func(metav1.Object) bool, adopt, release func(metav1.Object) error) (bool, error) {
	controllerRef := GetControllerOf(obj)
	if controllerRef != nil {
		if controllerRef.UID != m.controller.GetUID() {
			// Owned by someone else. Ignore.
			return false, nil
		}
		if match(obj) {
			// We already own it and the selector matches.
			// Return true (successfully claimed) before checking deletion timestamp.
			// We're still allowed to claim things we already own while being deleted
			// because doing so requires taking no actions.
			return true, nil
		}
		// Owned by us but selector doesn't match.
		// Try to release, unless we're being deleted.
		if m.controller.GetDeletionTimestamp() != nil {
			return false, nil
		}
		if err := release(obj); err != nil {
			// If the pod no longer exists, ignore the error.
			if errors.IsNotFound(err) {
				return false, nil
			}
			// Either someone else released it, or there was a transient error.
			// The controller should requeue and try again if it's still stale.
			return false, err
		}
		// Successfully released.
		return false, nil
	}

	// It's an orphan.
	if m.controller.GetDeletionTimestamp() != nil || !match(obj) {
		// Ignore if we're being deleted or selector doesn't match.
		return false, nil
	}
	if obj.GetDeletionTimestamp() != nil {
		// Ignore if the object is being deleted
		return false, nil
	}
	// Selector matches. Try to adopt.
	if err := adopt(obj); err != nil {
		// If the pod no longer exists, ignore the error.
		if errors.IsNotFound(err) {
			return false, nil
		}
		// Either someone else claimed it first, or there was a transient error.
		// The controller should requeue and try again if it's still orphaned.
		return false, err
	}
	// Successfully adopted.
	return true, nil
}

type PodControllerRefManager struct {
	baseControllerRefManager
	controllerKind schema.GroupVersionKind
	podControl     PodControlInterface
}

// NewPodControllerRefManager returns a PodControllerRefManager that exposes
// methods to manage the controllerRef of pods.
//
// The canAdopt() function can be used to perform a potentially expensive check
// (such as a live GET from the API server) prior to the first adoption.
// It will only be called (at most once) if an adoption is actually attempted.
// If canAdopt() returns a non-nil error, all adoptions will fail.
//
// NOTE: Once canAdopt() is called, it will not be called again by the same
//       PodControllerRefManager instance. Create a new instance if it makes
//       sense to check canAdopt() again (e.g. in a different sync pass).
func NewPodControllerRefManager(
	podControl PodControlInterface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func() error,
) *PodControllerRefManager {
	return &PodControllerRefManager{
		baseControllerRefManager: baseControllerRefManager{
			controller:   controller,
			selector:     selector,
			canAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		podControl:     podControl,
	}
}

// ClaimPods tries to take ownership of a list of Pods.
//
// It will reconcile the following:
//   * Adopt orphans if the selector matches.
//   * Release owned objects if the selector no longer matches.
//
// Optional: If one or more filters are specified, a Pod will only be claimed if
// all filters return true.
//
// A non-nil error is returned if some form of reconciliation was attemped and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of Pods that you now own is returned.
func (m *PodControllerRefManager) ClaimPods(pods []*v1.Pod, filters ...func(*v1.Pod) bool) ([]*v1.Pod, error) {
	var claimed []*v1.Pod
	var errlist []error

	match := func(obj metav1.Object) bool {
		pod := obj.(*v1.Pod)
		// Check selector first so filters only run on potentially matching Pods.
		if !m.selector.Matches(labels.Set(pod.Labels)) {
			return false
		}
		for _, filter := range filters {
			if !filter(pod) {
				return false
			}
		}
		return true
	}
	adopt := func(obj metav1.Object) error {
		return m.AdoptPod(obj.(*v1.Pod))
	}
	release := func(obj metav1.Object) error {
		return m.ReleasePod(obj.(*v1.Pod))
	}

	for _, pod := range pods {
		ok, err := m.claimObject(pod, match, adopt, release)
		if err != nil {
			errlist = append(errlist, err)
			continue
		}
		if ok {
			claimed = append(claimed, pod)
		}
	}
	return claimed, utilerrors.NewAggregate(errlist)
}

// AdoptPod sends a patch to take control of the pod. It returns the error if
// the patching fails.
func (m *PodControllerRefManager) AdoptPod(pod *v1.Pod) error {
	if err := m.canAdopt(); err != nil {
		return fmt.Errorf("can't adopt Pod %v/%v (%v): %v", pod.Namespace, pod.Name, pod.UID, err)
	}
	// Note that ValidateOwnerReferences() will reject this patch if another
	// OwnerReference exists with controller=true.
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true,"blockOwnerDeletion":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controller.GetName(), m.controller.GetUID(), pod.UID)
	return m.podControl.PatchPod(pod.Namespace, pod.Name, []byte(addControllerPatch))
}

// ReleasePod sends a patch to free the pod from the control of the controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *PodControllerRefManager) ReleasePod(pod *v1.Pod) error {
	glog.V(2).Infof("patching pod %s_%s to remove its controllerRef to %s/%s:%s",
		pod.Namespace, pod.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controller.GetName())
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controller.GetUID(), pod.UID)
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
	baseControllerRefManager
	controllerKind schema.GroupVersionKind
	rsControl      RSControlInterface
}

// NewReplicaSetControllerRefManager returns a ReplicaSetControllerRefManager that exposes
// methods to manage the controllerRef of ReplicaSets.
//
// The canAdopt() function can be used to perform a potentially expensive check
// (such as a live GET from the API server) prior to the first adoption.
// It will only be called (at most once) if an adoption is actually attempted.
// If canAdopt() returns a non-nil error, all adoptions will fail.
//
// NOTE: Once canAdopt() is called, it will not be called again by the same
//       ReplicaSetControllerRefManager instance. Create a new instance if it
//       makes sense to check canAdopt() again (e.g. in a different sync pass).
func NewReplicaSetControllerRefManager(
	rsControl RSControlInterface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func() error,
) *ReplicaSetControllerRefManager {
	return &ReplicaSetControllerRefManager{
		baseControllerRefManager: baseControllerRefManager{
			controller:   controller,
			selector:     selector,
			canAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		rsControl:      rsControl,
	}
}

// ClaimReplicaSets tries to take ownership of a list of ReplicaSets.
//
// It will reconcile the following:
//   * Adopt orphans if the selector matches.
//   * Release owned objects if the selector no longer matches.
//
// A non-nil error is returned if some form of reconciliation was attemped and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of ReplicaSets that you now own is
// returned.
func (m *ReplicaSetControllerRefManager) ClaimReplicaSets(sets []*extensions.ReplicaSet) ([]*extensions.ReplicaSet, error) {
	var claimed []*extensions.ReplicaSet
	var errlist []error

	match := func(obj metav1.Object) bool {
		return m.selector.Matches(labels.Set(obj.GetLabels()))
	}
	adopt := func(obj metav1.Object) error {
		return m.AdoptReplicaSet(obj.(*extensions.ReplicaSet))
	}
	release := func(obj metav1.Object) error {
		return m.ReleaseReplicaSet(obj.(*extensions.ReplicaSet))
	}

	for _, rs := range sets {
		ok, err := m.claimObject(rs, match, adopt, release)
		if err != nil {
			errlist = append(errlist, err)
			continue
		}
		if ok {
			claimed = append(claimed, rs)
		}
	}
	return claimed, utilerrors.NewAggregate(errlist)
}

// AdoptReplicaSet sends a patch to take control of the ReplicaSet. It returns
// the error if the patching fails.
func (m *ReplicaSetControllerRefManager) AdoptReplicaSet(rs *extensions.ReplicaSet) error {
	if err := m.canAdopt(); err != nil {
		return fmt.Errorf("can't adopt ReplicaSet %v/%v (%v): %v", rs.Namespace, rs.Name, rs.UID, err)
	}
	// Note that ValidateOwnerReferences() will reject this patch if another
	// OwnerReference exists with controller=true.
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true,"blockOwnerDeletion":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controller.GetName(), m.controller.GetUID(), rs.UID)
	return m.rsControl.PatchReplicaSet(rs.Namespace, rs.Name, []byte(addControllerPatch))
}

// ReleaseReplicaSet sends a patch to free the ReplicaSet from the control of the Deployment controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *ReplicaSetControllerRefManager) ReleaseReplicaSet(replicaSet *extensions.ReplicaSet) error {
	glog.V(2).Infof("patching ReplicaSet %s_%s to remove its controllerRef to %s/%s:%s",
		replicaSet.Namespace, replicaSet.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controller.GetName())
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controller.GetUID(), replicaSet.UID)
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

// RecheckDeletionTimestamp returns a canAdopt() function to recheck deletion.
//
// The canAdopt() function calls getObject() to fetch the latest value,
// and denies adoption attempts if that object has a non-nil DeletionTimestamp.
func RecheckDeletionTimestamp(getObject func() (metav1.Object, error)) func() error {
	return func() error {
		obj, err := getObject()
		if err != nil {
			return fmt.Errorf("can't recheck DeletionTimestamp: %v", err)
		}
		if obj.GetDeletionTimestamp() != nil {
			return fmt.Errorf("%v/%v has just been deleted at %v", obj.GetNamespace(), obj.GetName(), obj.GetDeletionTimestamp())
		}
		return nil
	}
}

// ControllerRevisionControllerRefManager is used to manage controllerRef of ControllerRevisions.
// Three methods are defined on this object 1: Classify 2: AdoptControllerRevision and
// 3: ReleaseControllerRevision which are used to classify the ControllerRevisions into appropriate
// categories and accordingly adopt or release them. See comments on these functions
// for more details.
type ControllerRevisionControllerRefManager struct {
	baseControllerRefManager
	controllerKind schema.GroupVersionKind
	crControl      ControllerRevisionControlInterface
}

// NewControllerRevisionControllerRefManager returns a ControllerRevisionControllerRefManager that exposes
// methods to manage the controllerRef of ControllerRevisions.
//
// The canAdopt() function can be used to perform a potentially expensive check
// (such as a live GET from the API server) prior to the first adoption.
// It will only be called (at most once) if an adoption is actually attempted.
// If canAdopt() returns a non-nil error, all adoptions will fail.
//
// NOTE: Once canAdopt() is called, it will not be called again by the same
//       ControllerRevisionControllerRefManager instance. Create a new instance if it
//       makes sense to check canAdopt() again (e.g. in a different sync pass).
func NewControllerRevisionControllerRefManager(
	crControl ControllerRevisionControlInterface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func() error,
) *ControllerRevisionControllerRefManager {
	return &ControllerRevisionControllerRefManager{
		baseControllerRefManager: baseControllerRefManager{
			controller:   controller,
			selector:     selector,
			canAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		crControl:      crControl,
	}
}

// ClaimControllerRevisions tries to take ownership of a list of ControllerRevisions.
//
// It will reconcile the following:
//   * Adopt orphans if the selector matches.
//   * Release owned objects if the selector no longer matches.
//
// A non-nil error is returned if some form of reconciliation was attemped and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of ControllerRevisions that you now own is
// returned.
func (m *ControllerRevisionControllerRefManager) ClaimControllerRevisions(histories []*appsv1beta1.ControllerRevision) ([]*appsv1beta1.ControllerRevision, error) {
	var claimed []*appsv1beta1.ControllerRevision
	var errlist []error

	match := func(obj metav1.Object) bool {
		return m.selector.Matches(labels.Set(obj.GetLabels()))
	}
	adopt := func(obj metav1.Object) error {
		return m.AdoptControllerRevision(obj.(*appsv1beta1.ControllerRevision))
	}
	release := func(obj metav1.Object) error {
		return m.ReleaseControllerRevision(obj.(*appsv1beta1.ControllerRevision))
	}

	for _, h := range histories {
		ok, err := m.claimObject(h, match, adopt, release)
		if err != nil {
			errlist = append(errlist, err)
			continue
		}
		if ok {
			claimed = append(claimed, h)
		}
	}
	return claimed, utilerrors.NewAggregate(errlist)
}

// AdoptControllerRevision sends a patch to take control of the ControllerRevision. It returns the error if
// the patching fails.
func (m *ControllerRevisionControllerRefManager) AdoptControllerRevision(history *appsv1beta1.ControllerRevision) error {
	if err := m.canAdopt(); err != nil {
		return fmt.Errorf("can't adopt ControllerRevision %v/%v (%v): %v", history.Namespace, history.Name, history.UID, err)
	}
	// Note that ValidateOwnerReferences() will reject this patch if another
	// OwnerReference exists with controller=true.
	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true,"blockOwnerDeletion":true}],"uid":"%s"}}`,
		m.controllerKind.GroupVersion(), m.controllerKind.Kind,
		m.controller.GetName(), m.controller.GetUID(), history.UID)
	return m.crControl.PatchControllerRevision(history.Namespace, history.Name, []byte(addControllerPatch))
}

// ReleaseControllerRevision sends a patch to free the ControllerRevision from the control of its controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *ControllerRevisionControllerRefManager) ReleaseControllerRevision(history *appsv1beta1.ControllerRevision) error {
	glog.V(2).Infof("patching ControllerRevision %s_%s to remove its controllerRef to %s/%s:%s",
		history.Namespace, history.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.controller.GetName())
	deleteOwnerRefPatch := fmt.Sprintf(`{"metadata":{"ownerReferences":[{"$patch":"delete","uid":"%s"}],"uid":"%s"}}`, m.controller.GetUID(), history.UID)
	err := m.crControl.PatchControllerRevision(history.Namespace, history.Name, []byte(deleteOwnerRefPatch))
	if err != nil {
		if errors.IsNotFound(err) {
			// If the ControllerRevision no longer exists, ignore it.
			return nil
		}
		if errors.IsInvalid(err) {
			// Invalid error will be returned in two cases: 1. the ControllerRevision
			// has no owner reference, 2. the uid of the ControllerRevision doesn't
			// match, which means the ControllerRevision is deleted and then recreated.
			// In both cases, the error can be ignored.
			return nil
		}
	}
	return err
}
