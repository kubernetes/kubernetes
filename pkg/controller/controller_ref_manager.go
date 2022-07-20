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
	"context"
	"encoding/json"
	"fmt"
	"sync"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"
)

type BaseControllerRefManager struct {
	Controller metav1.Object
	Selector   labels.Selector

	canAdoptErr  error
	canAdoptOnce sync.Once
	CanAdoptFunc func(ctx context.Context) error
}

func (m *BaseControllerRefManager) CanAdopt(ctx context.Context) error {
	m.canAdoptOnce.Do(func() {
		if m.CanAdoptFunc != nil {
			m.canAdoptErr = m.CanAdoptFunc(ctx)
		}
	})
	return m.canAdoptErr
}

// ClaimObject tries to take ownership of an object for this controller.
//
// It will reconcile the following:
//   - Adopt orphans if the match function returns true.
//   - Release owned objects if the match function returns false.
//
// A non-nil error is returned if some form of reconciliation was attempted and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The returned boolean indicates whether you now
// own the object.
//
// No reconciliation will be attempted if the controller is being deleted.
func (m *BaseControllerRefManager) ClaimObject(ctx context.Context, obj metav1.Object, match func(metav1.Object) bool, adopt, release func(context.Context, metav1.Object) error) (bool, error) {
	controllerRef := metav1.GetControllerOfNoCopy(obj)
	if controllerRef != nil {
		if controllerRef.UID != m.Controller.GetUID() {
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
		if m.Controller.GetDeletionTimestamp() != nil {
			return false, nil
		}
		if err := release(ctx, obj); err != nil {
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
	if m.Controller.GetDeletionTimestamp() != nil || !match(obj) {
		// Ignore if we're being deleted or selector doesn't match.
		return false, nil
	}
	if obj.GetDeletionTimestamp() != nil {
		// Ignore if the object is being deleted
		return false, nil
	}

	if len(m.Controller.GetNamespace()) > 0 && m.Controller.GetNamespace() != obj.GetNamespace() {
		// Ignore if namespace not match
		return false, nil
	}

	// Selector matches. Try to adopt.
	if err := adopt(ctx, obj); err != nil {
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
	BaseControllerRefManager
	controllerKind schema.GroupVersionKind
	podControl     PodControlInterface
	finalizers     []string
}

// NewPodControllerRefManager returns a PodControllerRefManager that exposes
// methods to manage the controllerRef of pods.
//
// The CanAdopt() function can be used to perform a potentially expensive check
// (such as a live GET from the API server) prior to the first adoption.
// It will only be called (at most once) if an adoption is actually attempted.
// If CanAdopt() returns a non-nil error, all adoptions will fail.
//
// NOTE: Once CanAdopt() is called, it will not be called again by the same
//
//	PodControllerRefManager instance. Create a new instance if it makes
//	sense to check CanAdopt() again (e.g. in a different sync pass).
func NewPodControllerRefManager(
	podControl PodControlInterface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func(ctx context.Context) error,
	finalizers ...string,
) *PodControllerRefManager {
	return &PodControllerRefManager{
		BaseControllerRefManager: BaseControllerRefManager{
			Controller:   controller,
			Selector:     selector,
			CanAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		podControl:     podControl,
		finalizers:     finalizers,
	}
}

// ClaimPods tries to take ownership of a list of Pods.
//
// It will reconcile the following:
//   - Adopt orphans if the selector matches.
//   - Release owned objects if the selector no longer matches.
//
// Optional: If one or more filters are specified, a Pod will only be claimed if
// all filters return true.
//
// A non-nil error is returned if some form of reconciliation was attempted and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of Pods that you now own is returned.
func (m *PodControllerRefManager) ClaimPods(ctx context.Context, pods []*v1.Pod, filters ...func(*v1.Pod) bool) ([]*v1.Pod, error) {
	var claimed []*v1.Pod
	var errlist []error

	match := func(obj metav1.Object) bool {
		pod := obj.(*v1.Pod)
		// Check selector first so filters only run on potentially matching Pods.
		if !m.Selector.Matches(labels.Set(pod.Labels)) {
			return false
		}
		for _, filter := range filters {
			if !filter(pod) {
				return false
			}
		}
		return true
	}
	adopt := func(ctx context.Context, obj metav1.Object) error {
		return m.AdoptPod(ctx, obj.(*v1.Pod))
	}
	release := func(ctx context.Context, obj metav1.Object) error {
		return m.ReleasePod(ctx, obj.(*v1.Pod))
	}

	for _, pod := range pods {
		ok, err := m.ClaimObject(ctx, pod, match, adopt, release)
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
func (m *PodControllerRefManager) AdoptPod(ctx context.Context, pod *v1.Pod) error {
	if err := m.CanAdopt(ctx); err != nil {
		return fmt.Errorf("can't adopt Pod %v/%v (%v): %v", pod.Namespace, pod.Name, pod.UID, err)
	}
	// Note that ValidateOwnerReferences() will reject this patch if another
	// OwnerReference exists with controller=true.

	patchBytes, err := ownerRefControllerPatch(m.Controller, m.controllerKind, pod.UID, m.finalizers...)
	if err != nil {
		return err
	}
	return m.podControl.PatchPod(ctx, pod.Namespace, pod.Name, patchBytes)
}

// ReleasePod sends a patch to free the pod from the control of the controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *PodControllerRefManager) ReleasePod(ctx context.Context, pod *v1.Pod) error {
	klog.V(2).Infof("patching pod %s_%s to remove its controllerRef to %s/%s:%s",
		pod.Namespace, pod.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.Controller.GetName())
	patchBytes, err := GenerateDeleteOwnerRefStrategicMergeBytes(pod.UID, []types.UID{m.Controller.GetUID()}, m.finalizers...)
	if err != nil {
		return err
	}
	err = m.podControl.PatchPod(ctx, pod.Namespace, pod.Name, patchBytes)
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
	BaseControllerRefManager
	controllerKind schema.GroupVersionKind
	rsControl      RSControlInterface
}

// NewReplicaSetControllerRefManager returns a ReplicaSetControllerRefManager that exposes
// methods to manage the controllerRef of ReplicaSets.
//
// The CanAdopt() function can be used to perform a potentially expensive check
// (such as a live GET from the API server) prior to the first adoption.
// It will only be called (at most once) if an adoption is actually attempted.
// If CanAdopt() returns a non-nil error, all adoptions will fail.
//
// NOTE: Once CanAdopt() is called, it will not be called again by the same
//
//	ReplicaSetControllerRefManager instance. Create a new instance if it
//	makes sense to check CanAdopt() again (e.g. in a different sync pass).
func NewReplicaSetControllerRefManager(
	rsControl RSControlInterface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func(ctx context.Context) error,
) *ReplicaSetControllerRefManager {
	return &ReplicaSetControllerRefManager{
		BaseControllerRefManager: BaseControllerRefManager{
			Controller:   controller,
			Selector:     selector,
			CanAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		rsControl:      rsControl,
	}
}

// ClaimReplicaSets tries to take ownership of a list of ReplicaSets.
//
// It will reconcile the following:
//   - Adopt orphans if the selector matches.
//   - Release owned objects if the selector no longer matches.
//
// A non-nil error is returned if some form of reconciliation was attempted and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of ReplicaSets that you now own is
// returned.
func (m *ReplicaSetControllerRefManager) ClaimReplicaSets(ctx context.Context, sets []*apps.ReplicaSet) ([]*apps.ReplicaSet, error) {
	var claimed []*apps.ReplicaSet
	var errlist []error

	match := func(obj metav1.Object) bool {
		return m.Selector.Matches(labels.Set(obj.GetLabels()))
	}
	adopt := func(ctx context.Context, obj metav1.Object) error {
		return m.AdoptReplicaSet(ctx, obj.(*apps.ReplicaSet))
	}
	release := func(ctx context.Context, obj metav1.Object) error {
		return m.ReleaseReplicaSet(ctx, obj.(*apps.ReplicaSet))
	}

	for _, rs := range sets {
		ok, err := m.ClaimObject(ctx, rs, match, adopt, release)
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
func (m *ReplicaSetControllerRefManager) AdoptReplicaSet(ctx context.Context, rs *apps.ReplicaSet) error {
	if err := m.CanAdopt(ctx); err != nil {
		return fmt.Errorf("can't adopt ReplicaSet %v/%v (%v): %v", rs.Namespace, rs.Name, rs.UID, err)
	}
	// Note that ValidateOwnerReferences() will reject this patch if another
	// OwnerReference exists with controller=true.
	patchBytes, err := ownerRefControllerPatch(m.Controller, m.controllerKind, rs.UID)
	if err != nil {
		return err
	}
	return m.rsControl.PatchReplicaSet(ctx, rs.Namespace, rs.Name, patchBytes)
}

// ReleaseReplicaSet sends a patch to free the ReplicaSet from the control of the Deployment controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *ReplicaSetControllerRefManager) ReleaseReplicaSet(ctx context.Context, replicaSet *apps.ReplicaSet) error {
	klog.V(2).Infof("patching ReplicaSet %s_%s to remove its controllerRef to %s/%s:%s",
		replicaSet.Namespace, replicaSet.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.Controller.GetName())
	patchBytes, err := GenerateDeleteOwnerRefStrategicMergeBytes(replicaSet.UID, []types.UID{m.Controller.GetUID()})
	if err != nil {
		return err
	}
	err = m.rsControl.PatchReplicaSet(ctx, replicaSet.Namespace, replicaSet.Name, patchBytes)
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

// RecheckDeletionTimestamp returns a CanAdopt() function to recheck deletion.
//
// The CanAdopt() function calls getObject() to fetch the latest value,
// and denies adoption attempts if that object has a non-nil DeletionTimestamp.
func RecheckDeletionTimestamp(getObject func(context.Context) (metav1.Object, error)) func(context.Context) error {
	return func(ctx context.Context) error {
		obj, err := getObject(ctx)
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
	BaseControllerRefManager
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
//
//	ControllerRevisionControllerRefManager instance. Create a new instance if it
//	makes sense to check canAdopt() again (e.g. in a different sync pass).
func NewControllerRevisionControllerRefManager(
	crControl ControllerRevisionControlInterface,
	controller metav1.Object,
	selector labels.Selector,
	controllerKind schema.GroupVersionKind,
	canAdopt func(ctx context.Context) error,
) *ControllerRevisionControllerRefManager {
	return &ControllerRevisionControllerRefManager{
		BaseControllerRefManager: BaseControllerRefManager{
			Controller:   controller,
			Selector:     selector,
			CanAdoptFunc: canAdopt,
		},
		controllerKind: controllerKind,
		crControl:      crControl,
	}
}

// ClaimControllerRevisions tries to take ownership of a list of ControllerRevisions.
//
// It will reconcile the following:
//   - Adopt orphans if the selector matches.
//   - Release owned objects if the selector no longer matches.
//
// A non-nil error is returned if some form of reconciliation was attempted and
// failed. Usually, controllers should try again later in case reconciliation
// is still needed.
//
// If the error is nil, either the reconciliation succeeded, or no
// reconciliation was necessary. The list of ControllerRevisions that you now own is
// returned.
func (m *ControllerRevisionControllerRefManager) ClaimControllerRevisions(ctx context.Context, histories []*apps.ControllerRevision) ([]*apps.ControllerRevision, error) {
	var claimed []*apps.ControllerRevision
	var errlist []error

	match := func(obj metav1.Object) bool {
		return m.Selector.Matches(labels.Set(obj.GetLabels()))
	}
	adopt := func(ctx context.Context, obj metav1.Object) error {
		return m.AdoptControllerRevision(ctx, obj.(*apps.ControllerRevision))
	}
	release := func(ctx context.Context, obj metav1.Object) error {
		return m.ReleaseControllerRevision(ctx, obj.(*apps.ControllerRevision))
	}

	for _, h := range histories {
		ok, err := m.ClaimObject(ctx, h, match, adopt, release)
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
func (m *ControllerRevisionControllerRefManager) AdoptControllerRevision(ctx context.Context, history *apps.ControllerRevision) error {
	if err := m.CanAdopt(ctx); err != nil {
		return fmt.Errorf("can't adopt ControllerRevision %v/%v (%v): %v", history.Namespace, history.Name, history.UID, err)
	}
	// Note that ValidateOwnerReferences() will reject this patch if another
	// OwnerReference exists with controller=true.
	patchBytes, err := ownerRefControllerPatch(m.Controller, m.controllerKind, history.UID)
	if err != nil {
		return err
	}
	return m.crControl.PatchControllerRevision(ctx, history.Namespace, history.Name, patchBytes)
}

// ReleaseControllerRevision sends a patch to free the ControllerRevision from the control of its controller.
// It returns the error if the patching fails. 404 and 422 errors are ignored.
func (m *ControllerRevisionControllerRefManager) ReleaseControllerRevision(ctx context.Context, history *apps.ControllerRevision) error {
	klog.V(2).Infof("patching ControllerRevision %s_%s to remove its controllerRef to %s/%s:%s",
		history.Namespace, history.Name, m.controllerKind.GroupVersion(), m.controllerKind.Kind, m.Controller.GetName())
	patchBytes, err := GenerateDeleteOwnerRefStrategicMergeBytes(history.UID, []types.UID{m.Controller.GetUID()})
	if err != nil {
		return err
	}

	err = m.crControl.PatchControllerRevision(ctx, history.Namespace, history.Name, patchBytes)
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

type objectForAddOwnerRefPatch struct {
	Metadata objectMetaForPatch `json:"metadata"`
}

type objectMetaForPatch struct {
	OwnerReferences []metav1.OwnerReference `json:"ownerReferences"`
	UID             types.UID               `json:"uid"`
	Finalizers      []string                `json:"finalizers,omitempty"`
}

func ownerRefControllerPatch(controller metav1.Object, controllerKind schema.GroupVersionKind, uid types.UID, finalizers ...string) ([]byte, error) {
	blockOwnerDeletion := true
	isController := true
	addControllerPatch := objectForAddOwnerRefPatch{
		Metadata: objectMetaForPatch{
			UID: uid,
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion:         controllerKind.GroupVersion().String(),
					Kind:               controllerKind.Kind,
					Name:               controller.GetName(),
					UID:                controller.GetUID(),
					Controller:         &isController,
					BlockOwnerDeletion: &blockOwnerDeletion,
				},
			},
			Finalizers: finalizers,
		},
	}
	patchBytes, err := json.Marshal(&addControllerPatch)
	if err != nil {
		return nil, err
	}
	return patchBytes, nil
}

type objectForDeleteOwnerRefStrategicMergePatch struct {
	Metadata objectMetaForMergePatch `json:"metadata"`
}

type objectMetaForMergePatch struct {
	UID              types.UID           `json:"uid"`
	OwnerReferences  []map[string]string `json:"ownerReferences"`
	DeleteFinalizers []string            `json:"$deleteFromPrimitiveList/finalizers,omitempty"`
}

func GenerateDeleteOwnerRefStrategicMergeBytes(dependentUID types.UID, ownerUIDs []types.UID, finalizers ...string) ([]byte, error) {
	var ownerReferences []map[string]string
	for _, ownerUID := range ownerUIDs {
		ownerReferences = append(ownerReferences, ownerReference(ownerUID, "delete"))
	}
	patch := objectForDeleteOwnerRefStrategicMergePatch{
		Metadata: objectMetaForMergePatch{
			UID:              dependentUID,
			OwnerReferences:  ownerReferences,
			DeleteFinalizers: finalizers,
		},
	}
	patchBytes, err := json.Marshal(&patch)
	if err != nil {
		return nil, err
	}
	return patchBytes, nil
}

func ownerReference(uid types.UID, patchType string) map[string]string {
	return map[string]string{
		"$patch": patchType,
		"uid":    string(uid),
	}
}
