/*
Copyright 2025 The Kubernetes Authors.

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

package apicalls

import (
	"context"
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// PodStatusPatchCall is used to patch the pod status.
type PodStatusPatchCall struct {
	lock sync.Mutex
	// executed is set at the beginning of the call's Execute
	// and is used by Sync to know if the podStatus should be updated.
	executed bool

	// podUID is an UID of the pod.
	podUID types.UID
	// podRef is a reference to the pod.
	podRef klog.ObjectRef
	// podStatus contains the actual status of the pod.
	podStatus *v1.PodStatus
	// newCondition is a condition to update.
	newCondition *v1.PodCondition
	// nominatingInfo is a nominating info to update.
	nominatingInfo *fwk.NominatingInfo
}

func NewPodStatusPatchCall(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *fwk.NominatingInfo) *PodStatusPatchCall {
	return &PodStatusPatchCall{
		podUID:         pod.UID,
		podRef:         klog.KObj(pod),
		podStatus:      pod.Status.DeepCopy(),
		newCondition:   condition,
		nominatingInfo: nominatingInfo,
	}
}

func (psuc *PodStatusPatchCall) CallType() fwk.APICallType {
	return PodStatusPatch
}

func (psuc *PodStatusPatchCall) UID() types.UID {
	return psuc.podUID
}

// syncStatus syncs the given status with condition and nominatingInfo. It returns true if anything was actually updated.
func syncStatus(status *v1.PodStatus, condition *v1.PodCondition, nominatingInfo *fwk.NominatingInfo) bool {
	nnnNeedsUpdate := nominatingInfo.Mode() == fwk.ModeOverride && status.NominatedNodeName != nominatingInfo.NominatedNodeName
	if condition != nil {
		if !podutil.UpdatePodCondition(status, condition) && !nnnNeedsUpdate {
			return false
		}
	} else if !nnnNeedsUpdate {
		return false
	}
	if nnnNeedsUpdate {
		status.NominatedNodeName = nominatingInfo.NominatedNodeName
	}
	return true
}

func (psuc *PodStatusPatchCall) Execute(ctx context.Context, client clientset.Interface) error {
	psuc.lock.Lock()
	// Executed flag is set not to race with podStatus write in Sync afterwards.
	psuc.executed = true
	condition := psuc.newCondition.DeepCopy()
	podStatusCopy := psuc.podStatus.DeepCopy()
	psuc.lock.Unlock()

	logger := klog.FromContext(ctx)
	if condition != nil {
		logger.V(3).Info("Updating pod condition", "pod", psuc.podRef, "conditionType", condition.Type, "conditionStatus", condition.Status, "conditionReason", condition.Reason)
	}

	// Sync status to have the condition and nominatingInfo applied on a podStatusCopy.
	synced := syncStatus(podStatusCopy, condition, psuc.nominatingInfo)
	if !synced {
		logger.V(5).Info("Pod status patch call does not need to be executed because it has no effect", "pod", psuc.podRef)
		return nil
	}

	// It's safe to run PatchPodStatus even on outdated pod object.
	err := util.PatchPodStatus(ctx, client, psuc.podRef.Name, psuc.podRef.Namespace, psuc.podStatus, podStatusCopy)
	if err != nil {
		logger.Error(err, "Failed to patch pod status", "pod", psuc.podRef)
		return err
	}

	return nil
}

func (psuc *PodStatusPatchCall) Sync(obj metav1.Object) (metav1.Object, error) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		return obj, fmt.Errorf("unexpected error: object of type %T is not of type *v1.Pod", obj)
	}

	psuc.lock.Lock()
	if !psuc.executed {
		// Set podStatus only if the call execution haven't started yet,
		// because otherwise it's irrelevant and might race.
		psuc.podStatus = pod.Status.DeepCopy()
	}
	newCondition := psuc.newCondition.DeepCopy()
	psuc.lock.Unlock()

	podCopy := pod.DeepCopy()
	// Sync passed pod's status with the call's condition and nominatingInfo.
	synced := syncStatus(&podCopy.Status, newCondition, psuc.nominatingInfo)
	if !synced {
		return pod, nil
	}
	return podCopy, nil
}

func (psuc *PodStatusPatchCall) Merge(oldCall fwk.APICall) error {
	oldPsuc, ok := oldCall.(*PodStatusPatchCall)
	if !ok {
		return fmt.Errorf("unexpected error: call of type %T is not of type *PodStatusPatchCall", oldCall)
	}
	if psuc.nominatingInfo.Mode() == fwk.ModeNoop && oldPsuc.nominatingInfo.Mode() == fwk.ModeOverride {
		// Set a nominatingInfo from an old call if the new one is no-op.
		psuc.nominatingInfo = oldPsuc.nominatingInfo
	}
	if psuc.newCondition == nil && oldPsuc.newCondition != nil {
		// Set a condition from an old call if the new one is nil.
		psuc.newCondition = oldPsuc.newCondition
	}
	return nil
}

// conditionNeedsUpdate checks if the pod condition needs update.
func conditionNeedsUpdate(status *v1.PodStatus, condition *v1.PodCondition) bool {
	// Try to find this pod condition.
	_, oldCondition := podutil.GetPodCondition(status, condition.Type)
	if oldCondition == nil {
		return true
	}

	isEqual := condition.Status == oldCondition.Status &&
		condition.Reason == oldCondition.Reason &&
		condition.Message == oldCondition.Message &&
		condition.LastProbeTime.Equal(&oldCondition.LastProbeTime)

	// Return true if one of the fields have changed.
	return !isEqual
}

func (psuc *PodStatusPatchCall) IsNoOp() bool {
	nnnNeedsUpdate := psuc.nominatingInfo.Mode() == fwk.ModeOverride && psuc.podStatus.NominatedNodeName != psuc.nominatingInfo.NominatedNodeName
	if nnnNeedsUpdate {
		return false
	}
	if psuc.newCondition == nil {
		return true
	}
	return !conditionNeedsUpdate(psuc.podStatus, psuc.newCondition)
}
