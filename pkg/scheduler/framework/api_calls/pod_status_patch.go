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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

type PodStatusPatchCall struct {
	lock     sync.Mutex
	executed bool

	podUID         types.UID
	podRef         klog.ObjectRef
	oldPodStatus   *v1.PodStatus
	condition      *v1.PodCondition
	nominatingInfo *framework.NominatingInfo
}

func NewPodStatusPatchCall(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *framework.NominatingInfo) *PodStatusPatchCall {
	return &PodStatusPatchCall{
		podUID:         pod.UID,
		podRef:         klog.KObj(pod),
		oldPodStatus:   pod.Status.DeepCopy(),
		condition:      condition,
		nominatingInfo: nominatingInfo,
	}
}

func (psuc *PodStatusPatchCall) CallType() fwk.APICallType {
	return PodStatusPatch
}

func (psuc *PodStatusPatchCall) UID() types.UID {
	return psuc.podUID
}

func anyPodConditionUpdated(status *v1.PodStatus, condition *v1.PodCondition) bool {
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

func (psuc *PodStatusPatchCall) syncStatus(status *v1.PodStatus, condition *v1.PodCondition) bool {
	nnnNeedsUpdate := psuc.nominatingInfo.Mode() == framework.ModeOverride && psuc.oldPodStatus.NominatedNodeName != psuc.nominatingInfo.NominatedNodeName
	if condition != nil {
		if !podutil.UpdatePodCondition(status, condition) && !nnnNeedsUpdate {
			return false
		}
	} else if !nnnNeedsUpdate {
		return false
	}
	if nnnNeedsUpdate {
		status.NominatedNodeName = psuc.nominatingInfo.NominatedNodeName
	}
	return true
}

func (psuc *PodStatusPatchCall) Execute(ctx context.Context, client clientset.Interface) error {
	psuc.lock.Lock()
	// Have to set an executed flag not to race with oldPodStatus write in Sync afterwards.
	psuc.executed = true
	condition := psuc.condition.DeepCopy()
	psuc.lock.Unlock()

	logger := klog.FromContext(ctx)
	podStatusCopy := psuc.oldPodStatus.DeepCopy()
	if condition != nil {
		logger.V(3).Info("Updating pod condition", "pod", psuc.podRef, "conditionType", condition.Type, "conditionStatus", condition.Status, "conditionReason", condition.Reason)
	}

	synced := psuc.syncStatus(podStatusCopy, condition)
	if !synced {
		logger.V(5).Info("Pod status patch call does not need to be executed because it has no effect", "pod", psuc.podRef)
		return nil
	}

	// It's safe to run PatchPodStatus even on outdated pod object.
	_, err := util.PatchPodStatus(ctx, client, psuc.podRef.Name, psuc.podRef.Namespace, psuc.oldPodStatus, podStatusCopy)
	if err != nil {
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
		psuc.oldPodStatus = pod.Status.DeepCopy()
	}
	psuc.lock.Unlock()

	podCopy := pod.DeepCopy()
	synced := psuc.syncStatus(&podCopy.Status, psuc.condition)
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
	if psuc.nominatingInfo.Mode() == framework.ModeNoop && oldPsuc.nominatingInfo.Mode() == framework.ModeOverride {
		psuc.nominatingInfo = oldPsuc.nominatingInfo
	}
	if psuc.condition == nil && oldPsuc.condition != nil {
		psuc.condition = oldPsuc.condition
	}
	return nil
}

func (psuc *PodStatusPatchCall) IsNoOp() bool {
	nnnNeedsUpdate := psuc.nominatingInfo.Mode() == framework.ModeOverride && psuc.oldPodStatus.NominatedNodeName != psuc.nominatingInfo.NominatedNodeName
	if nnnNeedsUpdate {
		return false
	}
	if psuc.condition == nil {
		return true
	}
	return !anyPodConditionUpdated(psuc.oldPodStatus, psuc.condition)
}
