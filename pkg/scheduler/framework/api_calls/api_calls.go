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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	PodStatusPatch fwk.APICallType = "pod_status_patch"
	PodBinding     fwk.APICallType = "pod_binding"
	PodPreemption  fwk.APICallType = "pod_preemption"
)

// APICallRelevances is a built-in mapping types to relevances.
// Types of the same relevance should only be defined for different object types.
var APICallRelevances = fwk.APICallRelevances{
	PodStatusPatch: 1,
	PodBinding:     2,
	PodPreemption:  3,
}

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

	podCopy := pod.DeepCopy()

	psuc.lock.Lock()
	if !psuc.executed {
		psuc.oldPodStatus = pod.Status.DeepCopy()
	}
	psuc.lock.Unlock()

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

type PodBindingCall struct {
	binding *v1.Binding
}

func NewPodBindingCall(binding *v1.Binding) *PodBindingCall {
	return &PodBindingCall{
		binding: binding,
	}
}

func (pbc *PodBindingCall) CallType() fwk.APICallType {
	return PodBinding
}

func (pbc *PodBindingCall) UID() types.UID {
	return pbc.binding.UID
}

func (pbc *PodBindingCall) Execute(ctx context.Context, client clientset.Interface) error {
	logger := klog.FromContext(ctx)
	logger.V(3).Info("Attempting to bind pod to node", "pod", klog.KObj(&pbc.binding.ObjectMeta), "node", pbc.binding.Target.Name)

	return client.CoreV1().Pods(pbc.binding.Namespace).Bind(ctx, pbc.binding, metav1.CreateOptions{})
}

func (pbc *PodBindingCall) Sync(obj metav1.Object) (metav1.Object, error) {
	// Don't need to store or update an object.
	return obj, nil
}

func (pbc *PodBindingCall) Merge(oldCall fwk.APICall) error {
	// Merge should just overwrite the previous call.
	return nil
}

func (pbc *PodBindingCall) IsNoOp() bool {
	return false
}

type PodPreemptionCall struct {
	victimUID       types.UID
	victimRef       klog.ObjectRef
	preemptorRef    klog.ObjectRef
	oldVictimStatus *v1.PodStatus
	condition       *v1.PodCondition
}

func NewPodPreemptionCall(victim *v1.Pod, preemptor *v1.Pod, condition *v1.PodCondition) *PodPreemptionCall {
	return &PodPreemptionCall{
		victimUID:       victim.UID,
		victimRef:       klog.KObj(victim),
		preemptorRef:    klog.KObj(victim),
		oldVictimStatus: victim.Status.DeepCopy(),
		condition:       condition,
	}
}

func (ppc *PodPreemptionCall) CallType() fwk.APICallType {
	return PodPreemption
}

func (ppc *PodPreemptionCall) UID() types.UID {
	return ppc.victimUID
}

func (ppc *PodPreemptionCall) Execute(ctx context.Context, client clientset.Interface) error {
	logger := klog.FromContext(ctx)

	newStatus := ppc.oldVictimStatus.DeepCopy()
	updated := podutil.UpdatePodCondition(newStatus, ppc.condition)
	if updated {
		if _, err := util.PatchPodStatus(ctx, client, ppc.victimRef.Name, ppc.victimRef.Namespace, ppc.oldVictimStatus, newStatus); err != nil {
			logger.Error(err, "Tried to preempted pod", "pod", ppc.victimRef, "preemptor", ppc.preemptorRef)
			return err
		}
	}
	if err := client.CoreV1().Pods(ppc.victimRef.Namespace).Delete(ctx, ppc.victimRef.Name, metav1.DeleteOptions{}); err != nil {
		if !apierrors.IsNotFound(err) {
			logger.Error(err, "Tried to preempted pod", "pod", ppc.victimRef, "preemptor", ppc.preemptorRef)
			return err
		}
		logger.V(2).Info("Victim Pod is already deleted", "victim", ppc.victimRef, "preemptor", ppc.preemptorRef)
		return nil
	}
	logger.V(2).Info("Preemptor Pod preempted victim Pod", "victim", ppc.victimRef, "preemptor", ppc.preemptorRef)
	return nil
}

func (ppc *PodPreemptionCall) Sync(obj metav1.Object) (metav1.Object, error) {
	// Don't need to store or update an object.
	return obj, nil
}

func (ppc *PodPreemptionCall) Merge(oldCall fwk.APICall) error {
	// Merge should just overwrite the previous call.
	return nil
}

func (ppc *PodPreemptionCall) IsNoOp() bool {
	return false
}
