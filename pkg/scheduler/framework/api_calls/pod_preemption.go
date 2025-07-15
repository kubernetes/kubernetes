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

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

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
