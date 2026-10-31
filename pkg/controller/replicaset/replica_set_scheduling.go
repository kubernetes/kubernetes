/*
Copyright The Kubernetes Authors.

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

package replicaset

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	apps "k8s.io/api/apps/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1beta1"
	"k8s.io/client-go/tools/cache"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

const (
	workloadByDeploymentNameIndex = "workloadByDeploymentName"
	// replicaSetPodGroupFinalizer keeps a scheduling-enabled ReplicaSet until the
	// controller has attempted cleanup of its owned PodGroup.
	replicaSetPodGroupFinalizer = "apps.kubernetes.io/replicaset-podgroup-cleanup"
)

func workloadByDeploymentNameIndexFunc(obj interface{}) ([]string, error) {
	workload, ok := obj.(*schedulingv1beta1.Workload)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T", obj)
	}
	if workload.Spec.ControllerRef == nil {
		return nil, nil
	}
	ref := workload.Spec.ControllerRef
	if ref.APIGroup != apps.GroupName || ref.Kind != "Deployment" || ref.Name == "" {
		return nil, nil
	}
	return []string{cache.NewObjectName(workload.Namespace, ref.Name).String()}, nil
}

func workloadSpecChanged(oldObj, newObj interface{}) bool {
	oldWorkload, oldOK := oldObj.(*schedulingv1beta1.Workload)
	newWorkload, newOK := newObj.(*schedulingv1beta1.Workload)
	return oldOK && newOK && !apiequality.Semantic.DeepEqual(oldWorkload.Spec, newWorkload.Spec)
}

func (rsc *ReplicaSetController) addSchedulingInformers(
	logger klog.Logger,
	workloadInformer schedulinginformers.WorkloadInformer,
	podGroupInformer schedulinginformers.PodGroupInformer,
) error {
	if workloadInformer == nil {
		return fmt.Errorf("workload informer is required for ReplicaSet Workload integration")
	}
	if podGroupInformer == nil {
		return fmt.Errorf("PodGroup informer is required for ReplicaSet Workload integration")
	}

	if err := workloadInformer.Informer().AddIndexers(cache.Indexers{
		workloadByDeploymentNameIndex: workloadByDeploymentNameIndexFunc,
	}); err != nil {
		return fmt.Errorf("adding Workload indexer: %w", err)
	}

	rsc.workloadLister = workloadInformer.Lister()
	rsc.workloadIndexer = workloadInformer.Informer().GetIndexer()
	rsc.workloadListerSynced = workloadInformer.Informer().HasSynced
	rsc.podGroupLister = podGroupInformer.Lister()
	rsc.podGroupListerSynced = podGroupInformer.Informer().HasSynced
	rsc.schedulingEnabled = true

	if _, err := workloadInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			rsc.enqueueWorkloadReplicaSets(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			if workloadSpecChanged(oldObj, newObj) {
				rsc.enqueueWorkloadReplicaSets(logger, newObj)
			}
		},
	}); err != nil {
		return fmt.Errorf("adding Workload event handler: %w", err)
	}
	return nil
}

func (rsc *ReplicaSetController) schedulingCachesSynced() []cache.InformerSynced {
	if !rsc.schedulingEnabled {
		return nil
	}
	return []cache.InformerSynced{rsc.workloadListerSynced, rsc.podGroupListerSynced}
}

// isDeploymentControllerOwner accepts only complete Deployment controller references.
func isDeploymentControllerOwner(owner *metav1.OwnerReference) bool {
	return owner != nil &&
		owner.APIVersion == apps.SchemeGroupVersion.String() &&
		owner.Kind == "Deployment" && owner.Name != "" && owner.UID != ""
}

// deploymentControllerOwnersMatch prevents a ReplicaSet from using a Workload from another Deployment instance.
func deploymentControllerOwnersMatch(owner, workloadOwner *metav1.OwnerReference) bool {
	return isDeploymentControllerOwner(owner) &&
		isDeploymentControllerOwner(workloadOwner) &&
		owner.Name == workloadOwner.Name && owner.UID == workloadOwner.UID
}

func (rsc *ReplicaSetController) enqueueWorkloadReplicaSets(logger klog.Logger, obj interface{}) {
	workload, ok := obj.(*schedulingv1beta1.Workload)
	if !ok {
		return
	}
	workloadOwner := metav1.GetControllerOf(workload)
	if !isDeploymentControllerOwner(workloadOwner) {
		return
	}

	objects, err := rsc.rsIndexer.ByIndex(controllerUIDIndex, string(workloadOwner.UID))
	if err != nil {
		logger.Error(err, "Failed to look up ReplicaSets for Workload event", "workload", klog.KObj(workload))
		return
	}
	for _, obj := range objects {
		rs, ok := obj.(*apps.ReplicaSet)
		if !ok || rs.Namespace != workload.Namespace {
			continue
		}
		if deploymentControllerOwnersMatch(metav1.GetControllerOf(rs), workloadOwner) {
			rsc.enqueueRS(rs)
		}
	}
}

func (rsc *ReplicaSetController) findWorkloadForReplicaSet(rs *apps.ReplicaSet) (*schedulingv1beta1.Workload, error) {
	owner := metav1.GetControllerOf(rs)
	if !isDeploymentControllerOwner(owner) {
		return nil, nil
	}

	objs, err := rsc.workloadIndexer.ByIndex(workloadByDeploymentNameIndex, cache.NewObjectName(rs.Namespace, owner.Name).String())
	if err != nil {
		return nil, err
	}
	if len(objs) == 0 {
		return nil, nil
	}

	var matched *schedulingv1beta1.Workload
	for _, obj := range objs {
		workload, ok := obj.(*schedulingv1beta1.Workload)
		if !ok {
			return nil, fmt.Errorf("workload index returned %T", obj)
		}
		if !deploymentControllerOwnersMatch(owner, metav1.GetControllerOf(workload)) {
			continue
		}
		if matched != nil {
			return nil, fmt.Errorf("multiple Workloads controlled by Deployment %s/%s UID %q", rs.Namespace, owner.Name, owner.UID)
		}
		matched = workload
	}
	if matched == nil {
		return nil, fmt.Errorf("no Workload controlled by Deployment %s/%s with UID %q", rs.Namespace, owner.Name, owner.UID)
	}
	return matched, nil
}

func replicaSetPodGroupOwner(rs *apps.ReplicaSet) metav1.OwnerReference {
	controller := true
	blockOwnerDeletion := false
	return metav1.OwnerReference{
		APIVersion:         apps.SchemeGroupVersion.String(),
		Kind:               "ReplicaSet",
		Name:               rs.Name,
		UID:                rs.UID,
		Controller:         &controller,
		BlockOwnerDeletion: &blockOwnerDeletion,
	}
}

func workloadPodGroupOwner(wl *schedulingv1beta1.Workload) metav1.OwnerReference {
	return metav1.OwnerReference{
		APIVersion: schedulingv1beta1.SchemeGroupVersion.String(),
		Kind:       "Workload",
		Name:       wl.Name,
		UID:        wl.UID,
	}
}

func hasReplicaSetPodGroupFinalizer(rs *apps.ReplicaSet) bool {
	return slices.Contains(rs.Finalizers, replicaSetPodGroupFinalizer)
}

func (rsc *ReplicaSetController) ensureReplicaSetPodGroupFinalizer(ctx context.Context, rs *apps.ReplicaSet) error {
	if !rsc.schedulingEnabled || rsc.Kind != "ReplicaSet" || !utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) || rs.DeletionTimestamp != nil || hasReplicaSetPodGroupFinalizer(rs) {
		return nil
	}

	workload, err := rsc.findWorkloadForReplicaSet(rs)
	if err != nil {
		return err
	}
	if workload == nil {
		return nil
	}

	updated := rs.DeepCopy()
	updated.Finalizers = append(updated.Finalizers, replicaSetPodGroupFinalizer)
	_, err = rsc.kubeClient.AppsV1().ReplicaSets(rs.Namespace).Update(ctx, updated, metav1.UpdateOptions{})
	return err
}

// cleanupReplicaSetPodGroup removes the PodGroup only when it is owned by this ReplicaSet UID.
func (rsc *ReplicaSetController) cleanupReplicaSetPodGroup(ctx context.Context, rs *apps.ReplicaSet) error {
	var podGroup *schedulingv1beta1.PodGroup
	var err error
	if rsc.podGroupLister != nil {
		podGroup, err = rsc.podGroupLister.PodGroups(rs.Namespace).Get(rs.Name)
	}
	if rsc.podGroupLister == nil || apierrors.IsNotFound(err) {
		// A cache miss can precede observation of a newly created PodGroup.
		podGroup, err = rsc.kubeClient.SchedulingV1beta1().PodGroups(rs.Namespace).Get(ctx, rs.Name, metav1.GetOptions{})
	}
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if hasReplicaSetOwner(podGroup, rs) {
		if err := rsc.kubeClient.SchedulingV1beta1().PodGroups(rs.Namespace).Delete(ctx, podGroup.Name, metav1.DeleteOptions{
			Preconditions: &metav1.Preconditions{
				UID: &podGroup.UID,
			},
		}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func (rsc *ReplicaSetController) removeReplicaSetPodGroupFinalizer(ctx context.Context, rs *apps.ReplicaSet) error {
	updated := rs.DeepCopy()
	finalizers := updated.Finalizers[:0]
	for _, finalizer := range updated.Finalizers {
		if finalizer != replicaSetPodGroupFinalizer {
			finalizers = append(finalizers, finalizer)
		}
	}
	updated.Finalizers = finalizers
	_, err := rsc.kubeClient.AppsV1().ReplicaSets(rs.Namespace).Update(ctx, updated, metav1.UpdateOptions{})
	return err
}

// cleanupDeletingReplicaSet attempts owner-checked PodGroup cleanup before
// removing the ReplicaSet finalizer, even when WorkloadWithDeployment is
// disabled, so existing finalizers don't strand ReplicaSets.
func (rsc *ReplicaSetController) cleanupDeletingReplicaSet(ctx context.Context, rs *apps.ReplicaSet) error {
	if rsc.Kind != "ReplicaSet" || !hasReplicaSetPodGroupFinalizer(rs) {
		return nil
	}
	if err := rsc.cleanupReplicaSetPodGroup(ctx, rs); err != nil {
		return err
	}
	return rsc.removeReplicaSetPodGroupFinalizer(ctx, rs)
}

func hasReplicaSetOwner(podGroup *schedulingv1beta1.PodGroup, rs *apps.ReplicaSet) bool {
	for _, owner := range podGroup.OwnerReferences {
		if owner.APIVersion == apps.SchemeGroupVersion.String() &&
			owner.Kind == "ReplicaSet" && owner.Name == rs.Name && owner.UID == rs.UID &&
			owner.Controller != nil && *owner.Controller {
			return true
		}
	}
	return false
}

func (rsc *ReplicaSetController) patchPodGroupMinCount(ctx context.Context, podGroup *schedulingv1beta1.PodGroup, minCount int32) error {
	patch, err := json.Marshal(map[string]interface{}{
		"metadata": map[string]interface{}{
			"uid": podGroup.UID,
		},
		"spec": map[string]interface{}{
			"schedulingPolicy": map[string]interface{}{
				"gang": map[string]interface{}{"minCount": minCount},
			},
		},
	})
	if err != nil {
		return err
	}
	_, err = rsc.kubeClient.SchedulingV1beta1().PodGroups(podGroup.Namespace).
		Patch(ctx, podGroup.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	return err
}

func (rsc *ReplicaSetController) deletePodGroupForReplicaSet(ctx context.Context, rs *apps.ReplicaSet) error {
	if !rsc.schedulingEnabled || rsc.Kind != "ReplicaSet" || !utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) {
		return nil
	}
	return rsc.cleanupReplicaSetPodGroup(ctx, rs)
}

// ensurePodGroupForReplicaSet ensures and returns the ReplicaSet-owned PodGroup
// before manageReplicas can create Pods. At zero replicas, it requests deletion
// rather than writing minCount=0. The PodGroup name matches the ReplicaSet
// name, which includes the pod-template-hash.
func (rsc *ReplicaSetController) ensurePodGroupForReplicaSet(
	ctx context.Context, rs *apps.ReplicaSet,
) (*schedulingv1beta1.PodGroup, error) {
	if !rsc.schedulingEnabled || rsc.Kind != "ReplicaSet" || !utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) || rs.DeletionTimestamp != nil {
		return nil, nil
	}
	if rs.Spec.Replicas == nil {
		return nil, nil
	}

	workload, err := rsc.findWorkloadForReplicaSet(rs)
	if err != nil {
		return nil, err
	}
	if workload == nil {
		return nil, nil
	}
	if len(workload.Spec.PodGroupTemplates) != 1 {
		return nil, fmt.Errorf("workload %s/%s has %d PodGroupTemplates, want 1",
			workload.Namespace, workload.Name, len(workload.Spec.PodGroupTemplates))
	}

	if *rs.Spec.Replicas == 0 {
		return nil, rsc.deletePodGroupForReplicaSet(ctx, rs)
	}

	template := workload.Spec.PodGroupTemplates[0]
	templateName := template.Name
	podGroupName := rs.Name
	desiredMinCount := int32(0)
	if template.SchedulingPolicy.Gang != nil {
		desiredMinCount = template.SchedulingPolicy.Gang.MinCount
	}
	podGroup, err := rsc.podGroupLister.PodGroups(rs.Namespace).Get(podGroupName)
	if apierrors.IsNotFound(err) {
		owner := replicaSetPodGroupOwner(rs)
		builder := workloadbuilder.NewBuilderFromExistingWorkload(workload, workloadbuilder.BuildOptions{Owner: &owner})
		desired, err := builder.NewPodGroup(podGroupName, templateName)
		if err != nil {
			return nil, fmt.Errorf("failed to build PodGroup %s/%s: %w",
				rs.Namespace, podGroupName, err)
		}
		desired.OwnerReferences = append(desired.OwnerReferences, workloadPodGroupOwner(workload))
		created, err := rsc.kubeClient.SchedulingV1beta1().PodGroups(rs.Namespace).
			Create(ctx, desired, metav1.CreateOptions{})
		if err != nil {
			return nil, err
		}
		return created, nil
	}
	if err != nil {
		return nil, err
	}

	if !hasReplicaSetOwner(podGroup, rs) {
		// Avoid wiring Pods to foreign or ownerless PodGroups.
		return nil, fmt.Errorf("PodGroup %s/%s is not controlled by ReplicaSet %s",
			podGroup.Namespace, podGroup.Name, rs.Name)
	}
	if podGroup.DeletionTimestamp != nil {
		return nil, fmt.Errorf("PodGroup %s/%s is being deleted", podGroup.Namespace, podGroup.Name)
	}
	if podGroup.Spec.SchedulingPolicy.Gang != nil &&
		podGroup.Spec.SchedulingPolicy.Gang.MinCount != desiredMinCount {
		if err := rsc.patchPodGroupMinCount(ctx, podGroup, desiredMinCount); err != nil {
			return nil, err
		}
	}
	return podGroup, nil
}
