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

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1beta1"
	"k8s.io/client-go/tools/cache"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

func (rsc *ReplicaSetController) addSchedulingInformers(
	workloadInformer schedulinginformers.WorkloadInformer,
	podGroupInformer schedulinginformers.PodGroupInformer,
) error {
	if workloadInformer == nil {
		return fmt.Errorf("workload informer is required for ReplicaSet Workload integration")
	}
	if podGroupInformer == nil {
		return fmt.Errorf("PodGroup informer is required for ReplicaSet Workload integration")
	}
	rsc.workloadLister = workloadInformer.Lister()
	rsc.workloadListerSynced = workloadInformer.Informer().HasSynced
	rsc.podGroupLister = podGroupInformer.Lister()
	rsc.podGroupListerSynced = podGroupInformer.Informer().HasSynced

	if _, err := workloadInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) { rsc.enqueueWorkloadReplicaSets(obj) },
		UpdateFunc: func(oldObj, newObj interface{}) {
			if workloadSpecChanged(oldObj, newObj) {
				rsc.enqueueWorkloadReplicaSets(newObj)
			}
		},
	}); err != nil {
		return fmt.Errorf("adding Workload event handler: %w", err)
	}

	if _, err := podGroupInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { rsc.enqueueReplicaSetForPodGroup(obj) },
		UpdateFunc: func(_, newObj interface{}) { rsc.enqueueReplicaSetForPodGroup(newObj) },
		DeleteFunc: func(obj interface{}) { rsc.enqueueReplicaSetForPodGroup(obj) },
	}); err != nil {
		return fmt.Errorf("adding PodGroup event handler: %w", err)
	}
	return nil
}

func (rsc *ReplicaSetController) schedulingCachesSynced() []cache.InformerSynced {
	if !rsc.controllerFeatures.EnableWorkloadScheduling || !utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) {
		return nil
	}
	return []cache.InformerSynced{rsc.workloadListerSynced, rsc.podGroupListerSynced}
}

// hasDelegatedPodGroup reports whether the ReplicaSet selects a template in a
// Deployment-owned Workload via the workload-name or group-template-name annotation.
func hasDelegatedPodGroup(rs *apps.ReplicaSet) bool {
	_, hasWorkload := rs.Annotations[apischeduling.WorkloadNameAnnotation]
	_, hasTemplate := rs.Annotations[apischeduling.GroupTemplateNameAnnotation]
	return hasWorkload || hasTemplate
}

func (rsc *ReplicaSetController) shouldManagePodGroupForReplicaSet(rs *apps.ReplicaSet) bool {
	return rsc.controllerFeatures.EnableWorkloadScheduling &&
		utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) &&
		rs.Spec.Template.Spec.SchedulingGroup == nil &&
		hasDelegatedPodGroup(rs)
}

func (rsc *ReplicaSetController) getWorkload(namespace, name string) (*schedulingv1beta1.Workload, error) {
	if rsc.workloadLister == nil {
		return nil, fmt.Errorf("workload lister is not configured")
	}
	wl, err := rsc.workloadLister.Workloads(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return wl, err
}

func (rsc *ReplicaSetController) getPodGroup(namespace, name string) (*schedulingv1beta1.PodGroup, error) {
	if rsc.podGroupLister == nil {
		return nil, fmt.Errorf("podgroup lister is not configured")
	}
	pg, err := rsc.podGroupLister.PodGroups(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return pg, err
}

func getPodGroupTemplateByName(workload *schedulingv1beta1.Workload, name string) *schedulingv1beta1.PodGroupTemplate {
	for i := range workload.Spec.PodGroupTemplates {
		if workload.Spec.PodGroupTemplates[i].Name == name {
			return &workload.Spec.PodGroupTemplates[i]
		}
	}
	return nil
}

func podGroupMatchesTemplate(pg *schedulingv1beta1.PodGroup, workloadName, templateName string) bool {
	ref := pg.Spec.WorkloadRef
	return ref != nil && ref.WorkloadName == workloadName && ref.TemplateName == templateName
}

func (rsc *ReplicaSetController) newBuilderFromExistingWorkload(rs *apps.ReplicaSet, workload *schedulingv1beta1.Workload) *workloadbuilder.Builder {
	return workloadbuilder.NewBuilderFromExistingWorkload(workload, workloadbuilder.BuildOptions{
		Owner: metav1.NewControllerRef(rs, rsc.GroupVersionKind),
	})
}

// createPodGroup creates the runtime PodGroup for the ReplicaSet from the
// Workload's PodGroupTemplate. The PodGroup carries only a controller ownerRef
// to the ReplicaSet, so standard garbage collection removes it with the RS.
func (rsc *ReplicaSetController) createPodGroup(ctx context.Context, rs *apps.ReplicaSet, workload *schedulingv1beta1.Workload, templateName, pgName string) (*schedulingv1beta1.PodGroup, error) {
	desired, err := rsc.newBuilderFromExistingWorkload(rs, workload).NewPodGroup(pgName, templateName)
	if err != nil {
		return nil, fmt.Errorf("failed to build PodGroup %s/%s: %w", rs.Namespace, pgName, err)
	}
	// priorityClassName is per-revision, so source it from this ReplicaSet's
	// template rather than the shared Workload template.
	// Clear inherited priority and preemptionPolicy values so admission resolves
	// them from the ReplicaSet's priority class.
	desired.Spec.PriorityClassName = rs.Spec.Template.Spec.PriorityClassName
	desired.Spec.Priority = nil
	desired.Spec.PreemptionPolicy = nil
	created, err := rsc.kubeClient.SchedulingV1beta1().PodGroups(rs.Namespace).Create(ctx, desired, metav1.CreateOptions{})
	if err != nil {
		rsc.recorder.Eventf(rs, v1.EventTypeWarning, "FailedPodGroupCreate", "Failed to create PodGroup %s for ReplicaSet %s: %v", pgName, rs.Name, err)
		return nil, err
	}
	rsc.recorder.Eventf(rs, v1.EventTypeNormal, "PodGroupCreated", "Created PodGroup %s for ReplicaSet %s", created.Name, rs.Name)
	return created, nil
}

// syncPodGroupMinCount patches the runtime PodGroup's gang minCount to the
// desired value when the ReplicaSet controls it.
func (rsc *ReplicaSetController) syncPodGroupMinCount(ctx context.Context, rs *apps.ReplicaSet, podGroup *schedulingv1beta1.PodGroup, desiredMinCount int32) error {
	if podGroup.DeletionTimestamp != nil ||
		!metav1.IsControlledBy(podGroup, rs) ||
		podGroup.Spec.SchedulingPolicy.Gang == nil ||
		podGroup.Spec.SchedulingPolicy.Gang.MinCount == desiredMinCount {
		return nil
	}
	return rsc.patchPodGroupMinCount(ctx, podGroup, desiredMinCount)
}

func (rsc *ReplicaSetController) patchPodGroupMinCount(ctx context.Context, podGroup *schedulingv1beta1.PodGroup, minCount int32) error {
	updatedPG := podGroup.DeepCopy()
	updatedPG.Spec.SchedulingPolicy.Gang.MinCount = minCount
	patch, err := createStrategicMergePatch(podGroup, updatedPG, &schedulingv1beta1.PodGroup{})
	if err != nil {
		return err
	}
	_, err = rsc.kubeClient.SchedulingV1beta1().PodGroups(podGroup.Namespace).
		Patch(ctx, podGroup.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	return err
}

// createStrategicMergePatch returns the strategic merge patch that turns original
// into updated. dataStruct must be an empty object of their type.
func createStrategicMergePatch(original, updated, dataStruct any) ([]byte, error) {
	oldData, err := json.Marshal(original)
	if err != nil {
		return nil, err
	}
	newData, err := json.Marshal(updated)
	if err != nil {
		return nil, err
	}
	return strategicpatch.CreateTwoWayMergePatch(oldData, newData, dataStruct)
}

// deletePodGroupForReplicaSet deletes the RS-owned PodGroup named pgName. Used
// when the ReplicaSet scales to zero; standard GC handles ReplicaSet deletion.
func (rsc *ReplicaSetController) deletePodGroupForReplicaSet(ctx context.Context, rs *apps.ReplicaSet, pgName string) error {
	podGroup, err := rsc.getPodGroup(rs.Namespace, pgName)
	if err != nil {
		return err
	}
	if podGroup == nil {
		return nil
	}
	if !metav1.IsControlledBy(podGroup, rs) {
		return nil
	}
	if err := rsc.kubeClient.SchedulingV1beta1().PodGroups(rs.Namespace).Delete(ctx, podGroup.Name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{UID: &podGroup.UID},
	}); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// ensurePodGroupForReplicaSet discovers the Deployment-owned Workload named by the
// ReplicaSet's delegation annotations and ensures the runtime PodGroup before
// manageReplicas creates Pods. At zero replicas it deletes the PodGroup. When the
// annotations are present but the Workload is not yet cached, it returns an error
// so syncReplicaSet requeues instead of creating ungrouped Pods.
func (rsc *ReplicaSetController) ensurePodGroupForReplicaSet(ctx context.Context, rs *apps.ReplicaSet) (*schedulingv1beta1.PodGroup, error) {
	if !rsc.shouldManagePodGroupForReplicaSet(rs) || rs.DeletionTimestamp != nil || rs.Spec.Replicas == nil {
		return nil, nil
	}
	workloadName := rs.Annotations[apischeduling.WorkloadNameAnnotation]
	templateName := rs.Annotations[apischeduling.GroupTemplateNameAnnotation]
	if workloadName == "" || templateName == "" {
		return nil, fmt.Errorf("replicaset %s/%s has incomplete delegation annotations", rs.Namespace, rs.Name)
	}
	pgName := workloadbuilder.GeneratePodGroupName(workloadName, templateName, rs.UID)

	if *rs.Spec.Replicas == 0 {
		return nil, rsc.deletePodGroupForReplicaSet(ctx, rs, pgName)
	}

	workload, err := rsc.getWorkload(rs.Namespace, workloadName)
	if err != nil {
		return nil, err
	}
	if workload == nil {
		return nil, fmt.Errorf("workload %s/%s not found yet for replicaset %s", rs.Namespace, workloadName, rs.Name)
	}
	template := getPodGroupTemplateByName(workload, templateName)
	if template == nil {
		return nil, fmt.Errorf("podgrouptemplate %q not found in workload %s/%s", templateName, rs.Namespace, workloadName)
	}

	desiredMinCount := int32(0)
	if template.SchedulingPolicy.Gang != nil {
		desiredMinCount = template.SchedulingPolicy.Gang.MinCount
	}
	podGroup, err := rsc.getPodGroup(rs.Namespace, pgName)
	if err != nil {
		return nil, err
	}
	if podGroup == nil {
		return rsc.createPodGroup(ctx, rs, workload, templateName, pgName)
	}
	if !podGroupMatchesTemplate(podGroup, workload.Name, templateName) {
		return nil, fmt.Errorf("podgroup %s/%s does not reference workload %q template %q", rs.Namespace, pgName, workload.Name, templateName)
	}
	if podGroup.DeletionTimestamp != nil {
		return nil, fmt.Errorf("podgroup %s/%s is being deleted", podGroup.Namespace, podGroup.Name)
	}
	if err := rsc.syncPodGroupMinCount(ctx, rs, podGroup, desiredMinCount); err != nil {
		return nil, err
	}
	return podGroup, nil
}

// enqueueReplicaSetForPodGroup re-enqueues the ReplicaSet that controls the PodGroup.
func (rsc *ReplicaSetController) enqueueReplicaSetForPodGroup(obj interface{}) {
	podGroup, ok := obj.(*schedulingv1beta1.PodGroup)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return
		}
		podGroup, ok = tombstone.Obj.(*schedulingv1beta1.PodGroup)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a PodGroup %#v", obj))
			return
		}
	}
	controllerRef := metav1.GetControllerOf(podGroup)
	if controllerRef == nil || controllerRef.Kind != rsc.Kind {
		return
	}
	rs, err := rsc.rsLister.ReplicaSets(podGroup.Namespace).Get(controllerRef.Name)
	if err != nil || rs.UID != controllerRef.UID {
		return
	}
	rsc.enqueueRS(rs)
}

// enqueueWorkloadReplicaSets enqueues the ReplicaSets delegating to the changed
// Workload, so a minCount update on the Workload template reaches their PodGroups.
func (rsc *ReplicaSetController) enqueueWorkloadReplicaSets(obj interface{}) {
	workload, ok := obj.(*schedulingv1beta1.Workload)
	if !ok {
		return
	}
	replicaSets, err := rsc.rsLister.ReplicaSets(workload.Namespace).List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("looking up ReplicaSets for Workload %s/%s: %w", workload.Namespace, workload.Name, err))
		return
	}
	for _, rs := range replicaSets {
		if rs.Annotations[apischeduling.WorkloadNameAnnotation] != workload.Name {
			continue
		}
		rsc.enqueueRS(rs)
	}
}

// workloadSpecChanged reports whether the Workload spec differs between the two
// informer objects, filtering out metadata-only updates.
func workloadSpecChanged(oldObj, newObj interface{}) bool {
	oldWorkload, oldOK := oldObj.(*schedulingv1beta1.Workload)
	newWorkload, newOK := newObj.(*schedulingv1beta1.Workload)
	return oldOK && newOK && !apiequality.Semantic.DeepEqual(oldWorkload.Spec, newWorkload.Spec)
}
