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

package job

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1beta1"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/klog/v2"
	jobutil "k8s.io/kubernetes/pkg/api/job"
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// ReasonSchedulingDependencyNotFound means an external Workload or its
	// PodGroupTemplate selected by the Job is not available yet.
	ReasonSchedulingDependencyNotFound = "SchedulingDependencyNotFound"
	// ReasonInvalidSchedulingObjects means the Job's delegation annotations or
	// an existing Workload or PodGroup are invalid for the Job.
	ReasonInvalidSchedulingObjects = "InvalidSchedulingObjects"
	// ReasonSchedulingObjectsValid means SchedulingBlocked was cleared.
	ReasonSchedulingObjectsValid = "SchedulingObjectsValid"

	// the controller manages no scheduling objects, either the feature
	// is off, the user brought their own PodGroup, or a parent controller
	// owns both objects without delegating.
	manageNone managementMode = iota

	// the controller owns both Workload and PodGroup.
	manageBoth

	// the Job selects a template in an external Workload (ancestor-owned or
	// user pre-created) and the controller owns only the runtime PodGroup.
	managePodGroupOnly
)

// managementMode describes which scheduling objects the Job controller
// manages for a given Job.
type managementMode int

// schedulingBlockedError is returned when the Job's scheduling objects block pod
// creation until the user fixes them.
type schedulingBlockedError struct {
	reason  string
	message string
}

func (e *schedulingBlockedError) Error() string {
	return fmt.Sprintf("scheduling blocked (%s): %s", e.reason, e.message)
}

// getManagementMode computes which scheduling objects the Job controller
// manages for the Job.
func getManagementMode(job *batch.Job) managementMode {
	if !feature.DefaultFeatureGate.Enabled(features.WorkloadWithJob) {
		return manageNone
	}
	if job.Spec.Template.Spec.SchedulingGroup != nil {
		return manageNone
	}
	// The delegation annotations select an external Workload (parent-delegated
	// or user pre-created). Checked before ownership so a standalone BYO Job
	// does not compile its Workload.
	if hasDelegatedPodGroup(job) {
		return managePodGroupOnly
	}
	// spec.scheduling is explicit intent, honored regardless of the owner.
	if job.Spec.Scheduling != nil {
		return manageBoth
	}
	// An owned Job that asked for nothing is left to its owner.
	if metav1.GetControllerOf(job) != nil {
		return manageNone
	}
	return manageBoth
}

// shouldManageWorkloadForJob reports whether the Job controller should
// materialize scheduling objects (Workload and PodGroup) for the Job.
func shouldManageWorkloadForJob(job *batch.Job) bool {
	return getManagementMode(job) != manageNone
}

// hasDelegatedPodGroup reports whether the Job selects a template in an
// external Workload via the workload-name or group-template-name annotation.
func hasDelegatedPodGroup(job *batch.Job) bool {
	_, hasWorkload := job.Annotations[apischeduling.WorkloadNameAnnotation]
	_, hasTemplate := job.Annotations[apischeduling.GroupTemplateNameAnnotation]
	return hasWorkload || hasTemplate
}

// ensureWorkloadAndPodGroup discovers or creates the scheduling objects the
// controller manages for the Job and returns the runtime PodGroup its pods
// should reference. It returns nil when the controller manages nothing for the
// Job, when the Job is suspended, or when no PodGroup exists and the Job
// already has active pods.
func (jm *Controller) ensureWorkloadAndPodGroup(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1beta1.PodGroup, error) {
	mode := getManagementMode(job)
	if mode == manageNone {
		return nil, nil
	}

	// Case 1 - manage the PodGroup only.
	if mode == managePodGroupOnly {
		return jm.ensureDelegatedPodGroup(ctx, job, canCreatePodGroup(job, pods))
	}

	// Case 2 - manage both Workload and PodGroup (root Job).
	// Runs before the Workload lookup so a suspended Job's PodGroup is
	// deleted even when its Workload is gone.
	if jobSuspended(job) {
		if err := jm.deleteSuspendedRootPodGroup(ctx, job); err != nil {
			return nil, err
		}
	}
	wl, err := jm.getOrCreateWorkload(ctx, job, canCreateWorkload(job, pods))
	if err != nil || wl == nil {
		return nil, err
	}

	// Propagate a changed gang minCount (elastic scaling) to the Workload
	// before the PodGroup is looked up, so a PodGroup recreated on resume is
	// built from the current size instead of being created stale and patched.
	// Runs regardless of pod state so a running gang Job can be resized, and
	// while suspended so the retained Workload stays current.
	wl, err = jm.syncWorkloadGangMinCount(ctx, job, wl)
	if err != nil {
		return nil, err
	}
	if jobSuspended(job) {
		return nil, nil
	}
	template := wl.Spec.PodGroupTemplates[0]
	pg, err := jm.getOrCreatePodGroup(ctx, job, wl, template.Name, canCreatePodGroup(job, pods), true)
	if err != nil {
		return nil, err
	}
	if template.SchedulingPolicy.Gang != nil {
		if err := jm.syncPodGroupMinCount(ctx, job, pg, template.SchedulingPolicy.Gang.MinCount); err != nil {
			return nil, err
		}
	}
	return pg, nil
}

// syncWorkloadGangMinCount propagates an updated gang size to the
// controller-owned Workload. It recompiles the desired Workload from the
// current Job spec and, when the resolved policy is Gang, patches only the
// minCount on the live Workload if it differs.
//
// Basic policies are a no-op. Only minCount is changed; every other
// Workload field is immutable at the API.
func (jm *Controller) syncWorkloadGangMinCount(ctx context.Context, job *batch.Job,
	workload *schedulingv1beta1.Workload) (*schedulingv1beta1.Workload, error) {
	if job.Spec.Scheduling == nil ||
		job.Spec.Scheduling.SchedulingPolicy == nil ||
		job.Spec.Scheduling.SchedulingPolicy.Gang == nil {
		return workload, nil
	}

	desiredWorkload, err := generateWorkload(job)
	if err != nil || len(desiredWorkload.Spec.PodGroupTemplates) == 0 {
		return workload, err
	}
	gang := desiredWorkload.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
	if gang == nil {
		return workload, nil
	}
	desiredMinCount := gang.MinCount

	// Mutation is allowed only when this Job is the controller owner.
	currentPGTemplate := workload.Spec.PodGroupTemplates
	if metav1.IsControlledBy(workload, job) &&
		len(currentPGTemplate) == 1 &&
		currentPGTemplate[0].SchedulingPolicy.Gang != nil &&
		currentPGTemplate[0].SchedulingPolicy.Gang.MinCount != desiredMinCount {

		patched, err := jm.patchWorkloadMinCount(ctx, workload, desiredMinCount)
		if err != nil {
			return nil, err
		}
		workload = patched
	}
	return workload, nil
}

func (jm *Controller) syncPodGroupMinCount(ctx context.Context, job *batch.Job,
	podGroup *schedulingv1beta1.PodGroup, desiredMinCount int32) error {
	// A terminating PodGroup is about to be replaced and patching it
	// would race with its deletion.
	if podGroup == nil || podGroup.DeletionTimestamp != nil ||
		!metav1.IsControlledBy(podGroup, job) ||
		podGroup.Spec.SchedulingPolicy.Gang == nil ||
		podGroup.Spec.SchedulingPolicy.Gang.MinCount == desiredMinCount {
		return nil
	}
	return jm.patchPodGroupMinCount(ctx, podGroup, desiredMinCount)
}

// patchWorkloadMinCount sets the gang minCount on the Workload's single
// PodGroupTemplate and returns the patched Workload. minCount is the only
// mutable field on a compiled Workload.
func (jm *Controller) patchWorkloadMinCount(ctx context.Context, workload *schedulingv1beta1.Workload, minCount int32) (*schedulingv1beta1.Workload, error) {
	updatedW := workload.DeepCopy()
	updatedW.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = minCount

	patch, err := createStrategicMergePatch(workload, updatedW, &schedulingv1beta1.Workload{})
	if err != nil {
		return nil, err
	}
	return jm.kubeClient.SchedulingV1beta1().Workloads(workload.Namespace).
		Patch(ctx, workload.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
}

// patchPodGroupMinCount sets the gang minCount on the runtime PodGroup.
func (jm *Controller) patchPodGroupMinCount(ctx context.Context, podGroup *schedulingv1beta1.PodGroup, minCount int32) error {
	updatedPG := podGroup.DeepCopy()
	updatedPG.Spec.SchedulingPolicy.Gang.MinCount = minCount

	patch, err := createStrategicMergePatch(podGroup, updatedPG, &schedulingv1beta1.PodGroup{})
	if err != nil {
		return err
	}
	_, err = jm.kubeClient.SchedulingV1beta1().PodGroups(podGroup.Namespace).
		Patch(ctx, podGroup.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	return err
}

// createStrategicMergePatch returns the strategic merge patch that turns
// original into updated. dataStruct must be an empty object of their type.
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

// ensureSchedulingObjectsAndCondition reports whether the condition changed.
// The Warning event is emitted only on that change.
func (jm *Controller) ensureSchedulingObjectsAndCondition(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1beta1.PodGroup, bool, error) {
	pg, err := jm.ensureWorkloadAndPodGroup(ctx, job, pods)
	blockedErr, blocked := errors.AsType[*schedulingBlockedError](err)
	if err != nil && !blocked {
		return nil, false, err
	}

	var changed bool
	if blocked {
		job.Status.Conditions, changed = ensureJobConditionStatus(job.Status.Conditions, batch.JobSchedulingBlocked,
			v1.ConditionTrue, blockedErr.reason, blockedErr.message, jm.clock.Now())
		if changed {
			jm.recorder.Event(job, v1.EventTypeWarning, blockedErr.reason, blockedErr.message)
		}
	} else {
		// No-op unless SchedulingBlocked=True was recorded by an earlier sync.
		job.Status.Conditions, changed = ensureJobConditionStatus(job.Status.Conditions, batch.JobSchedulingBlocked,
			v1.ConditionFalse, ReasonSchedulingObjectsValid, "Scheduling objects for the Job are valid", jm.clock.Now())
	}
	return pg, changed, err
}

func (jm *Controller) getWorkload(namespace, name string) (*schedulingv1beta1.Workload, error) {
	if jm.workloadLister == nil {
		return nil, fmt.Errorf("workload lister is not configured")
	}
	wl, err := jm.workloadLister.Workloads(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return wl, err
}

func (jm *Controller) getPodGroup(namespace, name string) (*schedulingv1beta1.PodGroup, error) {
	if jm.podGroupLister == nil {
		return nil, fmt.Errorf("podgroup lister is not configured")
	}
	pg, err := jm.podGroupLister.PodGroups(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return pg, err
}

// ensureDelegatedPodGroup discovers or creates the runtime PodGroup for a
// Job that selects a template in an external Workload, or deletes it while the
// Job is suspended. The PodGroup references the Workload named by the
// workload-name annotation and the template named by the group-template-name
// annotation, and carries only a controller ownerRef to the Job.
func (jm *Controller) ensureDelegatedPodGroup(ctx context.Context,
	job *batch.Job, canCreate bool) (*schedulingv1beta1.PodGroup, error) {
	logger := klog.FromContext(ctx)

	workloadName := job.Annotations[apischeduling.WorkloadNameAnnotation]
	templateName := job.Annotations[apischeduling.GroupTemplateNameAnnotation]
	if workloadName == "" {
		return nil, &schedulingBlockedError{
			reason:  ReasonInvalidSchedulingObjects,
			message: fmt.Sprintf("Job %s/%s is missing or has an empty %s annotation", job.Namespace, job.Name, apischeduling.WorkloadNameAnnotation),
		}
	}
	if templateName == "" {
		return nil, &schedulingBlockedError{
			reason:  ReasonInvalidSchedulingObjects,
			message: fmt.Sprintf("Job %s/%s is missing or has an empty %s annotation", job.Namespace, job.Name, apischeduling.GroupTemplateNameAnnotation),
		}
	}
	// The PodGroup name depends only on the annotations, so a suspended Job's
	// PodGroup is deleted even when the external Workload or its template has
	// gone away and the Job is blocked below.
	if jobSuspended(job) {
		pgName := workloadbuilder.GeneratePodGroupName(workloadName, templateName, job.UID)
		if err := jm.deleteSuspendedPodGroup(ctx, job, pgName); err != nil {
			return nil, err
		}
	}
	parentWorkload, err := jm.getWorkload(job.Namespace, workloadName)
	if err != nil {
		return nil, err
	}
	if parentWorkload == nil {
		logger.V(2).Info("Named Workload not found yet for delegated Job, will retry",
			"job", klog.KObj(job), "workload", workloadName)
		return nil, &schedulingBlockedError{
			reason:  ReasonSchedulingDependencyNotFound,
			message: fmt.Sprintf("Workload %s/%s not found for Job %s", job.Namespace, workloadName, job.Name),
		}
	}

	template := getPodGroupTemplateByName(parentWorkload, templateName)
	if template == nil {
		logger.V(2).Info("Delegated PodGroupTemplate not found in Workload, will retry",
			"job", klog.KObj(job), "workload", klog.KObj(parentWorkload), "template", templateName)
		return nil, &schedulingBlockedError{
			reason:  ReasonSchedulingDependencyNotFound,
			message: fmt.Sprintf("PodGroupTemplate %q not found in Workload %s/%s for Job %s", templateName, parentWorkload.Namespace, parentWorkload.Name, job.Name),
		}
	}
	// The PodGroup was already deleted on suspend
	if jobSuspended(job) {
		return nil, nil
	}

	// The external Workload is never linked as an owner.
	updatedPG, err := jm.getOrCreatePodGroup(ctx, job, parentWorkload, template.Name, canCreate, false)
	if err != nil {
		return nil, err
	}
	// The parent controller may resize the gang on its Workload's template
	// after the PodGroup exists. The parent's template is the source of truth
	// in the delegated mode, so follow it.
	if template.SchedulingPolicy.Gang != nil {
		if err := jm.syncPodGroupMinCount(ctx, job, updatedPG, template.SchedulingPolicy.Gang.MinCount); err != nil {
			return nil, err
		}
	}
	return updatedPG, nil
}

// canCreateWorkload reports whether the Job has never run where it owns
// no pods, active or terminal.
func canCreateWorkload(job *batch.Job, pods []*v1.Pod) bool {
	return len(pods) == 0 &&
		job.Status.StartTime == nil &&
		job.Status.Succeeded == 0 &&
		job.Status.Failed == 0
}

// canCreatePodGroup reports whether the Job is running and has no active pods,
// so grouped and ungrouped pods are not mixed. Suspend deletes the active pods,
// which lets resume recreate the PodGroup deleted on suspend.
func canCreatePodGroup(job *batch.Job, pods []*v1.Pod) bool {
	return !jobSuspended(job) && !slices.ContainsFunc(pods, controller.IsPodActive)
}

// deleteSuspendedRootPodGroup deletes the PodGroup of a suspended root Job.
// The PodGroup is named after the Workload's template, which a reused Workload
// may name differently from the Job. The Job's own template name covers a
// Workload that is gone.
func (jm *Controller) deleteSuspendedRootPodGroup(ctx context.Context, job *batch.Job) error {
	workloadName := workloadbuilder.GenerateWorkloadName(job.Name, job.UID)
	wl, err := jm.getWorkload(job.Namespace, workloadName)
	if err != nil {
		return err
	}
	templateName := podGroupTemplateName(job)
	if wl != nil && len(wl.Spec.PodGroupTemplates) == 1 {
		templateName = wl.Spec.PodGroupTemplates[0].Name
	}
	return jm.deleteSuspendedPodGroup(ctx, job, workloadbuilder.GeneratePodGroupName(workloadName, templateName, job.UID))
}

// deleteSuspendedPodGroup deletes the named PodGroup of a suspended Job so the
// Job is scheduled as a new gang on resume. A PodGroup the Job does not control
// is left untouched, and the Workload is never deleted.
func (jm *Controller) deleteSuspendedPodGroup(ctx context.Context, job *batch.Job, name string) error {
	podGroup, err := jm.getPodGroup(job.Namespace, name)
	if err != nil {
		return err
	}
	if podGroup == nil || podGroup.DeletionTimestamp != nil || !metav1.IsControlledBy(podGroup, job) {
		return nil
	}
	res, err := jm.kubeClient.SchedulingV1beta1().PodGroups(podGroup.Namespace).DeleteWithResult(ctx, podGroup.Name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &podGroup.UID,
		},
	})
	switch {
	case apierrors.IsNotFound(err):
		// Deleted by another actor since the lister was read.
		return nil
	case err != nil:
		return fmt.Errorf("deleting PodGroup %s/%s of suspended Job %s/%s: %w", podGroup.Namespace, podGroup.Name, job.Namespace, job.Name, err)
	}
	jm.recorder.Eventf(job, v1.EventTypeNormal, "PodGroupDeleted",
		"Deleted PodGroup %s for suspended Job %s", podGroup.Name, job.Name)

	// Record the write so a quick resume waits for the informer to observe
	// the deletion instead of reusing the cached PodGroup. A finalizer-protected
	// PodGroup comes back terminating; an immediate delete returns a Status.
	// Both carry the resource version of the delete.
	var rv string
	if obj, getErr := res.Get(); getErr == nil {
		switch o := obj.(type) {
		case *schedulingv1beta1.PodGroup:
			rv = o.ResourceVersion
		case *metav1.Status:
			rv = o.ResourceVersion
		}
	}
	if rv == "" {
		klog.FromContext(ctx).V(2).Info("PodGroup delete returned no resource version, not recording the write",
			"job", klog.KObj(job), "podGroup", klog.KObj(podGroup))
		return nil
	}
	jm.consistencyStore.WroteAt(
		types.NamespacedName{Namespace: job.Namespace, Name: job.Name},
		job.UID,
		podGroupsGroupResource,
		rv,
	)
	return nil
}

// getOrCreateWorkload returns the Workload for this root Job, looked up by the
// deterministic name. An unsupported object at that name blocks the Job.
// Returns (nil, nil) when no Workload exists and canCreate is false.
func (jm *Controller) getOrCreateWorkload(ctx context.Context,
	job *batch.Job, canCreate bool) (*schedulingv1beta1.Workload, error) {
	name := workloadbuilder.GenerateWorkloadName(job.Name, job.UID)
	wl, err := jm.getWorkload(job.Namespace, name)
	if err != nil {
		return nil, err
	}
	if wl == nil {
		if !canCreate {
			return nil, nil
		}
		return jm.createWorkloadForJob(ctx, job)
	}
	if err := validateWorkloadStructure(job, wl); err != nil {
		return nil, &schedulingBlockedError{reason: ReasonInvalidSchedulingObjects, message: err.Error()}
	}
	return wl, nil
}

// validateWorkloadStructure returns an error when a root Job's Workload does not have
// exactly one template or its controllerRef does not point at this Job.
func validateWorkloadStructure(job *batch.Job, wl *schedulingv1beta1.Workload) error {
	var errMessages []string
	if n := len(wl.Spec.PodGroupTemplates); n != 1 {
		errMessages = append(errMessages, fmt.Sprintf("has %d PodGroupTemplates, want exactly 1", n))
	}
	if !workloadControllerRefMatchesJob(wl, job) {
		errMessages = append(errMessages, fmt.Sprintf("controllerRef does not point at job %s/%s", job.Namespace, job.Name))
	}
	if len(errMessages) == 0 {
		return nil
	}
	return fmt.Errorf("workload %s/%s for job %s/%s: %s",
		wl.Namespace, wl.Name, job.Namespace, job.Name, strings.Join(errMessages, ", "))
}

func workloadControllerRefMatchesJob(wl *schedulingv1beta1.Workload, job *batch.Job) bool {
	ref := wl.Spec.ControllerRef
	return ref != nil &&
		ref.APIGroup == controllerKind.Group &&
		ref.Kind == controllerKind.Kind &&
		ref.Name == job.Name
}

func podGroupMatchesTemplate(pg *schedulingv1beta1.PodGroup, workloadName, templateName string) bool {
	ref := pg.Spec.WorkloadRef
	return ref != nil && ref.WorkloadName == workloadName && ref.TemplateName == templateName
}

// getOrCreatePodGroup returns the Job's PodGroup for the Workload's template
// templateName, looked up by the deterministic name. An object at that name
// that does not reference the template blocks the Job. A terminating PodGroup
// is returned as is and pod creation waits for it to go away.
func (jm *Controller) getOrCreatePodGroup(ctx context.Context, job *batch.Job,
	workload *schedulingv1beta1.Workload, templateName string,
	canCreate, linkWorkloadOwner bool) (*schedulingv1beta1.PodGroup, error) {
	// The Job UID keeps sibling Jobs that select the same template on
	// distinct PodGroups.
	pgName := workloadbuilder.GeneratePodGroupName(workload.Name, templateName, job.UID)
	pg, err := jm.getPodGroup(job.Namespace, pgName)
	if err != nil {
		return nil, err
	}
	if pg != nil && !podGroupMatchesTemplate(pg, workload.Name, templateName) {
		return nil, &schedulingBlockedError{
			reason: ReasonInvalidSchedulingObjects,
			message: fmt.Sprintf("PodGroup %s/%s does not reference Workload %q template %q",
				pg.Namespace, pg.Name, workload.Name, templateName),
		}
	}
	if pg == nil && canCreate {
		return jm.createPodGroup(ctx, job, workload, templateName, linkWorkloadOwner)
	}
	return pg, nil
}

// podGroupTemplateName returns the name of the single PodGroupTemplate the
// controller compiles for a Job. A Job-owned Workload has exactly one template.
func podGroupTemplateName(job *batch.Job) string {
	const suffix = "-pgt-0"
	templateName := strings.ReplaceAll(job.Name, ".", "-") + suffix
	// Preserve the suffix while keeping the name within the DNS label limit.
	if len(templateName) > validation.DNS1123LabelMaxLength {
		templateName = templateName[:validation.DNS1123LabelMaxLength-len(suffix)] + suffix
	}
	return templateName
}

// buildWorkloadItem assembles the single-node logical workload tree for a Job.
// The Job defaults to Basic scheduling; the user's spec.scheduling overrides it.
func buildWorkloadItem(job *batch.Job) *workloadbuilder.WorkloadItem {
	var input workloadbuilder.WorkloadInput
	if s := job.Spec.Scheduling; s != nil {
		input = jobutil.WorkloadInputForJobV1(s)
	}
	return jobutil.WorkloadItemForJob(podGroupTemplateName(job),
		job.Spec.Template.Spec.PriorityClassName, job.Spec.Parallelism, input, nil)
}

// generateWorkload compiles the Job's spec.scheduling into a Workload via the
// shared workloadbuilder library. BuildWorkload also sets the controller
// ownerReference and spec.controllerRef pointing at the Job.
func generateWorkload(job *batch.Job) (*schedulingv1beta1.Workload, error) {
	return workloadbuilder.NewBuilder(buildWorkloadItem(job), workloadbuilder.BuildOptions{
		Name:      workloadbuilder.GenerateWorkloadName(job.Name, job.UID),
		Namespace: job.Namespace,
		Owner:     metav1.NewControllerRef(job, controllerKind),
	}).BuildWorkload()
}

// createWorkloadForJob compiles and creates a new Workload object for the given Job.
func (jm *Controller) createWorkloadForJob(ctx context.Context, job *batch.Job) (*schedulingv1beta1.Workload, error) {
	workload, err := generateWorkload(job)
	if err != nil {
		return nil, fmt.Errorf("compiling Workload for Job %s/%s: %w", job.Namespace, job.Name, err)
	}

	created, err := jm.kubeClient.SchedulingV1beta1().Workloads(job.Namespace).Create(ctx, workload, metav1.CreateOptions{})
	if err != nil {
		jm.recorder.Eventf(job, v1.EventTypeWarning, "FailedWorkloadCreate",
			"Failed to create Workload for Job %s: %v", job.Name, err)
		return nil, err
	}

	jm.recorder.Eventf(job, v1.EventTypeNormal, "WorkloadCreated",
		"Created Workload %s for Job %s", created.Name, job.Name)
	return created, nil
}

// getPodGroupTemplateByName returns the named PodGroupTemplate from the Workload,
// or nil if there is none.
func getPodGroupTemplateByName(workload *schedulingv1beta1.Workload, name string) *schedulingv1beta1.PodGroupTemplate {
	for i := range workload.Spec.PodGroupTemplates {
		if workload.Spec.PodGroupTemplates[i].Name == name {
			return &workload.Spec.PodGroupTemplates[i]
		}
	}
	return nil
}

// newBuilderFromExistingWorkload returns a builder that materializes PodGroups
// from an already-persisted Workload, stamping a controller ownerRef to the Job.
func newBuilderFromExistingWorkload(job *batch.Job, workload *schedulingv1beta1.Workload) *workloadbuilder.Builder {
	return workloadbuilder.NewBuilderFromExistingWorkload(workload, workloadbuilder.BuildOptions{
		Owner: metav1.NewControllerRef(job, controllerKind),
	})
}

// createPodGroup creates a runtime PodGroup for the Job from the Workload's
// PodGroupTemplate named templateName. The persisted Workload is authoritative,
// so the PodGroup is created from it rather than recompiled from the Job's spec.
// It always carries a controller ownerRef to the Job. When linkWorkloadOwner is
// true, it gets a non-controller ownerRef to the Workload. BlockOwnerDeletion is
// left unset so Workload deletion is never blocked by surviving PodGroups.
func (jm *Controller) createPodGroup(ctx context.Context, job *batch.Job, workload *schedulingv1beta1.Workload,
	templateName string, linkWorkloadOwner bool) (*schedulingv1beta1.PodGroup, error) {
	podGroup, err := newBuilderFromExistingWorkload(job, workload).
		NewPodGroup(workloadbuilder.GeneratePodGroupName(workload.Name, templateName, job.UID), templateName)
	if err != nil {
		return nil, err
	}
	if parent := job.Annotations[apischeduling.ParentCompositePodGroupAnnotation]; parent != "" {
		podGroup.Spec.ParentCompositePodGroupName = &parent
	}
	if linkWorkloadOwner {
		podGroup.OwnerReferences = append(podGroup.OwnerReferences, metav1.OwnerReference{
			APIVersion: schedulingv1beta1.SchemeGroupVersion.String(),
			Kind:       "Workload",
			Name:       workload.Name,
			UID:        workload.UID,
		})
	}

	created, err := jm.kubeClient.SchedulingV1beta1().PodGroups(job.Namespace).Create(ctx, podGroup, metav1.CreateOptions{})
	if err != nil {
		jm.recorder.Eventf(job, v1.EventTypeWarning, "FailedPodGroupCreate",
			"Failed to create PodGroup %s for Job %s: %v", podGroup.Name, job.Name, err)
		return nil, err
	}

	jm.recorder.Eventf(job, v1.EventTypeNormal, "PodGroupCreated",
		"Created PodGroup %s for Job %s", created.Name, job.Name)
	return created, nil
}

// addSchedulingInformers wires Workload and PodGroup listers. Objects are
// discovered by name, so no secondary indexes are required.
func (jm *Controller) addSchedulingInformers(workloadInformer schedulinginformers.WorkloadInformer, podGroupInformer schedulinginformers.PodGroupInformer) error {
	if workloadInformer == nil {
		return fmt.Errorf("workload informer is required when the feature gate %q is enabled", features.WorkloadWithJob)
	}
	if podGroupInformer == nil {
		return fmt.Errorf("pod group informer is required when the feature gate %q is enabled", features.WorkloadWithJob)
	}
	jm.workloadLister = workloadInformer.Lister()
	jm.workloadStoreSynced = workloadInformer.Informer().HasSynced

	jm.podGroupLister = podGroupInformer.Lister()
	jm.podGroupStoreSynced = podGroupInformer.Informer().HasSynced

	return nil
}
