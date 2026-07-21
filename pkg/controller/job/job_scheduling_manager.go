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
	"fmt"
	"hash/fnv"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1alpha3"
	"k8s.io/client-go/tools/cache"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/klog/v2"
	jobutil "k8s.io/kubernetes/pkg/api/job"
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

const (
	// Workload.spec.controllerRef and PodGroup.spec.workloadRef
	// carry only names (no UIDs), so these indexes use namespace/name as
	// the key. Callers must still verify ownership via ownerRef UID checks
	// before acting on results.
	workloadByJobNameIndexKey      = "workloadByJobName"
	podGroupByWorkloadNameIndexKey = "podGroupByWorkloadName"
)

// managementMode describes which scheduling objects the Job controller
// manages for a given Job.
type managementMode int

const (
	// the controller manages no scheduling objects, either the feature
	// is off, the user brought their own PodGroup, or a parent controller
	// owns both objects without delegating.
	manageNone managementMode = iota

	// the controller owns both Workload and PodGroup.
	manageBoth

	// a non-root Job whose parent owns the Workload and delegated
	// runtime PodGroup management to the Job controller.
	managePodGroupOnly
)

// getManagementMode computes which scheduling objects the Job controller
// manages for the Job.
func getManagementMode(job *batch.Job) managementMode {
	if !feature.DefaultFeatureGate.Enabled(features.WorkloadWithJob) {
		return manageNone
	}
	if job.Spec.Template.Spec.SchedulingGroup != nil {
		return manageNone
	}
	// A Job with no controller owner is a root workload node.
	if metav1.GetControllerOf(job) == nil {
		return manageBoth
	}
	if hasDelegatedPodGroup(job) {
		return managePodGroupOnly
	}
	return manageNone
}

// shouldManageWorkloadForJob reports whether the Job controller should
// materialize scheduling objects (Workload and PodGroup) for the Job.
func shouldManageWorkloadForJob(job *batch.Job) bool {
	return getManagementMode(job) != manageNone
}

// hasDelegatedPodGroup reports whether a parent controller delegated runtime
// PodGroup management to the Job controller via the podGroupTemplate annotation.
func hasDelegatedPodGroup(job *batch.Job) bool {
	_, ok := job.Annotations[apischeduling.GroupTemplateNameAnnotation]
	return ok
}

// filterControlledByJob returns only the objects whose controller ownerRef
// points at the given Job, i.e. the ones this controller created and owns.
func filterControlledByJob[T metav1.Object](objs []T, job *batch.Job) []T {
	matched := make([]T, 0, len(objs))
	for _, obj := range objs {
		if metav1.IsControlledBy(obj, job) {
			matched = append(matched, obj)
		}
	}
	return matched
}

// ensureWorkloadAndPodGroup discovers or creates Workload and PodGroup for the given Job.
// Returns both objects when workload integration is active, or nils when the Job
// should fall back to default scheduling / defers ownership to a parent.
func (jm *Controller) ensureWorkloadAndPodGroup(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1alpha3.Workload, *schedulingv1alpha3.PodGroup, error) {
	mode := getManagementMode(job)
	if mode == manageNone {
		return nil, nil, nil
	}

	// Objects are only created for a Job that has never started, so a
	// Job that predates the feature gate being enabled is left untouched.
	// An already-started Job discovers and reuses existing objects
	// but never creates new ones.
	newJob := isNewJob(job, pods)

	// Case 1 - manage the PodGroup only.
	if mode == managePodGroupOnly {
		pg, err := jm.getOrCreateDelegatedPodGroup(ctx, job, newJob)
		if err != nil {
			return nil, nil, err
		}
		return nil, pg, nil
	}

	// Case 2 - manage both Workload and PodGroup (root Job).
	wl, err := jm.getOrCreateWorkload(ctx, job, newJob)
	if err != nil || wl == nil {
		return nil, nil, err
	}
	pg, err := jm.getOrCreatePodGroup(ctx, job, wl, newJob)
	if err != nil {
		return nil, nil, err
	}

	// propagate a changed gang minCount (elastic scaling) to the controller-owned
	// Workload/PodGroup. Runs regardless of pod state so a running gang Job can
	// be resized.
	err = jm.syncGangMinCount(ctx, job, wl, pg)
	if err != nil {
		return nil, nil, err
	}
	return wl, pg, nil
}

// syncGangMinCount propagates an updated gang size to the controller-owned
// Workload and runtime PodGroup. It recompiles the desired Workload from
// the current Job spec. When the resolved policy is Gang, it updates only
// the minCount on the live objects if it differs.
//
// Basic policies are a no-op. Only minCount is changed; every other
// Workload/PodGroup field is immutable at the API.
func (jm *Controller) syncGangMinCount(ctx context.Context, job *batch.Job,
	workload *schedulingv1alpha3.Workload, podGroup *schedulingv1alpha3.PodGroup) error {

	desiredWorkload, err := jm.generateWorkload(job)
	if err != nil || len(desiredWorkload.Spec.PodGroupTemplates) == 0 {
		return err
	}
	gang := desiredWorkload.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
	if gang == nil {
		// Basic (non-gang) policies have no minCount to reconcile, so this is a no-op.
		return nil
	}
	desiredMinCount := gang.MinCount

	currentPGTemplate := workload.Spec.PodGroupTemplates
	if len(currentPGTemplate) == 1 &&
		currentPGTemplate[0].SchedulingPolicy.Gang != nil &&
		currentPGTemplate[0].SchedulingPolicy.Gang.MinCount != desiredMinCount {

		err := jm.patchWorkloadMinCount(ctx, workload, desiredMinCount)
		if err != nil {
			return err
		}
	}

	if podGroup != nil &&
		podGroup.Spec.SchedulingPolicy.Gang != nil &&
		podGroup.Spec.SchedulingPolicy.Gang.MinCount != desiredMinCount {

		err := jm.patchPodGroupMinCount(ctx, podGroup, desiredMinCount)
		if err != nil {
			return err
		}
	}
	return nil
}

// patchWorkloadMinCount sets the gang minCount on the Workload's single
// PodGroupTemplate. minCount is the only mutable field on a compiled Workload.
func (jm *Controller) patchWorkloadMinCount(ctx context.Context, workload *schedulingv1alpha3.Workload, minCount int32) error {
	updatedW := workload.DeepCopy()
	updatedW.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = minCount

	oldData, err := json.Marshal(workload)
	if err != nil {
		return err
	}

	newData, err := json.Marshal(updatedW)
	if err != nil {
		return err
	}
	patch, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &schedulingv1alpha3.Workload{})
	if err != nil {
		return err
	}
	_, err = jm.kubeClient.SchedulingV1alpha3().Workloads(workload.Namespace).
		Patch(ctx, workload.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	return err
}

// patchPodGroupMinCount sets the gang minCount on the runtime PodGroup.
func (jm *Controller) patchPodGroupMinCount(ctx context.Context, podGroup *schedulingv1alpha3.PodGroup, minCount int32) error {
	updatedPG := podGroup.DeepCopy()
	updatedPG.Spec.SchedulingPolicy.Gang.MinCount = minCount

	oldData, err := json.Marshal(podGroup)
	if err != nil {
		return err
	}

	newData, err := json.Marshal(updatedPG)
	if err != nil {
		return err
	}

	patch, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &schedulingv1alpha3.PodGroup{})
	if err != nil {
		return err
	}
	_, err = jm.kubeClient.SchedulingV1alpha3().PodGroups(podGroup.Namespace).
		Patch(ctx, podGroup.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	return err
}

// findParentWorkload returns the parent-owned Workload for a non-root Job: the
// Workload in the Job's namespace whose spec.controllerRef points to the Job's
// parent controller.
func (jm *Controller) findParentWorkload(job *batch.Job) (*schedulingv1alpha3.Workload, error) {
	if jm.workloadLister == nil {
		return nil, fmt.Errorf("workload lister is not configured")
	}
	parent := metav1.GetControllerOf(job)
	if parent == nil {
		return nil, nil
	}
	gv, err := schema.ParseGroupVersion(parent.APIVersion)
	if err != nil {
		return nil, nil
	}
	workloads, err := jm.workloadLister.Workloads(job.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	for _, wl := range workloads {
		// metav1.IsControlledBy can't be used here: it matches an object's
		// metadata.ownerReferences by UID, but spec.ControllerRef is a name-based
		// TypedLocalObjectReference that points at the Job's parent controller
		// rather than at the Job itself.
		if ref := wl.Spec.ControllerRef; ref != nil &&
			ref.APIGroup == gv.Group && ref.Kind == parent.Kind && ref.Name == parent.Name {
			return wl, nil
		}
	}
	return nil, nil
}

// getOrCreateDelegatedPodGroup discovers or creates the runtime PodGroup for a
// non-root Job whose parent delegates PodGroup management. The PodGroup maps to
// the parent-owned Workload via the named PodGroupTemplate from the delegation
// annotation and carries only a controller ownerRef to the Job.
func (jm *Controller) getOrCreateDelegatedPodGroup(ctx context.Context,
	job *batch.Job, newJob bool) (*schedulingv1alpha3.PodGroup, error) {
	logger := klog.FromContext(ctx)

	parentWorkload, err := jm.findParentWorkload(job)
	if err != nil {
		return nil, err
	}
	if parentWorkload == nil {
		logger.V(2).Info("Parent-owned Workload not found yet for delegated Job, will retry",
			"job", klog.KObj(job))
		return nil, nil
	}

	templateName := job.Annotations[apischeduling.GroupTemplateNameAnnotation]
	template, err := getPodGroupTemplateByName(parentWorkload, templateName)
	if err != nil {
		logger.V(2).Info("Delegated PodGroupTemplate not found in parent Workload, skipping",
			"job", klog.KObj(job), "workload", klog.KObj(parentWorkload), "template", templateName)
		return nil, nil
	}

	// The parent owns and compiled the Workload, so materialize the PodGroup
	// from its persisted template rather than recompiling from this Job's spec.
	builder := newBuilderFromExistingWorkload(job, parentWorkload)

	matched, err := jm.listPodGroupsForWorkload(parentWorkload)
	if err != nil {
		return nil, err
	}
	var updatedPG *schedulingv1alpha3.PodGroup
	for _, pg := range matched {
		if metav1.IsControlledBy(pg, job) {
			updatedPG = pg
			break
		}
	}
	// Create only when the Job has no pods yet and none exists; otherwise
	// discover-only. A create that races with another actor returns
	// AlreadyExists, which requeues the Job so the next sync discovers it.
	if updatedPG == nil && newJob {
		// A delegated Job maps to a parent-owned Workload, so the PodGroup does
		// not link the Workload as an owner (linkWorkloadOwner=false).
		updatedPG, err = jm.createPodGroup(ctx, job, parentWorkload, templateName, false, builder)
		if err != nil {
			return nil, err
		}
	}
	// The parent controller may resize the gang on its Workload's template
	// after the PodGroup exists; the controller owns PodGroup management in
	// the delegated mode, so follow the template.
	if err := jm.syncDelegatedPodGroupMinCount(ctx, template, updatedPG); err != nil {
		return nil, err
	}
	return updatedPG, nil
}

// syncDelegatedPodGroupMinCount propagates a gang minCount change made by the
// parent controller on its Workload's PodGroupTemplate to the delegated
// runtime PodGroup. The parent's template is the source of truth in the
// delegated mode.
func (jm *Controller) syncDelegatedPodGroupMinCount(ctx context.Context,
	template *schedulingv1alpha3.PodGroupTemplate, podGroup *schedulingv1alpha3.PodGroup) error {
	if podGroup == nil ||
		template.SchedulingPolicy.Gang == nil || podGroup.Spec.SchedulingPolicy.Gang == nil {
		return nil
	}
	desiredMinCount := template.SchedulingPolicy.Gang.MinCount
	if podGroup.Spec.SchedulingPolicy.Gang.MinCount == desiredMinCount {
		return nil
	}
	return jm.patchPodGroupMinCount(ctx, podGroup, desiredMinCount)
}

// isNewJob returns true if the Job has never started. This prevents creating
// scheduling objects for Jobs that predate the feature gate being enabled.
//
// This heuristic is intentionally conservative for alpha and will be replaced
// by a more robust mechanism before beta.
//
// We check pods, StartTime, terminal counters, and the Suspended condition
// because each alone has gaps, pods may be GC'd, StartTime is reset on
// suspend, but Succeeded/Failed persist. A Job that was suspended early
// would pass the other checks, so the Suspended condition catches that case.
func isNewJob(job *batch.Job, pods []*v1.Pod) bool {
	if len(pods) != 0 ||
		job.Status.StartTime != nil ||
		job.Status.Succeeded != 0 ||
		job.Status.Failed != 0 {
		return false
	}
	return findConditionByType(job.Status.Conditions, batch.JobSuspended) == nil
}

// getOrCreateWorkload returns the Workload for this Job, creating one if needed.
// Returns (nil, nil) when the Job should fall back to default scheduling
// (i.e., ambiguous matches, unsupported structure, or
// not a new Job).
func (jm *Controller) getOrCreateWorkload(ctx context.Context,
	job *batch.Job, newJob bool) (*schedulingv1alpha3.Workload, error) {
	allWorkloads, err := jm.listWorkloadsForJob(job)
	if err != nil {
		return nil, err
	}

	// Only Workloads this controller created are managed, identified by a
	// controller ownerRef to the Job. User pre-created (BYO) Workloads are
	// ignored so the controller always owns the objects it schedules against.
	matched := filterControlledByJob(allWorkloads, job)

	switch len(matched) {
	case 0:
		// Create only when the Job has no pods yet; otherwise discover-only.
		// A create that races with another actor returns AlreadyExists, which
		// requeues the Job so the next sync discovers the existing Workload.
		if !newJob {
			return nil, nil
		}
		return jm.createWorkloadForJob(ctx, job)
	case 1:
		if err := jm.validateWorkloadStructure(job, matched[0]); err != nil {
			return nil, err
		}
		return matched[0], nil
	default:
		utilruntime.HandleError(fmt.Errorf("found %d Workloads referencing Job %s/%s, falling back to default scheduling",
			len(matched), job.Namespace, job.Name))
		return nil, nil
	}
}

// validateWorkloadStructure checks alpha constraints on a controller-owned
// Workload. It returns an error when the structure is unsupported so the Job is
// requeued and the problem is surfaced, rather than silently degrading to
// default scheduling.
//
// TODO(beta): This only validates the initial shape. A user (or another
// controller) could mutate the Workload after creation, breaking our
// assumptions (e.g., adding/removing PodGroupTemplates, changing the
// scheduling policy). Before beta we need to decide how to handle such
// mutations; overwrite them back to the expected state, fall back to
// default scheduling, or reject the update via admission.
func (jm *Controller) validateWorkloadStructure(job *batch.Job, wl *schedulingv1alpha3.Workload) error {
	if len(wl.Spec.PodGroupTemplates) != 1 {
		return fmt.Errorf("workload %s/%s for job %s/%s has %d PodGroupTemplates, want exactly 1",
			wl.Namespace, wl.Name, job.Namespace, job.Name, len(wl.Spec.PodGroupTemplates))
	}
	return nil
}

// getOrCreatePodGroup returns the PodGroup for this Workload, creating one if needed.
// Returns (nil, nil) when the Job should fall back to default scheduling
// (ambiguous matches or not owned by this controller).
func (jm *Controller) getOrCreatePodGroup(ctx context.Context, job *batch.Job,
	workload *schedulingv1alpha3.Workload, newJob bool) (*schedulingv1alpha3.PodGroup, error) {
	builder := newBuilderFromExistingWorkload(job, workload)

	allPodGroups, err := jm.listPodGroupsForWorkload(workload)
	if err != nil {
		return nil, err
	}
	// Only PodGroups this controller created are managed, identified by a
	// controller ownerRef to the Job. User pre-created (BYO) PodGroups are
	// ignored so the controller always owns the objects it schedules against.
	matched := filterControlledByJob(allPodGroups, job)

	switch len(matched) {
	case 0:
		// Create only when the Job has no pods yet; otherwise discover-only.
		// This also completes a partial set (Workload exists, PodGroup missing)
		// left by a mid-flow restart, but only before pods exist.
		if !newJob {
			return nil, nil
		}
		// A root Job owns its Workload, so the PodGroup links it as a
		// non-controller owner (linkWorkloadOwner=true). An empty template
		// name selects the Workload's single template.
		return jm.createPodGroup(ctx, job, workload, "", true, builder)
	case 1:
		return matched[0], nil
	default:
		utilruntime.HandleError(fmt.Errorf("found %d PodGroups referencing Workload %s/%s, falling back to default scheduling",
			len(matched), workload.Namespace, workload.Name))
		return nil, nil
	}
}

// listWorkloadsForJob returns all Workloads in the Job's namespace whose
// spec.controllerRef points to the given Job.
func (jm *Controller) listWorkloadsForJob(job *batch.Job) ([]*schedulingv1alpha3.Workload, error) {
	if jm.workloadIndexer == nil {
		return nil, fmt.Errorf("workload indexer is not configured")
	}
	key := namespacedNameIndexKey(job.Namespace, job.Name)
	all, err := jm.workloadIndexer.ByIndex(workloadByJobNameIndexKey, key)
	if err != nil {
		return nil, err
	}
	matched := make([]*schedulingv1alpha3.Workload, 0, len(all))
	for _, wl := range all {
		workload, ok := wl.(*schedulingv1alpha3.Workload)
		if !ok {
			continue
		}
		matched = append(matched, workload)
	}
	return matched, nil
}

// listPodGroupsForWorkload returns all PodGroups in the Workload's namespace
// whose spec.workloadRef references the given Workload.
func (jm *Controller) listPodGroupsForWorkload(workload *schedulingv1alpha3.Workload) ([]*schedulingv1alpha3.PodGroup, error) {
	if jm.podGroupIndexer == nil {
		return nil, fmt.Errorf("podgroup indexer is not configured")
	}
	key := namespacedNameIndexKey(workload.Namespace, workload.Name)
	all, err := jm.podGroupIndexer.ByIndex(podGroupByWorkloadNameIndexKey, key)
	if err != nil {
		return nil, err
	}
	matched := make([]*schedulingv1alpha3.PodGroup, 0, len(all))
	for _, pg := range all {
		podGroup, ok := pg.(*schedulingv1alpha3.PodGroup)
		if !ok {
			continue
		}
		matched = append(matched, podGroup)
	}
	return matched, nil
}

// podGroupTemplateName returns the name of the single PodGroupTemplate the
// controller compiles for a Job. For alpha there is exactly one template per Job.
func podGroupTemplateName(job *batch.Job) string {
	return fmt.Sprintf("%s-pgt-%d", job.Name, 0)
}

// buildWorkloadItem assembles the single-node logical workload tree for a Job.
// The Job defaults to Basic scheduling; the user's spec.scheduling overrides it.
func buildWorkloadItem(job *batch.Job) *workloadbuilder.WorkloadItem {
	var input workloadbuilder.WorkloadInput
	if s := job.Spec.Scheduling; s != nil {
		input = jobutil.WorkloadInputForJobV1(s)
	}
	return jobutil.WorkloadItemForJob(podGroupTemplateName(job),
		job.Spec.Template.Spec.PriorityClassName, job.Spec.Parallelism, input)
}

// generateWorkload compiles the Job's spec.scheduling into a Workload via the
// shared workloadbuilder library. BuildWorkload also sets the controller
// ownerReference and spec.controllerRef pointing at the Job.
func (jm *Controller) generateWorkload(job *batch.Job) (*schedulingv1alpha3.Workload, error) {
	return workloadbuilder.NewBuilder(buildWorkloadItem(job), workloadbuilder.BuildOptions{
		Name:      computeWorkloadName(job),
		Namespace: job.Namespace,
		Owner:     metav1.NewControllerRef(job, controllerKind),
	}).BuildWorkload()
}

// createWorkloadForJob compiles and creates a new Workload object for the given Job.
func (jm *Controller) createWorkloadForJob(ctx context.Context, job *batch.Job) (*schedulingv1alpha3.Workload, error) {
	workload, err := jm.generateWorkload(job)
	if err != nil {
		jm.recorder.Eventf(job, v1.EventTypeWarning, "FailedWorkloadCompilation",
			"Failed to compile Workload for Job %s: %v", job.Name, err)
		return nil, err
	}

	created, err := jm.kubeClient.SchedulingV1alpha3().Workloads(job.Namespace).Create(ctx, workload, metav1.CreateOptions{})
	if err != nil {
		jm.recorder.Eventf(job, v1.EventTypeWarning, "FailedWorkloadCreate",
			"Failed to create Workload for Job %s: %v", job.Name, err)
		return nil, err
	}

	jm.recorder.Eventf(job, v1.EventTypeNormal, "WorkloadCreated",
		"Created Workload %s for Job %s", created.Name, job.Name)
	return created, nil
}

// getPodGroupTemplateByName returns the named PodGroupTemplate from the Workload.
// An empty name selects the Workload's single template.
func getPodGroupTemplateByName(workload *schedulingv1alpha3.Workload, name string) (*schedulingv1alpha3.PodGroupTemplate, error) {
	if len(workload.Spec.PodGroupTemplates) == 0 {
		return nil, fmt.Errorf("workload %s/%s has no PodGroupTemplates", workload.Namespace, workload.Name)
	}
	if name == "" {
		return &workload.Spec.PodGroupTemplates[0], nil
	}
	for i := range workload.Spec.PodGroupTemplates {
		if workload.Spec.PodGroupTemplates[i].Name == name {
			return &workload.Spec.PodGroupTemplates[i], nil
		}
	}
	return nil, fmt.Errorf("PodGroupTemplate %q not found in workload %s/%s", name, workload.Namespace, workload.Name)
}

// newBuilderFromExistingWorkload returns a builder that materializes PodGroups from
// an already-persisted Workload. A delegated Job's Workload is owned and
// compiled by a parent, so its persisted template is the source of truth and the
// Job's own spec.scheduling is irrelevant here. Only the Job controllerRef and
// the supplied Workload are needed.
func newBuilderFromExistingWorkload(job *batch.Job, workload *schedulingv1alpha3.Workload) *workloadbuilder.Builder {
	return workloadbuilder.NewBuilderFromExistingWorkload(workload, workloadbuilder.BuildOptions{
		Owner: metav1.NewControllerRef(job, controllerKind),
	})
}

// createPodGroup creates a runtime PodGroup for the Job from the Workload's
// PodGroupTemplate named templateName. It always carries a controller ownerRef
// to the Job. When linkWorkloadOwner is true, it gets a non-controller ownerRef
// to the Workload; a delegated (parent-owned) Workload gets none, leaving the
// parent as the Workload's sole owner. BlockOwnerDeletion is left unset so
// Workload deletion is never blocked by surviving PodGroups.
func (jm *Controller) createPodGroup(ctx context.Context, job *batch.Job, workload *schedulingv1alpha3.Workload,
	templateName string, linkWorkloadOwner bool, builder *workloadbuilder.Builder) (*schedulingv1alpha3.PodGroup, error) {
	template, err := getPodGroupTemplateByName(workload, templateName)
	if err != nil {
		return nil, err
	}

	podGroup, err := builder.NewPodGroup(computePodGroupName(workload.Name, template.Name), template.Name)
	if err != nil {
		return nil, err
	}
	if linkWorkloadOwner {
		podGroup.OwnerReferences = append(podGroup.OwnerReferences, metav1.OwnerReference{
			APIVersion: schedulingv1alpha3.SchemeGroupVersion.String(),
			Kind:       "Workload",
			Name:       workload.Name,
			UID:        workload.UID,
		})
	}

	created, err := jm.kubeClient.SchedulingV1alpha3().PodGroups(job.Namespace).Create(ctx, podGroup, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}

	jm.recorder.Eventf(job, v1.EventTypeNormal, "PodGroupCreated",
		"Created PodGroup %s for Job %s", created.Name, job.Name)
	return created, nil
}

// computeWorkloadName generates a deterministic name for a Workload associated with a Job.
// The pattern is: <truncated-job-name>-<hash>
// The hash is derived from the Job UID to ensure uniqueness and stability across restarts.
func computeWorkloadName(job *batch.Job) string {
	hasher := fnv.New32a()
	hashutil.DeepHashObject(hasher, job.UID)
	hash := rand.SafeEncodeString(fmt.Sprint(hasher.Sum32()))

	// hash is ~10 chars; include separator "-"
	maxPrefixLen := validation.DNS1123SubdomainMaxLength - len(hash) - 1
	prefix := job.Name
	if len(prefix) > maxPrefixLen {
		prefix = prefix[:maxPrefixLen]
	}
	return fmt.Sprintf("%s-%s", prefix, hash)
}

// computePodGroupName generates a deterministic name for a PodGroup associated with a Workload.
// The pattern is: <truncated-workload-name>-<truncated-template-name>-<hash>
// The hash is derived from the workload name and template name for uniqueness.
func computePodGroupName(workloadName, templateName string) string {
	hasher := fnv.New32a()
	// hash the full combination for uniqueness.
	hasher.Write([]byte(workloadName))
	hasher.Write([]byte(templateName))
	hash := rand.SafeEncodeString(fmt.Sprint(hasher.Sum32()))

	// Truncate workloadName and templateName to fit within the DNS subdomain
	// max length, reserving space for the hash and two "-" separators.
	maxAvailable := validation.DNS1123SubdomainMaxLength - len(hash) - 2
	wl := workloadName
	tpl := templateName

	if len(wl)+len(tpl) > maxAvailable {
		// only truncate the part that exceeds its fair share.
		// give each part up to half the space; if one is short, the other
		// gets the remainder.
		half := maxAvailable / 2
		switch {
		case len(wl) <= half:
			// workload name fits in its half, give template the rest.
			tpl = tpl[:maxAvailable-len(wl)]
		case len(tpl) <= half:
			// template name fits in its half, give workload the rest.
			wl = wl[:maxAvailable-len(tpl)]
		default:
			// both exceed half, split evenly (workload gets the extra char if odd).
			wl = wl[:maxAvailable-half]
			tpl = tpl[:half]
		}
	}

	return fmt.Sprintf("%s-%s-%s", wl, tpl, hash)
}

// addSchedulingInformers wires up Workload and PodGroup informers
// and indexers so that changes to those objects re-enqueue the owning Job.
func (jm *Controller) addSchedulingInformers(logger klog.Logger, workloadInformer schedulinginformers.TypedWorkloadInformer, podGroupInformer schedulinginformers.TypedPodGroupInformer) error {
	if workloadInformer == nil {
		return fmt.Errorf("workload informer is required when the feature gate %q is enabled", features.WorkloadWithJob)
	}
	if podGroupInformer == nil {
		return fmt.Errorf("pod group informer is required when the feature gate %q is enabled", features.WorkloadWithJob)
	}
	if err := workloadInformer.Informer().AddIndexers(cache.Indexers{
		workloadByJobNameIndexKey: workloadByJobNameIndexFunc,
	}); err != nil {
		return fmt.Errorf("adding Workload indexer: %w", err)
	}
	if err := podGroupInformer.Informer().AddIndexers(cache.Indexers{
		podGroupByWorkloadNameIndexKey: podGroupByWorkloadNameIndexFunc,
	}); err != nil {
		return fmt.Errorf("adding PodGroup indexer: %w", err)
	}
	jm.workloadLister = workloadInformer.Lister()
	jm.workloadIndexer = workloadInformer.Informer().GetIndexer()
	jm.workloadStoreSynced = workloadInformer.Informer().HasSynced

	jm.podGroupLister = podGroupInformer.Lister()
	jm.podGroupIndexer = podGroupInformer.Informer().GetIndexer()
	jm.podGroupStoreSynced = podGroupInformer.Informer().HasSynced

	return nil
}

func namespacedNameIndexKey(namespace, name string) string {
	return cache.NewObjectName(namespace, name).String()
}

// workloadControllerRefJobName returns the Job name from the Workload's
// spec.controllerRef if it points to a batch Job, or "" otherwise.
func workloadControllerRefJobName(wl *schedulingv1alpha3.Workload) string {
	if wl.Spec.ControllerRef == nil ||
		wl.Spec.ControllerRef.APIGroup != batch.SchemeGroupVersion.Group ||
		wl.Spec.ControllerRef.Kind != "Job" ||
		wl.Spec.ControllerRef.Name == "" {
		return ""
	}
	return wl.Spec.ControllerRef.Name
}

// podGroupWorkloadName returns the workload name from the PodGroup's
// spec.workloadRef, or "" if the reference is not set.
func podGroupWorkloadName(pg *schedulingv1alpha3.PodGroup) string {
	if pg.Spec.WorkloadRef == nil ||
		pg.Spec.WorkloadRef.WorkloadName == "" {
		return ""
	}
	return pg.Spec.WorkloadRef.WorkloadName
}

func workloadByJobNameIndexFunc(obj interface{}) ([]string, error) {
	wl, ok := obj.(*schedulingv1alpha3.Workload)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T", obj)
	}
	if name := workloadControllerRefJobName(wl); name != "" {
		return []string{namespacedNameIndexKey(wl.Namespace, name)}, nil
	}
	return nil, nil
}

func podGroupByWorkloadNameIndexFunc(obj interface{}) ([]string, error) {
	pg, ok := obj.(*schedulingv1alpha3.PodGroup)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T", obj)
	}
	if name := podGroupWorkloadName(pg); name != "" {
		return []string{namespacedNameIndexKey(pg.Namespace, name)}, nil
	}
	return nil, nil
}
