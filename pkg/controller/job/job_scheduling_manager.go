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
	"fmt"
	"hash/fnv"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1alpha3"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/klog/v2"
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/utils/ptr"
)

const (
	// Workload.spec.controllerRef and PodGroup.spec.workloadRef
	// carry only names (no UIDs), so these indexes use namespace/name as
	// the key. Callers must still verify ownership via ownerRef UID checks
	// before acting on results.
	workloadByJobNameIndexKey      = "workloadByJobName"
	podGroupByWorkloadNameIndexKey = "podGroupByWorkloadName"
)

// addWorkloadManagedByAnnotation marks the object's metadata as managed by the
// built-in Job controller, so controller-created Workload/PodGroup objects can
// be told apart from user-provided (BYO) ones that may carry identical
// ownerReferences before the controller updates them or relies on GC.
func addWorkloadManagedByAnnotation(meta *metav1.ObjectMeta) {
	if meta.Annotations == nil {
		meta.Annotations = make(map[string]string, 1)
	}
	meta.Annotations[batch.JobWorkloadManagedByAnnotation] = batch.JobControllerName
}

// isManagedByJobController reports whether the object was created by the Job controller.
func isManagedByJobController(obj metav1.Object) bool {
	return obj.GetAnnotations()[batch.JobWorkloadManagedByAnnotation] == batch.JobControllerName
}

// shouldManageWorkloadForJob reports whether the Job controller should
// materialize scheduling objects for the Job. With the WorkloadWithJob gate
// enabled, every eligible Job gets a Workload/PodGroup, defaulting to Basic
// when spec.scheduling is omitted. The only Job skipped entirely is one
// whose parent controller owns the Workload and does not delegate the PodGroup.
func shouldManageWorkloadForJob(job *batch.Job) bool {
	if !feature.DefaultFeatureGate.Enabled(features.WorkloadWithJob) {
		return false
	}
	// the user wired pods to their own PodGroup (BYO) via the pod
	// template (spec.template.spec.schedulingGroup), so the controller
	// does not manage any scheduling objects for this Job.
	if job.Spec.Template.Spec.SchedulingGroup != nil {
		return false
	}
	// Parent owns the Workload and does not delegate the PodGroup: manage neither.
	if hasParentWorkloadOwner(job) && !hasDelegatedPodGroup(job) {
		return false
	}
	return true
}

// hasParentWorkloadOwner reports whether the Job is a non-root workload node,
// i.e. it carries a controller OwnerReference to a parent controller that
// compiles and owns the Workload (e.g. JobSet). Jobs created by a CronJob are
// standalone roots (CronJob stamps out independent Jobs rather than owning a
// shared Workload) and are not treated as parent-owned.
func hasParentWorkloadOwner(job *batch.Job) bool {
	ref := metav1.GetControllerOf(job)
	if ref == nil {
		return false
	}
	gv, err := schema.ParseGroupVersion(ref.APIVersion)
	if err != nil {
		return false
	}
	return gv.Group != batch.SchemeGroupVersion.Group || ref.Kind != "CronJob"
}

// hasDelegatedPodGroup reports whether a parent controller delegated runtime
// PodGroup management to the Job controller via the podGroupTemplate annotation.
func hasDelegatedPodGroup(job *batch.Job) bool {
	_, ok := job.Annotations[apischeduling.GroupTemplateNameAnnotation]
	return ok
}

// ensureWorkloadAndPodGroup discovers or creates Workload and PodGroup for the given Job.
// Returns both objects when workload integration is active, or nils when the Job
// should fall back to default scheduling / defers ownership to a parent.
//
// Routing:
//   - Root Job: the controller owns both the Workload and the PodGroup.
//   - Non-root Job with a delegated PodGroup: the parent owns the Workload; the
//     controller creates only the runtime PodGroup mapped to the parent's template.
//   - Non-root Job without delegation: the parent owns both; the controller
//     creates neither (already filtered out by shouldManageWorkloadForJob).
func (jm *Controller) ensureWorkloadAndPodGroup(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1alpha3.Workload, *schedulingv1alpha3.PodGroup, error) {
	// Objects are only created for a Job that has never started, so a
	// Job that predates the feature gate being enabled is left untouched.
	// An already-started Job discovers and reuses existing objects
	// but never creates new ones.
	newJob := isNewJob(job, pods)

	// A non-root Job's parent owns the Workload. Without a delegated PodGroup
	// the controller manages nothing.
	if hasParentWorkloadOwner(job) {
		if !hasDelegatedPodGroup(job) {
			return nil, nil, nil
		}
		pg, err := jm.getOrCreateDelegatedPodGroup(ctx, job, newJob)
		if err != nil {
			return nil, nil, err
		}
		return nil, pg, nil
	}

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
	if err := jm.syncGangMinCount(ctx, job, wl, pg); err != nil {
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
	// Only mutate objects the controller created, identified by a controller
	// ownerRef to the Job and the managed-by annotation. BYO Workload/PodGroup
	// is never updated even on a minCount change.
	if workload == nil || !metav1.IsControlledBy(workload, job) ||
		!isManagedByJobController(workload) {
		return nil
	}
	desired, err := jm.generateWorkload(job)
	if err != nil || len(desired.Spec.PodGroupTemplates) == 0 {
		return nil
	}
	gang := desired.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
	if gang == nil {
		return nil
	}
	desiredMinCount := gang.MinCount

	if cur := workload.Spec.PodGroupTemplates; len(cur) == 1 &&
		cur[0].SchedulingPolicy.Gang != nil && cur[0].SchedulingPolicy.Gang.MinCount != desiredMinCount {
		if err := jm.updateWorkloadMinCount(ctx, workload.Namespace, workload.Name, desiredMinCount); err != nil {
			return err
		}
	}
	if podGroup != nil && metav1.IsControlledBy(podGroup, job) &&
		isManagedByJobController(podGroup) &&
		podGroup.Spec.SchedulingPolicy.Gang != nil && podGroup.Spec.SchedulingPolicy.Gang.MinCount != desiredMinCount {
		if err := jm.updatePodGroupMinCount(ctx, podGroup.Namespace, podGroup.Name, desiredMinCount); err != nil {
			return err
		}
	}
	return nil
}

// updateWorkloadMinCount sets the gang minCount on the Workload's single
// PodGroupTemplate.
func (jm *Controller) updateWorkloadMinCount(ctx context.Context, namespace, name string, minCount int32) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		cur, err := jm.kubeClient.SchedulingV1alpha3().Workloads(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if len(cur.Spec.PodGroupTemplates) != 1 {
			return nil
		}
		gang := cur.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang
		if gang == nil || gang.MinCount == minCount {
			return nil
		}
		gang.MinCount = minCount
		_, err = jm.kubeClient.SchedulingV1alpha3().Workloads(namespace).Update(ctx, cur, metav1.UpdateOptions{})
		return err
	})
}

// updatePodGroupMinCount sets the gang minCount on the runtime PodGroup.
func (jm *Controller) updatePodGroupMinCount(ctx context.Context, namespace, name string, minCount int32) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		cur, err := jm.kubeClient.SchedulingV1alpha3().PodGroups(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		gang := cur.Spec.SchedulingPolicy.Gang
		if gang == nil || gang.MinCount == minCount {
			return nil
		}
		gang.MinCount = minCount
		_, err = jm.kubeClient.SchedulingV1alpha3().PodGroups(namespace).Update(ctx, cur, metav1.UpdateOptions{})
		return err
	})
}

// findParentWorkload returns the parent-owned Workload for a non-root Job: the
// Workload in the Job's namespace whose spec.controllerRef points to the Job's
// parent controller. Returns (nil, nil) when it cannot be found yet.
func (jm *Controller) findParentWorkload(job *batch.Job) (*schedulingv1alpha3.Workload, error) {
	parent := metav1.GetControllerOf(job)
	if parent == nil {
		return nil, nil
	}
	gv, err := schema.ParseGroupVersion(parent.APIVersion)
	if err != nil {
		return nil, nil
	}
	if jm.workloadLister == nil {
		return nil, fmt.Errorf("workload lister is not configured")
	}
	workloads, err := jm.workloadLister.Workloads(job.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	for _, wl := range workloads {
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
	if _, err := getPodGroupTemplateByName(parentWorkload, templateName); err != nil {
		logger.V(2).Info("Delegated PodGroupTemplate not found in parent Workload, skipping",
			"job", klog.KObj(job), "workload", klog.KObj(parentWorkload), "template", templateName)
		return nil, nil
	}

	var result *schedulingv1alpha3.PodGroup
	err = retry.OnError(retry.DefaultRetry, apierrors.IsAlreadyExists, func() error {
		matched, err := jm.listPodGroupsForWorkload(parentWorkload)
		if err != nil {
			logger.V(2).Info("Failed to list PodGroups for delegated Job, will retry",
				"job", klog.KObj(job), "workload", klog.KObj(parentWorkload), "err", err)
			return nil
		}
		for _, pg := range matched {
			if metav1.IsControlledBy(pg, job) {
				result = pg
				return nil
			}
		}
		// Create only when the Job has no pods yet; otherwise discover-only.
		if !newJob {
			return nil
		}
		// A delegated Job maps to a parent-owned Workload, so the PodGroup does
		// not link the Workload as an owner (linkWorkloadOwner=false).
		pg, err := jm.createPodGroup(ctx, job, parentWorkload, templateName, false)
		if err != nil {
			return err
		}
		result = pg
		return nil
	})
	return result, err
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
	logger := klog.FromContext(ctx)

	var result *schedulingv1alpha3.Workload
	err := retry.OnError(retry.DefaultRetry, apierrors.IsAlreadyExists, func() error {
		matched, err := jm.listWorkloadsForJob(job)
		if err != nil {
			logger.V(2).Info("Failed to list Workloads, falling back to default scheduling",
				"job", klog.KObj(job), "err", err)
			return nil
		}
		// Adopt any Workload referencing this Job, whether the controller created
		// it or a user pre-created it (BYO), e.g. a higher-level controller that
		// provisions the Workload for quota/admission gating before the Job
		// exists. Ownership (controller ownerRef / managed-by) governs only
		// update and GC, not whether the Workload is adopted here.

		switch len(matched) {
		case 0:
			// Create only when the Job has no pods yet; otherwise discover-only.
			if !newJob {
				return nil
			}
			wl, err := jm.createWorkloadForJob(ctx, job)
			if err != nil {
				return err
			}
			result = wl
			return nil
		case 1:
			result, err = jm.validateWorkloadStructure(logger, job, matched[0])
			return err
		default:
			utilruntime.HandleError(fmt.Errorf("found %d Workloads referencing Job %s/%s, falling back to default scheduling",
				len(matched), job.Namespace, job.Name))
			return nil
		}
	})
	return result, err
}

// validateWorkloadStructure checks alpha constraints on the Workload.
// Returns (nil, nil) if the structure is unsupported.
//
// TODO(beta): This only validates the initial shape. A user (or another
// controller) could mutate the Workload after creation, breaking our
// assumptions (e.g., adding/removing PodGroupTemplates, changing the
// scheduling policy). Before beta we need to decide how to handle such
// mutations; overwrite them back to the expected state, fall back to
// default scheduling, or reject the update via admission.
func (jm *Controller) validateWorkloadStructure(logger klog.Logger, job *batch.Job,
	wl *schedulingv1alpha3.Workload) (*schedulingv1alpha3.Workload, error) {
	// A BYO Workload may carry multiple templates; the Job disambiguates which
	// one to bind by naming it via the podGroupTemplate annotation. When the
	// annotation is set we only require that named template to exist. Without
	// it, the controller can only bind an unambiguous single-template Workload.
	if name, ok := job.Annotations[apischeduling.GroupTemplateNameAnnotation]; ok {
		if _, err := getPodGroupTemplateByName(wl, name); err != nil {
			logger.V(2).Info("Named PodGroupTemplate not found in Workload, falling back to default scheduling",
				"job", klog.KObj(job), "workload", klog.KObj(wl), "template", name)
			return nil, nil
		}
		return wl, nil
	}
	if len(wl.Spec.PodGroupTemplates) != 1 {
		logger.V(2).Info("Workload has unsupported number of PodGroupTemplates and no podGroupTemplate annotation, falling back to default scheduling",
			"job", klog.KObj(job), "workload", klog.KObj(wl), "count", len(wl.Spec.PodGroupTemplates))
		return nil, nil
	}
	return wl, nil
}

// getOrCreatePodGroup returns the PodGroup for this Workload, creating one if needed.
// Returns (nil, nil) when the Job should fall back to default scheduling
// (ambiguous matches or not owned by this controller).
func (jm *Controller) getOrCreatePodGroup(ctx context.Context, job *batch.Job,
	workload *schedulingv1alpha3.Workload, newJob bool) (*schedulingv1alpha3.PodGroup, error) {
	logger := klog.FromContext(ctx)

	var result *schedulingv1alpha3.PodGroup
	err := retry.OnError(retry.DefaultRetry, apierrors.IsAlreadyExists, func() error {
		matched, err := jm.listPodGroupsForWorkload(workload)
		if err != nil {
			logger.V(2).Info("Failed to list PodGroups, falling back to default scheduling",
				"job", klog.KObj(job), "workload", klog.KObj(workload), "err", err)
			return nil
		}
		// Adopt any PodGroup referencing this Workload, whether controller-created
		// or BYO. Ownership only governs update/GC, not adoption.

		switch len(matched) {
		case 0:
			// Create only when the Job has no pods yet; otherwise discover-only.
			// This also completes a partial set (Workload exists, PodGroup
			// missing) left by a mid-flow restart, but only before pods exist.
			if !newJob {
				return nil
			}
			// A root Job owns its Workload, so the PodGroup links it as a
			// non-controller owner (linkWorkloadOwner=true). The Job may name a
			// specific template (BYO multi-template Workload) via the
			// podGroupTemplate annotation; an empty name selects the Workload's
			// single template.
			templateName := job.Annotations[apischeduling.GroupTemplateNameAnnotation]
			pg, err := jm.createPodGroup(ctx, job, workload, templateName, true)
			if err != nil {
				return err
			}
			result = pg
			return nil
		case 1:
			result = matched[0]
			return nil
		default:
			utilruntime.HandleError(fmt.Errorf("found %d PodGroups referencing Workload %s/%s, falling back to default scheduling",
				len(matched), workload.Namespace, workload.Name))
			return nil
		}
	})
	return result, err
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

// mapSchedulingConfig translates the Job's user-facing spec.scheduling block
// into the workloadbuilder intermediate representation. It returns nil for a
// nil block so the builder falls back to the controller's default config.
func mapSchedulingConfig(cfg *batch.JobSchedulingConfiguration) *workloadbuilder.SchedulingConfig {
	if cfg == nil {
		return nil
	}
	return workloadbuilder.MapPodGroupConfig(cfg.Policy, cfg.Constraints, cfg.DisruptionMode, cfg.ResourceClaims)
}

// defaultMinCountForJob returns a callback that defaults an unset gang MinCount
// to the Job's parallelism. It mutates only the resolved config, never writing
// the derived value back onto the Job's spec.
func defaultMinCountForJob(job *batch.Job) workloadbuilder.SchedulingConfigFunc {
	return func(cfg *workloadbuilder.SchedulingConfig) {
		if cfg == nil || cfg.Policy == nil || cfg.Policy.Gang == nil {
			return
		}
		if cfg.Policy.Gang.MinCount == nil {
			cfg.Policy.Gang.MinCount = new(ptr.Deref(job.Spec.Parallelism, 1))
		}
	}
}

// buildWorkloadItem assembles the single-node logical workload tree for a Job.
// The Job defaults to Basic scheduling; the user's spec.scheduling overrides it.
func buildWorkloadItem(job *batch.Job) *workloadbuilder.WorkloadItem {
	return &workloadbuilder.WorkloadItem{
		Name: podGroupTemplateName(job),
		DefaultConfig: &workloadbuilder.SchedulingConfig{
			Policy: &workloadbuilder.SchedulingPolicy{Basic: &workloadbuilder.BasicSchedulingPolicy{}},
		},
		UserConfig: mapSchedulingConfig(job.Spec.Scheduling),
		Callbacks:  []workloadbuilder.SchedulingConfigFunc{defaultMinCountForJob(job)},
	}
}

// generateWorkload compiles the Job's spec.scheduling into a Workload via the
// shared workloadbuilder library. Build also sets the controller ownerReference
// and spec.controllerRef pointing at the Job.
func (jm *Controller) generateWorkload(job *batch.Job) (*schedulingv1alpha3.Workload, error) {
	wl, err := workloadbuilder.Build(buildWorkloadItem(job), workloadbuilder.BuildOptions{
		Name:      computeWorkloadName(job),
		Namespace: job.Namespace,
		Owner:     metav1.NewControllerRef(job, controllerKind),
	})
	if err != nil {
		return nil, fmt.Errorf("invalid spec.scheduling: %w", err)
	}
	return wl, nil
}

// createWorkloadForJob compiles and creates a new Workload object for the given Job.
func (jm *Controller) createWorkloadForJob(ctx context.Context, job *batch.Job) (*schedulingv1alpha3.Workload, error) {
	workload, err := jm.generateWorkload(job)
	if err != nil {
		jm.recorder.Eventf(job, v1.EventTypeWarning, "FailedWorkloadCompilation",
			"Failed to compile Workload for Job %s: %v", job.Name, err)
		return nil, err
	}

	addWorkloadManagedByAnnotation(&workload.ObjectMeta)

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

// podGroupSpecFromTemplate builds the runtime PodGroupSpec from a Workload's
// PodGroupTemplate. Nil optional fields are copied as nil.
func podGroupSpecFromTemplate(workloadName string, template schedulingv1alpha3.PodGroupTemplate) schedulingv1alpha3.PodGroupSpec {
	spec := schedulingv1alpha3.PodGroupSpec{
		WorkloadRef: &schedulingv1alpha3.WorkloadReference{
			WorkloadName: workloadName,
			TemplateName: template.Name,
		},
		SchedulingPolicy:      *template.SchedulingPolicy.DeepCopy(),
		SchedulingConstraints: template.SchedulingConstraints.DeepCopy(),
		DisruptionMode:        template.DisruptionMode.DeepCopy(),
	}
	if len(template.ResourceClaims) > 0 {
		spec.ResourceClaims = make([]schedulingv1alpha3.PodGroupResourceClaim, len(template.ResourceClaims))
		for i := range template.ResourceClaims {
			template.ResourceClaims[i].DeepCopyInto(&spec.ResourceClaims[i])
		}
	}
	return spec
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

// createPodGroup creates a runtime PodGroup for the Job from the Workload's
// PodGroupTemplate named templateName. It always carries a controller ownerRef
// to the Job. When linkWorkloadOwner is true, it gets a non-controller ownerRef
// to the Workload; a delegated (parent-owned) Workload gets none, leaving the
// parent as the Workload's sole owner. BlockOwnerDeletion is left unset so
// Workload deletion is never blocked by surviving PodGroups.
func (jm *Controller) createPodGroup(ctx context.Context, job *batch.Job, workload *schedulingv1alpha3.Workload,
	templateName string, linkWorkloadOwner bool) (*schedulingv1alpha3.PodGroup, error) {
	template, err := getPodGroupTemplateByName(workload, templateName)
	if err != nil {
		return nil, err
	}

	ownerRefs := []metav1.OwnerReference{*metav1.NewControllerRef(job, controllerKind)}
	if linkWorkloadOwner {
		ownerRefs = append(ownerRefs, metav1.OwnerReference{
			APIVersion: schedulingv1alpha3.SchemeGroupVersion.String(),
			Kind:       "Workload",
			Name:       workload.Name,
			UID:        workload.UID,
		})
	}

	podGroup := &schedulingv1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:            computePodGroupName(workload.Name, template.Name),
			Namespace:       job.Namespace,
			OwnerReferences: ownerRefs,
		},
		Spec: podGroupSpecFromTemplate(workload.Name, *template),
	}
	addWorkloadManagedByAnnotation(&podGroup.ObjectMeta)

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
func (jm *Controller) addSchedulingInformers(logger klog.Logger, workloadInformer schedulinginformers.WorkloadInformer, podGroupInformer schedulinginformers.PodGroupInformer) error {
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
