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
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apiserver/pkg/util/feature"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1alpha2"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	jobutil "k8s.io/kubernetes/pkg/controller/job/util"
	"k8s.io/kubernetes/pkg/features"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

const (
	// Workload.spec.controllerRef and PodGroup.spec.podGroupTemplateRef
	// carry only names (no UIDs), so these indexes use namespace/name as
	// the key. Callers must still verify ownership via ownerRef UID checks
	// before acting on results.
	workloadByJobNameIndexKey      = "workloadByJobName"
	podGroupByWorkloadNameIndexKey = "podGroupByWorkloadName"
)

// shouldManageWorkloadForJob returns true if the Job should have Workload and PodGroup
// created. It checks:
//   - parallelism > 1
//   - completionMode == Indexed
//   - completions == parallelism
//   - schedulingGroup is not already set on the pod template
//   - EnableWorkloadWithJob feature gate is enabled
func shouldManageWorkloadForJob(job *batch.Job) bool {
	if !feature.DefaultFeatureGate.Enabled(features.EnableWorkloadWithJob) {
		return false
	}
	return jobutil.IsGangSchedulingCandidate(
		job.Spec.Parallelism,
		job.Spec.Completions,
		isIndexedJob(job),
		job.Spec.Template.Spec.SchedulingGroup != nil,
	)
}

// ensureWorkloadAndPodGroup discovers or creates Workload and PodGroup for the given Job.
// Returns both objects when workload integration is active, or (nil, nil, nil)
// when the Job should fall back to default scheduling.
//
// The flow is:
//  1. Get or create a Workload owned by this controller.
//  2. Get or create a PodGroup owned by this controller.
func (jm *Controller) ensureWorkloadAndPodGroup(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1alpha2.Workload, *schedulingv1alpha2.PodGroup, error) {
	wl, err := jm.getOrCreateWorkload(ctx, job, pods)
	if err != nil || wl == nil {
		return nil, nil, err
	}
	pg, err := jm.getOrCreatePodGroup(ctx, job, wl)
	return wl, pg, err
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
// (i.e., ambiguous matches, not owned by this controller, unsupported structure, or
// not a new Job).
func (jm *Controller) getOrCreateWorkload(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1alpha2.Workload, error) {
	logger := klog.FromContext(ctx)

	var result *schedulingv1alpha2.Workload
	err := retry.OnError(retry.DefaultRetry, apierrors.IsAlreadyExists, func() error {
		matched, err := jm.listWorkloadsForJob(job)
		if err != nil {
			logger.V(2).Info("Failed to list Workloads, falling back to default scheduling",
				"job", klog.KObj(job), "err", err)
			return nil
		}

		switch len(matched) {
		case 0:
			if !isNewJob(job, pods) {
				return nil
			}
			wl, err := jm.createWorkloadForJob(ctx, job)
			if err != nil {
				return err
			}
			result = wl
			return nil
		case 1:
			wl := matched[0]
			if !metav1.IsControlledBy(wl, job) {
				logger.V(2).Info("Workload not owned by this Job, skipping workload integration",
					"job", klog.KObj(job), "workload", klog.KObj(wl))
				return nil
			}
			result, err = jm.validateWorkloadStructure(logger, job, wl)
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
func (jm *Controller) validateWorkloadStructure(logger klog.Logger, job *batch.Job, wl *schedulingv1alpha2.Workload) (*schedulingv1alpha2.Workload, error) {
	if len(wl.Spec.PodGroupTemplates) != 1 {
		logger.V(2).Info("Workload has unsupported number of PodGroupTemplates, falling back to default scheduling",
			"job", klog.KObj(job), "workload", klog.KObj(wl), "count", len(wl.Spec.PodGroupTemplates))
		return nil, nil
	}
	return wl, nil
}

// getOrCreatePodGroup returns the PodGroup for this Workload, creating one if needed.
// Returns (nil, nil) when the Job should fall back to default scheduling
// (ambiguous matches or not owned by this controller).
func (jm *Controller) getOrCreatePodGroup(ctx context.Context, job *batch.Job, workload *schedulingv1alpha2.Workload) (*schedulingv1alpha2.PodGroup, error) {
	logger := klog.FromContext(ctx)

	var result *schedulingv1alpha2.PodGroup
	err := retry.OnError(retry.DefaultRetry, apierrors.IsAlreadyExists, func() error {
		matched, err := jm.listPodGroupsForWorkload(workload)
		if err != nil {
			logger.V(2).Info("Failed to list PodGroups, falling back to default scheduling",
				"job", klog.KObj(job), "workload", klog.KObj(workload), "err", err)
			return nil
		}

		switch len(matched) {
		case 0:
			pg, err := jm.createPodGroupForWorkload(ctx, job, workload)
			if err != nil {
				return err
			}
			result = pg
			return nil
		case 1:
			pg := matched[0]
			if !metav1.IsControlledBy(pg, job) {
				logger.V(2).Info("PodGroup not owned by this Job, skipping workload integration",
					"job", klog.KObj(job), "workload", klog.KObj(workload), "podGroup", klog.KObj(pg))
				return nil
			}
			result = pg
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
func (jm *Controller) listWorkloadsForJob(job *batch.Job) ([]*schedulingv1alpha2.Workload, error) {
	if jm.workloadIndexer == nil {
		return nil, fmt.Errorf("workload indexer is not configured")
	}
	key := namespacedNameIndexKey(job.Namespace, job.Name)
	all, err := jm.workloadIndexer.ByIndex(workloadByJobNameIndexKey, key)
	if err != nil {
		return nil, err
	}
	matched := make([]*schedulingv1alpha2.Workload, 0, len(all))
	for _, wl := range all {
		workload, ok := wl.(*schedulingv1alpha2.Workload)
		if !ok {
			continue
		}
		matched = append(matched, workload)
	}
	return matched, nil
}

// listPodGroupsForWorkload returns all PodGroups in the Workload's namespace
// whose spec.podGroupTemplateRef references the given Workload.
func (jm *Controller) listPodGroupsForWorkload(workload *schedulingv1alpha2.Workload) ([]*schedulingv1alpha2.PodGroup, error) {
	if jm.podGroupIndexer == nil {
		return nil, fmt.Errorf("podgroup indexer is not configured")
	}
	key := namespacedNameIndexKey(workload.Namespace, workload.Name)
	all, err := jm.podGroupIndexer.ByIndex(podGroupByWorkloadNameIndexKey, key)
	if err != nil {
		return nil, err
	}
	matched := make([]*schedulingv1alpha2.PodGroup, 0, len(all))
	for _, pg := range all {
		podGroup, ok := pg.(*schedulingv1alpha2.PodGroup)
		if !ok {
			continue
		}
		matched = append(matched, podGroup)
	}
	return matched, nil
}

// buildPodGroupTemplates constructs the PodGroupTemplate slice for a Job's Workload.
// For alpha, a single template is created covering all of the Job's pods.
// This is the single place to extend when supporting multiple PodGroupTemplates per
// Workload in future.
func buildPodGroupTemplates(job *batch.Job) []schedulingv1alpha2.PodGroupTemplate {
	return []schedulingv1alpha2.PodGroupTemplate{
		{
			Name: fmt.Sprintf("%s-pgt-%d", job.Name, 0),
			SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha2.GangSchedulingPolicy{
					MinCount: *job.Spec.Parallelism,
				},
			},
		},
	}
}

// createWorkloadForJob creates a new Workload object for the given Job.
func (jm *Controller) createWorkloadForJob(ctx context.Context, job *batch.Job) (*schedulingv1alpha2.Workload, error) {
	workloadName := computeWorkloadName(job)
	templates := buildPodGroupTemplates(job)

	workload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      workloadName,
			Namespace: job.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(job, controllerKind),
			},
		},
		Spec: schedulingv1alpha2.WorkloadSpec{
			ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
				APIGroup: batch.SchemeGroupVersion.Group,
				Kind:     "Job",
				Name:     job.Name,
			},
			PodGroupTemplates: templates,
		},
	}

	created, err := jm.kubeClient.SchedulingV1alpha2().Workloads(job.Namespace).Create(ctx, workload, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}

	jm.recorder.Eventf(job, v1.EventTypeNormal, "WorkloadCreated",
		"Created Workload %s for Job %s", created.Name, job.Name)
	return created, nil
}

// createPodGroupForWorkload creates a new PodGroup for the given Workload.
func (jm *Controller) createPodGroupForWorkload(ctx context.Context, job *batch.Job, workload *schedulingv1alpha2.Workload) (*schedulingv1alpha2.PodGroup, error) {
	if len(workload.Spec.PodGroupTemplates) == 0 {
		return nil, fmt.Errorf("workload %s/%s has no PodGroupTemplates", workload.Namespace, workload.Name)
	}

	template := workload.Spec.PodGroupTemplates[0]
	podGroupName := computePodGroupName(workload.Name, template.Name)

	podGroup := &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podGroupName,
			Namespace: job.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				// Controller ownerRef to Job
				*metav1.NewControllerRef(job, controllerKind),
				// Non-controller ownerRef to Workload.
				// BlockOwnerDeletion is intentionally left nil (false) so that
				// Workload deletion is not blocked by surviving PodGroups.
				{
					APIVersion: schedulingv1alpha2.SchemeGroupVersion.String(),
					Kind:       "Workload",
					Name:       workload.Name,
					UID:        workload.UID,
				},
			},
		},
		Spec: schedulingv1alpha2.PodGroupSpec{
			PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
				Workload: &schedulingv1alpha2.WorkloadPodGroupTemplateReference{
					WorkloadName:         workload.Name,
					PodGroupTemplateName: template.Name,
				},
			},
			SchedulingPolicy: *template.SchedulingPolicy.DeepCopy(),
		},
	}

	return jm.kubeClient.SchedulingV1alpha2().PodGroups(job.Namespace).Create(ctx, podGroup, metav1.CreateOptions{})
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
		return fmt.Errorf("workload informer is required when the feature gate %q is enabled", features.EnableWorkloadWithJob)
	}
	if podGroupInformer == nil {
		return fmt.Errorf("pod group informer is required when the feature gate %q is enabled", features.EnableWorkloadWithJob)
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
func workloadControllerRefJobName(wl *schedulingv1alpha2.Workload) string {
	if wl.Spec.ControllerRef == nil ||
		wl.Spec.ControllerRef.APIGroup != batch.SchemeGroupVersion.Group ||
		wl.Spec.ControllerRef.Kind != "Job" ||
		wl.Spec.ControllerRef.Name == "" {
		return ""
	}
	return wl.Spec.ControllerRef.Name
}

// podGroupWorkloadName returns the workload name from the PodGroup's
// spec.podGroupTemplateRef, or "" if the reference is not set.
func podGroupWorkloadName(pg *schedulingv1alpha2.PodGroup) string {
	if pg.Spec.PodGroupTemplateRef == nil ||
		pg.Spec.PodGroupTemplateRef.Workload == nil ||
		pg.Spec.PodGroupTemplateRef.Workload.WorkloadName == "" {
		return ""
	}
	return pg.Spec.PodGroupTemplateRef.Workload.WorkloadName
}

func workloadByJobNameIndexFunc(obj interface{}) ([]string, error) {
	wl, ok := obj.(*schedulingv1alpha2.Workload)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T", obj)
	}
	if name := workloadControllerRefJobName(wl); name != "" {
		return []string{namespacedNameIndexKey(wl.Namespace, name)}, nil
	}
	return nil, nil
}

func podGroupByWorkloadNameIndexFunc(obj interface{}) ([]string, error) {
	pg, ok := obj.(*schedulingv1alpha2.PodGroup)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T", obj)
	}
	if name := podGroupWorkloadName(pg); name != "" {
		return []string{namespacedNameIndexKey(pg.Namespace, name)}, nil
	}
	return nil, nil
}
