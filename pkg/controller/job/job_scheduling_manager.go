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
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1alpha2"
	"k8s.io/client-go/tools/cache"
	retry "k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

// isGangSchedulingEligible returns true if the Job should have Workload and PodGroup
// created. It checks:
//   - parallelism > 1
//   - completionMode == Indexed
//   - completions == parallelism
//   - schedulingGroup is not already set on the pod template (opt-out mechanism)
func isGangSchedulingEligible(job *batch.Job) bool {
	if job.Spec.Parallelism == nil || *job.Spec.Parallelism <= 1 {
		return false
	}
	if !isIndexedJob(job) {
		return false
	}
	if job.Spec.Completions == nil || *job.Spec.Completions != *job.Spec.Parallelism {
		return false
	}
	if job.Spec.Template.Spec.SchedulingGroup != nil {
		return false
	}
	return true
}

// ensureWorkloadAndPodGroup discovers or creates Workload and PodGroup for the given Job.
// It returns the PodGroup if the Job has associated scheduling objects, or nil if the
// Job is not eligible or no objects were found and creation is not applicable.
//
// The method first attempts to discover an existing Workload (which may have been
// created by the Job controller, a user, or a higher-level controller).
// If a Workload is found, it discovers the associated PodGroup and returns it.
//
// If no Workload is found, creation is only attempted when the Job has no pods
// (no active or terminal pods). This prevents creating scheduling objects for Jobs
// that predate the feature gate being enabled.
func (jm *Controller) ensureWorkloadAndPodGroup(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1alpha2.PodGroup, error) {
	if !isGangSchedulingEligible(job) {
		return nil, nil
	}

	// Try to discover existing Workload and PodGroup.
	workload, err := jm.discoverWorkloadForJob(ctx, job)
	if err != nil {
		return nil, fmt.Errorf("discovering Workload for Job %s/%s: %w", job.Namespace, job.Name, err)
	}
	if workload != nil {
		podGroup, err := jm.discoverPodGroupForWorkload(ctx, job, workload)
		if err != nil {
			return nil, fmt.Errorf("discovering PodGroup for Workload %s/%s: %w", workload.Namespace, workload.Name, err)
		}
		return podGroup, nil
	}

	// No existing Workload found. Only create when the Job has no pods.
	if len(pods) > 0 {
		return nil, nil
	}

	workload, err = jm.ensureWorkload(ctx, job)
	if err != nil || workload == nil {
		return nil, err
	}
	return jm.ensurePodGroup(ctx, job, workload)
}

// ensureWorkload creates a new Workload for the Job. If the create call returns
// AlreadyExists, it falls back to discovery. Returns nil without error when the
// Workload has an unsupported structure (PodGroupTemplates != 1 for alpha).
func (jm *Controller) ensureWorkload(ctx context.Context, job *batch.Job) (*schedulingv1alpha2.Workload, error) {
	logger := klog.FromContext(ctx)

	var workload *schedulingv1alpha2.Workload
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		var err error
		workload, err = jm.createWorkloadForJob(ctx, job)
		if apierrors.IsAlreadyExists(err) {
			if workload, err = jm.discoverWorkloadForJob(ctx, job); err != nil {
				return err
			}
		}
		return err
	})
	if err != nil {
		logger.V(2).Error(err, "Failed to create Workload for Job", "job", klog.KObj(job))
		return nil, err
	}

	// validate the Workload has a supported structure for alpha.
	if len(workload.Spec.PodGroupTemplates) != 1 {
		logger.V(2).Info("workload has unsupported number of PodGroupTemplates, falling back to normal scheduling",
			"job", klog.KObj(job), "workload", klog.KObj(workload), "PodGroupTemplateCount", len(workload.Spec.PodGroupTemplates))
		return nil, nil
	}

	return workload, nil
}

// ensurePodGroup creates a PodGroup for the Workload.
func (jm *Controller) ensurePodGroup(ctx context.Context, job *batch.Job, workload *schedulingv1alpha2.Workload) (*schedulingv1alpha2.PodGroup, error) {
	logger := klog.FromContext(ctx)

	var podGroup *schedulingv1alpha2.PodGroup
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		var err error
		podGroup, err = jm.createPodGroupForWorkload(ctx, job, workload)
		if apierrors.IsAlreadyExists(err) {
			if podGroup, err = jm.discoverPodGroupForWorkload(ctx, job, workload); err != nil {
				return err
			}
		}
		return err
	})
	if err != nil {
		logger.V(2).Error(err, "Failed to create PodGroup for Workload", "workload", klog.KObj(workload), "job", klog.KObj(job))
		return nil, err
	}

	return podGroup, nil
}

// workloadReferencesJob returns true if the Workload's spec.controllerRef
// points to the given Job.
func workloadReferencesJob(wl *schedulingv1alpha2.Workload, job *batch.Job) bool {
	ref := wl.Spec.ControllerRef
	return ref != nil &&
		ref.APIGroup == "batch" &&
		ref.Kind == "Job" &&
		ref.Name == job.Name
}

// discoverWorkloadForJob finds an existing Workload for the given Job by listing
// all Workloads in the namespace and matching on spec.controllerRef.
// Returns nil if no matching Workload is found. Returns nil and emits an event
// if more than one Workload references the same Job (ambiguous).
func (jm *Controller) discoverWorkloadForJob(ctx context.Context, job *batch.Job) (*schedulingv1alpha2.Workload, error) {
	allWorkloads, err := jm.workloadLister.Workloads(job.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	var matched []*schedulingv1alpha2.Workload
	for _, wl := range allWorkloads {
		if workloadReferencesJob(wl, job) {
			matched = append(matched, wl)
		}
	}
	if len(matched) > 1 {
		jm.recorder.Eventf(job, v1.EventTypeWarning, "AmbiguousWorkload",
			"Found %d Workloads referencing Job %s, falling back to normal scheduling for job %s/%s",
			len(matched), job.Name, job.Namespace, job.Name)
		return nil, nil
	}
	if len(matched) == 1 {
		return matched[0], nil
	}
	return nil, nil
}

// discoverPodGroupForWorkload finds an existing PodGroup for the given Workload
// by listing all PodGroups in the namespace and matching on
// spec.podGroupTemplateRef.workloadName.
func (jm *Controller) discoverPodGroupForWorkload(ctx context.Context, job *batch.Job, workload *schedulingv1alpha2.Workload) (*schedulingv1alpha2.PodGroup, error) {
	allPodGroups, err := jm.podGroupLister.PodGroups(workload.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	var matched []*schedulingv1alpha2.PodGroup
	for _, pg := range allPodGroups {
		if pg.Spec.PodGroupTemplateRef != nil &&
			pg.Spec.PodGroupTemplateRef.WorkloadName == workload.Name {
			matched = append(matched, pg)
		}
	}

	// for alpha, we only support one PodGroup per Workload.
	if len(matched) == 1 {
		return matched[0], nil
	}
	if len(matched) > 1 {
		jm.recorder.Eventf(workload, v1.EventTypeWarning, "AmbiguousPodGroup",
			"Found %d PodGroups referencing Workload %s, falling back to normal scheduling for job %s/%s",
			len(matched), workload.Name, job.Namespace, job.Name)
		return nil, nil
	}
	return nil, nil
}

// buildPodGroupTemplates constructs the PodGroupTemplate slice for a Job's Workload.
// For alpha, a single template is created covering all of the Job's pods.
// This is the single place to extend when supporting multiple PodGroupTemplates per
// Workload in future.
func buildPodGroupTemplates(job *batch.Job) []schedulingv1alpha2.PodGroupTemplate {
	return []schedulingv1alpha2.PodGroupTemplate{
		{
			Name: fmt.Sprintf("worker-%d", 0),
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
				APIGroup: "batch",
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
				// Non-controller ownerRef to Workload
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
				WorkloadName:         workload.Name,
				PodGroupTemplateName: template.Name,
			},
			SchedulingPolicy: *template.SchedulingPolicy.DeepCopy(),
		},
	}

	created, err := jm.kubeClient.SchedulingV1alpha2().PodGroups(job.Namespace).Create(ctx, podGroup, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}

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

// addSchedulingInformers wires up Workload and PodGroup informers so that
// changes to those objects re-enqueue the owning Job.
func (jm *Controller) addSchedulingInformers(logger klog.Logger, workloadInformer schedulinginformers.WorkloadInformer, podGroupInformer schedulinginformers.PodGroupInformer) error {
	if _, err := workloadInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { jm.enqueueJobFromOwnerRef(logger, obj) },
		UpdateFunc: func(old, new interface{}) { jm.enqueueJobFromOwnerRef(logger, new) },
		DeleteFunc: func(obj interface{}) { jm.enqueueJobFromOwnerRef(logger, obj) },
	}); err != nil {
		return fmt.Errorf("adding Workload event handler: %w", err)
	}
	jm.workloadLister = workloadInformer.Lister()
	jm.workloadStoreSynced = workloadInformer.Informer().HasSynced

	if _, err := podGroupInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { jm.enqueueJobFromOwnerRef(logger, obj) },
		UpdateFunc: func(old, new interface{}) { jm.enqueueJobFromOwnerRef(logger, new) },
		DeleteFunc: func(obj interface{}) { jm.enqueueJobFromOwnerRef(logger, obj) },
	}); err != nil {
		return fmt.Errorf("adding PodGroup event handler: %w", err)
	}
	jm.podGroupLister = podGroupInformer.Lister()
	jm.podGroupStoreSynced = podGroupInformer.Informer().HasSynced

	return nil
}

// enqueueJobFromOwnerRef enqueues the owning Job by resolving the object's
// ownerReferences. It handles tombstone-wrapped objects and is used as the
// event handler for Workload and PodGroup informers.
func (jm *Controller) enqueueJobFromOwnerRef(logger klog.Logger, obj interface{}) {
	// Unwrap tombstone if needed, then use ObjectMetaAccessor (implemented by
	// both Workload and PodGroup) to read namespace and ownerReferences.
	raw := obj
	if tombstone, ok := raw.(cache.DeletedFinalStateUnknown); ok {
		raw = tombstone.Obj
	}
	accessor, ok := raw.(metav1.ObjectMetaAccessor)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type %T", raw))
		return
	}
	meta := accessor.GetObjectMeta()

	for _, ref := range meta.GetOwnerReferences() {
		if ref.Kind == "Job" && ref.APIVersion == "batch/v1" {
			jm.queue.Add(meta.GetNamespace() + "/" + ref.Name)
			return
		}
	}
}
