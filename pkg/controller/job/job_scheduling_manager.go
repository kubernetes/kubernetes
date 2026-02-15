/*
Copyright 2026 The Kubernetes Authors.

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
	schedulinginformers "k8s.io/client-go/informers/scheduling/v1alpha2"
	"k8s.io/client-go/tools/cache"
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
	// All pods must run concurrently for gang scheduling to apply.
	if job.Spec.Completions == nil || *job.Spec.Completions != *job.Spec.Parallelism {
		return false
	}
	if job.Spec.Template.Spec.SchedulingGroup != nil {
		return false
	}
	return true
}

// ensureWorkloadAndPodGroup discovers or creates Workload and PodGroup for the given Job.
// It returns the PodGroup if the Job has associated scheduling objects, or nil if:
// - the feature is not applicable (not eligible, has pods, etc.)
// - creation is not needed and no existing objects were found
//
// The method only creates objects when the Job has no pods yet.
func (jm *Controller) ensureWorkloadAndPodGroup(ctx context.Context, job *batch.Job, pods []*v1.Pod) (*schedulingv1alpha2.PodGroup, error) {
	if !isGangSchedulingEligible(job) {
		return nil, nil
	}

	hasPods := len(pods) > 0

	workload, err := jm.ensureWorkload(ctx, job, hasPods)
	if err != nil {
		return nil, err
	}
	if workload == nil {
		return nil, nil
	}

	return jm.ensurePodGroup(ctx, job, workload, hasPods)
}

// ensureWorkload discovers an existing Workload for the Job, or creates one if
// the Job has no pods yet. Returns nil without error when creation should be
// skipped (e.g., Job already has pods) or when the Workload has an unsupported structure.
func (jm *Controller) ensureWorkload(ctx context.Context, job *batch.Job, hasPods bool) (*schedulingv1alpha2.Workload, error) {
	logger := klog.FromContext(ctx)

	workload, err := jm.discoverWorkloadForJob(ctx, job)
	if err != nil {
		return nil, fmt.Errorf("discovering Workload for Job %s/%s: %w", job.Namespace, job.Name, err)
	}

	// if no Workload found and Job already has pods, don't create one.
	if workload == nil && hasPods {
		return nil, nil
	}

	// if no Workload found and no pods, create one.
	if workload == nil {
		workload, err = jm.createWorkloadForJob(ctx, job)
		if err != nil {
			if apierrors.IsAlreadyExists(err) {
				// another actor created it concurrently, re-discover.
				logger.V(4).Info("Workload already exists, re-discovering", "job", klog.KObj(job))
				workload, err = jm.discoverWorkloadForJob(ctx, job)
				if err != nil {
					return nil, fmt.Errorf("re-discovering Workload for Job %s/%s after AlreadyExists: %w", job.Namespace, job.Name, err)
				}
				if workload == nil {
					return nil, fmt.Errorf("Workload for Job %s/%s not found after AlreadyExists", job.Namespace, job.Name)
				}
			} else {
				return nil, fmt.Errorf("creating Workload for Job %s/%s: %w", job.Namespace, job.Name, err)
			}
		}
	}

	// validate the Workload has a supported structure.
	if len(workload.Spec.PodGroupTemplates) != 1 {
		logger.V(2).Info("Workload has unsupported number of PodGroupTemplates, falling back to normal scheduling",
			"job", klog.KObj(job), "workload", klog.KObj(workload), "templateCount", len(workload.Spec.PodGroupTemplates))
		jm.recorder.Eventf(job, v1.EventTypeWarning, "UnsupportedWorkloadStructure",
			"Workload %s has %d PodGroupTemplates (expected 1), falling back to normal scheduling",
			workload.Name, len(workload.Spec.PodGroupTemplates))
		return nil, nil
	}

	return workload, nil
}

// ensurePodGroup discovers an existing PodGroup for the Workload, or creates one
// if the Job has no pods yet. Returns nil without error when creation should be
// skipped (e.g., Job already has pods but no PodGroup exists yet).
func (jm *Controller) ensurePodGroup(ctx context.Context, job *batch.Job, workload *schedulingv1alpha2.Workload, hasPods bool) (*schedulingv1alpha2.PodGroup, error) {
	logger := klog.FromContext(ctx)

	podGroup, err := jm.discoverPodGroupForWorkload(ctx, workload)
	if err != nil {
		return nil, fmt.Errorf("discovering PodGroup for Workload %s/%s: %w", workload.Namespace, workload.Name, err)
	}

	// if no PodGroup found and Job already has pods, return nil.
	// The pods may have been created before PodGroup creation completed.
	if podGroup == nil && hasPods {
		return nil, nil
	}

	// if no PodGroup found and no pods, create one.
	if podGroup == nil {
		podGroup, err = jm.createPodGroupForWorkload(ctx, job, workload)
		if err != nil {
			if apierrors.IsAlreadyExists(err) {
				logger.V(4).Info("PodGroup already exists, re-discovering", "job", klog.KObj(job))
				podGroup, err = jm.discoverPodGroupForWorkload(ctx, workload)
				if err != nil {
					return nil, fmt.Errorf("re-discovering PodGroup for Workload %s/%s after AlreadyExists: %w", workload.Namespace, workload.Name, err)
				}
				if podGroup == nil {
					return nil, fmt.Errorf("PodGroup for Workload %s/%s not found after AlreadyExists", workload.Namespace, workload.Name)
				}
			} else {
				return nil, fmt.Errorf("creating PodGroup for Workload %s/%s: %w", job.Namespace, workload.Name, err)
			}
		}
	}

	return podGroup, nil
}

// discoverWorkloadForJob finds an existing Workload for the given Job by looking
// up the deterministic name produced by computeWorkloadName.
func (jm *Controller) discoverWorkloadForJob(ctx context.Context, job *batch.Job) (*schedulingv1alpha2.Workload, error) {
	workload, err := jm.workloadLister.Workloads(job.Namespace).Get(computeWorkloadName(job))
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return workload, err
}

// discoverPodGroupForWorkload finds an existing PodGroup for the given Workload
// by looking up the deterministic name produced by computePodGroupName.
func (jm *Controller) discoverPodGroupForWorkload(ctx context.Context, workload *schedulingv1alpha2.Workload) (*schedulingv1alpha2.PodGroup, error) {
	if len(workload.Spec.PodGroupTemplates) == 0 {
		return nil, nil
	}
	pgName := computePodGroupName(workload.Name, workload.Spec.PodGroupTemplates[0].Name)
	podGroup, err := jm.podGroupLister.PodGroups(workload.Namespace).Get(pgName)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return podGroup, err
}

// buildPodGroupTemplates constructs the PodGroupTemplate slice for a Job's Workload.
// For alpha, a single template is created covering all of the Job's pods.
// This is the single place to extend when supporting multiple PodGroupTemplates per
// Workload in future.
func buildPodGroupTemplates(job *batch.Job) []schedulingv1alpha2.PodGroupTemplate {
	return []schedulingv1alpha2.PodGroupTemplate{
		{
			Name: fmt.Sprintf("%s-worker-0", job.Name),
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
		"Created Workload %s with gang scheduling (minCount=%d)", created.Name, *job.Spec.Parallelism)
	return created, nil
}

// createPodGroupForWorkload creates a new PodGroup for the given Workload.
func (jm *Controller) createPodGroupForWorkload(ctx context.Context, job *batch.Job, workload *schedulingv1alpha2.Workload) (*schedulingv1alpha2.PodGroup, error) {
	if len(workload.Spec.PodGroupTemplates) == 0 {
		return nil, fmt.Errorf("Workload %s/%s has no PodGroupTemplates", workload.Namespace, workload.Name)
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

	jm.recorder.Eventf(job, v1.EventTypeNormal, "PodGroupCreated",
		"Created PodGroup %s for Workload %s", created.Name, workload.Name)
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
