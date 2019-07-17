/*
Copyright 2016 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"
	"reflect"
	"time"

	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	policyclient "k8s.io/client-go/kubernetes/typed/policy/v1beta1"
	"k8s.io/client-go/util/retry"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
)

const (
	// MaxDisruptedPodSize is the max size of PodDisruptionBudgetStatus.DisruptedPods. API server eviction
	// subresource handler will refuse to evict pods covered by the corresponding PDB
	// if the size of the map exceeds this value. It means a large number of
	// evictions have been approved by the API server but not noticed by the PDB controller yet.
	// This situation should self-correct because the PDB controller removes
	// entries from the map automatically after the PDB DeletionTimeout regardless.
	MaxDisruptedPodSize = 2000
)

// EvictionsRetry is the retry for a conflict where multiple clients
// are making changes to the same resource.
var EvictionsRetry = wait.Backoff{
	Steps:    20,
	Duration: 500 * time.Millisecond,
	Factor:   1.0,
	Jitter:   0.1,
}

func newEvictionStorage(store *genericregistry.Store, podDisruptionBudgetClient policyclient.PodDisruptionBudgetsGetter) *EvictionREST {
	return &EvictionREST{store: store, podDisruptionBudgetClient: podDisruptionBudgetClient}
}

// EvictionREST implements the REST endpoint for evicting pods from nodes
type EvictionREST struct {
	store                     *genericregistry.Store
	podDisruptionBudgetClient policyclient.PodDisruptionBudgetsGetter
}

var _ = rest.Creater(&EvictionREST{})
var _ = rest.GroupVersionKindProvider(&EvictionREST{})

// GroupVersionKind specifies a particular GroupVersionKind to discovery
func (r *EvictionREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "policy", Version: "v1beta1", Kind: "Eviction"}
}

// New creates a new eviction resource
func (r *EvictionREST) New() runtime.Object {
	return &policy.Eviction{}
}

// Propagate dry-run takes the dry-run option from the request and pushes it into the eviction object.
// It returns an error if they have non-matching dry-run options.
func propagateDryRun(eviction *policy.Eviction, options *metav1.CreateOptions) (*metav1.DeleteOptions, error) {
	if eviction.DeleteOptions == nil {
		return &metav1.DeleteOptions{DryRun: options.DryRun}, nil
	}
	if len(eviction.DeleteOptions.DryRun) == 0 {
		eviction.DeleteOptions.DryRun = options.DryRun
		return eviction.DeleteOptions, nil
	}
	if len(options.DryRun) == 0 {
		return eviction.DeleteOptions, nil
	}

	if !reflect.DeepEqual(options.DryRun, eviction.DeleteOptions.DryRun) {
		return nil, fmt.Errorf("Non-matching dry-run options in request and content: %v and %v", options.DryRun, eviction.DeleteOptions.DryRun)
	}
	return eviction.DeleteOptions, nil
}

// Create attempts to create a new eviction.  That is, it tries to evict a pod.
func (r *EvictionREST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	eviction := obj.(*policy.Eviction)

	deletionOptions, err := propagateDryRun(eviction, options)
	if err != nil {
		return nil, err
	}

	obj, err = r.store.Get(ctx, eviction.Name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	pod := obj.(*api.Pod)

	if createValidation != nil {
		if err := createValidation(eviction.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	// Evicting a terminal pod should result in direct deletion of pod as it already caused disruption by the time we are evicting.
	// There is no need to check for pdb.
	if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
		_, _, err = r.store.Delete(ctx, eviction.Name, rest.ValidateAllObjectFunc, deletionOptions)
		if err != nil {
			return nil, err
		}
		return &metav1.Status{
			Status: metav1.StatusSuccess}, nil
	}
	var rtStatus *metav1.Status
	var pdbName string
	err = func() error {
		pdbs, err := r.getPodDisruptionBudgets(ctx, pod)
		if err != nil {
			return err
		}

		if len(pdbs) > 1 {
			rtStatus = &metav1.Status{
				Status:  metav1.StatusFailure,
				Message: "This pod has more than one PodDisruptionBudget, which the eviction subresource does not support.",
				Code:    500,
			}
			return nil
		}
		if len(pdbs) == 0 {
			return nil
		}

		pdb := &pdbs[0]
		pdbName = pdb.Name
		refresh := false
		err = retry.RetryOnConflict(EvictionsRetry, func() error {
			if refresh {
				pdb, err = r.podDisruptionBudgetClient.PodDisruptionBudgets(pod.Namespace).Get(pdbName, metav1.GetOptions{})
				if err != nil {
					return err
				}
			}
			// Try to verify-and-decrement

			// If it was false already, or if it becomes false during the course of our retries,
			// raise an error marked as a 429.
			if err = r.checkAndDecrement(pod.Namespace, pod.Name, *pdb, dryrun.IsDryRun(deletionOptions.DryRun)); err != nil {
				refresh = true
				return err
			}
			return nil
		})
		return err
	}()
	if err == wait.ErrWaitTimeout {
		err = errors.NewTimeoutError(fmt.Sprintf("couldn't update PodDisruptionBudget %q due to conflicts", pdbName), 10)
	}
	if err != nil {
		return nil, err
	}

	if rtStatus != nil {
		return rtStatus, nil
	}

	// At this point there was either no PDB or we succeeded in decrementing

	// Try the delete
	_, _, err = r.store.Delete(ctx, eviction.Name, rest.ValidateAllObjectFunc, deletionOptions)
	if err != nil {
		return nil, err
	}

	// Success!
	return &metav1.Status{Status: metav1.StatusSuccess}, nil
}

// checkAndDecrement checks if the provided PodDisruptionBudget allows any disruption.
func (r *EvictionREST) checkAndDecrement(namespace string, podName string, pdb policyv1beta1.PodDisruptionBudget, dryRun bool) error {
	if pdb.Status.ObservedGeneration < pdb.Generation {
		// TODO(mml): Add a Retry-After header.  Once there are time-based
		// budgets, we can sometimes compute a sensible suggested value.  But
		// even without that, we can give a suggestion (10 minutes?) that
		// prevents well-behaved clients from hammering us.
		err := errors.NewTooManyRequests("Cannot evict pod as it would violate the pod's disruption budget.", 0)
		err.ErrStatus.Details.Causes = append(err.ErrStatus.Details.Causes, metav1.StatusCause{Type: "DisruptionBudget", Message: fmt.Sprintf("The disruption budget %s is still being processed by the server.", pdb.Name)})
		return err
	}
	if pdb.Status.PodDisruptionsAllowed < 0 {
		return errors.NewForbidden(policy.Resource("poddisruptionbudget"), pdb.Name, fmt.Errorf("pdb disruptions allowed is negative"))
	}
	if len(pdb.Status.DisruptedPods) > MaxDisruptedPodSize {
		return errors.NewForbidden(policy.Resource("poddisruptionbudget"), pdb.Name, fmt.Errorf("DisruptedPods map too big - too many evictions not confirmed by PDB controller"))
	}
	if pdb.Status.PodDisruptionsAllowed == 0 {
		err := errors.NewTooManyRequests("Cannot evict pod as it would violate the pod's disruption budget.", 0)
		err.ErrStatus.Details.Causes = append(err.ErrStatus.Details.Causes, metav1.StatusCause{Type: "DisruptionBudget", Message: fmt.Sprintf("The disruption budget %s needs %d healthy pods and has %d currently", pdb.Name, pdb.Status.DesiredHealthy, pdb.Status.CurrentHealthy)})
		return err
	}

	pdb.Status.PodDisruptionsAllowed--
	// If this is a dry-run, we don't need to go any further than that.
	if dryRun == true {
		return nil
	}

	if pdb.Status.DisruptedPods == nil {
		pdb.Status.DisruptedPods = make(map[string]metav1.Time)
	}

	// Eviction handler needs to inform the PDB controller that it is about to delete a pod
	// so it should not consider it as available in calculations when updating PodDisruptions allowed.
	// If the pod is not deleted within a reasonable time limit PDB controller will assume that it won't
	// be deleted at all and remove it from DisruptedPod map.
	pdb.Status.DisruptedPods[podName] = metav1.Time{Time: time.Now()}
	if _, err := r.podDisruptionBudgetClient.PodDisruptionBudgets(namespace).UpdateStatus(&pdb); err != nil {
		return err
	}

	return nil
}

// getPodDisruptionBudgets returns any PDBs that match the pod or err if there's an error.
func (r *EvictionREST) getPodDisruptionBudgets(ctx context.Context, pod *api.Pod) ([]policyv1beta1.PodDisruptionBudget, error) {
	if len(pod.Labels) == 0 {
		return nil, nil
	}

	pdbList, err := r.podDisruptionBudgetClient.PodDisruptionBudgets(pod.Namespace).List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var pdbs []policyv1beta1.PodDisruptionBudget
	for _, pdb := range pdbList.Items {
		if pdb.Namespace != pod.Namespace {
			continue
		}
		selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
		if err != nil {
			continue
		}
		// If a PDB with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}

		pdbs = append(pdbs, pdb)
	}

	return pdbs, nil
}
