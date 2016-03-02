/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package resourcequota

import (
	"fmt"
	"io"
	"math/rand"
	"strings"
	"time"

	"github.com/hashicorp/golang-lru"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("ResourceQuota",
		func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
			registry := install.NewRegistry(client)
			return NewResourceQuota(client, registry)
		})
}

// quotaAdmission implements an admission controller that can enforce quota constraints
type quotaAdmission struct {
	*admission.Handler
	// must be able to read/write ResourceQuota
	client clientset.Interface
	// indexer that holds quota objects by namespace
	indexer cache.Indexer
	// registry that knows how to measure usage for objects
	registry quota.Registry

	// liveLookups holds the last few live lookups we've done to help ammortize cost on repeated lookup failures.
	// This let's us handle the case of latent caches, by looking up actual results for a namespace on cache miss/no results.
	// We track the lookup result here so that for repeated requests, we don't look it up very often.
	liveLookupCache *lru.Cache
	liveTTL         time.Duration
}

type liveLookupEntry struct {
	expiry time.Time
	items  []*api.ResourceQuota
}

// NewResourceQuota configures an admission controller that can enforce quota constraints
// using the provided registry.  The registry must have the capability to handle group/kinds that
// are persisted by the server this admission controller is intercepting
func NewResourceQuota(client clientset.Interface, registry quota.Registry) (admission.Interface, error) {
	liveLookupCache, err := lru.New(100)
	if err != nil {
		return nil, err
	}
	lw := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return client.Core().ResourceQuotas(api.NamespaceAll).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return client.Core().ResourceQuotas(api.NamespaceAll).Watch(options)
		},
	}
	indexer, reflector := cache.NewNamespaceKeyedIndexerAndReflector(lw, &api.ResourceQuota{}, 0)
	reflector.Run()
	return &quotaAdmission{
		Handler:         admission.NewHandler(admission.Create, admission.Update),
		client:          client,
		indexer:         indexer,
		registry:        registry,
		liveLookupCache: liveLookupCache,
		liveTTL:         time.Duration(30 * time.Second),
	}, nil
}

// Admit makes admission decisions while enforcing quota
func (q *quotaAdmission) Admit(a admission.Attributes) (err error) {
	// ignore all operations that correspond to sub-resource actions
	if a.GetSubresource() != "" {
		return nil
	}

	// if we do not know how to evaluate use for this kind, just ignore
	evaluators := q.registry.Evaluators()
	evaluator, found := evaluators[a.GetKind()]
	if !found {
		return nil
	}

	// for this kind, check if the operation could mutate any quota resources
	// if no resources tracked by quota are impacted, then just return
	op := a.GetOperation()
	operationResources := evaluator.OperationResources(op)
	if len(operationResources) == 0 {
		return nil
	}

	// determine if there are any quotas in this namespace
	// if there are no quotas, we don't need to do anything
	namespace, name := a.GetNamespace(), a.GetName()
	items, err := q.indexer.Index("namespace", &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Namespace: namespace, Name: ""}})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("Error resolving quota."))
	}
	// if there are no items held in our indexer, check our live-lookup LRU, if that misses, do the live lookup to prime it.
	if len(items) == 0 {
		lruItemObj, ok := q.liveLookupCache.Get(a.GetNamespace())
		if !ok || lruItemObj.(liveLookupEntry).expiry.Before(time.Now()) {
			liveList, err := q.client.Core().ResourceQuotas(namespace).List(api.ListOptions{})
			if err != nil {
				return admission.NewForbidden(a, err)
			}
			newEntry := liveLookupEntry{expiry: time.Now().Add(q.liveTTL)}
			for i := range liveList.Items {
				newEntry.items = append(newEntry.items, &liveList.Items[i])
			}
			q.liveLookupCache.Add(a.GetNamespace(), newEntry)
			lruItemObj = newEntry
		}
		lruEntry := lruItemObj.(liveLookupEntry)
		for i := range lruEntry.items {
			items = append(items, lruEntry.items[i])
		}
	}
	// if there are still no items, we can return
	if len(items) == 0 {
		return nil
	}

	// find the set of quotas that are pertinent to this request
	// reject if we match the quota, but usage is not calculated yet
	// reject if the input object does not satisfy quota constraints
	// if there are no pertinent quotas, we can just return
	inputObject := a.GetObject()
	resourceQuotas := []*api.ResourceQuota{}
	for i := range items {
		resourceQuota := items[i].(*api.ResourceQuota)
		match := evaluator.Matches(resourceQuota, inputObject)
		if !match {
			continue
		}
		hardResources := quota.ResourceNames(resourceQuota.Status.Hard)
		evaluatorResources := evaluator.MatchesResources()
		requiredResources := quota.Intersection(hardResources, evaluatorResources)
		err := evaluator.Constraints(requiredResources, inputObject)
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("Failed quota: %s: %v", resourceQuota.Name, err))
		}
		if !hasUsageStats(resourceQuota) {
			return admission.NewForbidden(a, fmt.Errorf("Status unknown for quota: %s", resourceQuota.Name))
		}
		resourceQuotas = append(resourceQuotas, resourceQuota)
	}
	if len(resourceQuotas) == 0 {
		return nil
	}

	// there is at least one quota that definitely matches our object
	// as a result, we need to measure the usage of this object for quota
	// on updates, we need to subtract the previous measured usage
	// if usage shows no change, just return since it has no impact on quota
	deltaUsage := evaluator.Usage(inputObject)
	if admission.Update == op {
		prevItem, err := evaluator.Get(namespace, name)
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("Unable to get previous: %v", err))
		}
		prevUsage := evaluator.Usage(prevItem)
		deltaUsage = quota.Subtract(deltaUsage, prevUsage)
	}
	if quota.IsZero(deltaUsage) {
		return nil
	}

	// TODO: Move to a bucketing work queue
	// If we guaranteed that we processed the request in order it was received to server, we would reduce quota conflicts.
	// Until we have the bucketing work queue, we jitter requests and retry on conflict.
	numRetries := 10
	interval := time.Duration(rand.Int63n(90)+int64(10)) * time.Millisecond

	// seed the retry loop with the initial set of quotas to process (should reduce each iteration)
	resourceQuotasToProcess := resourceQuotas
	for retry := 1; retry <= numRetries; retry++ {
		// the list of quotas we will try again if there is a version conflict
		tryAgain := []*api.ResourceQuota{}

		// check that we pass all remaining quotas so we do not prematurely charge
		// for each quota, mask the usage to the set of resources tracked by the quota
		// if request + used > hard, return an error describing the failure
		updatedUsage := map[string]api.ResourceList{}
		for _, resourceQuota := range resourceQuotasToProcess {
			hardResources := quota.ResourceNames(resourceQuota.Status.Hard)
			requestedUsage := quota.Mask(deltaUsage, hardResources)
			newUsage := quota.Add(resourceQuota.Status.Used, requestedUsage)
			if allowed, exceeded := quota.LessThanOrEqual(newUsage, resourceQuota.Status.Hard); !allowed {
				failedRequestedUsage := quota.Mask(requestedUsage, exceeded)
				failedUsed := quota.Mask(resourceQuota.Status.Used, exceeded)
				failedHard := quota.Mask(resourceQuota.Status.Hard, exceeded)
				return admission.NewForbidden(a,
					fmt.Errorf("Exceeded quota: %s, requested: %s, used: %s, limited: %s",
						resourceQuota.Name,
						prettyPrint(failedRequestedUsage),
						prettyPrint(failedUsed),
						prettyPrint(failedHard)))
			}
			updatedUsage[resourceQuota.Name] = newUsage
		}

		// update the status for each quota with its new usage
		// if we get a conflict, get updated quota, and enqueue
		for i, resourceQuota := range resourceQuotasToProcess {
			newUsage := updatedUsage[resourceQuota.Name]
			quotaToUpdate := &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Name:            resourceQuota.Name,
					Namespace:       resourceQuota.Namespace,
					ResourceVersion: resourceQuota.ResourceVersion,
				},
				Status: api.ResourceQuotaStatus{
					Hard: quota.Add(api.ResourceList{}, resourceQuota.Status.Hard),
					Used: newUsage,
				},
			}
			_, err = q.client.Core().ResourceQuotas(quotaToUpdate.Namespace).UpdateStatus(quotaToUpdate)
			if err != nil {
				if !errors.IsConflict(err) {
					return admission.NewForbidden(a, fmt.Errorf("Unable to update quota status: %s %v", resourceQuota.Name, err))
				}
				// if we get a conflict, we get the latest copy of the quota documents that were not yet modified so we retry all with latest state.
				for fetchIndex := i; fetchIndex < len(resourceQuotasToProcess); fetchIndex++ {
					latestQuota, err := q.client.Core().ResourceQuotas(namespace).Get(resourceQuotasToProcess[fetchIndex].Name)
					if err != nil {
						return admission.NewForbidden(a, fmt.Errorf("Unable to get quota: %s %v", resourceQuotasToProcess[fetchIndex].Name, err))
					}
					tryAgain = append(tryAgain, latestQuota)
				}
				break
			}
		}

		// all quotas were updated, so we can return
		if len(tryAgain) == 0 {
			return nil
		}

		// we have concurrent requests to update quota, so look to retry if needed
		// next iteration, we need to process the items that have to try again
		// pause the specified interval to encourage jitter
		if retry == numRetries {
			names := []string{}
			for _, quota := range tryAgain {
				names = append(names, quota.Name)
			}
			return admission.NewForbidden(a, fmt.Errorf("Unable to update status for quota: %s, ", strings.Join(names, ",")))
		}
		resourceQuotasToProcess = tryAgain
		time.Sleep(interval)
	}
	return nil
}

// prettyPrint formats a resource list for usage in errors
func prettyPrint(item api.ResourceList) string {
	parts := []string{}
	for key, value := range item {
		constraint := string(key) + "=" + value.String()
		parts = append(parts, constraint)
	}
	return strings.Join(parts, ",")
}

// hasUsageStats returns true if for each hard constraint there is a value for its current usage
func hasUsageStats(resourceQuota *api.ResourceQuota) bool {
	for resourceName := range resourceQuota.Status.Hard {
		if _, found := resourceQuota.Status.Used[resourceName]; !found {
			return false
		}
	}
	return true
}
