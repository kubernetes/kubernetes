/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"sync"
	"time"

	"github.com/golang/glog"
	lru "github.com/hashicorp/golang-lru"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/etcd"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

type quotaEvaluator struct {
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
	// updatedQuotas holds a cache of quotas that we've updated.  This is used to pull the "really latest" during back to
	// back quota evaluations that touch the same quota doc.  This only works because we can compare etcd resourceVersions
	// for the same resource as integers.  Before this change: 22 updates with 12 conflicts.  after this change: 15 updates with 0 conflicts
	updatedQuotas *lru.Cache

	// TODO these are used together to bucket items by namespace and then batch them up for processing.
	// The technique is valuable for rollup activities to avoid fanout and reduce resource contention.
	// We could move this into a library if another component needed it.
	// queue is indexed by namespace, so that we bundle up on a per-namespace basis
	queue      *workqueue.Type
	workLock   sync.Mutex
	work       map[string][]*admissionWaiter
	dirtyWork  map[string][]*admissionWaiter
	inProgress sets.String
}

type admissionWaiter struct {
	attributes admission.Attributes
	finished   chan struct{}
	result     error
}

type defaultDeny struct{}

func (defaultDeny) Error() string {
	return "DEFAULT DENY"
}

func IsDefaultDeny(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(defaultDeny)
	return ok
}

func newAdmissionWaiter(a admission.Attributes) *admissionWaiter {
	return &admissionWaiter{
		attributes: a,
		finished:   make(chan struct{}),
		result:     defaultDeny{},
	}
}

// newQuotaEvaluator configures an admission controller that can enforce quota constraints
// using the provided registry.  The registry must have the capability to handle group/kinds that
// are persisted by the server this admission controller is intercepting
func newQuotaEvaluator(client clientset.Interface, registry quota.Registry) (*quotaEvaluator, error) {
	liveLookupCache, err := lru.New(100)
	if err != nil {
		return nil, err
	}
	updatedCache, err := lru.New(100)
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
	return &quotaEvaluator{
		client:          client,
		indexer:         indexer,
		registry:        registry,
		liveLookupCache: liveLookupCache,
		liveTTL:         time.Duration(30 * time.Second),
		updatedQuotas:   updatedCache,

		queue:      workqueue.New(),
		work:       map[string][]*admissionWaiter{},
		dirtyWork:  map[string][]*admissionWaiter{},
		inProgress: sets.String{},
	}, nil
}

// Run begins watching and syncing.
func (e *quotaEvaluator) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	for i := 0; i < workers; i++ {
		go wait.Until(e.doWork, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down quota evaluator")
	e.queue.ShutDown()
}

func (e *quotaEvaluator) doWork() {
	workFunc := func() bool {
		ns, admissionAttributes, quit := e.getWork()
		if quit {
			return true
		}
		defer e.completeWork(ns)
		if len(admissionAttributes) == 0 {
			return false
		}
		e.checkAttributes(ns, admissionAttributes)
		return false
	}
	for {
		if quit := workFunc(); quit {
			glog.Infof("quota evaluator worker shutdown")
			return
		}
	}
}

// checkAttributes iterates evaluates all the waiting admissionAttributes.  It will always notify all waiters
// before returning.  The default is to deny.
func (e *quotaEvaluator) checkAttributes(ns string, admissionAttributes []*admissionWaiter) {
	// notify all on exit
	defer func() {
		for _, admissionAttribute := range admissionAttributes {
			close(admissionAttribute.finished)
		}
	}()

	quotas, err := e.getQuotas(ns)
	if err != nil {
		for _, admissionAttribute := range admissionAttributes {
			admissionAttribute.result = err
		}
		return
	}
	if len(quotas) == 0 {
		for _, admissionAttribute := range admissionAttributes {
			admissionAttribute.result = nil
		}
		return
	}

	e.checkQuotas(quotas, admissionAttributes, 3)
}

// checkQuotas checks the admission atttributes against the passed quotas.  If a quota applies, it will attempt to update it
// AFTER it has checked all the admissionAttributes.  The method breaks down into phase like this:
// 0. make a copy of the quotas to act as a "running" quota so we know what we need to update and can still compare against the
//    originals
// 1. check each admission attribute to see if it fits within *all* the quotas.  If it doesn't fit, mark the waiter as failed
//    and the running quota don't change.  If it did fit, check to see if any quota was changed.  It there was no quota change
//    mark the waiter as succeeded.  If some quota did change, update the running quotas
// 2. If no running quota was changed, return now since no updates are needed.
// 3. for each quota that has changed, attempt an update.  If all updates succeeded, update all unset waiters to success status and return.  If the some
//    updates failed on conflict errors and we have retries left, re-get the failed quota from our cache for the latest version
//    and recurse into this method with the subset.  It's safe for us to evaluate ONLY the subset, because the other quota
//    documents for these waiters have already been evaluated.  Step 1, will mark all the ones that should already have succeeded.
func (e *quotaEvaluator) checkQuotas(quotas []api.ResourceQuota, admissionAttributes []*admissionWaiter, remainingRetries int) {
	// yet another copy to compare against originals to see if we actually have deltas
	originalQuotas := make([]api.ResourceQuota, len(quotas), len(quotas))
	copy(originalQuotas, quotas)

	atLeastOneChanged := false
	for i := range admissionAttributes {
		admissionAttribute := admissionAttributes[i]
		newQuotas, err := e.checkRequest(quotas, admissionAttribute.attributes)
		if err != nil {
			admissionAttribute.result = err
			continue
		}

		// if the new quotas are the same as the old quotas, then this particular one doesn't issue any updates
		// that means that no quota docs applied, so it can get a pass
		atLeastOneChangeForThisWaiter := false
		for j := range newQuotas {
			if !quota.Equals(originalQuotas[j].Status.Used, newQuotas[j].Status.Used) {
				atLeastOneChanged = true
				atLeastOneChangeForThisWaiter = true
				break
			}
		}

		if !atLeastOneChangeForThisWaiter {
			admissionAttribute.result = nil
		}
		quotas = newQuotas
	}

	// if none of the requests changed anything, there's no reason to issue an update, just fail them all now
	if !atLeastOneChanged {
		return
	}

	// now go through and try to issue updates.  Things get a little weird here:
	// 1. check to see if the quota changed.  If not, skip.
	// 2. if the quota changed and the update passes, be happy
	// 3. if the quota changed and the update fails, add the original to a retry list
	var updatedFailedQuotas []api.ResourceQuota
	var lastErr error
	for i := range quotas {
		newQuota := quotas[i]
		// if this quota didn't have its status changed, skip it
		if quota.Equals(originalQuotas[i].Status.Used, newQuota.Status.Used) {
			continue
		}

		if updatedQuota, err := e.client.Core().ResourceQuotas(newQuota.Namespace).UpdateStatus(&newQuota); err != nil {
			updatedFailedQuotas = append(updatedFailedQuotas, newQuota)
			lastErr = err

		} else {
			// update our cache
			e.updateCache(updatedQuota)

		}
	}

	if len(updatedFailedQuotas) == 0 {
		// all the updates succeeded.  At this point, anything with the default deny error was just waiting to
		// get a successful update, so we can mark and notify
		for _, admissionAttribute := range admissionAttributes {
			if IsDefaultDeny(admissionAttribute.result) {
				admissionAttribute.result = nil
			}
		}
		return
	}

	// at this point, errors are fatal.  Update all waiters without status to failed and return
	if remainingRetries <= 0 {
		for _, admissionAttribute := range admissionAttributes {
			if IsDefaultDeny(admissionAttribute.result) {
				admissionAttribute.result = lastErr
			}
		}
		return
	}

	// this retry logic has the same bug that its possible to be checking against quota in a state that never actually exists where
	// you've added a new documented, then updated an old one, your resource matches both and you're only checking one
	// updates for these quota names failed.  Get the current quotas in the namespace, compare by name, check to see if the
	// resource versions have changed.  If not, we're going to fall through an fail everything.  If they all have, then we can try again
	newQuotas, err := e.getQuotas(quotas[0].Namespace)
	if err != nil {
		// this means that updates failed.  Anything with a default deny error has failed and we need to let them know
		for _, admissionAttribute := range admissionAttributes {
			if IsDefaultDeny(admissionAttribute.result) {
				admissionAttribute.result = lastErr
			}
		}
		return
	}

	// this logic goes through our cache to find the new version of all quotas that failed update.  If something has been removed
	// it is skipped on this retry.  After all, you removed it.
	quotasToCheck := []api.ResourceQuota{}
	for _, newQuota := range newQuotas {
		for _, oldQuota := range updatedFailedQuotas {
			if newQuota.Name == oldQuota.Name {
				quotasToCheck = append(quotasToCheck, newQuota)
				break
			}
		}
	}
	e.checkQuotas(quotasToCheck, admissionAttributes, remainingRetries-1)
}

// checkRequest verifies that the request does not exceed any quota constraint. it returns back a copy of quotas not yet persisted
// that capture what the usage would be if the request succeeded.  It return an error if the is insufficient quota to satisfy the request
func (e *quotaEvaluator) checkRequest(quotas []api.ResourceQuota, a admission.Attributes) ([]api.ResourceQuota, error) {
	namespace := a.GetNamespace()
	evaluators := e.registry.Evaluators()
	evaluator, found := evaluators[a.GetKind().GroupKind()]
	if !found {
		return quotas, nil
	}

	op := a.GetOperation()
	operationResources := evaluator.OperationResources(op)
	if len(operationResources) == 0 {
		return quotas, nil
	}

	// find the set of quotas that are pertinent to this request
	// reject if we match the quota, but usage is not calculated yet
	// reject if the input object does not satisfy quota constraints
	// if there are no pertinent quotas, we can just return
	inputObject := a.GetObject()
	interestingQuotaIndexes := []int{}
	for i := range quotas {
		resourceQuota := quotas[i]
		match := evaluator.Matches(&resourceQuota, inputObject)
		if !match {
			continue
		}

		hardResources := quota.ResourceNames(resourceQuota.Status.Hard)
		evaluatorResources := evaluator.MatchesResources()
		requiredResources := quota.Intersection(hardResources, evaluatorResources)
		err := evaluator.Constraints(requiredResources, inputObject)
		if err != nil {
			return nil, admission.NewForbidden(a, fmt.Errorf("Failed quota: %s: %v", resourceQuota.Name, err))
		}
		if !hasUsageStats(&resourceQuota) {
			return nil, admission.NewForbidden(a, fmt.Errorf("Status unknown for quota: %s", resourceQuota.Name))
		}

		interestingQuotaIndexes = append(interestingQuotaIndexes, i)
	}
	if len(interestingQuotaIndexes) == 0 {
		return quotas, nil
	}

	// Usage of some resources cannot be counted in isolation. For example when
	// the resource represents a number of unique references to external
	// resource. In such a case an evaluator needs to process other objects in
	// the same namespace which needs to be known.
	if accessor, err := meta.Accessor(inputObject); namespace != "" && err == nil {
		if accessor.GetNamespace() == "" {
			accessor.SetNamespace(namespace)
		}
	}

	// there is at least one quota that definitely matches our object
	// as a result, we need to measure the usage of this object for quota
	// on updates, we need to subtract the previous measured usage
	// if usage shows no change, just return since it has no impact on quota
	deltaUsage := evaluator.Usage(inputObject)
	if admission.Update == op {
		prevItem := a.GetOldObject()
		if prevItem == nil {
			return nil, admission.NewForbidden(a, fmt.Errorf("Unable to get previous usage since prior version of object was not found"))
		}
		prevUsage := evaluator.Usage(prevItem)
		deltaUsage = quota.Subtract(deltaUsage, prevUsage)
	}
	if quota.IsZero(deltaUsage) {
		return quotas, nil
	}

	for _, index := range interestingQuotaIndexes {
		resourceQuota := quotas[index]

		hardResources := quota.ResourceNames(resourceQuota.Status.Hard)
		requestedUsage := quota.Mask(deltaUsage, hardResources)
		newUsage := quota.Add(resourceQuota.Status.Used, requestedUsage)
		if allowed, exceeded := quota.LessThanOrEqual(newUsage, resourceQuota.Status.Hard); !allowed {
			failedRequestedUsage := quota.Mask(requestedUsage, exceeded)
			failedUsed := quota.Mask(resourceQuota.Status.Used, exceeded)
			failedHard := quota.Mask(resourceQuota.Status.Hard, exceeded)
			return nil, admission.NewForbidden(a,
				fmt.Errorf("Exceeded quota: %s, requested: %s, used: %s, limited: %s",
					resourceQuota.Name,
					prettyPrint(failedRequestedUsage),
					prettyPrint(failedUsed),
					prettyPrint(failedHard)))
		}

		// update to the new usage number
		quotas[index].Status.Used = newUsage
	}

	return quotas, nil
}

func (e *quotaEvaluator) evaluate(a admission.Attributes) error {
	waiter := newAdmissionWaiter(a)

	e.addWork(waiter)

	// wait for completion or timeout
	select {
	case <-waiter.finished:
	case <-time.After(10 * time.Second):
		return fmt.Errorf("timeout")
	}

	return waiter.result
}

func (e *quotaEvaluator) addWork(a *admissionWaiter) {
	e.workLock.Lock()
	defer e.workLock.Unlock()

	ns := a.attributes.GetNamespace()
	// this Add can trigger a Get BEFORE the work is added to a list, but this is ok because the getWork routine
	// waits the worklock before retrieving the work to do, so the writes in this method will be observed
	e.queue.Add(ns)

	if e.inProgress.Has(ns) {
		e.dirtyWork[ns] = append(e.dirtyWork[ns], a)
		return
	}

	e.work[ns] = append(e.work[ns], a)
}

func (e *quotaEvaluator) completeWork(ns string) {
	e.workLock.Lock()
	defer e.workLock.Unlock()

	e.queue.Done(ns)
	e.work[ns] = e.dirtyWork[ns]
	delete(e.dirtyWork, ns)
	e.inProgress.Delete(ns)
}

func (e *quotaEvaluator) getWork() (string, []*admissionWaiter, bool) {
	uncastNS, shutdown := e.queue.Get()
	if shutdown {
		return "", []*admissionWaiter{}, shutdown
	}
	ns := uncastNS.(string)

	e.workLock.Lock()
	defer e.workLock.Unlock()
	// at this point, we know we have a coherent view of e.work.  It is entirely possible
	// that our workqueue has another item requeued to it, but we'll pick it up early.  This ok
	// because the next time will go into our dirty list

	work := e.work[ns]
	delete(e.work, ns)
	delete(e.dirtyWork, ns)

	if len(work) != 0 {
		e.inProgress.Insert(ns)
		return ns, work, false
	}

	e.queue.Done(ns)
	e.inProgress.Delete(ns)
	return ns, []*admissionWaiter{}, false
}

func (e *quotaEvaluator) updateCache(quota *api.ResourceQuota) {
	key := quota.Namespace + "/" + quota.Name
	e.updatedQuotas.Add(key, quota)
}

var etcdVersioner = etcd.APIObjectVersioner{}

// checkCache compares the passed quota against the value in the look-aside cache and returns the newer
// if the cache is out of date, it deletes the stale entry.  This only works because of etcd resourceVersions
// being monotonically increasing integers
func (e *quotaEvaluator) checkCache(quota *api.ResourceQuota) *api.ResourceQuota {
	key := quota.Namespace + "/" + quota.Name
	uncastCachedQuota, ok := e.updatedQuotas.Get(key)
	if !ok {
		return quota
	}
	cachedQuota := uncastCachedQuota.(*api.ResourceQuota)

	if etcdVersioner.CompareResourceVersion(quota, cachedQuota) >= 0 {
		e.updatedQuotas.Remove(key)
		return quota
	}
	return cachedQuota
}

func (e *quotaEvaluator) getQuotas(namespace string) ([]api.ResourceQuota, error) {
	// determine if there are any quotas in this namespace
	// if there are no quotas, we don't need to do anything
	items, err := e.indexer.Index("namespace", &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Namespace: namespace, Name: ""}})
	if err != nil {
		return nil, fmt.Errorf("Error resolving quota.")
	}

	// if there are no items held in our indexer, check our live-lookup LRU, if that misses, do the live lookup to prime it.
	if len(items) == 0 {
		lruItemObj, ok := e.liveLookupCache.Get(namespace)
		if !ok || lruItemObj.(liveLookupEntry).expiry.Before(time.Now()) {
			// TODO: If there are multiple operations at the same time and cache has just expired,
			// this may cause multiple List operations being issued at the same time.
			// If there is already in-flight List() for a given namespace, we should wait until
			// it is finished and cache is updated instead of doing the same, also to avoid
			// throttling - see #22422 for details.
			liveList, err := e.client.Core().ResourceQuotas(namespace).List(api.ListOptions{})
			if err != nil {
				return nil, err
			}
			newEntry := liveLookupEntry{expiry: time.Now().Add(e.liveTTL)}
			for i := range liveList.Items {
				newEntry.items = append(newEntry.items, &liveList.Items[i])
			}
			e.liveLookupCache.Add(namespace, newEntry)
			lruItemObj = newEntry
		}
		lruEntry := lruItemObj.(liveLookupEntry)
		for i := range lruEntry.items {
			items = append(items, lruEntry.items[i])
		}
	}

	resourceQuotas := []api.ResourceQuota{}
	for i := range items {
		quota := items[i].(*api.ResourceQuota)
		quota = e.checkCache(quota)
		// always make a copy.  We're going to muck around with this and we should never mutate the originals
		resourceQuotas = append(resourceQuotas, *quota)
	}

	return resourceQuotas, nil
}
