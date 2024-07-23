/*
Copyright 2024 The Kubernetes Authors.

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

package leaderelection

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationv1informers "k8s.io/client-go/informers/coordination/v1"
	coordinationv1alpha1 "k8s.io/client-go/informers/coordination/v1alpha1"
	coordinationv1client "k8s.io/client-go/kubernetes/typed/coordination/v1"
	coordinationv1alpha1client "k8s.io/client-go/kubernetes/typed/coordination/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

const (
	controllerName = "leader-election-controller"

	// Requeue interval is the interval at which a Lease is requeued to verify that it is being renewed properly.
	defaultRequeueInterval            = 5 * time.Second
	noRequeue                         = 0
	defaultLeaseDurationSeconds int32 = 5

	electionDuration = 5 * time.Second

	leaseCandidateValidDuration = 30 * time.Minute
)

// Controller is the leader election controller, which observes component identity leases for
// components that have self-nominated as candidate leaders for leases and elects leaders
// for those leases, favoring candidates with higher versions.
type Controller struct {
	leaseInformer     coordinationv1informers.LeaseInformer
	leaseClient       coordinationv1client.CoordinationV1Interface
	leaseRegistration cache.ResourceEventHandlerRegistration

	leaseCandidateInformer     coordinationv1alpha1.LeaseCandidateInformer
	leaseCandidateClient       coordinationv1alpha1client.CoordinationV1alpha1Interface
	leaseCandidateRegistration cache.ResourceEventHandlerRegistration

	queue workqueue.TypedRateLimitingInterface[types.NamespacedName]
}

func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer func() {
		err := c.leaseInformer.Informer().RemoveEventHandler(c.leaseRegistration)
		if err != nil {
			klog.Warning("error removing leaseInformer eventhandler")
		}
		err = c.leaseCandidateInformer.Informer().RemoveEventHandler(c.leaseCandidateRegistration)
		if err != nil {
			klog.Warning("error removing leaseCandidateInformer eventhandler")
		}
	}()

	if !cache.WaitForNamedCacheSync(controllerName, ctx.Done(), c.leaseRegistration.HasSynced, c.leaseCandidateRegistration.HasSynced) {
		return
	}

	// This controller is leader elected and may start after informers have already started. List on startup.
	lcs, err := c.leaseCandidateInformer.Lister().List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	for _, lc := range lcs {
		c.processCandidate(lc)
	}

	klog.Infof("Workers: %d", workers)
	for i := 0; i < workers; i++ {
		klog.Infof("Starting worker")
		go wait.UntilWithContext(ctx, c.runElectionWorker, time.Second)
	}
	<-ctx.Done()
}

func NewController(leaseInformer coordinationv1informers.LeaseInformer, leaseCandidateInformer coordinationv1alpha1.LeaseCandidateInformer, leaseClient coordinationv1client.CoordinationV1Interface, leaseCandidateClient coordinationv1alpha1client.CoordinationV1alpha1Interface) (*Controller, error) {
	c := &Controller{
		leaseInformer:          leaseInformer,
		leaseCandidateInformer: leaseCandidateInformer,
		leaseClient:            leaseClient,
		leaseCandidateClient:   leaseCandidateClient,

		queue: workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[types.NamespacedName](), workqueue.TypedRateLimitingQueueConfig[types.NamespacedName]{Name: controllerName}),
	}
	leaseSynced, err := leaseInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.processLease(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.processLease(newObj)
		},
		DeleteFunc: func(oldObj interface{}) {
			c.processLease(oldObj)
		},
	})

	if err != nil {
		return nil, err
	}
	leaseCandidateSynced, err := leaseCandidateInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.processCandidate(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.processCandidate(newObj)
		},
		DeleteFunc: func(oldObj interface{}) {
			c.processCandidate(oldObj)
		},
	})

	if err != nil {
		return nil, err
	}
	c.leaseRegistration = leaseSynced
	c.leaseCandidateRegistration = leaseCandidateSynced
	return c, nil
}

func (c *Controller) runElectionWorker(ctx context.Context) {
	for c.processNextElectionItem(ctx) {
	}
}

func (c *Controller) processNextElectionItem(ctx context.Context) bool {
	key, shutdown := c.queue.Get()
	if shutdown {
		return false
	}

	intervalForRequeue, err := c.reconcileElectionStep(ctx, key)
	utilruntime.HandleError(err)
	if intervalForRequeue != noRequeue {
		defer c.queue.AddAfter(key, intervalForRequeue)
	}
	c.queue.Done(key)
	return true
}

func (c *Controller) processCandidate(obj any) {
	lc, ok := obj.(*v1alpha1.LeaseCandidate)
	if !ok {
		return
	}
	if lc == nil {
		return
	}
	// Ignore candidates that transitioned to Pending because reelection is already in progress
	if lc.Spec.PingTime != nil {
		return
	}
	c.queue.Add(types.NamespacedName{Namespace: lc.Namespace, Name: lc.Spec.LeaseName})
}

func (c *Controller) processLease(obj any) {
	lease, ok := obj.(*v1.Lease)
	if !ok {
		return
	}
	c.queue.Add(types.NamespacedName{Namespace: lease.Namespace, Name: lease.Name})
}

func (c *Controller) electionNeeded(candidates []*v1alpha1.LeaseCandidate, leaseNN types.NamespacedName) (bool, error) {
	lease, err := c.leaseInformer.Lister().Leases(leaseNN.Namespace).Get(leaseNN.Name)
	if err != nil && !apierrors.IsNotFound(err) {
		return false, fmt.Errorf("error reading lease")
	} else if apierrors.IsNotFound(err) {
		return true, nil
	}

	if isLeaseExpired(lease) || lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity == "" {
		return true, nil
	}

	for _, candidate := range candidates {
		if candidate.Spec.RenewTime != nil && candidate.Spec.RenewTime.Add(leaseCandidateValidDuration/2).Before(time.Now()) {
			return true, nil
		}
	}

	prelimStrategy, err := pickBestStrategy(candidates)
	if err != nil {
		return false, err
	}
	if prelimStrategy != v1.OldestEmulationVersion {
		klog.V(2).Infof("strategy %s is not recognized by CLE.", prelimStrategy)
		return false, nil
	}

	prelimElectee := pickBestLeaderOldestEmulationVersion(candidates)
	if prelimElectee == nil {
		return false, nil
	} else if lease != nil && lease.Spec.HolderIdentity != nil && prelimElectee.Name == *lease.Spec.HolderIdentity {
		klog.V(2).Infof("Leader %s is already most optimal for lease %s %s", prelimElectee.Name, lease.Namespace, lease.Name)
		return false, nil
	}
	return true, nil
}

// reconcileElectionStep steps through a step in an election.
// A step looks at the current state of Lease and LeaseCandidates and takes one of the following action
// - do nothing (because leader is already optimal or still waiting for an event)
// - request ack from candidates (update LeaseCandidate PingTime)
// - finds the most optimal candidate and elect (update the Lease object)
// Instead of keeping a map and lock on election, the state is
// calculated every time by looking at the lease, and set of available candidates.
// PingTime + electionDuration > time.Now: We just asked all candidates to ack and are still waiting for response
// PingTime + electionDuration < time.Now: Candidate has not responded within the appropriate PingTime. Continue the election.
// RenewTime + 5 seconds > time.Now: All candidates acked in the last 5 seconds, continue the election.
func (c *Controller) reconcileElectionStep(ctx context.Context, leaseNN types.NamespacedName) (requeue time.Duration, err error) {
	now := time.Now()

	candidates, err := c.listAdmissableCandidates(leaseNN)
	if err != nil {
		return defaultRequeueInterval, err
	} else if len(candidates) == 0 {
		return noRequeue, nil
	}
	klog.V(4).Infof("reconcileElectionStep %s, candidates: %d", leaseNN, len(candidates))

	// Check if an election is really needed by looking at the current lease
	// and set of candidates
	needElection, err := c.electionNeeded(candidates, leaseNN)
	if !needElection {
		return noRequeue, err
	}
	if err != nil {
		return defaultRequeueInterval, err
	}

	fastTrackElection := false

	for _, candidate := range candidates {
		// If a candidate has a PingTime within the election duration, they have not acked
		// and we should wait until we receive their response
		if candidate.Spec.PingTime != nil {
			if candidate.Spec.PingTime.Add(electionDuration).After(now) {
				// continue waiting for the election to timeout
				return noRequeue, nil
			} else {
				// election timed out without ack. Clear and start election.
				fastTrackElection = true
				clone := candidate.DeepCopy()
				clone.Spec.PingTime = nil
				_, err := c.leaseCandidateClient.LeaseCandidates(clone.Namespace).Update(ctx, clone, metav1.UpdateOptions{})
				if err != nil {
					return noRequeue, err
				}
				break
			}
		}
	}

	if !fastTrackElection {
		continueElection := true
		for _, candidate := range candidates {
			// if renewTime of a candidate is longer than electionDuration old, we have to ping.
			if candidate.Spec.RenewTime != nil && candidate.Spec.RenewTime.Add(electionDuration).Before(now) {
				continueElection = false
				break
			}
		}
		if !continueElection {
			// Send an "are you alive" signal to all candidates
			for _, candidate := range candidates {
				clone := candidate.DeepCopy()
				clone.Spec.PingTime = &metav1.MicroTime{Time: time.Now()}
				_, err := c.leaseCandidateClient.LeaseCandidates(clone.Namespace).Update(ctx, clone, metav1.UpdateOptions{})
				if err != nil {
					return noRequeue, err
				}
			}
			return defaultRequeueInterval, nil
		}
	}

	var ackedCandidates []*v1alpha1.LeaseCandidate
	for _, candidate := range candidates {
		if candidate.Spec.RenewTime.Add(electionDuration).After(now) {
			ackedCandidates = append(ackedCandidates, candidate)
		}
	}
	if len(ackedCandidates) == 0 {
		return noRequeue, fmt.Errorf("no available candidates")
	}

	strategy, err := pickBestStrategy(ackedCandidates)
	if err != nil {
		return noRequeue, err
	}

	if strategy != v1.OldestEmulationVersion {
		klog.V(2).Infof("strategy %s is not recognized by CLE.", strategy)
		return noRequeue, nil
	}
	electee := pickBestLeaderOldestEmulationVersion(ackedCandidates)

	if electee == nil {
		return noRequeue, fmt.Errorf("should not happen, could not find suitable electee")
	}

	electeeName := electee.Name
	// create the leader election lease
	leaderLease := &v1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: leaseNN.Namespace,
			Name:      leaseNN.Name,
		},
		Spec: v1.LeaseSpec{
			HolderIdentity:       &electeeName,
			Strategy:             &strategy,
			LeaseDurationSeconds: ptr.To(defaultLeaseDurationSeconds),
			RenewTime:            &metav1.MicroTime{Time: time.Now()},
		},
	}
	_, err = c.leaseClient.Leases(leaseNN.Namespace).Create(ctx, leaderLease, metav1.CreateOptions{})
	// If the create was successful, then we can return here.
	if err == nil {
		klog.Infof("Created lease %s for %q", leaseNN, electee.Name)
		return defaultRequeueInterval, nil
	}

	// If there was an error, return
	if !apierrors.IsAlreadyExists(err) {
		return noRequeue, err
	}

	existingLease, err := c.leaseClient.Leases(leaseNN.Namespace).Get(ctx, leaseNN.Name, metav1.GetOptions{})
	if err != nil {
		return noRequeue, err
	}
	leaseClone := existingLease.DeepCopy()

	// Update the Lease if it either does not have a holder or is expired
	isExpired := isLeaseExpired(existingLease)
	if leaseClone.Spec.HolderIdentity == nil || *leaseClone.Spec.HolderIdentity == "" || (isExpired && *leaseClone.Spec.HolderIdentity != electeeName) {
		klog.Infof("lease %s is expired, resetting it and setting holder to %q", leaseNN, electee.Name)
		leaseClone.Spec.Strategy = &strategy
		leaseClone.Spec.PreferredHolder = nil
		leaseClone.Spec.HolderIdentity = &electeeName

		leaseClone.Spec.RenewTime = &metav1.MicroTime{Time: time.Now()}
		leaseClone.Spec.LeaseDurationSeconds = ptr.To(defaultLeaseDurationSeconds)
		leaseClone.Spec.AcquireTime = nil
		_, err = c.leaseClient.Leases(leaseNN.Namespace).Update(ctx, leaseClone, metav1.UpdateOptions{})
		if err != nil {
			return time.Until(leaseClone.Spec.RenewTime.Time), err
		}
	} else if leaseClone.Spec.HolderIdentity != nil && *leaseClone.Spec.HolderIdentity != electeeName {
		klog.Infof("lease %s already exists for holder %q but should be held by %q, marking preferredHolder", leaseNN, *leaseClone.Spec.HolderIdentity, electee.Name)
		leaseClone.Spec.PreferredHolder = &electeeName
		leaseClone.Spec.Strategy = &strategy
		_, err = c.leaseClient.Leases(leaseNN.Namespace).Update(ctx, leaseClone, metav1.UpdateOptions{})
		if err != nil {
			return noRequeue, err
		}
		return time.Until(leaseClone.Spec.RenewTime.Time), nil
	}
	return defaultRequeueInterval, nil
}

func (c *Controller) listAdmissableCandidates(leaseNN types.NamespacedName) ([]*v1alpha1.LeaseCandidate, error) {
	leases, err := c.leaseCandidateInformer.Lister().LeaseCandidates(leaseNN.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	var results []*v1alpha1.LeaseCandidate
	for _, l := range leases {
		if l.Spec.LeaseName != leaseNN.Name {
			continue
		}
		if !isLeaseCandidateExpired(l) {
			results = append(results, l)
		} else {
			klog.Infof("LeaseCandidate %s is expired", l.Name)
		}
	}
	return results, nil
}
