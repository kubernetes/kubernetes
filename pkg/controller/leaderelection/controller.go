/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
	"time"

	"github.com/blang/semver/v4"

	v1 "k8s.io/api/coordination/v1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationv1 "k8s.io/client-go/informers/coordination/v1"
	coordinationv1client "k8s.io/client-go/kubernetes/typed/coordination/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
)

const controllerName = "leader-election-controller" // TODO: make exported?

// TODO: multi-valued labels are problematic.. label matching gets broken.. should we use a label
// per leader lease? Do we need to add a spec field?
const CanLeadLeasesAnnotationName = "coordination.k8s.io/can-lead-leases"

const CompatibilityVersionAnnotationName = "coordination.k8s.io/compatibility-version"
const BinaryVersionAnnotationName = "coordination.k8s.io/binary-version"

const EndOfTermAnnotationName = "coordination.k8s.io/end-of-term"

const ElectedByAnnotationName = "coordination.k8s.io/elected-by" // Value should be set to controllerName

const electionDuration = time.Second * 1

// Controller is the leader election controller, which observes component identity leases for
// components that have self-nominated as candidate leaders for leases and elects leaders
// for those leases, favoring candidates with higher versions.
type Controller struct {
	leaseInformer coordinationv1.LeaseInformer
	leaseQueue    workqueue.RateLimitingInterface
	leaseSynced   cache.InformerSynced
	leaseClient   coordinationv1client.CoordinationV1Interface

	electionCh chan election
}

type election struct {
	leaderLeaseId leaderLeaseId
	electionStart time.Time
}

type leaderLeaseId struct {
	namespace, name string
}

func parseLeaderLeaseId(id string) (leaderLeaseId, error) {
	parts := strings.Split(id, "/")
	if len(parts) != 2 {
		return leaderLeaseId{}, fmt.Errorf("expected single '/' in leader lease identifier but got '%s'", id)
	}
	return leaderLeaseId{parts[0], parts[1]}, nil
}

func (l leaderLeaseId) String() string {
	return l.namespace + "/" + l.name
}

func (c *Controller) Run(ctx context.Context, workers int) {
	klog.Infof("Running")
	defer utilruntime.HandleCrash()

	klog.Infof("Start WaitForNamedCacheSync")
	if !cache.WaitForNamedCacheSync(controllerName, ctx.Done(), c.leaseSynced) {
		klog.Infof("Failed WaitForNamedCacheSync")
		return
	}
	klog.Infof("Done WaitForNamedCacheSync")

	defer c.leaseQueue.ShutDown()
	klog.Infof("Workers: %d", workers)
	for i := 0; i < workers; i++ {
		klog.Infof("Starting worker")
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
	klog.Infof("Done Running")
}

func NewController(leaseInformer coordinationv1.LeaseInformer, leaseClient coordinationv1client.CoordinationV1Interface) (*Controller, error) {
	klog.Infof("NewController")
	c := &Controller{
		leaseInformer: leaseInformer,
		leaseQueue:    workqueue.NewRateLimitingQueueWithConfig(workqueue.DefaultControllerRateLimiter(), workqueue.RateLimitingQueueConfig{Name: controllerName}),
		leaseClient:   leaseClient,

		// TODO: What to do if the size limit is reached?
		// Need to write producer code to handle? It is unsafe to block on send.
		electionCh: make(chan election, 1000),
	}
	reg, err := leaseInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueLease(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.enqueueLease(newObj)
		},
		DeleteFunc: func(oldObj interface{}) {
			c.leaseDeleted(oldObj)
		},
	})
	if err != nil {
		return nil, err
	}
	c.leaseSynced = reg.HasSynced
	return c, nil
}

func (c *Controller) enqueueLease(obj any) {
	if lease, ok := obj.(*v1.Lease); ok {
		// TODO: handle namespaces
		key, err := controller.KeyFunc(lease)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("cannot get name of object %v: %w", lease, err))
		}
		c.leaseQueue.Add(key)
	}
}

func (c *Controller) leaseDeleted(oldObj any) {
	klog.Infof("leaseDeleted")
	if lease, ok := oldObj.(*v1.Lease); ok {
		// TODO: Check if this is a lease that needs a leader election? We may need to maintain
		// a set of leader election leases?  For prototype purposes can we just scan the leases?

		err := c.scheduleElection(leaderLeaseId{namespace: lease.Namespace, name: lease.Name})
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("cannot start election for %v: %w", lease, err))
		}
	}
}

func (c *Controller) runWorker(ctx context.Context) {
	klog.Infof("runWorker")
	go c.runElectionLoop(ctx.Done())
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.leaseQueue.Get()
	if shutdown {
		return false
	}
	defer c.leaseQueue.Done(key)

	err := func() error {
		key, ok := key.(string)
		if !ok {
			return fmt.Errorf("expect a string but got %v", key)
		}
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			return err
		}
		lease, err := c.leaseInformer.Lister().Leases(namespace).Get(name)
		if err != nil {
			if kerrors.IsNotFound(err) {
				klog.Infof("processNextWorkItem lease not found")
				// If not found, the lease is being deleted, do nothing.
				return nil
			}
			return err
		}
		return c.reconcile(ctx, lease)
	}()

	if err == nil {
		c.leaseQueue.Forget(key)
		return true
	}

	utilruntime.HandleError(err)
	klog.Infof("processNextWorkItem.AddRateLimited: %v", key)
	c.leaseQueue.AddRateLimited(key)

	return true
}

func (c *Controller) runElectionLoop(stopCh <-chan struct{}) {
	klog.Infof("runElectionLoop")
	ctx := context.Background()
	for {
		select {
		case <-stopCh:
			return
		case e := <-c.electionCh:
			select {
			case <-stopCh:
				return
			case <-time.After(e.electionStart.Sub(time.Now())): // elections are time ordered in the channel
				err := c.runElection(ctx, e.leaderLeaseId)
				utilruntime.HandleError(err)
			}
		}
	}
}

func (c *Controller) reconcile(ctx context.Context, lease *v1.Lease) error {
	if lease == nil {
		return nil
	}
	klog.Infof("reconcile for lease namespace=%q, name=%q", lease.Namespace, lease.Name)

	if canLead, ok := lease.Annotations[CanLeadLeasesAnnotationName]; ok {
		klog.Infof("reconcile found canLead label namespace=%q, name=%q: %q", lease.Namespace, lease.Name, canLead)
		for _, leadeLeaseId := range strings.Split(canLead, ",") {
			leaderLeaseId, err := parseLeaderLeaseId(leadeLeaseId)
			if err != nil {
				return err
			}

			// Check if the lease has a current leader, and if that leader is an ideal leader
			leader, ok, err := c.activeLeader(ctx, leaderLeaseId)
			if err != nil {
				return err
			}
			if ok {
				klog.Infof("finding candidates for lease namespace=%q, name=%q", lease.Namespace, lease.Name)
				candidates, err := c.listCandidates(leaderLeaseId)
				if err != nil {
					return err
				}
				if !shouldReelect(candidates, leader) {
					klog.Infof("shouldReelect returned false")
					continue
				}
			}

			err = c.scheduleElection(leaderLeaseId)
			if err != nil {
				return err
			}
		}
	} else {
		isExpired := isLeaseExpired(lease)
		clone := lease.DeepCopy()
		if isExpired && lease.Annotations[ElectedByAnnotationName] == controllerName && lease.Spec.HolderIdentity != nil && clone.Spec.RenewTime != nil && clone.Spec.LeaseDurationSeconds != nil && clone.Spec.AcquireTime != nil {
			delete(clone.Annotations, EndOfTermAnnotationName)
			clone.ObjectMeta.Annotations[ElectedByAnnotationName] = controllerName
			clone.Spec.HolderIdentity = nil
			clone.Spec.RenewTime = nil
			clone.Spec.LeaseDurationSeconds = nil
			clone.Spec.AcquireTime = nil
		}
		_, err := c.leaseClient.Leases(clone.Namespace).Update(ctx, clone, metav1.UpdateOptions{})
		if err != nil {
			return err
		}
		err = c.scheduleElection(leaderLeaseId{namespace: lease.Namespace, name: lease.Name})
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *Controller) activeLeader(ctx context.Context, leaderLeaseId leaderLeaseId) (*v1.Lease, bool, error) {
	klog.Infof("activeLeader checking for lease namespace=%q, name=%q", leaderLeaseId.namespace, leaderLeaseId.name)
	leaderLease, err := c.leaseInformer.Lister().Leases(leaderLeaseId.namespace).Get(leaderLeaseId.name)
	if err != nil {
		if kerrors.IsNotFound(err) {
			klog.Infof("activeLeader not found for lease namespace=%q, name=%q", leaderLeaseId.namespace, leaderLeaseId.name)
			return nil, false, nil
		} else {
			return nil, false, err
		}
	}
	holder := leaderLease.Spec.HolderIdentity
	if holder == nil {
		return nil, false, nil
	}
	// TODO: What namespace to use?
	holderIdentityLease, err := c.leaseInformer.Lister().Leases(leaderLeaseId.namespace).Get(*holder)
	if err != nil {
		if kerrors.IsNotFound(err) {
			klog.Infof("activeLeader holder identity not found for lease namespace=%q, name=%q", leaderLeaseId.namespace, leaderLeaseId.name)
			return nil, false, nil
		} else {
			return nil, false, err
		}
	}

	return holderIdentityLease, !isLeaseExpired(leaderLease), nil
}

func (c *Controller) scheduleElection(leaderLeaseId leaderLeaseId) error {
	klog.Infof("scheduleElection")
	// TODO: add a set to track ongoing elections and avoid requesting an election if one has already been kicked off?
	// This might not be needed..
	c.electionCh <- election{
		leaderLeaseId: leaderLeaseId,
		electionStart: time.Now(),
	}
	return nil
}

func (c *Controller) runElection(ctx context.Context, leaderLeaseId leaderLeaseId) error {
	klog.Infof("runElection %q %q", leaderLeaseId.namespace, leaderLeaseId.name)
	candidates, err := c.listCandidates(leaderLeaseId)
	if err != nil {
		return err
	}

	electee := pickLeader(candidates)
	if electee == nil {
		return nil
	}

	klog.Infof("pickLeader found %q %q", electee.Namespace, electee.Name)

	klog.Infof("Creating lease %q %q for %q", leaderLeaseId.namespace, leaderLeaseId.name, electee.Spec.HolderIdentity)
	// create the leader election lease
	leaderLease := &v1.Lease{
		// TODO: fill out all lease fields
		ObjectMeta: metav1.ObjectMeta{
			Namespace: leaderLeaseId.namespace,
			Name:      leaderLeaseId.name,
			Annotations: map[string]string{
				ElectedByAnnotationName: controllerName,
			},
		},
		Spec: v1.LeaseSpec{
			HolderIdentity: electee.Spec.HolderIdentity,
		},
	}
	_, err = c.leaseClient.Leases(leaderLeaseId.namespace).Create(ctx, leaderLease, metav1.CreateOptions{})
	if err != nil {
		if kerrors.IsAlreadyExists(err) {
			lease, err := c.leaseClient.Leases(leaderLeaseId.namespace).Get(ctx, leaderLeaseId.name, metav1.GetOptions{})
			if err != nil {
				return err
			}

			// If the lease has expired and the holder identity is inaccurate, reset it
			isExpired := isLeaseExpired(lease)
			if isExpired && electee.Spec.HolderIdentity != nil && (lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != *electee.Spec.HolderIdentity) {
				klog.Infof("lease %q %q is expired, resetting it", leaderLeaseId.namespace, leaderLeaseId.name)
				delete(lease.Annotations, EndOfTermAnnotationName)
				lease.ObjectMeta.Annotations[ElectedByAnnotationName] = controllerName
				lease.Spec.HolderIdentity = electee.Spec.HolderIdentity

				// TODO: we don't really need to clear these, right?
				lease.Spec.RenewTime = nil
				lease.Spec.LeaseDurationSeconds = nil
				lease.Spec.AcquireTime = nil
				_, err = c.leaseClient.Leases(leaderLeaseId.namespace).Update(ctx, lease, metav1.UpdateOptions{})
				if err != nil {
					return err
				}
			} else if lease.Spec.HolderIdentity != nil && electee.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != *electee.Spec.HolderIdentity {
				klog.Infof("lease %q %q already exists for holder %q but should be held by %q, marking end of term", leaderLeaseId.namespace, leaderLeaseId.name, *lease.Spec.HolderIdentity, *electee.Spec.HolderIdentity)
				lease.Annotations[EndOfTermAnnotationName] = "true"
				_, err = c.leaseClient.Leases(leaderLeaseId.namespace).Update(ctx, lease, metav1.UpdateOptions{})
				if err != nil {
					return err
				}

				// Hack: force reconcilation
				key, err := controller.KeyFunc(lease)
				if err != nil {
					utilruntime.HandleError(fmt.Errorf("cannot get name of object %v: %w", lease, err))
				}
				c.leaseQueue.AddRateLimited(key)
			}

			return nil
		}
		return err
	}
	return nil
}

func pickLeader(candidates []*v1.Lease) *v1.Lease {
	var electee *v1.Lease
	for _, c := range candidates {
		if electee == nil || compare(electee, c) > 0 {
			electee = c
		}
	}
	if electee == nil {
		klog.Infof("pickLeader: none found")
	} else {
		klog.Infof("pickLeader: %s %s", electee.Namespace, electee.Name)
	}
	return electee
}

func shouldReelect(candidates []*v1.Lease, currentLeader *v1.Lease) bool {
	klog.Infof("shouldReelect for candidates: %+v", candidates)
	pickedLeader := pickLeader(candidates)
	if pickedLeader == nil {
		return false
	}
	return compare(currentLeader, pickedLeader) > 0
}

func getCompatibilityVersion(l *v1.Lease) semver.Version {
	if value, ok := l.Annotations[CompatibilityVersionAnnotationName]; ok {
		v, err := semver.ParseTolerant(value)
		if err != nil {
			return semver.Version{}
		}
		return v
	}
	return semver.Version{}
}

func getBinaryVersion(l *v1.Lease) semver.Version {
	if value, ok := l.Annotations[BinaryVersionAnnotationName]; ok {
		v, err := semver.ParseTolerant(value)
		if err != nil {
			return semver.Version{}
		}
		return v
	}
	return semver.Version{}
}

func compare(lhs, rhs *v1.Lease) int {
	lhsVersion := getCompatibilityVersion(lhs)
	rhsVersion := getCompatibilityVersion(rhs)
	result := lhsVersion.Compare(rhsVersion)
	if result == 0 {
		lhsVersion := getBinaryVersion(lhs)
		rhsVersion := getBinaryVersion(rhs)
		result = lhsVersion.Compare(rhsVersion)
	}
	return result
}

func (c *Controller) listCandidates(leaderLeaseId leaderLeaseId) ([]*v1.Lease, error) {
	leases, err := c.leaseInformer.Lister().Leases(leaderLeaseId.namespace).List(labels.Everything()) // TODO: somwhow filter
	if err != nil {
		return nil, err
	}
	var results []*v1.Lease
	for _, l := range leases {
		if canLead(l, leaderLeaseId.String()) && !isLeaseExpired(l) {
			results = append(results, l)
		}
	}
	return results, nil
}

func canLead(lease *v1.Lease, leaderLeaseName string) bool {
	for _, l := range listCanLead(lease) {
		if l == leaderLeaseName {
			return true
		}
	}
	return false
}

func listCanLead(lease *v1.Lease) []string {
	if lease == nil {
		return nil
	}
	if canLead, ok := lease.Annotations[CanLeadLeasesAnnotationName]; ok {
		return strings.Split(canLead, ",")
	}
	return nil
}

func isLeaseExpired(lease *v1.Lease) bool {
	currentTime := time.Now()
	// Leases created by the apiserver lease controller should have non-nil renew time
	// and lease duration set. Leases without these fields set are invalid and should
	// be GC'ed.
	return lease.Spec.RenewTime == nil ||
		lease.Spec.LeaseDurationSeconds == nil ||
		lease.Spec.RenewTime.Add(time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second).Before(currentTime)
}
