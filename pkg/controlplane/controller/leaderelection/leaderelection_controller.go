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
	"strings"
	"time"

	"github.com/blang/semver/v4"

	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationv1 "k8s.io/client-go/informers/coordination/v1"
	coordinationv1alpha1 "k8s.io/client-go/informers/coordination/v1alpha1"
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

// Controller is the leader election controller, which observes component identity leases for
// components that have self-nominated as candidate leaders for leases and elects leaders
// for those leases, favoring candidates with higher versions.
type Controller struct {
	leaseInformer coordinationv1.LeaseInformer
	leaseQueue    workqueue.TypedRateLimitingInterface[string]
	leaseSynced   cache.InformerSynced
	leaseClient   coordinationv1client.CoordinationV1Interface

	identityLeaseInformer coordinationv1alpha1.IdentityLeaseInformer
	// Maybe be useful if identity leases are written to in the future
	// identityLeaseQueue    workqueue.RateLimitingInterface
	// identityLeaseClient   coordinationv1alpha1client.CoordinationV1alpha1Interface
	// identityLeaseSynced   cache.InformerSynced

	electionCh chan election
}

type election struct {
	leaderLeaseID leaderLeaseID
	electionStart time.Time
}

type leaderLeaseID struct {
	namespace, name string
}

func parseLeaderLeaseID(id string) (leaderLeaseID, error) {
	parts := strings.Split(id, "/")
	if len(parts) != 2 {
		return leaderLeaseID{}, fmt.Errorf("expected single '/' in leader lease identifier but got '%s'", id)
	}
	return leaderLeaseID{parts[0], parts[1]}, nil
}

func (l leaderLeaseID) String() string {
	return l.namespace + "/" + l.name
}

func (c *Controller) Sync(ctx context.Context) {
	if !cache.WaitForNamedCacheSync(controllerName, ctx.Done(), c.leaseSynced) {
		return
	}
}

func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	defer c.leaseQueue.ShutDown()
	klog.Infof("Workers: %d", workers)
	for i := 0; i < workers; i++ {
		klog.Infof("Starting worker")
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func NewController(leaseInformer coordinationv1.LeaseInformer, identityLeaseInformer coordinationv1alpha1.IdentityLeaseInformer, leaseClient coordinationv1client.CoordinationV1Interface) (*Controller, error) {
	klog.Infof("NewController")
	c := &Controller{
		leaseInformer:         leaseInformer,
		identityLeaseInformer: identityLeaseInformer,
		leaseQueue:            workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[string](), workqueue.TypedRateLimitingQueueConfig[string]{Name: controllerName}),
		leaseClient:           leaseClient,

		// TODO: What to do if the size limit is reached?
		// realistically we'd have a lot less than 1000 leases that all need to be updated at the same time.
		// Is there any harm to naively spinning a goroutine for each election?
		// I don't estimate the load on the apiserver to be that much
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
	_, err = identityLeaseInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueIdentity(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			c.enqueueIdentity(newObj)
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

func (c *Controller) enqueueIdentity(obj any) {
	if lease, ok := obj.(*v1alpha1.IdentityLease); ok {
		// TODO: handle namespaces
		// key, err := controller.KeyFunc(lease)
		// if err != nil {
		// 	utilruntime.HandleError(fmt.Errorf("cannot get name of object %v: %w", lease, err))
		// }
		_ = c.reconcileIdentityLease(lease)
		// How to handle invalid lease?
	}
}

func (c *Controller) leaseDeleted(oldObj any) {
	klog.Infof("leaseDeleted")
	if lease, ok := oldObj.(*v1.Lease); ok {
		// TODO: Check if this is a lease that needs a leader election? We may need to maintain
		// a set of leader election leases?  For prototype purposes can we just scan the leases?

		err := c.scheduleElection(leaderLeaseID{namespace: lease.Namespace, name: lease.Name})
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
		return c.reconcile(ctx, key)
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
			// TODO; Isn't this always true? Is this for dealing with time drift?
			case <-time.After(time.Until(e.electionStart)): // elections are time ordered in the channel
				err := c.runElection(ctx, e.leaderLeaseID)
				utilruntime.HandleError(err)
			}
		}
	}
}

func (c *Controller) reconcileIdentityLease(lease *v1alpha1.IdentityLease) error {
	canLead := lease.Spec.CanLeadLease
	if canLead == "" {
		return nil
	}
	klog.Infof("reconcile found canLead label namespace=%q, name=%q: %q", lease.Namespace, lease.Name, canLead)
	leaderLeaseID, err := parseLeaderLeaseID(canLead)
	if err != nil {
		return err
	}
	klog.Infof("Adding lease %s", leaderLeaseID.String())
	c.leaseQueue.Add(leaderLeaseID.String())
	return nil
}

func (c *Controller) reconcile(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	lease, err := c.leaseInformer.Lister().Leases(namespace).Get(name)

	if err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
	}
	if lease != nil {
		if lease.Annotations[ElectedByAnnotationName] == controllerName {
			klog.Infof("reconcile for lease namespace=%q, name=%q", lease.Namespace, lease.Name)
		} else if isLeaseExpired(lease) {
			clone := lease.DeepCopy()
			if lease.Annotations[ElectedByAnnotationName] == controllerName && lease.Spec.HolderIdentity != nil && clone.Spec.RenewTime != nil && clone.Spec.LeaseDurationSeconds != nil && clone.Spec.AcquireTime != nil {
				delete(clone.Annotations, EndOfTermAnnotationName)
				clone.ObjectMeta.Annotations[ElectedByAnnotationName] = controllerName
				clone.Spec.HolderIdentity = nil
				clone.Spec.RenewTime = nil
				clone.Spec.LeaseDurationSeconds = nil
				clone.Spec.AcquireTime = nil
				_, err := c.leaseClient.Leases(clone.Namespace).Update(ctx, clone, metav1.UpdateOptions{})
				if err != nil {
					return err
				}
			}
			return nil
		}
	}

	err = c.scheduleElection(leaderLeaseID{namespace: namespace, name: name})
	if err != nil {
		return err
	}
	return nil
}

// func (c *Controller) activeLeader(ctx context.Context, leaderLeaseID leaderLeaseID) (*v1alpha1.IdentityLease, bool, error) {
// 	leaderLease, err := c.leaseInformer.Lister().Leases(leaderLeaseID.namespace).Get(leaderLeaseID.name)
// 	if err != nil {
// 		if apierrors.IsNotFound(err) {
// 			klog.Infof("activeLeader not found for lease namespace=%q, name=%q", leaderLeaseID.namespace, leaderLeaseID.name)
// 			return nil, false, nil
// 		} else {
// 			return nil, false, err
// 		}
// 	}
// 	holder := leaderLease.Spec.HolderIdentity
// 	if holder == nil {
// 		return nil, false, nil
// 	}
// 	// TODO: What namespace to use?
// 	holderIdentityLease, err := c.identityLeaseInformer.Lister().IdentityLeases(leaderLeaseID.namespace).Get(*holder)
// 	if err != nil {
// 		if apierrors.IsNotFound(err) {
// 			klog.Infof("activeLeader holder identity not found for lease namespace=%q, name=%q", leaderLeaseID.namespace, leaderLeaseID.name)
// 			return nil, false, nil
// 		} else {
// 			return nil, false, err
// 		}
// 	}

// 	return holderIdentityLease, !isLeaseExpired(leaderLease), nil
// }

func (c *Controller) scheduleElection(leaderLeaseID leaderLeaseID) error {
	klog.Infof("scheduleElection")
	// TODO: add a set to track ongoing elections and avoid requesting an election if one has already been kicked off?
	// This might not be needed..
	c.electionCh <- election{
		leaderLeaseID: leaderLeaseID,
		electionStart: time.Now(),
	}
	return nil
}

func (c *Controller) runElection(ctx context.Context, leaderLeaseID leaderLeaseID) error {
	klog.Infof("runElection %q %q", leaderLeaseID.namespace, leaderLeaseID.name)
	candidates, err := c.listCandidates(leaderLeaseID)
	if err != nil {
		return err
	}

	electee := pickBestLeader(candidates)
	if electee == nil {
		return nil
	}

	klog.Infof("pickBestLeader found %q %q", electee.Namespace, electee.Name)

	// TODO: Is taking the pointer safe
	klog.Infof("Creating lease %q %q for %q", leaderLeaseID.namespace, leaderLeaseID.name, *electee.Spec.HolderIdentity)
	// create the leader election lease
	leaderLease := &v1.Lease{
		// TODO: fill out all lease fields
		ObjectMeta: metav1.ObjectMeta{
			Namespace: leaderLeaseID.namespace,
			Name:      leaderLeaseID.name,
			Annotations: map[string]string{
				ElectedByAnnotationName: controllerName,
			},
		},
		Spec: v1.LeaseSpec{
			HolderIdentity: electee.Spec.HolderIdentity,
		},
	}
	_, err = c.leaseClient.Leases(leaderLeaseID.namespace).Create(ctx, leaderLease, metav1.CreateOptions{})
	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			lease, err := c.leaseClient.Leases(leaderLeaseID.namespace).Get(ctx, leaderLeaseID.name, metav1.GetOptions{})
			if err != nil {
				return err
			}

			// If the lease has expired and the holder identity is inaccurate, reset it
			isExpired := isLeaseExpired(lease)
			if isExpired && electee.Spec.HolderIdentity != nil && (lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != *electee.Spec.HolderIdentity) {
				klog.Infof("lease %q %q is expired, resetting it", leaderLeaseID.namespace, leaderLeaseID.name)
				delete(lease.Annotations, EndOfTermAnnotationName)
				lease.ObjectMeta.Annotations[ElectedByAnnotationName] = controllerName
				lease.Spec.HolderIdentity = electee.Spec.HolderIdentity

				// TODO: we don't really need to clear these, right?
				lease.Spec.RenewTime = nil
				lease.Spec.LeaseDurationSeconds = nil
				lease.Spec.AcquireTime = nil
				_, err = c.leaseClient.Leases(leaderLeaseID.namespace).Update(ctx, lease, metav1.UpdateOptions{})
				if err != nil {
					return err
				}
			} else if lease.Spec.HolderIdentity != nil && electee.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != *electee.Spec.HolderIdentity {
				klog.Infof("lease %q %q already exists for holder %q but should be held by %q, marking end of term", leaderLeaseID.namespace, leaderLeaseID.name, *lease.Spec.HolderIdentity, *electee.Spec.HolderIdentity)
				if lease.Annotations == nil {
					lease.Annotations = make(map[string]string)
				}
				lease.Annotations[EndOfTermAnnotationName] = "true"
				_, err = c.leaseClient.Leases(leaderLeaseID.namespace).Update(ctx, lease, metav1.UpdateOptions{})
				if err != nil {
					return err
				}

				// Hack: force reconciliation
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

func pickBestLeader(candidates []*v1alpha1.IdentityLease) *v1alpha1.IdentityLease {
	var electee *v1alpha1.IdentityLease
	for _, c := range candidates {
		if electee == nil || compare(electee, c) > 0 {
			electee = c
		}
	}
	if electee == nil {
		klog.Infof("pickBestLeader: none found")
	} else {
		klog.Infof("pickBestLeader: %s %s", electee.Namespace, electee.Name)
	}
	return electee
}

func shouldReelect(candidates []*v1alpha1.IdentityLease, currentLeader *v1alpha1.IdentityLease) bool {
	klog.Infof("shouldReelect for candidates: %+v", candidates)
	pickedLeader := pickBestLeader(candidates)
	if pickedLeader == nil {
		return false
	}
	return compare(currentLeader, pickedLeader) > 0
}

func getCompatibilityVersion(l *v1alpha1.IdentityLease) semver.Version {
	value := l.Spec.CompatibilityVersion
	v, err := semver.ParseTolerant(value)
	if err != nil {
		return semver.Version{}
	}
	return v
}

func getBinaryVersion(l *v1alpha1.IdentityLease) semver.Version {
	value := l.Spec.BinaryVersion
	v, err := semver.ParseTolerant(value)
	if err != nil {
		return semver.Version{}
	}
	return v
}

func compare(lhs, rhs *v1alpha1.IdentityLease) int {
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

func (c *Controller) listCandidates(leaderLeaseID leaderLeaseID) ([]*v1alpha1.IdentityLease, error) {
	leases, err := c.identityLeaseInformer.Lister().IdentityLeases(leaderLeaseID.namespace).List(labels.Everything()) // TODO: somwhow filter
	klog.Infof("total candidates %d", len(leases))
	if err != nil {
		return nil, err
	}
	var results []*v1alpha1.IdentityLease

	for _, l := range leases {
		if canLead(l, leaderLeaseID.String()) {
			if !isIdentityLeaseExpired(l) {
				results = append(results, l)
			} else {
				klog.Infof("IdentityLease %s is expired", l.Name)
			}
		}
	}
	klog.Infof("total candidates after filter %d", len(results))
	return results, nil
}

func canLead(lease *v1alpha1.IdentityLease, leaderLeaseName string) bool {
	return lease.Spec.CanLeadLease == leaderLeaseName
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

func isIdentityLeaseExpired(lease *v1alpha1.IdentityLease) bool {
	currentTime := time.Now()
	// IdentityLeases created should have non-nil renew time
	// and lease duration set. Leases without these fields set are invalid and should
	// be GC'ed.
	return lease.Spec.RenewTime == nil ||
		lease.Spec.LeaseDurationSeconds == nil ||
		lease.Spec.RenewTime.Add(time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second).Before(currentTime)
}
