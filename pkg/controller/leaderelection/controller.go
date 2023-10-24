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
	"k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/controller"
)

const controllerName = "leader-election-controller"

// TODO: multi-valued labels are problematic.. label matching gets broken.. should we use a label
// per leader lease? Do we need to add a spec field?
const CanLeadLeasesLabelName = "coordination.k8s.io/can-lead-leases"

const CompatibilityVersionAnnotationName = "coordination.k8s.io/compatibility-version"
const BinaryVersionAnnotationName = "coordination.k8s.io/binary-version"

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
	defer utilruntime.HandleCrash()

	if !cache.WaitForNamedCacheSync(controllerName, ctx.Done(), c.leaseSynced) {
		return
	}

	defer c.leaseQueue.ShutDown()
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func NewController(leaseInformer coordinationv1.LeaseInformer, leaseClient coordinationv1client.CoordinationV1Interface) (*Controller, error) {
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
		// DeleteFunc: TODO: Start a new election, if needed
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

func (c *Controller) runWorker(ctx context.Context) {
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
	c.leaseQueue.AddRateLimited(key)

	return true
}

func (c *Controller) runElectionLoop(stopCh <-chan struct{}) {
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

	if canLead, ok := lease.Labels[CanLeadLeasesLabelName]; ok {
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
				candidates, err := c.listCandidates(leaderLeaseId)
				if err != nil {
					return err
				}
				if !shouldReelect(candidates, leader) {
					continue
				}

			}

			err = c.startElection(leaderLeaseId)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (c *Controller) activeLeader(ctx context.Context, leaderLeaseId leaderLeaseId) (*v1.Lease, bool, error) {
	leaderLease, err := c.leaseInformer.Lister().Leases(leaderLeaseId.namespace).Get(leaderLeaseId.name)
	if err != nil {
		if errors.IsNotFound(err) {
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

	return holderIdentityLease, !isLeaseExpired(leaderLease), nil
}

func (c *Controller) startElection(leaderLeaseId leaderLeaseId) error {
	// TODO: add a set to track ongoing elections and avoid requesting an election if one has already been kicked off?
	// This might not be needed..
	c.electionCh <- election{
		leaderLeaseId: leaderLeaseId,
		electionStart: time.Now(),
	}
	return nil
}

func (c *Controller) runElection(ctx context.Context, leaderLeaseId leaderLeaseId) error {
	candidates, err := c.listCandidates(leaderLeaseId)
	if err != nil {
		return err
	}

	electee := pickLeader(candidates)
	if electee == nil {
		return nil
	}

	// create the leader election lease
	leaderLease := &v1.Lease{
		// TODO: fill out all lease fields
		ObjectMeta: metav1.ObjectMeta{Namespace: leaderLeaseId.namespace, Name: leaderLeaseId.name},
		Spec: v1.LeaseSpec{
			HolderIdentity: electee.Spec.HolderIdentity,
		},
	}
	_, err = c.leaseClient.Leases(leaderLeaseId.namespace).Create(ctx, leaderLease, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	return nil
}

func pickLeader(candidates []*v1.Lease) *v1.Lease {
	var electee *v1.Lease
	for _, c := range candidates {
		if electee == nil || compare(electee, c) < 0 {
			electee = c
		}
	}
	return electee
}

func shouldReelect(candidates []*v1.Lease, currentLeader *v1.Lease) bool {
	pickedLeader := pickLeader(candidates)
	if pickedLeader == nil {
		return false
	}
	return compare(currentLeader, pickedLeader) < 0
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
		if canLead(l, leaderLeaseId.String()) {
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
	if canLead, ok := lease.Labels[CanLeadLeasesLabelName]; ok {
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
