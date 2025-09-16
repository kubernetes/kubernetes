/*
Copyright 2018 The Kubernetes Authors.

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

package lease

import (
	"context"
	"fmt"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	coordclientset "k8s.io/client-go/kubernetes/typed/coordination/v1"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"

	"k8s.io/klog/v2"
)

const (
	// maxUpdateRetries is the number of immediate, successive retries the controller will attempt
	// when renewing the lease before it waits for the renewal interval before trying again,
	// similar to what we do for node status retries
	maxUpdateRetries = 5
	// maxBackoff is the maximum sleep time during backoff (e.g. in backoffEnsureLease)
	maxBackoff = 7 * time.Second
)

// Controller manages creating and renewing the lease for this component (kube-apiserver, kubelet, etc.)
type Controller interface {
	Run(ctx context.Context)
}

// ProcessLeaseFunc processes the given lease in-place
type ProcessLeaseFunc func(*coordinationv1.Lease) error

type controller struct {
	client                     clientset.Interface
	leaseClient                coordclientset.LeaseInterface
	holderIdentity             string
	leaseName                  string
	leaseNamespace             string
	leaseDurationSeconds       int32
	renewInterval              time.Duration
	clock                      clock.Clock
	onRepeatedHeartbeatFailure func()

	// latestLease is the latest lease which the controller updated or created
	latestLease *coordinationv1.Lease

	// newLeasePostProcessFunc allows customizing a lease object (e.g. setting OwnerReference)
	// before every time the lease is created/refreshed(updated).
	// Note that an error will block the lease operation.
	newLeasePostProcessFunc ProcessLeaseFunc
}

// NewController constructs and returns a controller
func NewController(clock clock.Clock, client clientset.Interface, holderIdentity string, leaseDurationSeconds int32, onRepeatedHeartbeatFailure func(), renewInterval time.Duration, leaseName, leaseNamespace string, newLeasePostProcessFunc ProcessLeaseFunc) Controller {
	var leaseClient coordclientset.LeaseInterface
	if client != nil {
		leaseClient = client.CoordinationV1().Leases(leaseNamespace)
	}
	return &controller{
		client:                     client,
		leaseClient:                leaseClient,
		holderIdentity:             holderIdentity,
		leaseName:                  leaseName,
		leaseNamespace:             leaseNamespace,
		leaseDurationSeconds:       leaseDurationSeconds,
		renewInterval:              renewInterval,
		clock:                      clock,
		onRepeatedHeartbeatFailure: onRepeatedHeartbeatFailure,
		newLeasePostProcessFunc:    newLeasePostProcessFunc,
	}
}

// Run runs the controller
func (c *controller) Run(ctx context.Context) {
	if c.leaseClient == nil {
		klog.FromContext(ctx).Info("lease controller has nil lease client, will not claim or renew leases")
		return
	}
	wait.JitterUntilWithContext(ctx, c.sync, c.renewInterval, 0.04, true)
}

func (c *controller) sync(ctx context.Context) {
	if c.latestLease != nil {
		// As long as the lease is not (or very rarely) updated by any other agent than the component itself,
		// we can optimistically assume it didn't change since our last update and try updating
		// based on the version from that time. Thanks to it we avoid GET call and reduce load
		// on etcd and kube-apiserver.
		// If at some point other agents will also be frequently updating the Lease object, this
		// can result in performance degradation, because we will end up with calling additional
		// GET/PUT - at this point this whole "if" should be removed.
		err := c.retryUpdateLease(ctx, c.latestLease)
		if err == nil {
			return
		}
		klog.FromContext(ctx).Info("failed to update lease using latest lease, fallback to ensure lease", "err", err)
	}

	lease, created := c.backoffEnsureLease(ctx)
	c.latestLease = lease
	// we don't need to update the lease if we just created it
	if !created && lease != nil {
		if err := c.retryUpdateLease(ctx, lease); err != nil {
			klog.FromContext(ctx).Error(err, "Will retry updating lease", "interval", c.renewInterval)
		}
	}
}

// backoffEnsureLease attempts to create the lease if it does not exist,
// and uses exponentially increasing waits to prevent overloading the API server
// with retries. Returns the lease, and true if this call created the lease,
// false otherwise.
func (c *controller) backoffEnsureLease(ctx context.Context) (*coordinationv1.Lease, bool) {
	var (
		lease   *coordinationv1.Lease
		created bool
		err     error
	)
	sleep := 100 * time.Millisecond
	for {
		lease, created, err = c.ensureLease(ctx)
		if err == nil {
			break
		}
		sleep = minDuration(2*sleep, maxBackoff)
		klog.FromContext(ctx).Error(err, "Failed to ensure lease exists, will retry", "interval", sleep)
		// backoff wait with early return if the context gets canceled
		select {
		case <-ctx.Done():
			return nil, false
		case <-time.After(sleep):
		}
	}
	return lease, created
}

// ensureLease creates the lease if it does not exist. Returns the lease and
// a bool (true if this call created the lease), or any error that occurs.
func (c *controller) ensureLease(ctx context.Context) (*coordinationv1.Lease, bool, error) {
	lease, err := c.leaseClient.Get(ctx, c.leaseName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		// lease does not exist, create it.
		leaseToCreate, err := c.newLease(nil)
		// An error occurred during allocating the new lease (likely from newLeasePostProcessFunc).
		// Given that we weren't able to set the lease correctly, we simply
		// not create it this time - we will retry in the next iteration.
		if err != nil {
			return nil, false, nil
		}
		lease, err := c.leaseClient.Create(ctx, leaseToCreate, metav1.CreateOptions{})
		if err != nil {
			return nil, false, err
		}
		return lease, true, nil
	} else if err != nil {
		// unexpected error getting lease
		return nil, false, err
	}
	// lease already existed
	return lease, false, nil
}

// retryUpdateLease attempts to update the lease for maxUpdateRetries,
// call this once you're sure the lease has been created
func (c *controller) retryUpdateLease(ctx context.Context, base *coordinationv1.Lease) error {
	for i := 0; i < maxUpdateRetries; i++ {
		leaseToUpdate, err := c.newLease(base)
		if err != nil {
			klog.FromContext(ctx).Error(err, "Failed to prepare lease")
		} else {
			lease, err := c.leaseClient.Update(ctx, leaseToUpdate, metav1.UpdateOptions{})
			if err == nil {
				c.latestLease = lease
				return nil
			}
			klog.FromContext(ctx).Error(err, "Failed to update lease")
			// OptimisticLockError requires getting the newer version of lease to proceed.
			if apierrors.IsConflict(err) {
				base, _ = c.backoffEnsureLease(ctx)
				continue
			}
		}
		if i > 0 && c.onRepeatedHeartbeatFailure != nil {
			c.onRepeatedHeartbeatFailure()
		}
	}
	return fmt.Errorf("failed %d attempts to update lease", maxUpdateRetries)
}

// newLease constructs a new lease if base is nil, or returns a copy of base
// with desired state asserted on the copy.
// Note that an error will block lease CREATE, causing the CREATE to be retried in
// the next iteration; but the error won't block lease refresh (UPDATE).
func (c *controller) newLease(base *coordinationv1.Lease) (*coordinationv1.Lease, error) {
	// Use the bare minimum set of fields; other fields exist for debugging/legacy,
	// but we don't need to make component heartbeats more complicated by using them.
	var lease *coordinationv1.Lease
	if base == nil {
		lease = &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      c.leaseName,
				Namespace: c.leaseNamespace,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity: ptr.To(c.holderIdentity),
			},
		}
	} else {
		lease = base.DeepCopy()
	}
	// update the duration, the controller's config may have changed since lease creation
	lease.Spec.LeaseDurationSeconds = ptr.To(c.leaseDurationSeconds)
	lease.Spec.RenewTime = &metav1.MicroTime{Time: c.clock.Now()}

	if c.newLeasePostProcessFunc != nil {
		err := c.newLeasePostProcessFunc(lease)
		return lease, err
	}

	return lease, nil
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}
