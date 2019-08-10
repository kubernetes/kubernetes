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

package nodelease

import (
	"fmt"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	coordclientset "k8s.io/client-go/kubernetes/typed/coordination/v1"
	"k8s.io/utils/pointer"

	"k8s.io/klog"
)

const (
	// renewIntervalFraction is the fraction of lease duration to renew the lease
	renewIntervalFraction = 0.25
	// maxUpdateRetries is the number of immediate, successive retries the Kubelet will attempt
	// when renewing the lease before it waits for the renewal interval before trying again,
	// similar to what we do for node status retries
	maxUpdateRetries = 5
	// maxBackoff is the maximum sleep time during backoff (e.g. in backoffEnsureLease)
	maxBackoff = 7 * time.Second
)

// Controller manages creating and renewing the lease for this Kubelet
type Controller interface {
	Run(stopCh <-chan struct{})
}

type controller struct {
	client                     clientset.Interface
	leaseClient                coordclientset.LeaseInterface
	holderIdentity             string
	leaseDurationSeconds       int32
	renewInterval              time.Duration
	clock                      clock.Clock
	onRepeatedHeartbeatFailure func()

	// latestLease is the latest node lease which Kubelet updated or created
	latestLease *coordinationv1.Lease
}

// NewController constructs and returns a controller
func NewController(clock clock.Clock, client clientset.Interface, holderIdentity string, leaseDurationSeconds int32, onRepeatedHeartbeatFailure func()) Controller {
	var leaseClient coordclientset.LeaseInterface
	if client != nil {
		leaseClient = client.CoordinationV1().Leases(corev1.NamespaceNodeLease)
	}
	leaseDuration := time.Duration(leaseDurationSeconds) * time.Second
	return &controller{
		client:                     client,
		leaseClient:                leaseClient,
		holderIdentity:             holderIdentity,
		leaseDurationSeconds:       leaseDurationSeconds,
		renewInterval:              time.Duration(float64(leaseDuration) * renewIntervalFraction),
		clock:                      clock,
		onRepeatedHeartbeatFailure: onRepeatedHeartbeatFailure,
	}
}

// Run runs the controller
func (c *controller) Run(stopCh <-chan struct{}) {
	if c.leaseClient == nil {
		klog.Infof("node lease controller has nil lease client, will not claim or renew leases")
		return
	}
	wait.Until(c.sync, c.renewInterval, stopCh)
}

func (c *controller) sync() {
	if c.latestLease != nil {
		// As long as node lease is not (or very rarely) updated by any other agent than Kubelet,
		// we can optimistically assume it didn't change since our last update and try updating
		// based on the version from that time. Thanks to it we avoid GET call and reduce load
		// on etcd and kube-apiserver.
		// If at some point other agents will also be frequently updating the Lease object, this
		// can result in performance degradation, because we will end up with calling additional
		// GET/PUT - at this point this whole "if" should be removed.
		lease, err := c.leaseClient.Update(c.newLease(c.latestLease))
		if err == nil {
			c.latestLease = lease
			return
		}

		klog.Infof("failed to update lease using latest lease, fallback to ensure lease, err: %v", err)
	}

	lease, created := c.backoffEnsureLease()
	c.latestLease = lease
	// we don't need to update the lease if we just created it
	if !created {
		if err := c.retryUpdateLease(lease); err != nil {
			klog.Errorf("%v, will retry after %v", err, c.renewInterval)
		}
	}
}

// backoffEnsureLease attempts to create the lease if it does not exist,
// and uses exponentially increasing waits to prevent overloading the API server
// with retries. Returns the lease, and true if this call created the lease,
// false otherwise.
func (c *controller) backoffEnsureLease() (*coordinationv1.Lease, bool) {
	var (
		lease   *coordinationv1.Lease
		created bool
		err     error
	)
	sleep := 100 * time.Millisecond
	for {
		lease, created, err = c.ensureLease()
		if err == nil {
			break
		}
		sleep = minDuration(2*sleep, maxBackoff)
		klog.Errorf("failed to ensure node lease exists, will retry in %v, error: %v", sleep, err)
		// backoff wait
		c.clock.Sleep(sleep)
	}
	return lease, created
}

// ensureLease creates the lease if it does not exist. Returns the lease and
// a bool (true if this call created the lease), or any error that occurs.
func (c *controller) ensureLease() (*coordinationv1.Lease, bool, error) {
	lease, err := c.leaseClient.Get(c.holderIdentity, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		// lease does not exist, create it
		lease, err := c.leaseClient.Create(c.newLease(nil))
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
func (c *controller) retryUpdateLease(base *coordinationv1.Lease) error {
	for i := 0; i < maxUpdateRetries; i++ {
		lease, err := c.leaseClient.Update(c.newLease(base))
		if err == nil {
			c.latestLease = lease
			return nil
		}
		klog.Errorf("failed to update node lease, error: %v", err)
		// OptimisticLockError requires getting the newer version of lease to proceed.
		if apierrors.IsConflict(err) {
			base, _ = c.backoffEnsureLease()
			continue
		}
		if i > 0 && c.onRepeatedHeartbeatFailure != nil {
			c.onRepeatedHeartbeatFailure()
		}
	}
	return fmt.Errorf("failed %d attempts to update node lease", maxUpdateRetries)
}

// newLease constructs a new lease if base is nil, or returns a copy of base
// with desired state asserted on the copy.
func (c *controller) newLease(base *coordinationv1.Lease) *coordinationv1.Lease {
	// Use the bare minimum set of fields; other fields exist for debugging/legacy,
	// but we don't need to make node heartbeats more complicated by using them.
	var lease *coordinationv1.Lease
	if base == nil {
		lease = &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      c.holderIdentity,
				Namespace: corev1.NamespaceNodeLease,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity:       pointer.StringPtr(c.holderIdentity),
				LeaseDurationSeconds: pointer.Int32Ptr(c.leaseDurationSeconds),
			},
		}
	} else {
		lease = base.DeepCopy()
	}
	lease.Spec.RenewTime = &metav1.MicroTime{Time: c.clock.Now()}

	// Setting owner reference needs node's UID. Note that it is different from
	// kubelet.nodeRef.UID. When lease is initially created, it is possible that
	// the connection between master and node is not ready yet. So try to set
	// owner reference every time when renewing the lease, until successful.
	if lease.OwnerReferences == nil || len(lease.OwnerReferences) == 0 {
		if node, err := c.client.CoreV1().Nodes().Get(c.holderIdentity, metav1.GetOptions{}); err == nil {
			lease.OwnerReferences = []metav1.OwnerReference{
				{
					APIVersion: corev1.SchemeGroupVersion.WithKind("Node").Version,
					Kind:       corev1.SchemeGroupVersion.WithKind("Node").Kind,
					Name:       c.holderIdentity,
					UID:        node.UID,
				},
			}
		} else {
			klog.Errorf("failed to get node %q when trying to set owner ref to the node lease: %v", c.holderIdentity, err)
		}
	}

	return lease
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}
