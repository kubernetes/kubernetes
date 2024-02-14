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
	"time"

	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	coordinationv1alpha1client "k8s.io/client-go/kubernetes/typed/coordination/v1alpha1"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"

	identityleaseinformers "k8s.io/client-go/informers/coordination/v1alpha1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/client-go/util/workqueue"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

const requeueInterval = 10 * time.Second

type IdentityLease struct {
	LeaseClient           coordinationv1alpha1client.IdentityLeaseInterface
	HolderIdentity        string
	IdentityLeaseInformer identityleaseinformers.IdentityLeaseInformer

	// At most there will be one or two items in this Queue (since we only watch one item)
	// Is there a better data structure?
	leaseQueue workqueue.TypedRateLimitingInterface[int]

	// identity lease
	LeaseName      string
	LeaseNamespace string

	// controller lease
	CanLeadLeases string

	LeaseDurationSeconds int32
	RenewInterval        time.Duration
	Clock                clock.Clock

	BinaryVersion, CompatibilityVersion string
}

// TODO: Should create a constructor for this function since there are some critical fields that we should validate
func (c *IdentityLease) Run(ctx context.Context) {
	c.leaseQueue = workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[int](), workqueue.TypedRateLimitingQueueConfig[int]{Name: "test"})
	defer c.leaseQueue.ShutDown()
	c.enqueueLease(0)
	// TODO: Wait for ready?
	_, err := c.IdentityLeaseInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			if identitylease, ok := newObj.(*v1alpha1.IdentityLease); ok {
				// We're watching all changes and filtering for name match
				// Is there a more efficient way to only watch what we want?
				// This still needs to handle namespace
				if identitylease.Name == c.LeaseName {
					c.enqueueLease(0)
				}
			}
		},
	})
	if err != nil {
		// TODO handle error
		return
	}

	go c.runWorker(ctx)
	<-ctx.Done()

	// use the controller pattern instead of writing our own loop
	// c.acquireOrRenewLease(ctx)
}

func (c *IdentityLease) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *IdentityLease) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.leaseQueue.Get()
	if shutdown {
		return false
	}
	defer c.leaseQueue.Done(key)

	err := func() error {
		_, _, err := c.ensureLease(ctx)
		return err
	}()

	if err == nil {
		c.leaseQueue.AddAfter(key, requeueInterval)
		return true
	}

	utilruntime.HandleError(err)
	klog.Infof("processNextWorkItem.AddRateLimited: %v", key)
	c.leaseQueue.AddRateLimited(key)

	return true
}

func (c *IdentityLease) enqueueLease(n int) {
	c.leaseQueue.Add(n)
}

// func (c *IdentityLease) acquireOrRenewLease(ctx context.Context) {
// 	klog.Infof("Starting identity lease management")
// 	sleep := 5 * time.Second
// 	for {
// 		c.backoffEnsureLease(ctx)
// 		select {
// 		case <-ctx.Done():
// 			klog.Infof("Shutting down identity lease management")
// 			return
// 		case <-time.After(sleep):
// 		}
// 	}
// }

// // backoffEnsureLease attempts to create the lease if it does not exist,
// // and uses exponentially increasing waits to prevent overloading the API server
// // with retries. Returns the lease, and true if this call created the lease,
// // false otherwise.
// func (c *IdentityLease) backoffEnsureLease(ctx context.Context) (*v1alpha1.IdentityLease, bool) {
// 	var (
// 		lease   *v1alpha1.IdentityLease
// 		created bool
// 		err     error
// 	)
// 	sleep := 100 * time.Millisecond
// 	for {
// 		lease, created, err = c.ensureLease(ctx)
// 		if err == nil {
// 			break
// 		}
// 		sleep = minDuration(2*sleep, maxBackoff)
// 		klog.FromContext(ctx).Error(err, "Failed to ensure identity lease exists, will retry", "interval", sleep)
// 		// backoff wait with early return if the context gets canceled
// 		select {
// 		case <-ctx.Done():
// 			return nil, false
// 		case <-time.After(sleep):
// 		}
// 	}
// 	// klog.Infof("Shutting down identity lease management")
// 	return lease, created
// }
// const maxBackoff = 7 * time.Second

// func minDuration(a, b time.Duration) time.Duration {
// 	if a < b {
// 		return a
// 	}
// 	return b
// }

// ensureLease creates the lease if it does not exist and renew it if it exists. Returns the lease and
// a bool (true if this call created the lease), or any error that occurs.
func (c *IdentityLease) ensureLease(ctx context.Context) (*v1alpha1.IdentityLease, bool, error) {
	lease, err := c.LeaseClient.Get(ctx, c.LeaseName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		klog.Infof("Creating identity lease")
		// lease does not exist, create it.
		leaseToCreate, err := c.newLease(nil)
		// An error occurred during allocating the new lease (likely from newLeasePostProcessFunc).
		// Given that we weren't able to set the lease correctly, we simply
		// not create it this time - we will retry in the next iteration.
		if err != nil {
			return nil, false, nil
		}
		lease, err := c.LeaseClient.Create(ctx, leaseToCreate, metav1.CreateOptions{})
		if err != nil {
			return nil, false, err
		}
		klog.Infof("Created identity lease")
		return lease, true, nil
	} else if err != nil {
		// unexpected error getting lease
		return nil, false, err
	}
	klog.Infof("identity lease exists.. renewing")
	clone := lease.DeepCopy()
	clone.Spec.RenewTime = &metav1.MicroTime{Time: c.Clock.Now()}
	lease, err = c.LeaseClient.Update(ctx, clone, metav1.UpdateOptions{})
	if err != nil {
		return nil, false, err
	}
	return lease, false, nil
}

// newLease constructs a new lease if base is nil, or returns a copy of base
// with desired state asserted on the copy.
// Note that an error will block lease CREATE, causing the CREATE to be retried in
// the next iteration; but the error won't block lease refresh (UPDATE).
func (c *IdentityLease) newLease(base *v1alpha1.IdentityLease) (*v1alpha1.IdentityLease, error) {
	// Use the bare minimum set of fields; other fields exist for debugging/legacy,
	// but we don't need to make component heartbeats more complicated by using them.
	var lease *v1alpha1.IdentityLease
	if base == nil {
		lease = &v1alpha1.IdentityLease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      c.LeaseName,
				Namespace: c.LeaseNamespace,
			},
			Spec: v1alpha1.IdentityLeaseSpec{
				CanLeadLease:         c.CanLeadLeases,
				BinaryVersion:        c.BinaryVersion,
				CompatibilityVersion: c.CompatibilityVersion,
				HolderIdentity:       ptr.To(c.HolderIdentity),
				LeaseDurationSeconds: ptr.To(c.LeaseDurationSeconds),
			},
		}
	} else {
		lease = base.DeepCopy()
	}
	lease.Spec.RenewTime = &metav1.MicroTime{Time: c.Clock.Now()}

	return lease, nil
}
