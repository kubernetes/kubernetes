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

	leasecandidateinformers "k8s.io/client-go/informers/coordination/v1alpha1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/client-go/util/workqueue"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

const requeueInterval = 10 * time.Second

type LeaseCandidate struct {
	LeaseClient            coordinationv1alpha1client.LeaseCandidateInterface
	LeaseCandidateInformer leasecandidateinformers.LeaseCandidateInformer

	// At most there will be one or two items in this Queue (since we only watch one item)
	// Is there a better data structure?
	leaseQueue workqueue.TypedRateLimitingInterface[int]

	// identity lease
	LeaseName      string
	LeaseNamespace string

	// controller lease
	TargetLease string

	LeaseDurationSeconds int32
	RenewInterval        time.Duration
	Clock                clock.Clock

	BinaryVersion, CompatibilityVersion string
}

// TODO: Should create a constructor for this function since there are some critical fields that we should validate
func (c *LeaseCandidate) Run(ctx context.Context) {
	c.leaseQueue = workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[int](), workqueue.TypedRateLimitingQueueConfig[int]{Name: "leasecandidate"})
	defer c.leaseQueue.ShutDown()
	c.enqueueLease(0)
	// TODO: Wait for ready?
	_, err := c.LeaseCandidateInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			if leasecandidate, ok := newObj.(*v1alpha1.LeaseCandidate); ok {
				// We're watching all changes and filtering for name match
				// Is there a more efficient way to only watch what we want?
				// This still needs to handle namespace
				if leasecandidate.Name == c.LeaseName {
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

func (c *LeaseCandidate) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *LeaseCandidate) processNextWorkItem(ctx context.Context) bool {
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

func (c *LeaseCandidate) enqueueLease(n int) {
	c.leaseQueue.Add(n)
}

// ensureLease creates the lease if it does not exist and renew it if it exists. Returns the lease and
// a bool (true if this call created the lease), or any error that occurs.
func (c *LeaseCandidate) ensureLease(ctx context.Context) (*v1alpha1.LeaseCandidate, bool, error) {
	lease, err := c.LeaseClient.Get(ctx, c.LeaseName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		klog.Infof("Creating lease candidate")
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
		klog.Infof("Created lease candidate")
		return lease, true, nil
	} else if err != nil {
		// unexpected error getting lease
		return nil, false, err
	}
	klog.Infof("lease candidate exists.. renewing")
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
func (c *LeaseCandidate) newLease(base *v1alpha1.LeaseCandidate) (*v1alpha1.LeaseCandidate, error) {
	// Use the bare minimum set of fields; other fields exist for debugging/legacy,
	// but we don't need to make component heartbeats more complicated by using them.
	var lease *v1alpha1.LeaseCandidate
	if base == nil {
		lease = &v1alpha1.LeaseCandidate{
			ObjectMeta: metav1.ObjectMeta{
				Name:      c.LeaseName,
				Namespace: c.LeaseNamespace,
			},
			Spec: v1alpha1.LeaseCandidateSpec{
				TargetLease:          c.TargetLease,
				BinaryVersion:        c.BinaryVersion,
				CompatibilityVersion: c.CompatibilityVersion,
				LeaseDurationSeconds: ptr.To(c.LeaseDurationSeconds),
			},
		}
	} else {
		lease = base.DeepCopy()
	}
	lease.Spec.RenewTime = &metav1.MicroTime{Time: c.Clock.Now()}

	return lease, nil
}
