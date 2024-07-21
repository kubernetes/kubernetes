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

	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	coordinationv1alpha1client "k8s.io/client-go/kubernetes/typed/coordination/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const requeueInterval = 5 * time.Minute

type LeaseCandidate struct {
	LeaseClient            coordinationv1alpha1client.LeaseCandidateInterface
	LeaseCandidateInformer cache.SharedIndexInformer
	InformerFactory        informers.SharedInformerFactory
	HasSynced              cache.InformerSynced

	// At most there will be one item in this Queue (since we only watch one item)
	queue workqueue.TypedRateLimitingInterface[int]

	name      string
	namespace string

	// controller lease
	leaseName string

	Clock clock.Clock

	binaryVersion, emulationVersion string
	preferredStrategies             []v1.CoordinatedLeaseStrategy
}

func NewCandidate(clientset kubernetes.Interface,
	candidateName string,
	candidateNamespace string,
	targetLease string,
	clock clock.Clock,
	binaryVersion, emulationVersion string,
	preferredStrategies []v1.CoordinatedLeaseStrategy,
) (*LeaseCandidate, error) {
	fieldSelector := fields.OneTermEqualSelector("metadata.name", candidateName).String()
	// A separate informer factory is required because this must start before informerFactories
	// are started for leader elected components
	informerFactory := informers.NewSharedInformerFactoryWithOptions(
		clientset, 5*time.Minute,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			options.FieldSelector = fieldSelector
		}),
	)
	leaseCandidateInformer := informerFactory.Coordination().V1alpha1().LeaseCandidates().Informer()

	lc := &LeaseCandidate{
		LeaseClient:            clientset.CoordinationV1alpha1().LeaseCandidates(candidateNamespace),
		LeaseCandidateInformer: leaseCandidateInformer,
		InformerFactory:        informerFactory,
		name:                   candidateName,
		namespace:              candidateNamespace,
		leaseName:              targetLease,
		Clock:                  clock,
		binaryVersion:          binaryVersion,
		emulationVersion:       emulationVersion,
		preferredStrategies:    preferredStrategies,
	}
	lc.queue = workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[int](), workqueue.TypedRateLimitingQueueConfig[int]{Name: "leasecandidate"})

	synced, err := leaseCandidateInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			if leasecandidate, ok := newObj.(*v1alpha1.LeaseCandidate); ok {
				if leasecandidate.Spec.PingTime != nil {
					lc.enqueueLease()
				}
			}
		},
	})
	if err != nil {
		return nil, err
	}
	lc.HasSynced = synced.HasSynced

	return lc, nil
}

func (c *LeaseCandidate) Run(ctx context.Context) {
	defer c.queue.ShutDown()

	go c.InformerFactory.Start(ctx.Done())
	if !cache.WaitForNamedCacheSync("leasecandidateclient", ctx.Done(), c.HasSynced) {
		return
	}

	c.enqueueLease()
	go c.runWorker(ctx)
	<-ctx.Done()
}

func (c *LeaseCandidate) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *LeaseCandidate) processNextWorkItem(ctx context.Context) bool {
	key, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(key)

	err := c.ensureLease(ctx)
	if err == nil {
		c.queue.AddAfter(key, requeueInterval)
		return true
	}

	utilruntime.HandleError(err)
	klog.Infof("processNextWorkItem.AddRateLimited: %v", key)
	c.queue.AddRateLimited(key)

	return true
}

func (c *LeaseCandidate) enqueueLease() {
	c.queue.Add(0)
}

// ensureLease creates the lease if it does not exist and renew it if it exists. Returns the lease and
// a bool (true if this call created the lease), or any error that occurs.
func (c *LeaseCandidate) ensureLease(ctx context.Context) error {
	lease, err := c.LeaseClient.Get(ctx, c.name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		klog.V(2).Infof("Creating lease candidate")
		// lease does not exist, create it.
		leaseToCreate := c.newLease()
		_, err := c.LeaseClient.Create(ctx, leaseToCreate, metav1.CreateOptions{})
		if err != nil {
			return err
		}
		klog.V(2).Infof("Created lease candidate")
		return nil
	} else if err != nil {
		return err
	}
	klog.V(2).Infof("lease candidate exists.. renewing")
	clone := lease.DeepCopy()
	clone.Spec.RenewTime = &metav1.MicroTime{Time: c.Clock.Now()}
	clone.Spec.PingTime = nil
	_, err = c.LeaseClient.Update(ctx, clone, metav1.UpdateOptions{})
	if err != nil {
		return err
	}
	return nil
}

func (c *LeaseCandidate) newLease() *v1alpha1.LeaseCandidate {
	lease := &v1alpha1.LeaseCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      c.name,
			Namespace: c.namespace,
		},
		Spec: v1alpha1.LeaseCandidateSpec{
			LeaseName:           c.leaseName,
			BinaryVersion:       c.binaryVersion,
			EmulationVersion:    c.emulationVersion,
			PreferredStrategies: c.preferredStrategies,
		},
	}
	lease.Spec.RenewTime = &metav1.MicroTime{Time: c.Clock.Now()}
	return lease
}
