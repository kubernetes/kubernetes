/*
Copyright 2020 The Kubernetes Authors.

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

package apiserverleasegc

import (
	"context"
	"time"

	v1 "k8s.io/api/coordination/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/coordination/v1"
	"k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/coordination/v1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/klog/v2"
)

// Controller deletes expired apiserver leases.
type Controller struct {
	kubeclientset kubernetes.Interface

	leaseLister   listers.LeaseLister
	leaseInformer cache.SharedIndexInformer
	leasesSynced  cache.InformerSynced

	leaseNamespace string

	gcCheckPeriod time.Duration
}

// NewAPIServerLeaseGC creates a new Controller.
func NewAPIServerLeaseGC(clientset kubernetes.Interface, gcCheckPeriod time.Duration, leaseNamespace, leaseLabelSelector string) *Controller {
	// we construct our own informer because we need such a small subset of the information available.
	// Just one namespace with label selection.
	leaseInformer := informers.NewFilteredLeaseInformer(
		clientset,
		leaseNamespace,
		0,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(listOptions *metav1.ListOptions) {
			listOptions.LabelSelector = leaseLabelSelector
		})
	return &Controller{
		kubeclientset:  clientset,
		leaseLister:    listers.NewLeaseLister(leaseInformer.GetIndexer()),
		leaseInformer:  leaseInformer,
		leasesSynced:   leaseInformer.HasSynced,
		leaseNamespace: leaseNamespace,
		gcCheckPeriod:  gcCheckPeriod,
	}
}

// Run starts one worker.
//
//logcheck:context // RunWithContext should be used instead of Run in code which supports contextual logging.
func (c *Controller) Run(stopCh <-chan struct{}) {
	c.RunWithContext(wait.ContextForChannel(stopCh))
}

// RunWithContext starts one worker with a context.
func (c *Controller) RunWithContext(ctx context.Context) {
	logger := klog.FromContext(ctx)
	defer utilruntime.HandleCrashWithContext(ctx)
	defer logger.Info("Shutting down apiserver lease garbage collector")

	logger.Info("Starting apiserver lease garbage collector")

	// we have a personal informer that is narrowly scoped, start it.
	go c.leaseInformer.RunWithContext(ctx)

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.leasesSynced) {
		utilruntime.HandleErrorWithContext(ctx, nil, "Timed out waiting for caches to sync")
		return
	}

	go wait.UntilWithContext(ctx, c.gc, c.gcCheckPeriod)

	<-ctx.Done()
}

func (c *Controller) gc(ctx context.Context) {
	leases, err := c.leaseLister.Leases(c.leaseNamespace).List(labels.Everything())
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.Error(err, "Error while listing apiserver leases")
		return
	}
	for _, lease := range leases {
		// evaluate lease from cache
		if !isLeaseExpired(lease) {
			continue
		}
		// double check latest lease from apiserver before deleting
		lease, err := c.kubeclientset.CoordinationV1().Leases(c.leaseNamespace).Get(ctx, lease.Name, metav1.GetOptions{})
		if err != nil && !errors.IsNotFound(err) {
			logger.Error(err, "Error getting lease")
			continue
		}
		if errors.IsNotFound(err) || lease == nil {
			// In an HA cluster, this can happen if the lease was deleted
			// by the same GC controller in another apiserver, which is legit.
			// We don't expect other components to delete the lease.
			logger.V(4).Info("Cannot find apiserver lease", "err", err)
			continue
		}
		// evaluate lease from apiserver
		if !isLeaseExpired(lease) {
			continue
		}
		if err := c.kubeclientset.CoordinationV1().Leases(c.leaseNamespace).Delete(
			ctx, lease.Name, metav1.DeleteOptions{}); err != nil {
			if errors.IsNotFound(err) {
				// In an HA cluster, this can happen if the lease was deleted
				// by the same GC controller in another apiserver, which is legit.
				// We don't expect other components to delete the lease.
				logger.V(4).Info("Apiserver lease is gone already", "err", err)
			} else {
				logger.Error(err, "Error deleting lease")
			}
		}
	}
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
