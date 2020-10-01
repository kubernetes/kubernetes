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

package leasegarbagecollector

import (
	"context"
	"fmt"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/coordination/v1"
	"k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/coordination/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/klog/v2"
)

// Controller watches kube-apiserver leases and deletes expired leases.
type Controller struct {
	kubeclientset kubernetes.Interface

	leaseLister  listers.LeaseLister
	leasesSynced cache.InformerSynced

	// To allow injection for testing.
	syncFn func(string) error

	queue workqueue.RateLimitingInterface
	clock clock.Clock
}

// NewController creates a new Controller with input lease informer.
func NewController(clientset kubernetes.Interface, leaseInformer informers.LeaseInformer, clock clock.Clock, leaseResyncPeriod time.Duration) *Controller {
	c := &Controller{
		kubeclientset: clientset,
		leaseLister:   leaseInformer.Lister(),
		leasesSynced:  leaseInformer.Informer().HasSynced,
		queue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "lease_garbage_collector"),
		clock:         clock,
	}

	leaseInformer.Informer().AddEventHandlerWithResyncPeriod(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addLease,
		UpdateFunc: c.updateLease,
	}, leaseResyncPeriod)

	c.syncFn = c.sync
	return c
}

// Run starts one worker.
func (c *Controller) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()
	defer klog.Infof("Shutting down Lease garbage collector")

	klog.Infof("Starting Lease garbage collector")

	if !cache.WaitForCacheSync(stopCh, c.leasesSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// only start one worker thread
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

func (c *Controller) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *Controller) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncFn(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with: %v", key, err))
	c.queue.AddRateLimited(key)
	return true
}

func (c *Controller) sync(name string) error {
	lease, err := c.leaseLister.Leases(corev1.NamespaceAPIServerLease).Get(name)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	if errors.IsNotFound(err) || lease == nil {
		return nil
	}
	currentTime := c.clock.Now()
	if lease.Spec.RenewTime == nil ||
		lease.Spec.LeaseDurationSeconds == nil ||
		lease.Spec.RenewTime.Add(time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second).Before(currentTime) {
		if err := c.kubeclientset.CoordinationV1().Leases(corev1.NamespaceAPIServerLease).Delete(context.TODO(), name, metav1.DeleteOptions{}); err != nil {
			return err
		}
	}
	return nil
}

func (c *Controller) addLease(obj interface{}) {
	castObj := obj.(*coordinationv1.Lease)
	klog.V(4).Infof("Adding lease %s", castObj.Name)
	c.enqueue(castObj)
}

func (c *Controller) updateLease(oldObj, newObj interface{}) {
	castNewObj := newObj.(*coordinationv1.Lease)
	klog.V(4).Infof("Updating lease %s", castNewObj.Name)
	c.enqueue(castNewObj)
}

func (c *Controller) enqueue(obj *coordinationv1.Lease) {
	c.queue.Add(obj.Name)
}
