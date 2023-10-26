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

// Package certificates implements an abstract controller that is useful for
// building controllers that manage CSRs
package certificates

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/time/rate"

	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certificatesv1alpha1informers "k8s.io/client-go/informers/certificates/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	certificatesv1alpha1listers "k8s.io/client-go/listers/certificates/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

type PodCertificateRequestController struct {
	// name is an identifier for this particular controller instance.
	name string

	kubeClient clientset.Interface

	reqLister  certificatesv1alpha1listers.PodCertificateRequestLister
	reqsSynced cache.InformerSynced

	handler func(context.Context, *certificatesv1alpha1.PodCertificateRequest) error

	queue workqueue.TypedRateLimitingInterface[string]
}

func NewPodCertificateRequestController(
	ctx context.Context,
	name string,
	kubeClient clientset.Interface,
	reqInformer certificatesv1alpha1informers.PodCertificateRequestInformer,
	handler func(context.Context, *certificatesv1alpha1.PodCertificateRequest) error,
) *PodCertificateRequestController {
	logger := klog.FromContext(ctx)
	cc := &PodCertificateRequestController{
		name:       name,
		kubeClient: kubeClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.NewTypedMaxOfRateLimiter[string](
				workqueue.NewTypedItemExponentialFailureRateLimiter[string](200*time.Millisecond, 1000*time.Second),
				// 10 qps, 100 bucket size.  This is only for retry speed and its only the overall factor (not per item)
				&workqueue.TypedBucketRateLimiter[string]{Limiter: rate.NewLimiter(rate.Limit(10), 100)},
			),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "podcertificaterequest",
			},
		),
		handler: handler,
	}

	// Manage the addition/update of certificate requests
	reqInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			req := obj.(*certificatesv1alpha1.PodCertificateRequest)
			logger.V(4).Info("Adding PodCertificateRequest", "req", req.Name)
			cc.enqueueRequest(obj)
		},
		UpdateFunc: func(old, new interface{}) {
			oldReq := old.(*certificatesv1alpha1.PodCertificateRequest)
			logger.V(4).Info("Updating PodCertificateRequest", "old", oldReq.Name)
			cc.enqueueRequest(new)
		},
		DeleteFunc: func(obj interface{}) {
			req, ok := obj.(*certificatesv1alpha1.PodCertificateRequest)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					logger.V(2).Info("Couldn't get object from tombstone", "object", obj)
					return
				}
				req, ok = tombstone.Obj.(*certificatesv1alpha1.PodCertificateRequest)
				if !ok {
					logger.V(2).Info("Tombstone contained object that is not a PodCertificateRequest", "object", obj)
					return
				}
			}
			logger.V(4).Info("Deleting PodCertificateRequest", "req", req.Name)
			cc.enqueueRequest(obj)
		},
	})
	cc.reqLister = reqInformer.Lister()
	cc.reqsSynced = reqInformer.Informer().HasSynced
	return cc
}

// Run the main goroutine responsible for watching and syncing jobs.
func (cc *PodCertificateRequestController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer cc.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting certificate controller", "name", cc.name)
	defer logger.Info("Shutting down certificate controller", "name", cc.name)

	if !cache.WaitForNamedCacheSync(fmt.Sprintf("certificate-%s", cc.name), ctx.Done(), cc.reqsSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, cc.worker, time.Second)
	}

	<-ctx.Done()
}

// worker runs a thread that dequeues CSRs, handles them, and marks them done.
func (cc *PodCertificateRequestController) worker(ctx context.Context) {
	for cc.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (cc *PodCertificateRequestController) processNextWorkItem(ctx context.Context) bool {
	cKey, quit := cc.queue.Get()
	if quit {
		return false
	}
	defer cc.queue.Done(cKey)

	if err := cc.syncFunc(ctx, cKey); err != nil {
		cc.queue.AddRateLimited(cKey)
		if _, ignorable := err.(ignorableError); !ignorable {
			utilruntime.HandleError(fmt.Errorf("Sync %v failed with : %v", cKey, err))
		} else {
			klog.FromContext(ctx).V(4).Info("Sync certificate request failed", "csr", cKey, "err", err)
		}
		return true
	}

	cc.queue.Forget(cKey)
	return true

}

func (cc *PodCertificateRequestController) enqueueRequest(obj interface{}) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	cc.queue.Add(key)
}

func (cc *PodCertificateRequestController) syncFunc(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	startTime := time.Now()
	defer func() {
		logger.V(4).Info("Finished syncing certificate request", "csr", key, "elapsedTime", time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return fmt.Errorf("while splitting key: %w", err)
	}

	req, err := cc.reqLister.PodCertificateRequests(namespace).Get(name)
	if errors.IsNotFound(err) {
		logger.V(3).Info("csr has been deleted", "csr", key)
		return nil
	}
	if err != nil {
		return err
	}

	if len(req.Status.CertificateChain) > 0 {
		// no need to do anything because it already has a cert
		return nil
	}

	// need to operate on a copy so we don't mutate the csr in the shared cache
	req = req.DeepCopy()
	return cc.handler(ctx, req)
}
