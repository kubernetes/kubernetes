/*
Copyright 2016 The Kubernetes Authors.

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

	certificates "k8s.io/api/certificates/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certificatesinformers "k8s.io/client-go/informers/certificates/v1"
	clientset "k8s.io/client-go/kubernetes"
	certificateslisters "k8s.io/client-go/listers/certificates/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
)

type CertificateController struct {
	// name is an identifier for this particular controller instance.
	name string

	kubeClient clientset.Interface

	csrLister  certificateslisters.CertificateSigningRequestLister
	csrsSynced cache.InformerSynced

	handler func(context.Context, *certificates.CertificateSigningRequest) error

	queue workqueue.TypedRateLimitingInterface[string]
}

func NewCertificateController(
	ctx context.Context,
	name string,
	kubeClient clientset.Interface,
	csrInformer certificatesinformers.CertificateSigningRequestInformer,
	handler func(context.Context, *certificates.CertificateSigningRequest) error,
) *CertificateController {
	logger := klog.FromContext(ctx)
	cc := &CertificateController{
		name:       name,
		kubeClient: kubeClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.NewTypedMaxOfRateLimiter[string](
				workqueue.NewTypedItemExponentialFailureRateLimiter[string](200*time.Millisecond, 1000*time.Second),
				// 10 qps, 100 bucket size.  This is only for retry speed and its only the overall factor (not per item)
				&workqueue.TypedBucketRateLimiter[string]{Limiter: rate.NewLimiter(rate.Limit(10), 100)},
			),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "certificate",
			},
		),
		handler: handler,
	}

	// Manage the addition/update of certificate requests
	csrInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			csr := obj.(*certificates.CertificateSigningRequest)
			logger.V(4).Info("Adding certificate request", "csr", csr.Name)
			cc.enqueueCertificateRequest(obj)
		},
		UpdateFunc: func(old, new interface{}) {
			oldCSR := old.(*certificates.CertificateSigningRequest)
			logger.V(4).Info("Updating certificate request", "old", oldCSR.Name)
			cc.enqueueCertificateRequest(new)
		},
		DeleteFunc: func(obj interface{}) {
			csr, ok := obj.(*certificates.CertificateSigningRequest)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					logger.V(2).Info("Couldn't get object from tombstone", "object", obj)
					return
				}
				csr, ok = tombstone.Obj.(*certificates.CertificateSigningRequest)
				if !ok {
					logger.V(2).Info("Tombstone contained object that is not a CSR", "object", obj)
					return
				}
			}
			logger.V(4).Info("Deleting certificate request", "csr", csr.Name)
			cc.enqueueCertificateRequest(obj)
		},
	})
	cc.csrLister = csrInformer.Lister()
	cc.csrsSynced = csrInformer.Informer().HasSynced
	return cc
}

// Run the main goroutine responsible for watching and syncing jobs.
func (cc *CertificateController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer cc.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting certificate controller", "name", cc.name)
	defer logger.Info("Shutting down certificate controller", "name", cc.name)

	if !cache.WaitForNamedCacheSync(fmt.Sprintf("certificate-%s", cc.name), ctx.Done(), cc.csrsSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, cc.worker, time.Second)
	}

	<-ctx.Done()
}

// worker runs a thread that dequeues CSRs, handles them, and marks them done.
func (cc *CertificateController) worker(ctx context.Context) {
	for cc.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (cc *CertificateController) processNextWorkItem(ctx context.Context) bool {
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

func (cc *CertificateController) enqueueCertificateRequest(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	cc.queue.Add(key)
}

func (cc *CertificateController) syncFunc(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	startTime := time.Now()
	defer func() {
		logger.V(4).Info("Finished syncing certificate request", "csr", key, "elapsedTime", time.Since(startTime))
	}()
	csr, err := cc.csrLister.Get(key)
	if errors.IsNotFound(err) {
		logger.V(3).Info("csr has been deleted", "csr", key)
		return nil
	}
	if err != nil {
		return err
	}

	if len(csr.Status.Certificate) > 0 {
		// no need to do anything because it already has a cert
		return nil
	}

	// need to operate on a copy so we don't mutate the csr in the shared cache
	csr = csr.DeepCopy()
	return cc.handler(ctx, csr)
}

// IgnorableError returns an error that we shouldn't handle (i.e. log) because
// it's spammy and usually user error. Instead we will log these errors at a
// higher log level. We still need to throw these errors to signal that the
// sync should be retried.
func IgnorableError(s string, args ...interface{}) ignorableError {
	return ignorableError(fmt.Sprintf(s, args...))
}

type ignorableError string

func (e ignorableError) Error() string {
	return string(e)
}
