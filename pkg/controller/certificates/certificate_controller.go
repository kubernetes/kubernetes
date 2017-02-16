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

package certificates

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	certificatesinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/certificates/v1beta1"
	certificateslisters "k8s.io/kubernetes/pkg/client/listers/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

// err returned from these interfaces should indicate utter failure that
// should be retried. "Buisness logic" errors should be indicated by adding
// a condition to the CSRs status, not by returning an error.

type AutoApprover interface {
	AutoApprove(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error)
}

type Signer interface {
	Sign(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error)
}

type CertificateController struct {
	kubeClient clientset.Interface

	csrLister  certificateslisters.CertificateSigningRequestLister
	csrsSynced cache.InformerSynced

	syncHandler func(csrKey string) error

	approver AutoApprover
	signer   Signer

	queue workqueue.RateLimitingInterface
}

func NewCertificateController(kubeClient clientset.Interface, csrInformer certificatesinformers.CertificateSigningRequestInformer, signer Signer, approver AutoApprover) (*CertificateController, error) {
	// Send events to the apiserver
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.Core().RESTClient()).Events("")})

	cc := &CertificateController{
		kubeClient: kubeClient,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "certificate"),
		signer:     signer,
		approver:   approver,
	}

	// Manage the addition/update of certificate requests
	csrInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			csr := obj.(*certificates.CertificateSigningRequest)
			glog.V(4).Infof("Adding certificate request %s", csr.Name)
			cc.enqueueCertificateRequest(obj)
		},
		UpdateFunc: func(old, new interface{}) {
			oldCSR := old.(*certificates.CertificateSigningRequest)
			glog.V(4).Infof("Updating certificate request %s", oldCSR.Name)
			cc.enqueueCertificateRequest(new)
		},
		DeleteFunc: func(obj interface{}) {
			csr, ok := obj.(*certificates.CertificateSigningRequest)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					glog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
					return
				}
				csr, ok = tombstone.Obj.(*certificates.CertificateSigningRequest)
				if !ok {
					glog.V(2).Infof("Tombstone contained object that is not a CSR: %#v", obj)
					return
				}
			}
			glog.V(4).Infof("Deleting certificate request %s", csr.Name)
			cc.enqueueCertificateRequest(obj)
		},
	})
	cc.csrLister = csrInformer.Lister()
	cc.csrsSynced = csrInformer.Informer().HasSynced
	cc.syncHandler = cc.maybeSignCertificate
	return cc, nil
}

// Run the main goroutine responsible for watching and syncing jobs.
func (cc *CertificateController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer cc.queue.ShutDown()

	glog.Infof("Starting certificate controller manager")

	if !cache.WaitForCacheSync(stopCh, cc.csrsSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(cc.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down certificate controller")
}

// worker runs a thread that dequeues CSRs, handles them, and marks them done.
func (cc *CertificateController) worker() {
	for cc.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (cc *CertificateController) processNextWorkItem() bool {
	cKey, quit := cc.queue.Get()
	if quit {
		return false
	}
	defer cc.queue.Done(cKey)

	err := cc.syncHandler(cKey.(string))
	if err == nil {
		cc.queue.Forget(cKey)
		return true
	}

	cc.queue.AddRateLimited(cKey)
	utilruntime.HandleError(fmt.Errorf("Sync %v failed with : %v", cKey, err))
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

// maybeSignCertificate will inspect the certificate request and, if it has
// been approved and meets policy expectations, generate an X509 cert using the
// cluster CA assets. If successful it will update the CSR approve subresource
// with the signed certificate.
func (cc *CertificateController) maybeSignCertificate(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing certificate request %q (%v)", key, time.Now().Sub(startTime))
	}()
	csr, err := cc.csrLister.Get(key)
	if errors.IsNotFound(err) {
		glog.V(3).Infof("csr has been deleted: %v", key)
		return nil
	}
	if err != nil {
		return err
	}

	if csr.Status.Certificate != nil {
		// no need to do anything because it already has a cert
		return nil
	}

	// need to operate on a copy so we don't mutate the csr in the shared cache
	copy, err := api.Scheme.DeepCopy(csr)
	if err != nil {
		return err
	}
	csr = copy.(*certificates.CertificateSigningRequest)

	if cc.approver != nil {
		csr, err = cc.approver.AutoApprove(csr)
		if err != nil {
			return fmt.Errorf("error auto approving csr: %v", err)
		}
		_, err = cc.kubeClient.Certificates().CertificateSigningRequests().UpdateApproval(csr)
		if err != nil {
			return fmt.Errorf("error updating approval for csr: %v", err)
		}
	}

	// At this point, the controller needs to:
	// 1. Check the approval conditions
	// 2. Generate a signed certificate
	// 3. Update the Status subresource

	if cc.signer != nil && IsCertificateRequestApproved(csr) {
		csr, err := cc.signer.Sign(csr)
		if err != nil {
			return fmt.Errorf("error auto signing csr: %v", err)
		}
		_, err = cc.kubeClient.Certificates().CertificateSigningRequests().UpdateStatus(csr)
		if err != nil {
			return fmt.Errorf("error updating signature for csr: %v", err)
		}
	}

	return nil
}
