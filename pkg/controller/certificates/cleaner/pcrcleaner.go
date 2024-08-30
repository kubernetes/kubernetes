/*
Copyright 2025 The Kubernetes Authors.

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

package cleaner

import (
	"context"
	"fmt"
	"time"

	certsv1alpha1 "k8s.io/api/certificates/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1alpha1 "k8s.io/client-go/informers/certificates/v1alpha1"
	"k8s.io/client-go/kubernetes"
	certlistersv1alpha1 "k8s.io/client-go/listers/certificates/v1alpha1"
	"k8s.io/klog/v2"
)

// PCRCleanerController garbage-collects PodCertificateRequests older than 30
// minutes.
type PCRCleanerController struct {
	client    kubernetes.Interface
	pcrLister certlistersv1alpha1.PodCertificateRequestLister
}

// NewPCRCleanerController creates a PCRCleanerController.
func NewPCRCleanerController(
	client kubernetes.Interface,
	pcrLister certinformersv1alpha1.PodCertificateRequestInformer,
) *PCRCleanerController {
	return &PCRCleanerController{
		client:    client,
		pcrLister: pcrLister.Lister(),
	}
}

func (c *PCRCleanerController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting PCR cleaner controller")
	defer logger.Info("Shutting down PCR cleaner controller")

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.worker, pollingInterval)
	}

	<-ctx.Done()
}

func (c *PCRCleanerController) worker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	pcrs, err := c.pcrLister.List(labels.Everything())
	if err != nil {
		logger.Error(err, "Unable to list PCRs")
		return
	}
	for _, pcr := range pcrs {
		if err := c.handle(ctx, pcr); err != nil {
			logger.Error(err, "Error while attempting to clean PCR", "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
		}
	}
}

func (c PCRCleanerController) handle(ctx context.Context, pcr *certsv1alpha1.PodCertificateRequest) error {
	if pcr.ObjectMeta.CreationTimestamp.Time.After(time.Now().Add(-30 * time.Minute)) {
		return nil
	}

	if err := c.client.CertificatesV1alpha1().PodCertificateRequests(pcr.ObjectMeta.Namespace).Delete(ctx, pcr.ObjectMeta.Name, metav1.DeleteOptions{}); err != nil {
		return fmt.Errorf("unable to delete PCR %q: %w", pcr.ObjectMeta.Name, err)
	}
	return nil
}
