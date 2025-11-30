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

	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1beta1 "k8s.io/client-go/informers/certificates/v1beta1"
	"k8s.io/client-go/kubernetes"
	certlistersv1beta1 "k8s.io/client-go/listers/certificates/v1beta1"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

// PCRCleanerController garbage-collects PodCertificateRequests older than 30
// minutes.
type PCRCleanerController struct {
	client          kubernetes.Interface
	pcrLister       certlistersv1beta1.PodCertificateRequestLister
	clock           clock.PassiveClock
	threshold       time.Duration
	pollingInterval time.Duration
}

// NewPCRCleanerController creates a PCRCleanerController.
func NewPCRCleanerController(
	client kubernetes.Interface,
	pcrLister certinformersv1beta1.PodCertificateRequestInformer,
	clock clock.PassiveClock,
	threshold time.Duration,
	pollingInterval time.Duration,
) *PCRCleanerController {
	return &PCRCleanerController{
		client:          client,
		pcrLister:       pcrLister.Lister(),
		clock:           clock,
		threshold:       threshold,
		pollingInterval: pollingInterval,
	}
}

func (c *PCRCleanerController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting PodCertificateRequest cleaner controller")
	defer logger.Info("Shutting down PodCertificateRequest cleaner controller")

	wait.UntilWithContext(ctx, c.worker, c.pollingInterval)
}

func (c *PCRCleanerController) worker(ctx context.Context) {
	pcrs, err := c.pcrLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Unable to list PodCertificateRequests")
		return
	}
	for _, pcr := range pcrs {
		if err := c.handle(ctx, pcr); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Error while attempting to clean PodCertificateRequest", "pcr", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name)
		}
	}
}

func (c PCRCleanerController) handle(ctx context.Context, pcr *certsv1beta1.PodCertificateRequest) error {
	if c.clock.Now().Before(pcr.ObjectMeta.CreationTimestamp.Time.Add(c.threshold)) {
		return nil
	}

	opts := metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: ptr.To(pcr.ObjectMeta.UID),
		},
	}

	err := c.client.CertificatesV1beta1().PodCertificateRequests(pcr.ObjectMeta.Namespace).Delete(ctx, pcr.ObjectMeta.Name, opts)
	if k8serrors.IsNotFound(err) {
		// This is OK, we don't care if someone else already deleted it.
		return nil
	} else if err != nil {
		return fmt.Errorf("unable to delete PodCertificateRequest %q: %w", pcr.ObjectMeta.Namespace+"/"+pcr.ObjectMeta.Name, err)
	}

	return nil
}
