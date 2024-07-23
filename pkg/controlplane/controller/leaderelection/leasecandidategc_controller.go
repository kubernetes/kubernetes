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
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationv1alpha1informers "k8s.io/client-go/informers/coordination/v1alpha1"
	"k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/coordination/v1alpha1"
	"k8s.io/client-go/tools/cache"

	"k8s.io/klog/v2"
)

type LeaseCandidateGCController struct {
	kubeclientset kubernetes.Interface

	leaseCandidateLister   listers.LeaseCandidateLister
	leaseCandidateInformer coordinationv1alpha1informers.LeaseCandidateInformer
	leaseCandidatesSynced  cache.InformerSynced

	gcCheckPeriod time.Duration
}

// NewLeaseCandidateGC creates a new LeaseCandidateGCController.
func NewLeaseCandidateGC(clientset kubernetes.Interface, gcCheckPeriod time.Duration, leaseCandidateInformer coordinationv1alpha1informers.LeaseCandidateInformer) *LeaseCandidateGCController {
	return &LeaseCandidateGCController{
		kubeclientset:          clientset,
		leaseCandidateLister:   leaseCandidateInformer.Lister(),
		leaseCandidateInformer: leaseCandidateInformer,
		leaseCandidatesSynced:  leaseCandidateInformer.Informer().HasSynced,
		gcCheckPeriod:          gcCheckPeriod,
	}
}

// Run starts one worker.
func (c *LeaseCandidateGCController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer klog.Infof("Shutting down apiserver leasecandidate garbage collector")

	klog.Infof("Starting apiserver leasecandidate garbage collector")

	if !cache.WaitForCacheSync(stopCh, c.leaseCandidatesSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	go wait.Until(c.gc, c.gcCheckPeriod, stopCh)

	<-stopCh
}

func (c *LeaseCandidateGCController) gc() {
	lcs, err := c.leaseCandidateLister.List(labels.Everything())
	if err != nil {
		klog.ErrorS(err, "Error while listing lease candidates")
		return
	}
	for _, leaseCandidate := range lcs {
		// evaluate lease from cache
		if !isLeaseCandidateExpired(leaseCandidate) {
			continue
		}
		lc, err := c.kubeclientset.CoordinationV1alpha1().LeaseCandidates(leaseCandidate.Namespace).Get(context.TODO(), leaseCandidate.Name, metav1.GetOptions{})
		if err != nil {
			klog.ErrorS(err, "Error getting lc")
			continue
		}
		// evaluate lease from apiserver
		if !isLeaseCandidateExpired(lc) {
			continue
		}
		if err := c.kubeclientset.CoordinationV1alpha1().LeaseCandidates(lc.Namespace).Delete(
			context.TODO(), lc.Name, metav1.DeleteOptions{}); err != nil {
			klog.ErrorS(err, "Error deleting lease")
		}
	}
}
