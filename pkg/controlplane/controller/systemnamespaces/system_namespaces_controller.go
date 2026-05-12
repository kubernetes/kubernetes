/*
Copyright 2023 The Kubernetes Authors.

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

package systemnamespaces

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// Controller ensure system namespaces exist.
type Controller struct {
	client kubernetes.Interface

	namespaceLister listers.NamespaceLister
	namespaceSynced cache.InformerSynced

	systemNamespaces []string
	interval         time.Duration
}

// NewController creates a new Controller to ensure system namespaces exist.
func NewController(systemNamespaces []string, clientset kubernetes.Interface, namespaceInformer coreinformers.NamespaceInformer) *Controller {
	interval := 1 * time.Minute

	return &Controller{
		client:           clientset,
		namespaceLister:  namespaceInformer.Lister(),
		namespaceSynced:  namespaceInformer.Informer().HasSynced,
		systemNamespaces: systemNamespaces,
		interval:         interval,
	}
}

// Run starts one worker.
func (c *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)
	logger := klog.FromContext(ctx)
	defer logger.Info("Shutting down system namespaces controller")

	logger.Info("Starting system namespaces controller")

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.namespaceSynced) {
		return
	}

	go wait.UntilWithContext(ctx, c.sync, c.interval)

	<-ctx.Done()
}

func (c *Controller) sync(ctx context.Context) {
	// Loop the system namespace list, and create them if they do not exist
	for _, ns := range c.systemNamespaces {
		if err := c.createNamespaceIfNeeded(ctx, ns); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Unable to create required kubernetes system namespace", "namespace", ns)
		}
	}
}

func (c *Controller) createNamespaceIfNeeded(ctx context.Context, ns string) error {
	if _, err := c.namespaceLister.Get(ns); err == nil {
		// the namespace already exists
		return nil
	}
	newNs := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ns,
			Namespace: "",
		},
	}
	_, err := c.client.CoreV1().Namespaces().Create(ctx, newNs, metav1.CreateOptions{})
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}
