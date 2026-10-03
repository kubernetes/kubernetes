/*
Copyright 2019 The Kubernetes Authors.

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

package dynamiccertificates

import (
	"bytes"
	"context"
	"crypto/x509"
	"fmt"
	"sync/atomic"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// ConfigMapCAController provies a CAContentProvider that can dynamically react to configmap changes
// It also fulfills the authenticator interface to provide verifyoptions
type ConfigMapCAController struct {
	name string

	configmapLister    corev1listers.ConfigMapLister
	configmapNamespace string
	configmapName      string
	configmapKey       string
	// configMapInformer is tracked so that we can start these on Run
	configMapInformer cache.SharedIndexInformer

	// caBundle is a caBundleAndVerifier that contains the last read, non-zero length content of the file.
	// It is nil when the configmap has never been read and nil again once the configmap is deleted,
	// see loadCABundle, so those two states are indistinguishable.
	caBundle atomic.Pointer[caBundleAndVerifier]

	listeners []Listener

	queue workqueue.TypedRateLimitingInterface[string]
	// preRunCaches are the caches to sync before starting the work of this control loop
	preRunCaches []cache.InformerSynced
}

var _ CAContentProvider = &ConfigMapCAController{}
var _ ControllerRunner = &ConfigMapCAController{}

// NewDynamicCAFromConfigMapController returns a CAContentProvider based on a configmap that automatically reloads content.
// It is near-realtime via an informer.
func NewDynamicCAFromConfigMapController(purpose, namespace, name, key string, kubeClient kubernetes.Interface) (*ConfigMapCAController, error) {
	if len(purpose) == 0 {
		return nil, fmt.Errorf("missing purpose for ca bundle")
	}
	if len(namespace) == 0 {
		return nil, fmt.Errorf("missing namespace for ca bundle")
	}
	if len(name) == 0 {
		return nil, fmt.Errorf("missing name for ca bundle")
	}
	if len(key) == 0 {
		return nil, fmt.Errorf("missing key for ca bundle")
	}
	caContentName := fmt.Sprintf("%s::%s::%s::%s", purpose, namespace, name, key)

	// we construct our own informer because we need such a small subset of the information available.  Just one namespace.
	uncastConfigmapInformer := corev1informers.NewFilteredConfigMapInformer(kubeClient, namespace, 12*time.Hour, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, func(listOptions *v1.ListOptions) {
		listOptions.FieldSelector = fields.OneTermEqualSelector("metadata.name", name).String()
	})

	configmapLister := corev1listers.NewConfigMapLister(uncastConfigmapInformer.GetIndexer())

	c := &ConfigMapCAController{
		name:               caContentName,
		configmapNamespace: namespace,
		configmapName:      name,
		configmapKey:       key,
		configmapLister:    configmapLister,
		configMapInformer:  uncastConfigmapInformer,

		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: fmt.Sprintf("DynamicConfigMapCABundle-%s", purpose)},
		),
		preRunCaches: []cache.InformerSynced{uncastConfigmapInformer.HasSynced},
	}

	uncastConfigmapInformer.AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			if cast, ok := obj.(*corev1.ConfigMap); ok {
				return cast.Name == c.configmapName && cast.Namespace == c.configmapNamespace
			}
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				if cast, ok := tombstone.Obj.(*corev1.ConfigMap); ok {
					return cast.Name == c.configmapName && cast.Namespace == c.configmapNamespace
				}
			}
			return true // always return true just in case.  The checks are fairly cheap
		},
		Handler: cache.ResourceEventHandlerFuncs{
			// we have a filter, so any time we're called, we may as well queue. We only ever check one configmap
			// so we don't have to be choosy about our key.
			AddFunc: func(obj interface{}) {
				c.queue.Add(c.keyFn())
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				c.queue.Add(c.keyFn())
			},
			DeleteFunc: func(obj interface{}) {
				c.queue.Add(c.keyFn())
			},
		},
	})

	return c, nil
}

func (c *ConfigMapCAController) keyFn() string {
	// this format matches DeletionHandlingMetaNamespaceKeyFunc for our single key
	return c.configmapNamespace + "/" + c.configmapName
}

// AddListener adds a listener to be notified when the CA content changes.
func (c *ConfigMapCAController) AddListener(listener Listener) {
	c.listeners = append(c.listeners, listener)
}

// loadCABundle determines the next set of content for the file.
func (c *ConfigMapCAController) loadCABundle() error {
	configMap, err := c.configmapLister.ConfigMaps(c.configmapNamespace).Get(c.configmapName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// the configmap has been deleted, so stop verifying against a CA that no longer
			// exists. nothing is stored when nothing was ever loaded: RunOnce reads this lister
			// before Run starts the informer, so a cold start always lands here, and the
			// periodic resync must not re-notify listeners on every pass.
			if c.caBundle.Load() == nil {
				return nil
			}
			c.storeAndNotify(nil)
			klog.InfoS("Cleared CA bundle, configmap was deleted", "name", c.Name())

			return nil
		}
		return err
	}
	caBundle := configMap.Data[c.configmapKey]
	if len(caBundle) == 0 {
		return fmt.Errorf("missing content for CA bundle %q", c.Name())
	}

	// check to see if we have a change. If the values are the same, do nothing.
	if !c.hasCAChanged([]byte(caBundle)) {
		return nil
	}

	caBundleAndVerifier, err := newCABundleAndVerifier(c.Name(), []byte(caBundle))
	if err != nil {
		return err
	}
	c.storeAndNotify(caBundleAndVerifier)

	return nil
}

// storeAndNotify publishes the given bundle and wakes all listeners so dependent TLS
// configuration can pick it up.
func (c *ConfigMapCAController) storeAndNotify(caBundleAndVerifier *caBundleAndVerifier) {
	c.caBundle.Store(caBundleAndVerifier)

	for _, listener := range c.listeners {
		listener.Enqueue()
	}
}

// hasCAChanged returns true if the caBundle is different than the current.
func (c *ConfigMapCAController) hasCAChanged(caBundle []byte) bool {
	existing := c.caBundle.Load()
	if existing == nil {
		return true
	}

	// check to see if we have a change. If the values are the same, do nothing.
	if !bytes.Equal(existing.caBundle, caBundle) {
		return true
	}

	return false
}

// RunOnce runs a single sync loop
func (c *ConfigMapCAController) RunOnce(ctx context.Context) error {
	// Ignore the error when running once because when using a dynamically loaded ca file, because we think it's better to have nothing for
	// a brief time than completely crash.  If crashing is necessary, higher order logic like a healthcheck and cause failures.
	_ = c.loadCABundle()
	return nil
}

// Run starts the kube-apiserver and blocks until stopCh is closed.
func (c *ConfigMapCAController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer c.queue.ShutDown()

	klog.InfoS("Starting controller", "name", c.name)
	defer klog.InfoS("Shutting down controller", "name", c.name)

	// we have a personal informer that is narrowly scoped, start it.
	go c.configMapInformer.Run(ctx.Done())

	// wait for your secondary caches to fill before starting your work
	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.preRunCaches...) {
		return
	}

	// doesn't matter what workers say, only start one.
	go wait.Until(c.runWorker, time.Second, ctx.Done())

	// start timer that rechecks every minute, just in case.  this also serves to prime the controller quickly.
	go wait.PollImmediateUntil(FileRefreshDuration, func() (bool, error) {
		c.queue.Add(workItemKey)
		return false, nil
	}, ctx.Done())

	<-ctx.Done()
}

func (c *ConfigMapCAController) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *ConfigMapCAController) processNextWorkItem() bool {
	dsKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(dsKey)

	err := c.loadCABundle()
	if err == nil {
		c.queue.Forget(dsKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", dsKey, err))
	c.queue.AddRateLimited(dsKey)

	return true
}

// Name is just an identifier
func (c *ConfigMapCAController) Name() string {
	return c.name
}

// CurrentCABundleContent provides ca bundle byte content
func (c *ConfigMapCAController) CurrentCABundleContent() []byte {
	caBundle := c.caBundle.Load()
	if caBundle == nil {
		return nil // this can happen if we've been unable load data from the apiserver for some reason
	}
	return caBundle.caBundle
}

// VerifyOptions provides verifyoptions compatible with authenticators
func (c *ConfigMapCAController) VerifyOptions() (x509.VerifyOptions, bool) {
	caBundle := c.caBundle.Load()
	if caBundle == nil {
		// This covers both a configmap that was never loaded and one that has been deleted.
		// A zero x509.VerifyOptions has nil Roots, which certificate verification treats as
		// "use the host trust store", so returning it here would widen trust rather than
		// remove it. Reporting unavailable makes verification fail closed instead.
		return x509.VerifyOptions{}, false
	}

	return caBundle.verifyOptions, true
}
