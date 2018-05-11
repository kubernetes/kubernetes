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

package bootstrap

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	bootstrapapi "k8s.io/client-go/tools/bootstrap/token/api"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
)

// TokenCleanerOptions contains options for the TokenCleaner
type TokenCleanerOptions struct {
	// TokenSecretNamespace string is the namespace for token Secrets.
	TokenSecretNamespace string

	// SecretResync is the time.Duration at which to fully re-list secrets.
	// If zero, re-list will be delayed as long as possible
	SecretResync time.Duration
}

// DefaultTokenCleanerOptions returns a set of default options for creating a
// TokenCleaner
func DefaultTokenCleanerOptions() TokenCleanerOptions {
	return TokenCleanerOptions{
		TokenSecretNamespace: api.NamespaceSystem,
	}
}

// TokenCleaner is a controller that deletes expired tokens
type TokenCleaner struct {
	tokenSecretNamespace string

	client clientset.Interface

	// secretLister is able to list/get secrets and is populated by the shared informer passed to NewTokenCleaner.
	secretLister corelisters.SecretLister

	// secretSynced returns true if the secret shared informer has been synced at least once.
	secretSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface
}

// NewTokenCleaner returns a new *NewTokenCleaner.
func NewTokenCleaner(cl clientset.Interface, secrets coreinformers.SecretInformer, options TokenCleanerOptions) (*TokenCleaner, error) {
	e := &TokenCleaner{
		client:               cl,
		secretLister:         secrets.Lister(),
		secretSynced:         secrets.Informer().HasSynced,
		tokenSecretNamespace: options.TokenSecretNamespace,
		queue:                workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "token_cleaner"),
	}

	if cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := metrics.RegisterMetricAndTrackRateLimiterUsage("token_cleaner", cl.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	secrets.Informer().AddEventHandlerWithResyncPeriod(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Secret:
					return t.Type == bootstrapapi.SecretTypeBootstrapToken && t.Namespace == e.tokenSecretNamespace
				default:
					utilruntime.HandleError(fmt.Errorf("object passed to %T that is not expected: %T", e, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    e.enqueueSecrets,
				UpdateFunc: func(oldSecret, newSecret interface{}) { e.enqueueSecrets(newSecret) },
			},
		},
		options.SecretResync,
	)

	return e, nil
}

// Run runs controller loops and returns when they are done
func (tc *TokenCleaner) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer tc.queue.ShutDown()

	glog.Infof("Starting token cleaner controller")
	defer glog.Infof("Shutting down token cleaner controller")

	if !controller.WaitForCacheSync("token_cleaner", stopCh, tc.secretSynced) {
		return
	}

	go wait.Until(tc.worker, 10*time.Second, stopCh)

	<-stopCh
}

func (tc *TokenCleaner) enqueueSecrets(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	tc.queue.Add(key)
}

// worker runs a thread that dequeues secrets, handles them, and marks them done.
func (tc *TokenCleaner) worker() {
	for tc.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (tc *TokenCleaner) processNextWorkItem() bool {
	key, quit := tc.queue.Get()
	if quit {
		return false
	}
	defer tc.queue.Done(key)

	if err := tc.syncFunc(key.(string)); err != nil {
		tc.queue.AddRateLimited(key)
		utilruntime.HandleError(fmt.Errorf("Sync %v failed with : %v", key, err))
		return true
	}

	tc.queue.Forget(key)
	return true
}

func (tc *TokenCleaner) syncFunc(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing secret %q (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	ret, err := tc.secretLister.Secrets(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		glog.V(3).Infof("secret has been deleted: %v", key)
		return nil
	}

	if err != nil {
		return err
	}

	if ret.Type == bootstrapapi.SecretTypeBootstrapToken {
		tc.evalSecret(ret)
	}
	return nil
}

func (tc *TokenCleaner) evalSecret(o interface{}) {
	secret := o.(*v1.Secret)
	if isSecretExpired(secret) {
		glog.V(3).Infof("Deleting expired secret %s/%s", secret.Namespace, secret.Name)
		var options *metav1.DeleteOptions
		if len(secret.UID) > 0 {
			options = &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &secret.UID}}
		}
		err := tc.client.CoreV1().Secrets(secret.Namespace).Delete(secret.Name, options)
		// NotFound isn't a real error (it's already been deleted)
		// Conflict isn't a real error (the UID precondition failed)
		if err != nil && !apierrors.IsConflict(err) && !apierrors.IsNotFound(err) {
			glog.V(3).Infof("Error deleting Secret: %v", err)
		}
	}
}
