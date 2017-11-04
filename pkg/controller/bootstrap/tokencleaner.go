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
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
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

	secrets           cache.Store
	secretsController cache.Controller
}

// NewTokenCleaner returns a new *NewTokenCleaner.
//
// TODO: Switch to shared informers
func NewTokenCleaner(cl clientset.Interface, options TokenCleanerOptions) *TokenCleaner {
	e := &TokenCleaner{
		client:               cl,
		tokenSecretNamespace: options.TokenSecretNamespace,
	}
	if cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("token_cleaner", cl.CoreV1().RESTClient().GetRateLimiter())
	}

	secretSelector := fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(bootstrapapi.SecretTypeBootstrapToken)})
	e.secrets, e.secretsController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(lo metav1.ListOptions) (runtime.Object, error) {
				lo.FieldSelector = secretSelector.String()
				return e.client.CoreV1().Secrets(e.tokenSecretNamespace).List(lo)
			},
			WatchFunc: func(lo metav1.ListOptions) (watch.Interface, error) {
				lo.FieldSelector = secretSelector.String()
				return e.client.CoreV1().Secrets(e.tokenSecretNamespace).Watch(lo)
			},
		},
		&v1.Secret{},
		options.SecretResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    e.evalSecret,
			UpdateFunc: func(oldSecret, newSecret interface{}) { e.evalSecret(newSecret) },
		},
	)
	return e
}

// Run runs controller loops and returns when they are done
func (tc *TokenCleaner) Run(stopCh <-chan struct{}) {
	go tc.secretsController.Run(stopCh)
	go wait.Until(tc.evalSecrets, 10*time.Second, stopCh)
	<-stopCh
}

func (tc *TokenCleaner) evalSecrets() {
	for _, obj := range tc.secrets.List() {
		tc.evalSecret(obj)
	}
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
