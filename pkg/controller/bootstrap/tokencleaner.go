/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/watch"
)

// TokenCleanerOptions contains options for the TokenCleaner
type TokenCleanerOptions struct {

	// SecretResync is the time.Duration at which to fully re-list secrets.
	// If zero, re-list will be delayed as long as possible
	SecretResync time.Duration

	// MaxRetries controls the maximum number of times a particular key is retried before giving up
	// If zero, a default max is used
	MaxRetries int
}

// DefaultBootstrapSignerOptions returns a set of default options for creating a
// TokenCleaner
func DefaultTokenCleanerOptions() TokenCleanerOptions {
	return TokenCleanerOptions{}
}

// TokenCleaner is a controller that deletes expired tokens
type TokenCleaner struct {
	stopChan chan struct{}

	client clientset.Interface

	secrets           cache.Store
	secretsController *cache.Controller

	maxRetries int
}

func NewTokenCleaner(cl clientset.Interface, options TokenCleanerOptions) *TokenCleaner {
	maxRetries := options.MaxRetries
	if maxRetries == 0 {
		maxRetries = 10
	}

	e := &TokenCleaner{
		client:     cl,
		maxRetries: maxRetries,
	}
	if cl != nil && cl.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("token_cleaner", cl.Core().RESTClient().GetRateLimiter())
	}

	secretSelector := fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(api.SecretTypeBootstrapToken)})
	e.secrets, e.secretsController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(lo api.ListOptions) (runtime.Object, error) {
				lo.FieldSelector = secretSelector
				return e.client.Core().Secrets(api.NamespaceSystem).List(lo)
			},
			WatchFunc: func(lo api.ListOptions) (watch.Interface, error) {
				lo.FieldSelector = secretSelector
				return e.client.Core().Secrets(api.NamespaceSystem).Watch(lo)
			},
		},
		&api.Secret{},
		options.SecretResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    e.evalSecret,
			UpdateFunc: func(old, new interface{}) { e.evalSecret(new) },
		},
	)
	return e
}

func (e *TokenCleaner) evalSecret(o interface{}) {
	secret := o.(*api.Secret)
	if isSecretExpired(secret) {
		glog.V(3).Infof("Deleting expired secret %s/%s", secret.Namespace, secret.Name)
		for i := 0; i < e.maxRetries; i++ {
			var options *api.DeleteOptions
			if len(secret.UID) > 0 {
				options = &api.DeleteOptions{Preconditions: &api.Preconditions{UID: &secret.UID}}
			}
			err := e.client.Core().Secrets(secret.Namespace).Delete(secret.Name, options)
			// NotFound doesn't need a retry (it's already been deleted)
			// Conflict doesn't need a retry (the UID precondition failed)
			if err == nil || apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
				return
			}
			glog.Errorf("Error deleting secret. Try %d/%d. Error: %v", i+1, e.maxRetries, err)
		}
	}
}
