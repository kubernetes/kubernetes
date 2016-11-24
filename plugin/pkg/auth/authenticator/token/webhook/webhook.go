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

// Package webhook implements the authenticator.Token interface using HTTP webhooks.
package webhook

import (
	"time"

	_ "k8s.io/kubernetes/pkg/apis/authentication/install"
	authentication "k8s.io/kubernetes/pkg/apis/authentication/v1beta1"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	authenticationclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authentication/v1beta1"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/cache"
	"k8s.io/kubernetes/plugin/pkg/webhook"
)

var (
	groupVersions = []schema.GroupVersion{authentication.SchemeGroupVersion}
)

const retryBackoff = 500 * time.Millisecond

// Ensure WebhookTokenAuthenticator implements the authenticator.Token interface.
var _ authenticator.Token = (*WebhookTokenAuthenticator)(nil)

type WebhookTokenAuthenticator struct {
	tokenReview    authenticationclient.TokenReviewInterface
	responseCache  *cache.LRUExpireCache
	ttl            time.Duration
	initialBackoff time.Duration
}

// NewFromInterface creates a webhook authenticator using the given tokenReview client
func NewFromInterface(tokenReview authenticationclient.TokenReviewInterface, ttl time.Duration) (*WebhookTokenAuthenticator, error) {
	return newWithBackoff(tokenReview, ttl, retryBackoff)
}

// New creates a new WebhookTokenAuthenticator from the provided kubeconfig file.
func New(kubeConfigFile string, ttl time.Duration) (*WebhookTokenAuthenticator, error) {
	tokenReview, err := tokenReviewInterfaceFromKubeconfig(kubeConfigFile)
	if err != nil {
		return nil, err
	}
	return newWithBackoff(tokenReview, ttl, retryBackoff)
}

// newWithBackoff allows tests to skip the sleep.
func newWithBackoff(tokenReview authenticationclient.TokenReviewInterface, ttl, initialBackoff time.Duration) (*WebhookTokenAuthenticator, error) {
	return &WebhookTokenAuthenticator{tokenReview, cache.NewLRUExpireCache(1024), ttl, initialBackoff}, nil
}

// AuthenticateToken implements the authenticator.Token interface.
func (w *WebhookTokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	r := &authentication.TokenReview{
		Spec: authentication.TokenReviewSpec{Token: token},
	}
	if entry, ok := w.responseCache.Get(r.Spec); ok {
		r.Status = entry.(authentication.TokenReviewStatus)
	} else {
		var (
			result *authentication.TokenReview
			err    error
		)
		webhook.WithExponentialBackoff(w.initialBackoff, func() error {
			result, err = w.tokenReview.Create(r)
			return err
		})
		if err != nil {
			return nil, false, err
		}
		r.Status = result.Status
		w.responseCache.Add(r.Spec, result.Status, w.ttl)
	}
	if !r.Status.Authenticated {
		return nil, false, nil
	}

	var extra map[string][]string
	if r.Status.User.Extra != nil {
		extra = map[string][]string{}
		for k, v := range r.Status.User.Extra {
			extra[k] = v
		}
	}

	return &user.DefaultInfo{
		Name:   r.Status.User.Username,
		UID:    r.Status.User.UID,
		Groups: r.Status.User.Groups,
		Extra:  extra,
	}, true, nil
}

// tokenReviewInterfaceFromKubeconfig builds a client from the specified kubeconfig file,
// and returns a TokenReviewInterface that uses that client. Note that the client submits TokenReview
// requests to the exact path specified in the kubeconfig file, so arbitrary non-API servers can be targeted.
func tokenReviewInterfaceFromKubeconfig(kubeConfigFile string) (authenticationclient.TokenReviewInterface, error) {
	gw, err := webhook.NewGenericWebhook(kubeConfigFile, groupVersions, 0)
	if err != nil {
		return nil, err
	}
	return &tokenReviewClient{gw}, nil
}

type tokenReviewClient struct {
	w *webhook.GenericWebhook
}

func (t *tokenReviewClient) Create(tokenReview *authentication.TokenReview) (*authentication.TokenReview, error) {
	result := &authentication.TokenReview{}
	err := t.w.RestClient.Post().Body(tokenReview).Do().Into(result)
	return result, err
}
