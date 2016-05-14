/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/authentication.k8s.io/v1beta1"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/cache"
	"k8s.io/kubernetes/plugin/pkg/webhook"

	_ "k8s.io/kubernetes/pkg/apis/authentication.k8s.io/install"
)

var (
	groupVersions = []unversioned.GroupVersion{v1beta1.SchemeGroupVersion}
)

// Ensure WebhookTokenAuthenticator implements the authenticator.Token interface.
var _ authenticator.Token = (*WebhookTokenAuthenticator)(nil)

type WebhookTokenAuthenticator struct {
	*webhook.GenericWebhook
	responseCache *cache.LRUExpireCache
	ttl           time.Duration
}

// New creates a new WebhookTokenAuthenticator from the provided kubeconfig file.
func New(kubeConfigFile string, ttl time.Duration) (*WebhookTokenAuthenticator, error) {
	gw, err := webhook.NewGenericWebhook(kubeConfigFile, groupVersions)
	if err != nil {
		return nil, err
	}
	return &WebhookTokenAuthenticator{gw, cache.NewLRUExpireCache(1024), ttl}, nil
}

// AuthenticateToken implements the authenticator.Token interface.
func (w *WebhookTokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	r := &v1beta1.TokenReview{
		Spec: v1beta1.TokenReviewSpec{Token: token},
	}
	if entry, ok := w.responseCache.Get(r.Spec); ok {
		r.Status = entry.(v1beta1.TokenReviewStatus)
	} else {
		result := w.RestClient.Post().Body(r).Do()
		if err := result.Error(); err != nil {
			return nil, false, err
		}
		spec := r.Spec
		if err := result.Into(r); err != nil {
			return nil, false, err
		}
		go w.responseCache.Add(spec, r.Status, w.ttl)
	}
	if !r.Status.Authenticated {
		return nil, false, nil
	}
	return &user.DefaultInfo{
		Name:   r.Status.User.Username,
		UID:    r.Status.User.UID,
		Groups: r.Status.User.Groups,
	}, true, nil
}
