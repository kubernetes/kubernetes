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
	"context"
	"errors"
	"time"

	authentication "k8s.io/api/authentication/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/kubernetes/scheme"
	authenticationclient "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	"k8s.io/klog"
)

var (
	groupVersions = []schema.GroupVersion{authentication.SchemeGroupVersion}
)

const retryBackoff = 500 * time.Millisecond

// Ensure WebhookTokenAuthenticator implements the authenticator.Token interface.
var _ authenticator.Token = (*WebhookTokenAuthenticator)(nil)

type WebhookTokenAuthenticator struct {
	tokenReview    authenticationclient.TokenReviewInterface
	initialBackoff time.Duration
	implicitAuds   authenticator.Audiences
}

// NewFromInterface creates a webhook authenticator using the given tokenReview
// client. It is recommend to wrap this authenticator with the token cache
// authenticator implemented in
// k8s.io/apiserver/pkg/authentication/token/cache.
func NewFromInterface(tokenReview authenticationclient.TokenReviewInterface, implicitAuds authenticator.Audiences) (*WebhookTokenAuthenticator, error) {
	return newWithBackoff(tokenReview, retryBackoff, implicitAuds)
}

// New creates a new WebhookTokenAuthenticator from the provided kubeconfig
// file. It is recommend to wrap this authenticator with the token cache
// authenticator implemented in
// k8s.io/apiserver/pkg/authentication/token/cache.
func New(kubeConfigFile string, implicitAuds authenticator.Audiences) (*WebhookTokenAuthenticator, error) {
	tokenReview, err := tokenReviewInterfaceFromKubeconfig(kubeConfigFile)
	if err != nil {
		return nil, err
	}
	return newWithBackoff(tokenReview, retryBackoff, implicitAuds)
}

// newWithBackoff allows tests to skip the sleep.
func newWithBackoff(tokenReview authenticationclient.TokenReviewInterface, initialBackoff time.Duration, implicitAuds authenticator.Audiences) (*WebhookTokenAuthenticator, error) {
	return &WebhookTokenAuthenticator{tokenReview, initialBackoff, implicitAuds}, nil
}

// AuthenticateToken implements the authenticator.Token interface.
func (w *WebhookTokenAuthenticator) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	// We take implicit audiences of the API server at WebhookTokenAuthenticator
	// construction time. The outline of how we validate audience here is:
	//
	// * if the ctx is not audience limited, don't do any audience validation.
	// * if ctx is audience-limited, add the audiences to the tokenreview spec
	//   * if the tokenreview returns with audiences in the status that intersect
	//     with the audiences in the ctx, copy into the response and return success
	//   * if the tokenreview returns without an audience in the status, ensure
	//     the ctx audiences intersect with the implicit audiences, and set the
	//     intersection in the response.
	//   * otherwise return unauthenticated.
	wantAuds, checkAuds := authenticator.AudiencesFrom(ctx)
	r := &authentication.TokenReview{
		Spec: authentication.TokenReviewSpec{
			Token:     token,
			Audiences: wantAuds,
		},
	}
	var (
		result *authentication.TokenReview
		err    error
		auds   authenticator.Audiences
	)
	webhook.WithExponentialBackoff(w.initialBackoff, func() error {
		result, err = w.tokenReview.Create(r)
		return err
	})
	if err != nil {
		// An error here indicates bad configuration or an outage. Log for debugging.
		klog.Errorf("Failed to make webhook authenticator request: %v", err)
		return nil, false, err
	}

	if checkAuds {
		gotAuds := w.implicitAuds
		if len(result.Status.Audiences) > 0 {
			gotAuds = result.Status.Audiences
		}
		auds = wantAuds.Intersect(gotAuds)
		if len(auds) == 0 {
			return nil, false, nil
		}
	}

	r.Status = result.Status
	if !r.Status.Authenticated {
		var err error
		if len(r.Status.Error) != 0 {
			err = errors.New(r.Status.Error)
		}
		return nil, false, err
	}

	var extra map[string][]string
	if r.Status.User.Extra != nil {
		extra = map[string][]string{}
		for k, v := range r.Status.User.Extra {
			extra[k] = v
		}
	}

	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name:   r.Status.User.Username,
			UID:    r.Status.User.UID,
			Groups: r.Status.User.Groups,
			Extra:  extra,
		},
		Audiences: auds,
	}, true, nil
}

// tokenReviewInterfaceFromKubeconfig builds a client from the specified kubeconfig file,
// and returns a TokenReviewInterface that uses that client. Note that the client submits TokenReview
// requests to the exact path specified in the kubeconfig file, so arbitrary non-API servers can be targeted.
func tokenReviewInterfaceFromKubeconfig(kubeConfigFile string) (authenticationclient.TokenReviewInterface, error) {
	localScheme := runtime.NewScheme()
	if err := scheme.AddToScheme(localScheme); err != nil {
		return nil, err
	}
	if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
		return nil, err
	}

	gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, kubeConfigFile, groupVersions, 0)
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
