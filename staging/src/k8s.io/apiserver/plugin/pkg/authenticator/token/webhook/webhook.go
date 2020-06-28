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
	"fmt"
	"time"

	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/kubernetes/scheme"
	authenticationv1client "k8s.io/client-go/kubernetes/typed/authentication/v1"
	"k8s.io/klog/v2"
)

const retryBackoff = 500 * time.Millisecond

// Ensure WebhookTokenAuthenticator implements the authenticator.Token interface.
var _ authenticator.Token = (*WebhookTokenAuthenticator)(nil)

type tokenReviewer interface {
	Create(ctx context.Context, review *authenticationv1.TokenReview, _ metav1.CreateOptions) (*authenticationv1.TokenReview, error)
}

type WebhookTokenAuthenticator struct {
	tokenReview    tokenReviewer
	initialBackoff time.Duration
	implicitAuds   authenticator.Audiences
}

// NewFromInterface creates a webhook authenticator using the given tokenReview
// client. It is recommend to wrap this authenticator with the token cache
// authenticator implemented in
// k8s.io/apiserver/pkg/authentication/token/cache.
func NewFromInterface(tokenReview authenticationv1client.TokenReviewInterface, implicitAuds authenticator.Audiences) (*WebhookTokenAuthenticator, error) {
	return newWithBackoff(tokenReview, retryBackoff, implicitAuds)
}

// New creates a new WebhookTokenAuthenticator from the provided kubeconfig
// file. It is recommend to wrap this authenticator with the token cache
// authenticator implemented in
// k8s.io/apiserver/pkg/authentication/token/cache.
func New(kubeConfigFile string, version string, implicitAuds authenticator.Audiences, customDial utilnet.DialFunc) (*WebhookTokenAuthenticator, error) {
	tokenReview, err := tokenReviewInterfaceFromKubeconfig(kubeConfigFile, version, customDial)
	if err != nil {
		return nil, err
	}
	return newWithBackoff(tokenReview, retryBackoff, implicitAuds)
}

// newWithBackoff allows tests to skip the sleep.
func newWithBackoff(tokenReview tokenReviewer, initialBackoff time.Duration, implicitAuds authenticator.Audiences) (*WebhookTokenAuthenticator, error) {
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
	r := &authenticationv1.TokenReview{
		Spec: authenticationv1.TokenReviewSpec{
			Token:     token,
			Audiences: wantAuds,
		},
	}
	var (
		result *authenticationv1.TokenReview
		err    error
		auds   authenticator.Audiences
	)
	webhook.WithExponentialBackoff(ctx, w.initialBackoff, func() error {
		result, err = w.tokenReview.Create(ctx, r, metav1.CreateOptions{})
		return err
	}, webhook.DefaultShouldRetry)
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
func tokenReviewInterfaceFromKubeconfig(kubeConfigFile string, version string, customDial utilnet.DialFunc) (tokenReviewer, error) {
	localScheme := runtime.NewScheme()
	if err := scheme.AddToScheme(localScheme); err != nil {
		return nil, err
	}

	switch version {
	case authenticationv1.SchemeGroupVersion.Version:
		groupVersions := []schema.GroupVersion{authenticationv1.SchemeGroupVersion}
		if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
			return nil, err
		}
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, kubeConfigFile, groupVersions, 0, customDial)
		if err != nil {
			return nil, err
		}
		return &tokenReviewV1Client{gw}, nil

	case authenticationv1beta1.SchemeGroupVersion.Version:
		groupVersions := []schema.GroupVersion{authenticationv1beta1.SchemeGroupVersion}
		if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
			return nil, err
		}
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, kubeConfigFile, groupVersions, 0, customDial)
		if err != nil {
			return nil, err
		}
		return &tokenReviewV1beta1Client{gw}, nil

	default:
		return nil, fmt.Errorf(
			"unsupported authentication webhook version %q, supported versions are %q, %q",
			version,
			authenticationv1.SchemeGroupVersion.Version,
			authenticationv1beta1.SchemeGroupVersion.Version,
		)
	}

}

type tokenReviewV1Client struct {
	w *webhook.GenericWebhook
}

func (t *tokenReviewV1Client) Create(ctx context.Context, review *authenticationv1.TokenReview, _ metav1.CreateOptions) (*authenticationv1.TokenReview, error) {
	result := &authenticationv1.TokenReview{}
	err := t.w.RestClient.Post().Body(review).Do(ctx).Into(result)
	return result, err
}

type tokenReviewV1beta1Client struct {
	w *webhook.GenericWebhook
}

func (t *tokenReviewV1beta1Client) Create(ctx context.Context, review *authenticationv1.TokenReview, _ metav1.CreateOptions) (*authenticationv1.TokenReview, error) {
	v1beta1Review := &authenticationv1beta1.TokenReview{Spec: v1SpecToV1beta1Spec(&review.Spec)}
	v1beta1Result := &authenticationv1beta1.TokenReview{}
	err := t.w.RestClient.Post().Body(v1beta1Review).Do(ctx).Into(v1beta1Result)
	if err != nil {
		return nil, err
	}
	review.Status = v1beta1StatusToV1Status(&v1beta1Result.Status)
	return review, nil
}

func v1SpecToV1beta1Spec(in *authenticationv1.TokenReviewSpec) authenticationv1beta1.TokenReviewSpec {
	return authenticationv1beta1.TokenReviewSpec{
		Token:     in.Token,
		Audiences: in.Audiences,
	}
}

func v1beta1StatusToV1Status(in *authenticationv1beta1.TokenReviewStatus) authenticationv1.TokenReviewStatus {
	return authenticationv1.TokenReviewStatus{
		Authenticated: in.Authenticated,
		User:          v1beta1UserToV1User(in.User),
		Audiences:     in.Audiences,
		Error:         in.Error,
	}
}

func v1beta1UserToV1User(u authenticationv1beta1.UserInfo) authenticationv1.UserInfo {
	var extra map[string]authenticationv1.ExtraValue
	if u.Extra != nil {
		extra = make(map[string]authenticationv1.ExtraValue, len(u.Extra))
		for k, v := range u.Extra {
			extra[k] = authenticationv1.ExtraValue(v)
		}
	}
	return authenticationv1.UserInfo{
		Username: u.Username,
		UID:      u.UID,
		Groups:   u.Groups,
		Extra:    extra,
	}
}
