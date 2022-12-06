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
	"strconv"
	"time"

	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/kubernetes/scheme"
	authenticationv1client "k8s.io/client-go/kubernetes/typed/authentication/v1"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

// DefaultRetryBackoff returns the default backoff parameters for webhook retry.
func DefaultRetryBackoff() *wait.Backoff {
	backoff := webhook.DefaultRetryBackoffWithInitialDelay(500 * time.Millisecond)
	return &backoff
}

// Ensure WebhookTokenAuthenticator implements the authenticator.Token interface.
var _ authenticator.Token = (*WebhookTokenAuthenticator)(nil)

type tokenReviewer interface {
	Create(ctx context.Context, review *authenticationv1.TokenReview, _ metav1.CreateOptions) (*authenticationv1.TokenReview, int, error)
}

type WebhookTokenAuthenticator struct {
	tokenReview    tokenReviewer
	retryBackoff   wait.Backoff
	implicitAuds   authenticator.Audiences
	requestTimeout time.Duration
	metrics        AuthenticatorMetrics
}

// NewFromInterface creates a webhook authenticator using the given tokenReview
// client. It is recommend to wrap this authenticator with the token cache
// authenticator implemented in
// k8s.io/apiserver/pkg/authentication/token/cache.
func NewFromInterface(tokenReview authenticationv1client.AuthenticationV1Interface, implicitAuds authenticator.Audiences, retryBackoff wait.Backoff, requestTimeout time.Duration, metrics AuthenticatorMetrics) (*WebhookTokenAuthenticator, error) {
	tokenReviewClient := &tokenReviewV1Client{tokenReview.RESTClient()}
	return newWithBackoff(tokenReviewClient, retryBackoff, implicitAuds, requestTimeout, metrics)
}

// New creates a new WebhookTokenAuthenticator from the provided rest
// config. It is recommend to wrap this authenticator with the token cache
// authenticator implemented in
// k8s.io/apiserver/pkg/authentication/token/cache.
func New(config *rest.Config, version string, implicitAuds authenticator.Audiences, retryBackoff wait.Backoff) (*WebhookTokenAuthenticator, error) {
	tokenReview, err := tokenReviewInterfaceFromConfig(config, version, retryBackoff)
	if err != nil {
		return nil, err
	}
	return newWithBackoff(tokenReview, retryBackoff, implicitAuds, time.Duration(0), AuthenticatorMetrics{
		RecordRequestTotal:   noopMetrics{}.RequestTotal,
		RecordRequestLatency: noopMetrics{}.RequestLatency,
	})
}

// newWithBackoff allows tests to skip the sleep.
func newWithBackoff(tokenReview tokenReviewer, retryBackoff wait.Backoff, implicitAuds authenticator.Audiences, requestTimeout time.Duration, metrics AuthenticatorMetrics) (*WebhookTokenAuthenticator, error) {
	return &WebhookTokenAuthenticator{
		tokenReview,
		retryBackoff,
		implicitAuds,
		requestTimeout,
		metrics,
	}, nil
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
		auds   authenticator.Audiences
		cancel context.CancelFunc
	)

	// set a hard timeout if it was defined
	// if the child has a shorter deadline then it will expire first,
	// otherwise if the parent has a shorter deadline then the parent will expire and it will be propagate to the child
	if w.requestTimeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, w.requestTimeout)
		defer cancel()
	}

	// WithExponentialBackoff will return tokenreview create error (tokenReviewErr) if any.
	if err := webhook.WithExponentialBackoff(ctx, w.retryBackoff, func() error {
		var tokenReviewErr error
		var statusCode int

		start := time.Now()
		result, statusCode, tokenReviewErr = w.tokenReview.Create(ctx, r, metav1.CreateOptions{})
		latency := time.Since(start)

		if statusCode != 0 {
			w.metrics.RecordRequestTotal(ctx, strconv.Itoa(statusCode))
			w.metrics.RecordRequestLatency(ctx, strconv.Itoa(statusCode), latency.Seconds())
			return tokenReviewErr
		}

		if tokenReviewErr != nil {
			w.metrics.RecordRequestTotal(ctx, "<error>")
			w.metrics.RecordRequestLatency(ctx, "<error>", latency.Seconds())
		}
		return tokenReviewErr
	}, webhook.DefaultShouldRetry); err != nil {
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

// tokenReviewInterfaceFromConfig builds a client from the specified kubeconfig file,
// and returns a TokenReviewInterface that uses that client. Note that the client submits TokenReview
// requests to the exact path specified in the kubeconfig file, so arbitrary non-API servers can be targeted.
func tokenReviewInterfaceFromConfig(config *rest.Config, version string, retryBackoff wait.Backoff) (tokenReviewer, error) {
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
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, config, groupVersions, retryBackoff)
		if err != nil {
			return nil, err
		}
		return &tokenReviewV1ClientGW{gw.RestClient}, nil

	case authenticationv1beta1.SchemeGroupVersion.Version:
		groupVersions := []schema.GroupVersion{authenticationv1beta1.SchemeGroupVersion}
		if err := localScheme.SetVersionPriority(groupVersions...); err != nil {
			return nil, err
		}
		gw, err := webhook.NewGenericWebhook(localScheme, scheme.Codecs, config, groupVersions, retryBackoff)
		if err != nil {
			return nil, err
		}
		return &tokenReviewV1beta1ClientGW{gw.RestClient}, nil

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
	client rest.Interface
}

// Create takes the representation of a tokenReview and creates it.  Returns the server's representation of the tokenReview, HTTP status code and an error, if there is any.
func (c *tokenReviewV1Client) Create(ctx context.Context, tokenReview *authenticationv1.TokenReview, opts metav1.CreateOptions) (result *authenticationv1.TokenReview, statusCode int, err error) {
	result = &authenticationv1.TokenReview{}

	restResult := c.client.Post().
		Resource("tokenreviews").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(tokenReview).
		Do(ctx)

	restResult.StatusCode(&statusCode)
	err = restResult.Into(result)
	return
}

// tokenReviewV1ClientGW used by the generic webhook, doesn't specify GVR.
type tokenReviewV1ClientGW struct {
	client rest.Interface
}

// Create takes the representation of a tokenReview and creates it.  Returns the server's representation of the tokenReview, HTTP status code and an error, if there is any.
func (c *tokenReviewV1ClientGW) Create(ctx context.Context, tokenReview *authenticationv1.TokenReview, opts metav1.CreateOptions) (result *authenticationv1.TokenReview, statusCode int, err error) {
	result = &authenticationv1.TokenReview{}

	restResult := c.client.Post().
		Body(tokenReview).
		Do(ctx)

	restResult.StatusCode(&statusCode)
	err = restResult.Into(result)
	return
}

// tokenReviewV1beta1ClientGW used by the generic webhook, doesn't specify GVR.
type tokenReviewV1beta1ClientGW struct {
	client rest.Interface
}

func (t *tokenReviewV1beta1ClientGW) Create(ctx context.Context, review *authenticationv1.TokenReview, _ metav1.CreateOptions) (*authenticationv1.TokenReview, int, error) {
	var statusCode int
	v1beta1Review := &authenticationv1beta1.TokenReview{Spec: v1SpecToV1beta1Spec(&review.Spec)}
	v1beta1Result := &authenticationv1beta1.TokenReview{}

	restResult := t.client.Post().Body(v1beta1Review).Do(ctx)
	restResult.StatusCode(&statusCode)
	err := restResult.Into(v1beta1Result)
	if err != nil {
		return nil, statusCode, err
	}
	review.Status = v1beta1StatusToV1Status(&v1beta1Result.Status)
	return review, statusCode, nil
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
