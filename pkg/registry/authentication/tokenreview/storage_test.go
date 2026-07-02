/*
Copyright 2026 The Kubernetes Authors.

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

package tokenreview

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/apis/authentication"
)

type testContextKey struct{}

func TestCreatePropagatesCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	storage := NewREST(authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		select {
		case <-req.Context().Done():
			return nil, false, req.Context().Err()
		default:
			return nil, false, errors.New("request context was not canceled")
		}
	}), nil)

	obj, err := storage.Create(ctx, &authentication.TokenReview{
		Spec: authentication.TokenReviewSpec{
			Token: "test-token",
		},
	}, nil, &metav1.CreateOptions{})
	require.NoError(t, err)

	tokenReview, ok := obj.(*authentication.TokenReview)
	require.True(t, ok)
	assert.Equal(t, context.Canceled.Error(), tokenReview.Status.Error)
}

func TestCreatePropagatesContextDeadline(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	storage := NewREST(authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		select {
		case <-req.Context().Done():
			return nil, false, req.Context().Err()
		default:
			return nil, false, errors.New("request context deadline was not propagated")
		}
	}), []string{"api"})

	obj, err := storage.Create(ctx, &authentication.TokenReview{
		Spec: authentication.TokenReviewSpec{
			Token: "test-token",
		},
	}, nil, &metav1.CreateOptions{})
	require.NoError(t, err)

	tokenReview, ok := obj.(*authentication.TokenReview)
	require.True(t, ok)
	assert.Equal(t, context.DeadlineExceeded.Error(), tokenReview.Status.Error)
}

func TestCreateStripsParentContextValues(t *testing.T) {
	ctx := context.WithValue(context.Background(), testContextKey{}, "sensitive-request-value")

	storage := NewREST(authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		if got := req.Context().Value(testContextKey{}); got != nil {
			return nil, false, fmt.Errorf("unexpected context value %v", got)
		}

		auds, ok := authenticator.AudiencesFrom(req.Context())
		if !ok {
			return nil, false, errors.New("expected audiences")
		}

		return &authenticator.Response{
			Audiences: auds,
			User:      &user.DefaultInfo{Name: "test-user"},
		}, true, nil
	}), []string{"api"})

	obj, err := storage.Create(ctx, &authentication.TokenReview{
		Spec: authentication.TokenReviewSpec{
			Token: "test-token",
		},
	}, nil, &metav1.CreateOptions{})
	require.NoError(t, err)

	tokenReview, ok := obj.(*authentication.TokenReview)
	require.True(t, ok)
	assert.True(t, tokenReview.Status.Authenticated)
	assert.Empty(t, tokenReview.Status.Error)
	assert.Equal(t, []string{"api"}, tokenReview.Status.Audiences)
}
