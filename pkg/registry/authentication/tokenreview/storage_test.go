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
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/apis/authentication"
)

type contextKey string

func TestCreatePropagatesCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	storage := NewREST(authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		if err := req.Context().Err(); !errors.Is(err, context.Canceled) {
			return nil, false, fmt.Errorf("expected canceled request context, got %v", err)
		}

		return nil, false, req.Context().Err()
	}), nil)

	obj, err := storage.Create(ctx, newTokenReview("test-token", nil), nil, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Create returned error: %v", err)
	}

	status := obj.(*authentication.TokenReview).Status
	if status.Authenticated {
		t.Fatal("expected unauthenticated TokenReview")
	}
	if status.Error != context.Canceled.Error() {
		t.Fatalf("expected status error %q, got %q", context.Canceled.Error(), status.Error)
	}
}

func TestCreatePropagatesContextDeadline(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	storage := NewREST(authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		if err := req.Context().Err(); !errors.Is(err, context.DeadlineExceeded) {
			return nil, false, fmt.Errorf("expected deadline exceeded request context, got %v", err)
		}

		return nil, false, req.Context().Err()
	}), nil)

	obj, err := storage.Create(ctx, newTokenReview("test-token", nil), nil, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Create returned error: %v", err)
	}

	status := obj.(*authentication.TokenReview).Status
	if status.Authenticated {
		t.Fatal("expected unauthenticated TokenReview")
	}
	if status.Error != context.DeadlineExceeded.Error() {
		t.Fatalf("expected status error %q, got %q", context.DeadlineExceeded.Error(), status.Error)
	}
}

func TestCreateStripsParentContextValuesAndPreservesAudiences(t *testing.T) {
	parentKey := contextKey("request-value")
	ctx := context.WithValue(context.Background(), parentKey, "value")
	wantAuds := []string{"api", "kubernetes"}

	storage := NewREST(authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		if got := req.Context().Value(parentKey); got != nil {
			return nil, false, fmt.Errorf("expected parent context value to be hidden, got %v", got)
		}

		auds, ok := authenticator.AudiencesFrom(req.Context())
		if !ok {
			return nil, false, errors.New("expected request audiences")
		}
		if !reflect.DeepEqual(auds, authenticator.Audiences(wantAuds)) {
			return nil, false, fmt.Errorf("expected audiences %v, got %v", wantAuds, auds)
		}

		return &authenticator.Response{
			Audiences: auds,
			User:      &user.DefaultInfo{Name: "test-user"},
		}, true, nil
	}), wantAuds)

	obj, err := storage.Create(ctx, newTokenReview("test-token", nil), nil, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Create returned error: %v", err)
	}

	status := obj.(*authentication.TokenReview).Status
	if !status.Authenticated {
		t.Fatal("expected authenticated TokenReview")
	}
	if status.Error != "" {
		t.Fatalf("expected no status error, got %q", status.Error)
	}
	if status.User.Username != "test-user" {
		t.Fatalf("expected username %q, got %q", "test-user", status.User.Username)
	}
	if !reflect.DeepEqual(status.Audiences, wantAuds) {
		t.Fatalf("expected status audiences %v, got %v", wantAuds, status.Audiences)
	}
}

func newTokenReview(token string, audiences []string) *authentication.TokenReview {
	return &authentication.TokenReview{
		Spec: authentication.TokenReviewSpec{
			Token:     token,
			Audiences: audiences,
		},
	}
}
