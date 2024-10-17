/*
Copyright 2024 The Kubernetes Authors.

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

package storage

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	"k8s.io/kubernetes/pkg/features"
)

func TestCreate_Token_NamespaceNotExist(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	// Enable JTI feature
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAccountTokenJTI, true)()

	ctx := context.Background()
	// Create a test service account
	serviceAccount := validNewServiceAccount("foo")

	// create an audit context to allow recording audit information
	ctx = audit.WithAuditContext(ctx)
	_, err := storage.Token.Create(ctx, serviceAccount.Name, &authenticationapi.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceAccount.Name,
			Namespace: serviceAccount.Namespace,
		},
		Spec: authenticationapi.TokenRequestSpec{ExpirationSeconds: 3600},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err == nil {
		t.Errorf("should create namespace first.")
	}
}

func TestCreate_Token_NamespaceNameNotMatched(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	// Enable JTI feature
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAccountTokenJTI, true)()

	ctx := context.Background()
	// Create a test service account
	serviceAccount := validNewServiceAccount("foo")
	// add the namespace to the context as it is required
	ctx = request.WithNamespace(ctx, serviceAccount.Namespace)
	_, err := storage.Store.Create(ctx, serviceAccount, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed creating test service account: %v", err)
	}

	// create an audit context to allow recording audit information
	ctx = audit.WithAuditContext(ctx)
	_, err = storage.Token.Create(ctx, serviceAccount.Name, &authenticationapi.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: serviceAccount.Namespace,
		},
		Spec: authenticationapi.TokenRequestSpec{ExpirationSeconds: 3600},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err == nil {
		t.Errorf("service account name should be aligned with name in request.")
	}

	_, err = storage.Token.Create(ctx, serviceAccount.Name, &authenticationapi.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: serviceAccount.Namespace,
		},
		Spec: authenticationapi.TokenRequestSpec{ExpirationSeconds: 3600},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err == nil {
		t.Errorf("service account namespace should be aligned with namespace in request.")
	}

	// test name/namespace populate
	_, err = storage.Token.Create(ctx, serviceAccount.Name, &authenticationapi.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "",
			Namespace: "",
		},
		Spec: authenticationapi.TokenRequestSpec{
			ExpirationSeconds: 3600,
		},
	}, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed calling /token endpoint for service account: %v", err)
	}
}
