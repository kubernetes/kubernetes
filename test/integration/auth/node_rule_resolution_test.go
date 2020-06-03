/*
Copyright 2020 The Kubernetes Authors.

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

package auth

import (
	"context"
	"testing"

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestNodeAuthorizerRuleResolutionWithNodeUser(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	// Test server has node_authorizer enabled by default
	_, kubeAPIServerClientConfig := framework.StartTestServer(t, stopCh, framework.TestServerSetup{})

	srr := &authorizationv1.SelfSubjectRulesReview{
		Spec: authorizationv1.SelfSubjectRulesReviewSpec{
			Namespace: "default",
		},
	}

	t.Run("Test node authorizer rule resolution with node user", func(t *testing.T) {

		kubeAPIServerClientConfig.Impersonate.UserName = "system:node:foo"
		kubeAPIServerClientConfig.Impersonate.Groups = []string{"system:nodes", "system:authenticated"}

		impersonationClient, _ := clientset.NewForConfig(kubeAPIServerClientConfig)

		result, err := impersonationClient.AuthorizationV1().SelfSubjectRulesReviews().Create(context.TODO(), srr, metav1.CreateOptions{})

		expectedIncompletenessResult := true
		expectedEvaluationResult := "node authorizer does not support user rule resolution"

		if err != nil {
			t.Fatal(err)
		}

		if result.Status.Incomplete != expectedIncompletenessResult {
			t.Errorf("Rule resolution incompleteness: expected %v, got %v", expectedIncompletenessResult, result.Status.Incomplete)
		}

		if result.Status.EvaluationError != expectedEvaluationResult {
			t.Errorf("Rule resolution evaluation error: expected %v, got %v", expectedEvaluationResult, result.Status.EvaluationError)
		}

	})
}

func TestNodeAuthorizerRuleResolutionWithNonNodeUser(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	_, kubeAPIServerClientConfig := framework.StartTestServer(t, stopCh, framework.TestServerSetup{})

	srr := &authorizationv1.SelfSubjectRulesReview{
		Spec: authorizationv1.SelfSubjectRulesReviewSpec{
			Namespace: "default",
		},
	}

	t.Run("Test node authorizer rule resolution with non-node/regular user", func(t *testing.T) {

		kubeAPIServerClientConfig.Impersonate.UserName = "user:foo"
		kubeAPIServerClientConfig.Impersonate.Groups = []string{"user:foo", "system:authenticated"}

		impersonationClient, _ := clientset.NewForConfig(kubeAPIServerClientConfig)

		result, err := impersonationClient.AuthorizationV1().SelfSubjectRulesReviews().Create(context.TODO(), srr, metav1.CreateOptions{})

		expectedIncompletenessResult := false
		expectedEvaluationResult := ""

		if err != nil {
			t.Fatal(err)
		}

		if result.Status.Incomplete != expectedIncompletenessResult {
			t.Errorf("Rule resolution incompleteness: expected %v, got %v", expectedIncompletenessResult, result.Status.Incomplete)
		}

		if result.Status.EvaluationError != expectedEvaluationResult {
			t.Errorf("Rule resolution evaluation error: expected %v, got %v", expectedEvaluationResult, result.Status.EvaluationError)
		}

	})
}
