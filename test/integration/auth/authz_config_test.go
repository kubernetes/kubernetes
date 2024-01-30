/*
Copyright 2023 The Kubernetes Authors.

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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestAuthzConfig(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthorizationConfiguration, true)()

	dir := t.TempDir()
	configFileName := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configFileName, []byte(`
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: AuthorizationConfiguration
authorizers:
- type: RBAC
  name: rbac
`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	server := kubeapiservertesting.StartTestServerOrDie(
		t,
		nil,
		[]string{"--authorization-config=" + configFileName},
		framework.SharedEtcd(),
	)
	t.Cleanup(server.TearDownFn)

	// Make sure anonymous requests work
	anonymousClient := clientset.NewForConfigOrDie(rest.AnonymousClientConfig(server.ClientConfig))
	healthzResult, err := anonymousClient.DiscoveryClient.RESTClient().Get().AbsPath("/healthz").Do(context.TODO()).Raw()
	if !bytes.Equal(healthzResult, []byte(`ok`)) {
		t.Fatalf("expected 'ok', got %s", string(healthzResult))
	}
	if err != nil {
		t.Fatal(err)
	}

	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	sar := &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Namespace: "foo",
			Verb:      "create",
			Group:     "",
			Version:   "v1",
			Resource:  "configmaps",
		},
	}}
	result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Status.Allowed {
		t.Fatal("expected denied, got allowed")
	}

	authutil.GrantUserAuthorization(t, context.TODO(), adminClient, "alice",
		rbacv1.PolicyRule{
			Verbs:     []string{"create"},
			APIGroups: []string{""},
			Resources: []string{"configmaps"},
		},
	)

	result, err = adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Status.Allowed {
		t.Fatal("expected allowed, got denied")
	}
}

func TestMultiWebhookAuthzConfig(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StructuredAuthorizationConfiguration, true)()

	dir := t.TempDir()

	kubeconfigTemplate := `
apiVersion: v1
kind: Config
clusters:
- name: integration
  cluster:
    server: %q
    insecure-skip-tls-verify: true
contexts:
- name: default-context
  context:
    cluster: integration
    user: test
current-context: default-context
users:
- name: test
`

	// returns malformed responses when called
	serverErrorCalled := atomic.Int32{}
	serverError := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverErrorCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverError", sar)
		if _, err := w.Write([]byte(`error response`)); err != nil {
			t.Error(err)
		}
	}))
	defer serverError.Close()
	serverErrorKubeconfigName := filepath.Join(dir, "serverError.yaml")
	if err := os.WriteFile(serverErrorKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverError.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// hangs for 2 seconds when called
	serverTimeoutCalled := atomic.Int32{}
	serverTimeout := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverTimeoutCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverTimeout", sar)
		time.Sleep(2 * time.Second)
	}))
	defer serverTimeout.Close()
	serverTimeoutKubeconfigName := filepath.Join(dir, "serverTimeout.yaml")
	if err := os.WriteFile(serverTimeoutKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverTimeout.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// returns a deny response when called
	serverDenyCalled := atomic.Int32{}
	serverDeny := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverDenyCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverDeny", sar)
		sar.Status.Allowed = false
		sar.Status.Denied = true
		sar.Status.Reason = "denied by webhook"
		if err := json.NewEncoder(w).Encode(sar); err != nil {
			t.Error(err)
		}
	}))
	defer serverDeny.Close()
	serverDenyKubeconfigName := filepath.Join(dir, "serverDeny.yaml")
	if err := os.WriteFile(serverDenyKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverDeny.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// returns a no opinion response when called
	serverNoOpinionCalled := atomic.Int32{}
	serverNoOpinion := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverNoOpinionCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverNoOpinion", sar)
		sar.Status.Allowed = false
		sar.Status.Denied = false
		if err := json.NewEncoder(w).Encode(sar); err != nil {
			t.Error(err)
		}
	}))
	defer serverNoOpinion.Close()
	serverNoOpinionKubeconfigName := filepath.Join(dir, "serverNoOpinion.yaml")
	if err := os.WriteFile(serverNoOpinionKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverNoOpinion.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// returns an allow response when called
	serverAllowCalled := atomic.Int32{}
	serverAllow := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverAllowCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverAllow", sar)
		sar.Status.Allowed = true
		sar.Status.Reason = "allowed by webhook"
		if err := json.NewEncoder(w).Encode(sar); err != nil {
			t.Error(err)
		}
	}))
	defer serverAllow.Close()
	serverAllowKubeconfigName := filepath.Join(dir, "serverAllow.yaml")
	if err := os.WriteFile(serverAllowKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverAllow.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	resetCounts := func() {
		serverErrorCalled.Store(0)
		serverTimeoutCalled.Store(0)
		serverDenyCalled.Store(0)
		serverNoOpinionCalled.Store(0)
		serverAllowCalled.Store(0)
	}
	assertCounts := func(errorCount, timeoutCount, denyCount, noOpinionCount, allowCount int32) {
		t.Helper()
		if e, a := errorCount, serverErrorCalled.Load(); e != a {
			t.Errorf("expected fail webhook calls: %d, got %d", e, a)
		}
		if e, a := timeoutCount, serverTimeoutCalled.Load(); e != a {
			t.Errorf("expected timeout webhook calls: %d, got %d", e, a)
		}
		if e, a := denyCount, serverDenyCalled.Load(); e != a {
			t.Errorf("expected deny webhook calls: %d, got %d", e, a)
		}
		if e, a := noOpinionCount, serverNoOpinionCalled.Load(); e != a {
			t.Errorf("expected noOpinion webhook calls: %d, got %d", e, a)
		}
		if e, a := allowCount, serverAllowCalled.Load(); e != a {
			t.Errorf("expected allow webhook calls: %d, got %d", e, a)
		}
		resetCounts()
	}

	configFileName := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configFileName, []byte(`
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: AuthorizationConfiguration
authorizers:
- type: Webhook
  name: error.example.com
  webhook:
    timeout: 5s
    failurePolicy: Deny
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverErrorKubeconfigName+`
    matchConditions:
    - expression: has(request.resourceAttributes)
    - expression: 'request.resourceAttributes.namespace == "fail"'
    - expression: 'request.resourceAttributes.name == "error"'

- type: Webhook
  name: timeout.example.com
  webhook:
    timeout: 1s
    failurePolicy: Deny
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverTimeoutKubeconfigName+`
    matchConditions:
    - expression: has(request.resourceAttributes)
    - expression: 'request.resourceAttributes.namespace == "fail"'
    - expression: 'request.resourceAttributes.name == "timeout"'

- type: Webhook
  name: deny.example.com
  webhook:
    timeout: 5s
    failurePolicy: NoOpinion
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverDenyKubeconfigName+`
    matchConditions:
    - expression: has(request.resourceAttributes)
    - expression: 'request.resourceAttributes.namespace == "fail"'

- type: Webhook
  name: noopinion.example.com
  webhook:
    timeout: 5s
    failurePolicy: Deny
    subjectAccessReviewVersion: v1
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverNoOpinionKubeconfigName+`

- type: Webhook
  name: allow.example.com
  webhook:
    timeout: 5s
    failurePolicy: Deny
    subjectAccessReviewVersion: v1
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverAllowKubeconfigName+`
`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	server := kubeapiservertesting.StartTestServerOrDie(
		t,
		nil,
		[]string{"--authorization-config=" + configFileName},
		framework.SharedEtcd(),
	)
	t.Cleanup(server.TearDownFn)

	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	// malformed webhook short circuits
	t.Log("checking error")
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb:      "get",
			Group:     "",
			Version:   "v1",
			Resource:  "configmaps",
			Namespace: "fail",
			Name:      "error",
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if result.Status.Allowed {
		t.Fatal("expected denied, got allowed")
	} else {
		t.Log(result.Status.Reason)
		assertCounts(1, 0, 0, 0, 0)
	}

	// timeout webhook short circuits
	t.Log("checking timeout")
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb:      "get",
			Group:     "",
			Version:   "v1",
			Resource:  "configmaps",
			Namespace: "fail",
			Name:      "timeout",
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if result.Status.Allowed {
		t.Fatal("expected denied, got allowed")
	} else {
		t.Log(result.Status.Reason)
		assertCounts(0, 1, 0, 0, 0)
	}

	// deny webhook short circuits
	t.Log("checking deny")
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb:      "list",
			Group:     "",
			Version:   "v1",
			Resource:  "configmaps",
			Namespace: "fail",
			Name:      "",
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if result.Status.Allowed {
		t.Fatal("expected denied, got allowed")
	} else {
		t.Log(result.Status.Reason)
		assertCounts(0, 0, 1, 0, 0)
	}

	// no-opinion webhook passes through, allow webhook allows
	t.Log("checking allow")
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb:      "list",
			Group:     "",
			Version:   "v1",
			Resource:  "configmaps",
			Namespace: "allow",
			Name:      "",
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if !result.Status.Allowed {
		t.Fatal("expected allowed, got denied")
	} else {
		t.Log(result.Status.Reason)
		assertCounts(0, 0, 0, 1, 1)
	}
}
