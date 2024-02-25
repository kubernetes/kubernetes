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
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	authorizationmetrics "k8s.io/apiserver/pkg/authorization/metrics"
	"k8s.io/apiserver/pkg/features"
	authzmetrics "k8s.io/apiserver/pkg/server/options/authorizationconfig/metrics"
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
	authzmetrics.ResetMetricsForTest()
	defer authzmetrics.ResetMetricsForTest()
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
	denyName := "deny.example.com"
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
	allowName := "allow.example.com"
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

	// returns an allow response when called
	allowReloadedName := "allowreloaded.example.com"
	serverAllowReloadedCalled := atomic.Int32{}
	serverAllowReloaded := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverAllowReloadedCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverAllowReloaded", sar)
		sar.Status.Allowed = true
		sar.Status.Reason = "allowed2 by webhook"
		if err := json.NewEncoder(w).Encode(sar); err != nil {
			t.Error(err)
		}
	}))
	defer serverAllowReloaded.Close()
	serverAllowReloadedKubeconfigName := filepath.Join(dir, "serverAllowReloaded.yaml")
	if err := os.WriteFile(serverAllowReloadedKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverAllowReloaded.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	resetCounts := func() {
		serverErrorCalled.Store(0)
		serverTimeoutCalled.Store(0)
		serverDenyCalled.Store(0)
		serverNoOpinionCalled.Store(0)
		serverAllowCalled.Store(0)
		serverAllowReloadedCalled.Store(0)
		authorizationmetrics.ResetMetricsForTest()
	}
	var adminClient *clientset.Clientset
	assertCounts := func(errorCount, timeoutCount, denyCount, noOpinionCount, allowCount, allowReloadedCount int32) {
		t.Helper()
		metrics, err := getMetrics(t, adminClient)
		if err != nil {
			t.Errorf("error getting metrics: %v", err)
		}
		if e, a := errorCount, serverErrorCalled.Load(); e != a {
			t.Errorf("expected fail webhook calls: %d, got %d", e, a)
		}
		if e, a := timeoutCount, serverTimeoutCalled.Load(); e != a {
			t.Errorf("expected timeout webhook calls: %d, got %d", e, a)
		}
		if e, a := denyCount, serverDenyCalled.Load(); e != a {
			t.Errorf("expected deny webhook calls: %d, got %d", e, a)
		}
		if e, a := denyCount, metrics.decisions[authorizerKey{authorizerType: "Webhook", authorizerName: denyName}]["denied"]; e != int32(a) {
			t.Errorf("expected deny webhook denied metrics calls: %d, got %d", e, a)
		}
		if e, a := noOpinionCount, serverNoOpinionCalled.Load(); e != a {
			t.Errorf("expected noOpinion webhook calls: %d, got %d", e, a)
		}
		if e, a := allowCount, serverAllowCalled.Load(); e != a {
			t.Errorf("expected allow webhook calls: %d, got %d", e, a)
		}
		if e, a := allowCount, metrics.decisions[authorizerKey{authorizerType: "Webhook", authorizerName: allowName}]["allowed"]; e != int32(a) {
			t.Errorf("expected allow webhook allowed metrics calls: %d, got %d", e, a)
		}
		if e, a := allowReloadedCount, serverAllowReloadedCalled.Load(); e != a {
			t.Errorf("expected allowReloaded webhook calls: %d, got %d", e, a)
		}
		if e, a := allowReloadedCount, metrics.decisions[authorizerKey{authorizerType: "Webhook", authorizerName: allowReloadedName}]["allowed"]; e != int32(a) {
			t.Errorf("expected allowReloaded webhook allowed metrics calls: %d, got %d", e, a)
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
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
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
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverTimeoutKubeconfigName+`
    matchConditions:
    - expression: has(request.resourceAttributes)
    - expression: 'request.resourceAttributes.namespace == "fail"'
    - expression: 'request.resourceAttributes.name == "timeout"'

- type: Webhook
  name: `+denyName+`
  webhook:
    timeout: 5s
    failurePolicy: NoOpinion
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
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
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverNoOpinionKubeconfigName+`

- type: Webhook
  name: `+allowName+`
  webhook:
    timeout: 5s
    failurePolicy: Deny
    subjectAccessReviewVersion: v1
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
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

	adminClient = clientset.NewForConfigOrDie(server.ClientConfig)

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
		assertCounts(1, 0, 0, 0, 0, 0)
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
		assertCounts(0, 1, 0, 0, 0, 0)
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
		assertCounts(0, 0, 1, 0, 0, 0)
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
		assertCounts(0, 0, 0, 1, 1, 0)
	}

	// check last loaded success/failure metric timestamps, ensure success is present, failure is not
	initialMetrics, err := getMetrics(t, adminClient)
	if err != nil {
		t.Fatal(err)
	}
	if initialMetrics.reloadSuccess == nil {
		t.Fatal("expected success timestamp, got none")
	}
	if initialMetrics.reloadFailure != nil {
		t.Fatal("expected no failure timestamp, got one")
	}

	// write bogus file
	if err := os.WriteFile(configFileName, []byte(`apiVersion: apiserver.config.k8s.io`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// wait for failure timestamp > success timestamp
	var reload1Metrics *metrics
	err = wait.PollUntilContextTimeout(context.TODO(), time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		reload1Metrics, err = getMetrics(t, adminClient)
		if err != nil {
			t.Fatal(err)
		}
		if reload1Metrics.reloadSuccess == nil {
			t.Fatal("expected success timestamp, got none")
		}
		if !reload1Metrics.reloadSuccess.Equal(*initialMetrics.reloadSuccess) {
			t.Fatalf("success timestamp changed from initial success %s to %s unexpectedly", initialMetrics.reloadSuccess.String(), reload1Metrics.reloadSuccess.String())
		}
		if reload1Metrics.reloadFailure == nil {
			t.Log("expected failure timestamp, got nil, retrying")
			return false, nil
		}
		if !reload1Metrics.reloadFailure.After(*reload1Metrics.reloadSuccess) {
			t.Fatalf("expected failure timestamp to be more recent than success timestamp, got %s <= %s", reload1Metrics.reloadFailure.String(), reload1Metrics.reloadSuccess.String())
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// ensure authz still works
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
		assertCounts(0, 0, 0, 1, 1, 0)
	}

	// write good config with different webhook
	if err := os.WriteFile(configFileName, []byte(`
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: AuthorizationConfiguration
authorizers:
- type: Webhook
  name: `+allowReloadedName+`
  webhook:
    timeout: 5s
    failurePolicy: Deny
    subjectAccessReviewVersion: v1
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverAllowReloadedKubeconfigName+`
`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// wait for success timestamp > reload1Metrics.reloadFailure timestamp
	var reload2Metrics *metrics
	err = wait.PollUntilContextTimeout(context.TODO(), time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		reload2Metrics, err = getMetrics(t, adminClient)
		if err != nil {
			t.Fatal(err)
		}
		if reload2Metrics.reloadFailure == nil {
			t.Log("expected failure timestamp, got nil, retrying")
			return false, nil
		}
		if !reload2Metrics.reloadFailure.Equal(*reload1Metrics.reloadFailure) {
			t.Fatalf("failure timestamp changed from reload1Metrics.reloadFailure %s to %s unexpectedly", reload1Metrics.reloadFailure.String(), reload2Metrics.reloadFailure.String())
		}
		if reload2Metrics.reloadSuccess == nil {
			t.Fatal("expected success timestamp, got none")
		}
		if reload2Metrics.reloadSuccess.Equal(*initialMetrics.reloadSuccess) {
			t.Log("success timestamp hasn't updated from initial success, retrying")
			return false, nil
		}
		if !reload2Metrics.reloadSuccess.After(*reload2Metrics.reloadFailure) {
			t.Fatalf("expected success timestamp to be more recent than failure, got %s <= %s", reload2Metrics.reloadSuccess.String(), reload2Metrics.reloadFailure.String())
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// ensure authz still works, new webhook is called
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
		assertCounts(0, 0, 0, 0, 0, 1)
	}

	// delete file (do this test last because it makes file watch fall back to one minute poll interval)
	if err := os.Remove(configFileName); err != nil {
		t.Fatal(err)
	}

	// wait for failure timestamp > success timestamp
	var reload3Metrics *metrics
	err = wait.PollUntilContextTimeout(context.TODO(), time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		reload3Metrics, err = getMetrics(t, adminClient)
		if err != nil {
			t.Fatal(err)
		}
		if reload3Metrics.reloadSuccess == nil {
			t.Fatal("expected success timestamp, got none")
		}
		if !reload3Metrics.reloadSuccess.Equal(*reload2Metrics.reloadSuccess) {
			t.Fatalf("success timestamp changed from %s to %s unexpectedly", reload2Metrics.reloadSuccess.String(), reload3Metrics.reloadSuccess.String())
		}
		if reload3Metrics.reloadFailure == nil {
			t.Log("expected failure timestamp, got nil, retrying")
			return false, nil
		}
		if reload3Metrics.reloadFailure.Equal(*reload2Metrics.reloadFailure) {
			t.Log("failure timestamp hasn't updated, retrying")
			return false, nil
		}
		if !reload3Metrics.reloadFailure.After(*reload3Metrics.reloadSuccess) {
			t.Fatalf("expected failure timestamp to be more recent than success, got %s <= %s", reload3Metrics.reloadFailure.String(), reload3Metrics.reloadSuccess.String())
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// ensure authz still works, new webhook is called
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
		assertCounts(0, 0, 0, 0, 0, 1)
	}
}

type metrics struct {
	reloadSuccess *time.Time
	reloadFailure *time.Time
	decisions     map[authorizerKey]map[string]int
}
type authorizerKey struct {
	authorizerType string
	authorizerName string
}

var decisionMetric = regexp.MustCompile(`apiserver_authorization_decisions_total\{decision="(.*?)",name="(.*?)",type="(.*?)"\} (\d+)`)

func getMetrics(t *testing.T, client *clientset.Clientset) (*metrics, error) {
	data, err := client.RESTClient().Get().AbsPath("/metrics").DoRaw(context.TODO())

	//  apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:4b86cfa719a83dd63a4dc6a9831edb2b59240d0f59cf215b2d51aacb3f5c395e",status="success"} 1.7002567356895502e+09
	//  apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:4b86cfa719a83dd63a4dc6a9831edb2b59240d0f59cf215b2d51aacb3f5c395e",status="failure"} 1.7002567356895502e+09
	//  apiserver_authorization_decisions_total{decision="allowed",name="allow.example.com",type="Webhook"} 2
	//  apiserver_authorization_decisions_total{decision="allowed",name="allowreloaded.example.com",type="Webhook"} 1
	//  apiserver_authorization_decisions_total{decision="denied",name="deny.example.com",type="Webhook"} 1
	//  apiserver_authorization_decisions_total{decision="denied",name="error.example.com",type="Webhook"} 1
	//  apiserver_authorization_decisions_total{decision="denied",name="timeout.example.com",type="Webhook"} 1
	if err != nil {
		return nil, err
	}

	var m metrics
	for _, line := range strings.Split(string(data), "\n") {
		if matches := decisionMetric.FindStringSubmatch(line); matches != nil {
			t.Log(line)
			if m.decisions == nil {
				m.decisions = map[authorizerKey]map[string]int{}
			}
			key := authorizerKey{authorizerType: matches[3], authorizerName: matches[2]}
			if m.decisions[key] == nil {
				m.decisions[key] = map[string]int{}
			}
			count, err := strconv.Atoi(matches[4])
			if err != nil {
				return nil, err
			}
			m.decisions[key][matches[1]] = count

		}
		if strings.HasPrefix(line, "apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds") {
			t.Log(line)
			values := strings.Split(line, " ")
			value, err := strconv.ParseFloat(values[len(values)-1], 64)
			if err != nil {
				return nil, err
			}
			seconds := int64(value)
			nanoseconds := int64((value - float64(seconds)) * 1000000000)
			tm := time.Unix(seconds, nanoseconds)
			if strings.Contains(line, `"success"`) {
				m.reloadSuccess = &tm
				t.Log("success", m.reloadSuccess.String())
			}
			if strings.Contains(line, `"failure"`) {
				m.reloadFailure = &tm
				t.Log("failure", m.reloadFailure.String())
			}
		}
	}
	return &m, nil
}
