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
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	authorizationv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	celmetrics "k8s.io/apiserver/pkg/authorization/cel"
	authorizationmetrics "k8s.io/apiserver/pkg/authorization/metrics"
	authzmetrics "k8s.io/apiserver/pkg/server/options/authorizationconfig/metrics"
	webhookmetrics "k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestAuthzConfig(t *testing.T) {
	dir := t.TempDir()
	configFileName := filepath.Join(dir, "config.yaml")
	if err := atomicWriteFile(configFileName, []byte(`
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
	errorName := "error.example.com"
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
	timeoutName := "timeout.example.com"
	serverTimeoutCalled := atomic.Int32{}
	serverTimeout := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverTimeoutCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverTimeout", sar)
		time.Sleep(2 * time.Second)
		sar.Status.Reason = "no opinion"
		if err := json.NewEncoder(w).Encode(sar); err != nil {
			t.Error(err)
		}
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
	noOpinionName := "noopinion.example.com"
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

	// returns malformed responses when called, which is then configured to fail open
	failOpenName := "failopen.example.com"
	serverFailOpenCalled := atomic.Int32{}
	serverFailOpen := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		serverFailOpenCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("serverFailOpen", sar)
		if _, err := w.Write([]byte(`malformed response`)); err != nil {
			t.Error(err)
		}
	}))
	defer serverFailOpen.Close()
	serverFailOpenKubeconfigName := filepath.Join(dir, "failOpen.yaml")
	if err := os.WriteFile(serverFailOpenKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, serverFailOpen.URL)), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// returns allow responses when called with a label selector containing an allow key, and records the selectors it saw
	selectorName := "selector.example.com"
	serverSelectorCalled := atomic.Int32{}
	var selectorLabelAttributes *authorizationv1.LabelSelectorAttributes
	var selectorFieldAttributes *authorizationv1.FieldSelectorAttributes
	selectorServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		selectorLabelAttributes = nil
		selectorFieldAttributes = nil
		serverSelectorCalled.Add(1)
		sar := &authorizationv1.SubjectAccessReview{}
		if err := json.NewDecoder(req.Body).Decode(sar); err != nil {
			t.Error(err)
		}
		t.Log("selector", sar)
		if sar.Spec.ResourceAttributes != nil {
			selectorLabelAttributes = sar.Spec.ResourceAttributes.LabelSelector.DeepCopy()
			selectorFieldAttributes = sar.Spec.ResourceAttributes.FieldSelector.DeepCopy()
			if sar.Spec.ResourceAttributes.LabelSelector != nil {
				for _, req := range sar.Spec.ResourceAttributes.LabelSelector.Requirements {
					if req.Key == "allow" {
						sar.Status.Allowed = true
					}
				}
			}
		}
		if err := json.NewEncoder(w).Encode(sar); err != nil {
			t.Error(err)
		}
	}))
	defer selectorServer.Close()
	serverSelectorKubeconfigName := filepath.Join(dir, "selector.yaml")
	if err := os.WriteFile(serverSelectorKubeconfigName, []byte(fmt.Sprintf(kubeconfigTemplate, selectorServer.URL)), os.FileMode(0644)); err != nil {
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
		serverFailOpenCalled.Store(0)
		serverSelectorCalled.Store(0)
		serverAllowCalled.Store(0)
		serverAllowReloadedCalled.Store(0)
		authorizationmetrics.ResetMetricsForTest()
		celmetrics.ResetMetricsForTest()
		webhookmetrics.ResetMetricsForTest()
	}
	var adminClient *clientset.Clientset
	type counts struct {
		errorCount, timeoutCount, denyCount, noOpinionCount, failOpenCount, selectorCount, allowCount, allowReloadedCount, webhookExclusionCount, evalErrorsCount int32
	}
	assertCounts := func(c counts) {
		t.Helper()
		metrics, err := getMetrics(t, adminClient)
		if err != nil {
			t.Fatalf("error getting metrics: %v", err)
		}

		assertCount := func(name string, expected int32, serverCalls *atomic.Int32) {
			t.Helper()
			if actual := serverCalls.Load(); expected != actual {
				t.Fatalf("expected %q webhook calls: %d, got %d", name, expected, actual)
			}
			if actual := int32(metrics.whTotal[name]); expected != actual {
				t.Fatalf("expected %q webhook metric call count: %d, got %d (%#v)", name, expected, actual, metrics.whTotal)
			}
			if actual := int32(metrics.whDurationCount[name]); expected != actual {
				t.Fatalf("expected %q webhook metric duration count: %d, got %d (%#v)", name, expected, actual, metrics.whDurationCount)
			}
		}

		assertCount(errorName, c.errorCount, &serverErrorCalled)
		assertCount(timeoutName, c.timeoutCount, &serverTimeoutCalled)
		expectedTimeoutCounts := map[string]int{}
		if c.timeoutCount > 0 {
			expectedTimeoutCounts[timeoutName] = int(c.timeoutCount)
		}
		if !reflect.DeepEqual(expectedTimeoutCounts, metrics.whTimeoutTotal) {
			t.Fatalf("expected timeouts %#v, got %#v", expectedTimeoutCounts, metrics.whTimeoutTotal)
		}
		assertCount(denyName, c.denyCount, &serverDenyCalled)
		if e, a := c.denyCount, metrics.decisions[authorizerKey{authorizerType: "Webhook", authorizerName: denyName}]["denied"]; e != int32(a) {
			t.Fatalf("expected deny webhook denied metrics calls: %d, got %d", e, a)
		}
		assertCount(selectorName, c.selectorCount, &serverSelectorCalled)
		assertCount(noOpinionName, c.noOpinionCount, &serverNoOpinionCalled)
		assertCount(failOpenName, c.failOpenCount, &serverFailOpenCalled)
		expectedFailOpenCounts := map[string]int{}
		if c.failOpenCount > 0 {
			expectedFailOpenCounts[failOpenName] = int(c.failOpenCount)
		}
		if !reflect.DeepEqual(expectedFailOpenCounts, metrics.whFailOpenTotal) {
			t.Fatalf("expected fail open %#v, got %#v", expectedFailOpenCounts, metrics.whFailOpenTotal)
		}

		assertCount(allowName, c.allowCount, &serverAllowCalled)
		if e, a := c.allowCount, metrics.decisions[authorizerKey{authorizerType: "Webhook", authorizerName: allowName}]["allowed"]; e != int32(a) {
			t.Fatalf("expected allow webhook allowed metrics calls: %d, got %d", e, a)
		}
		assertCount(allowReloadedName, c.allowReloadedCount, &serverAllowReloadedCalled)
		if e, a := c.allowReloadedCount, metrics.decisions[authorizerKey{authorizerType: "Webhook", authorizerName: allowReloadedName}]["allowed"]; e != int32(a) {
			t.Fatalf("expected allowReloaded webhook allowed metrics calls: %d, got %d", e, a)
		}
		if e, a := c.webhookExclusionCount, metrics.exclusions; e != int32(a) {
			t.Fatalf("expected webhook exclusions due to match conditions: %d, got %d", e, a)
		}
		if e, a := c.evalErrorsCount, metrics.evalErrors; e != int32(a) {
			t.Fatalf("expected webhook match condition eval errors: %d, got %d", e, a)
		}
		resetCounts()
	}

	configFileName := filepath.Join(dir, "config.yaml")
	if err := atomicWriteFile(configFileName, []byte(`
apiVersion: apiserver.config.k8s.io/v1beta1
kind: AuthorizationConfiguration
authorizers:
- type: Webhook
  name: `+errorName+`
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
  name: `+timeoutName+`
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
    # intentionally skip this check so we can trigger an eval error with a non-resource request
    # - expression: has(request.resourceAttributes)
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
  name: `+selectorName+`
  webhook:
    timeout: 5s
    failurePolicy: NoOpinion
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverSelectorKubeconfigName+`
    matchConditions:
    - expression: request.?resourceAttributes.labelSelector.requirements.orValue([]).exists(r, r.key=='testselector')

- type: Webhook
  name: `+noOpinionName+`
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
  name: `+failOpenName+`
  webhook:
    timeout: 5s
    failurePolicy: NoOpinion
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: `+serverFailOpenKubeconfigName+`


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

	impersonationConfig := rest.CopyConfig(server.ClientConfig)
	impersonationConfig.Impersonate.UserName = "alice"
	aliceClient := clientset.NewForConfigOrDie(impersonationConfig)

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
		assertCounts(counts{errorCount: 1})
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
		assertCounts(counts{timeoutCount: 1, webhookExclusionCount: 1})
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
		assertCounts(counts{denyCount: 1, webhookExclusionCount: 2})
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
		assertCounts(counts{noOpinionCount: 1, failOpenCount: 1, allowCount: 1, webhookExclusionCount: 4})
	}

	// the timeout webhook results in match condition eval errors when evaluating a non-resource request
	// failure policy is deny
	t.Log("checking match condition eval error")
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		NonResourceAttributes: &authorizationv1.NonResourceAttributes{
			Verb: "list",
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if result.Status.Allowed {
		t.Fatal("expected denied, got allowed")
	} else {
		t.Log(result.Status.Reason)
		// error webhook matchConditions skip non-resource request
		// timeout webhook matchConditions error on non-resource request
		assertCounts(counts{webhookExclusionCount: 1, evalErrorsCount: 1})
	}

	disorderedFieldSelector := "spec.nodeName=mynode,metadata.name!=b,metadata.name!=a"
	orderedFieldRequirements := &authorizationv1.FieldSelectorAttributes{Requirements: []metav1.FieldSelectorRequirement{
		{Key: "metadata.name", Operator: "NotIn", Values: []string{"a"}},
		{Key: "metadata.name", Operator: "NotIn", Values: []string{"b"}},
		{Key: "spec.nodeName", Operator: "In", Values: []string{"mynode"}}}}
	disorderedFieldRequirements := &authorizationv1.FieldSelectorAttributes{Requirements: []metav1.FieldSelectorRequirement{
		{Key: "spec.nodeName", Operator: "In", Values: []string{"mynode"}},
		{Key: "metadata.name", Operator: "NotIn", Values: []string{"b"}},
		{Key: "metadata.name", Operator: "NotIn", Values: []string{"a"}}}}
	disorderedUnknownFieldRequirements := disorderedFieldRequirements.DeepCopy()
	disorderedUnknownFieldRequirements.Requirements = append(disorderedUnknownFieldRequirements.Requirements, metav1.FieldSelectorRequirement{Key: "x", Operator: "Unknown"})
	disorderedLabelSelector := "testselector in (b,a),allow=true"
	orderedLabelRequirements := &authorizationv1.LabelSelectorAttributes{Requirements: []metav1.LabelSelectorRequirement{
		{Key: "allow", Operator: "In", Values: []string{"true"}},
		{Key: "testselector", Operator: "In", Values: []string{"a", "b"}}}}
	disorderedLabelRequirements := &authorizationv1.LabelSelectorAttributes{Requirements: []metav1.LabelSelectorRequirement{
		{Key: "testselector", Operator: "In", Values: []string{"b", "a"}},
		{Key: "allow", Operator: "In", Values: []string{"true"}}}}
	disorderedUnknownLabelRequirements := disorderedLabelRequirements.DeepCopy()
	disorderedUnknownLabelRequirements.Requirements = append(disorderedUnknownLabelRequirements.Requirements, metav1.LabelSelectorRequirement{Key: "x", Operator: "Unknown"})

	// make request matching selector webhook matchCondition
	// check fieldSelector and labelSelector are parsed and normalized
	_, err := aliceClient.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{FieldSelector: disorderedFieldSelector, LabelSelector: disorderedLabelSelector})
	assertCounts(counts{selectorCount: 1, webhookExclusionCount: 3})
	if err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}
	if e, a := orderedFieldRequirements.DeepCopy(), selectorFieldAttributes; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected diff:\n%s", cmp.Diff(a, e))
	}
	if e, a := orderedLabelRequirements.DeepCopy(), selectorLabelAttributes; !reflect.DeepEqual(e, a) {
		t.Fatalf("unexpected diff:\n%s", cmp.Diff(a, e))
	}
	selectorFieldAttributes = nil
	selectorLabelAttributes = nil

	// make subjectaccessreview request containing fieldSelector and labelSelector requirements
	// check known fieldSelector and labelSelector requirements get passed through to the webhook as-is
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb:          "list",
			Version:       "v1",
			Resource:      "pods",
			FieldSelector: disorderedUnknownFieldRequirements.DeepCopy(),
			LabelSelector: disorderedUnknownLabelRequirements.DeepCopy(),
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if !result.Status.Allowed {
		t.Fatal("expected allowed, got denied")
	} else {
		t.Log(result.Status.Reason)
		t.Log(result.Status.EvaluationError)
		assertCounts(counts{selectorCount: 1, webhookExclusionCount: 3})
		if e, a := disorderedFieldRequirements.DeepCopy(), selectorFieldAttributes; !reflect.DeepEqual(e, a) {
			t.Fatalf("unexpected diff:\n%s", cmp.Diff(a, e))
		}
		if e, a := disorderedLabelRequirements.DeepCopy(), selectorLabelAttributes; !reflect.DeepEqual(e, a) {
			t.Fatalf("unexpected diff:\n%s", cmp.Diff(a, e))
		}
	}
	selectorFieldAttributes = nil
	selectorLabelAttributes = nil

	// make subjectaccessreview request containing fieldSelector and labelSelector rawSelector
	// check fieldSelector and labelSelector rawSelector get parsed and passed to the webhook
	if result, err := adminClient.AuthorizationV1().SubjectAccessReviews().Create(context.TODO(), &authorizationv1.SubjectAccessReview{Spec: authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb:          "list",
			Version:       "v1",
			Resource:      "pods",
			FieldSelector: &authorizationv1.FieldSelectorAttributes{RawSelector: disorderedFieldSelector},
			LabelSelector: &authorizationv1.LabelSelectorAttributes{RawSelector: disorderedLabelSelector},
		},
	}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	} else if !result.Status.Allowed {
		t.Fatal("expected allowed, got denied")
	} else {
		t.Log(result.Status.Reason)
		t.Log(result.Status.EvaluationError)
		assertCounts(counts{selectorCount: 1, webhookExclusionCount: 3})
		if e, a := orderedFieldRequirements.DeepCopy(), selectorFieldAttributes; !reflect.DeepEqual(e, a) {
			t.Fatalf("unexpected diff:\n%s", cmp.Diff(a, e))
		}
		if e, a := orderedLabelRequirements.DeepCopy(), selectorLabelAttributes; !reflect.DeepEqual(e, a) {
			t.Fatalf("unexpected diff:\n%s", cmp.Diff(a, e))
		}
	}
	selectorFieldAttributes = nil
	selectorLabelAttributes = nil

	// check last loaded success/failure metric timestamps, ensure success is present, failure is not
	initialMetrics, err := getMetrics(t, adminClient)
	if err != nil {
		t.Fatal(err)
	}
	if initialMetrics.reloadSuccess != nil {
		t.Fatal("expected no success timestamp, got one")
	}
	if initialMetrics.reloadFailure != nil {
		t.Fatal("expected no failure timestamp, got one")
	}

	// write bogus file
	if err := atomicWriteFile(configFileName, []byte(`apiVersion: apiserver.config.k8s.io`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	// wait for failure timestamp > success timestamp
	var reload1Metrics *metrics
	err = wait.PollUntilContextTimeout(context.TODO(), time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		reload1Metrics, err = getMetrics(t, adminClient)
		if err != nil {
			t.Fatal(err)
		}
		if reload1Metrics.reloadSuccess != nil {
			t.Fatal("expected no success timestamp, got one")
		}
		if reload1Metrics.reloadFailure == nil {
			t.Log("expected failure timestamp, got nil, retrying")
			return false, nil
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
		assertCounts(counts{noOpinionCount: 1, failOpenCount: 1, allowCount: 1, webhookExclusionCount: 4})
	}

	// write good config with different webhook
	if err := atomicWriteFile(configFileName, []byte(`
apiVersion: apiserver.config.k8s.io/v1
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
			t.Log("expected success timestamp, got nil, retrying")
			return false, nil
		}
		if !reload2Metrics.reloadSuccess.After(*reload2Metrics.reloadFailure) {
			t.Fatalf("expected success timestamp to be more recent than failure, got %s <= %s", reload2Metrics.reloadSuccess.String(), reload2Metrics.reloadFailure.String())
		}
		if len(reload2Metrics.configHash) == 0 {
			t.Fatal("expected config hash, got none")
		}
		if reload2Metrics.configHash == initialMetrics.configHash {
			t.Logf("config hash %s is the same as initial %s, retrying", reload2Metrics.configHash, initialMetrics.configHash)
			return false, nil
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
		assertCounts(counts{allowReloadedCount: 1})
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
		if len(reload3Metrics.configHash) == 0 {
			t.Fatal("expected config hash, got none")
		}
		if reload3Metrics.configHash != reload2Metrics.configHash {
			t.Fatalf("expected config hash to be the same as reload2Metrics %s, got %s", reload2Metrics.configHash, reload3Metrics.configHash)
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
		assertCounts(counts{allowReloadedCount: 1})
	}
}

type metrics struct {
	reloadSuccess *time.Time
	reloadFailure *time.Time
	configHash    string
	decisions     map[authorizerKey]map[string]int
	exclusions    int
	evalErrors    int

	whTotal         map[string]int
	whTimeoutTotal  map[string]int
	whFailOpenTotal map[string]int
	whDurationCount map[string]int
}
type authorizerKey struct {
	authorizerType string
	authorizerName string
}

var decisionMetric = regexp.MustCompile(`apiserver_authorization_decisions_total\{decision="(.*?)",name="(.*?)",type="(.*?)"\} (\d+)`)
var webhookExclusionMetric = regexp.MustCompile(`apiserver_authorization_match_condition_exclusions_total\{name="(.*?)",type="(.*?)"\} (\d+)`)
var webhookMatchConditionEvalErrorMetric = regexp.MustCompile(`apiserver_authorization_match_condition_evaluation_errors_total\{name="(.*?)",type="(.*?)"\} (\d+)`)
var whTotalMetric = regexp.MustCompile(`apiserver_authorization_webhook_evaluations_total{name="(.*?)",result="(.*?)"} (\d+)`)
var webhookDurationMetric = regexp.MustCompile(`apiserver_authorization_webhook_duration_seconds_count{name="(.*?)",result="(.*?)"} (\d+)`)
var webhookFailOpenMetric = regexp.MustCompile(`apiserver_authorization_webhook_evaluations_fail_open_total{name="(.*?)",result="(.*?)"} (\d+)`)
var configInfoMetric = regexp.MustCompile(`apiserver_authorization_config_controller_last_config_info\{apiserver_id_hash="sha256:[^"]*",hash="([^"]*)"\} (\d+)`)

func getMetrics(t *testing.T, client *clientset.Clientset) (*metrics, error) {
	data, err := client.RESTClient().Get().AbsPath("/metrics").DoRaw(context.TODO())

	//  apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:4b86cfa719a83dd63a4dc6a9831edb2b59240d0f59cf215b2d51aacb3f5c395e",status="success"} 1.7002567356895502e+09
	//  apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:4b86cfa719a83dd63a4dc6a9831edb2b59240d0f59cf215b2d51aacb3f5c395e",status="failure"} 1.7002567356895502e+09
	//  apiserver_authorization_config_controller_automatic_reload_last_config_info{apiserver_id_hash="sha256:4b86cfa719a83dd63a4dc6a9831edb2b59240d0f59cf215b2d51aacb3f5c395e",hash="sha256:f309dd9c31fe24b3e594d2f9420419c48dfe954523245d5f35dc37739970d881"} 1
	//  apiserver_authorization_decisions_total{decision="allowed",name="allow.example.com",type="Webhook"} 2
	//  apiserver_authorization_decisions_total{decision="allowed",name="allowreloaded.example.com",type="Webhook"} 1
	//  apiserver_authorization_decisions_total{decision="denied",name="deny.example.com",type="Webhook"} 1
	//  apiserver_authorization_decisions_total{decision="denied",name="error.example.com",type="Webhook"} 1
	//  apiserver_authorization_decisions_total{decision="denied",name="timeout.example.com",type="Webhook"} 1
	//  apiserver_authorization_match_condition_exclusions_total{name="exclusion.example.com",type="webhook"} 1
	if err != nil {
		return nil, err
	}

	var m metrics

	m.whTotal = map[string]int{}
	m.whTimeoutTotal = map[string]int{}
	m.whFailOpenTotal = map[string]int{}
	m.whDurationCount = map[string]int{}
	m.exclusions = 0
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
		if matches := webhookExclusionMetric.FindStringSubmatch(line); matches != nil {
			t.Log(matches)
			count, err := strconv.Atoi(matches[3])
			if err != nil {
				return nil, err
			}
			t.Log(count)
			m.exclusions += count
		}
		if matches := webhookMatchConditionEvalErrorMetric.FindStringSubmatch(line); matches != nil {
			t.Log(matches)
			count, err := strconv.Atoi(matches[3])
			if err != nil {
				return nil, err
			}
			t.Log(count)
			m.evalErrors += count
		}
		if matches := whTotalMetric.FindStringSubmatch(line); matches != nil {
			t.Log(matches)
			count, err := strconv.Atoi(matches[3])
			if err != nil {
				return nil, err
			}
			t.Log(count)
			m.whTotal[matches[1]] += count
			if matches[2] == "timeout" {
				m.whTimeoutTotal[matches[1]] += count
			}
		}
		if matches := webhookDurationMetric.FindStringSubmatch(line); matches != nil {
			t.Log(matches)
			count, err := strconv.Atoi(matches[3])
			if err != nil {
				return nil, err
			}
			t.Log(count)
			m.whDurationCount[matches[1]] += count
		}
		if matches := webhookFailOpenMetric.FindStringSubmatch(line); matches != nil {
			t.Log(matches)
			count, err := strconv.Atoi(matches[3])
			if err != nil {
				return nil, err
			}
			t.Log(count)
			m.whFailOpenTotal[matches[1]] += count
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
		if matches := configInfoMetric.FindStringSubmatch(line); matches != nil {
			t.Logf("line: %s\n", line)
			t.Log(matches)
			m.configHash = matches[1]
			t.Logf("config hash: %s", m.configHash)
		}
	}
	return &m, nil
}

func atomicWriteFile(name string, data []byte, perm os.FileMode) error {
	tmp := name + ".tmp"
	if err := os.WriteFile(tmp, data, perm); err != nil {
		return err
	}
	return os.Rename(tmp, name)
}
