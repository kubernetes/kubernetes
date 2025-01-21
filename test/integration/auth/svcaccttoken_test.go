/*
Copyright 2017 The Kubernetes Authors.

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
	"encoding/base64"
	"encoding/json"
	"fmt"

	"io"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"
	"gopkg.in/go-jose/go-jose.v2/jwt"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/serviceaccount"
	svcacct "k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const (
	// This key is for testing purposes only and is not considered secure.
	ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEZmTmUhuanLjPA2CLquXivuwBDHTt5XYwgIr/kA1LtRoAoGCCqGSM49
AwEHoUQDQgAEH6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp/C/ASqiIGUeeKQtX0
/IR3qCXyThP/dbCiHrF3v1cuhBOHY8CLVg==
-----END EC PRIVATE KEY-----`

	tokenExpirationSeconds = 60*60 + 7
)

func TestServiceAccountAnnotationDeprecation(t *testing.T) {
	tCtx := ktesting.Init(t)
	// Start the server
	kubeClient, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{})
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(kubeClient, "myns", t)
	defer framework.DeleteNamespaceOrDie(kubeClient, ns, t)

	warningHandler := &recordingWarningHandler{}

	configWithWarningHandler := rest.CopyConfig(kubeConfig)
	configWithWarningHandler.WarningHandler = warningHandler
	cs, err := clientset.NewForConfig(configWithWarningHandler)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var (
		saWithAnnotation = &v1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-svcacct",
				Namespace: ns.Name,
			},
		}
		pod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pod",
				Namespace: saWithAnnotation.Namespace,
			},
			Spec: v1.PodSpec{
				ServiceAccountName: saWithAnnotation.Name,
				Containers:         []v1.Container{{Name: "test-container", Image: "nginx"}},
			},
		}
	)

	t.Run("service account with deprecated annotation", func(t *testing.T) {
		warningHandler.clear()
		// test warning is emitted when the service account has this deprecated annotation
		saWithAnnotation.Annotations = map[string]string{svcacct.EnforceMountableSecretsAnnotation: "true"}
		_, delSvcAcct := createDeleteSvcAcct(t, cs, saWithAnnotation)
		defer delSvcAcct()
		warningHandler.assertEqual(t, []string{fmt.Sprintf("metadata.annotations[%s]: deprecated in v1.32+; prefer separate namespaces to isolate access to mounted secrets", svcacct.EnforceMountableSecretsAnnotation)})

		warningHandler.clear()
		// no warning when a pod is created with a service account that has this deprecated annotation
		_, delPod := createDeletePod(t, cs, pod)
		defer delPod()
		warningHandler.assertEqual(t, nil)
	})
}

func TestServiceAccountTokenCreate(t *testing.T) {
	const iss = "https://foo.bar.example.com"
	aud := authenticator.Audiences{"api"}

	maxExpirationSeconds := int64(60 * 60 * 2)
	maxExpirationDuration, err := time.ParseDuration(fmt.Sprintf("%ds", maxExpirationSeconds))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	var tokenGenerator serviceaccount.TokenGenerator

	tCtx := ktesting.Init(t)

	// Start the server
	var serverAddress string
	kubeClient, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
			opts.Authorization.Modes = []string{"AlwaysAllow"}
			// Disable token cache so we can check reaction to service account deletion quickly
			opts.Authentication.TokenSuccessCacheTTL = 0
			opts.Authentication.TokenFailureCacheTTL = 0
			// Pin to fixed URLs for easier testing
			opts.Authentication.ServiceAccounts.JWKSURI = "https:///openid/v1/jwks"
			opts.Authentication.ServiceAccounts.Issuers = []string{iss}
			opts.Authentication.APIAudiences = aud
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			// extract token generator
			tokenGenerator = config.ControlPlane.Extra.ServiceAccountIssuer

			config.ControlPlane.Extra.ServiceAccountMaxExpiration = maxExpirationDuration
			config.ControlPlane.Extra.ExtendExpiration = true
		},
	})
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(kubeClient, "myns", t)
	defer framework.DeleteNamespaceOrDie(kubeClient, ns, t)

	warningHandler := &recordingWarningHandler{}

	configWithWarningHandler := rest.CopyConfig(kubeConfig)
	configWithWarningHandler.WarningHandler = warningHandler
	cs, err := clientset.NewForConfig(configWithWarningHandler)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	kubeConfig.NegotiatedSerializer = scheme.Codecs.WithoutConversion()
	rc, err := rest.UnversionedRESTClientFor(kubeConfig)
	if err != nil {
		t.Fatal(err)
	}

	var (
		sa = &v1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-svcacct",
				Namespace: ns.Name,
			},
		}
		node = &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node",
			},
		}
		pod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pod",
				Namespace: sa.Namespace,
			},
			Spec: v1.PodSpec{
				ServiceAccountName: sa.Name,
				Containers:         []v1.Container{{Name: "test-container", Image: "nginx"}},
			},
		}
		scheduledpod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pod",
				Namespace: sa.Namespace,
			},
			Spec: v1.PodSpec{
				ServiceAccountName: sa.Name,
				NodeName:           node.Name,
				Containers:         []v1.Container{{Name: "test-container", Image: "nginx"}},
			},
		}
		otherpod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "other-test-pod",
				Namespace: sa.Namespace,
			},
			Spec: v1.PodSpec{
				ServiceAccountName: "other-" + sa.Name,
				Containers:         []v1.Container{{Name: "test-container", Image: "nginx"}},
			},
		}
		secret = &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-secret",
				Namespace: sa.Namespace,
			},
		}
		wrongUID = types.UID("wrong")
		noUID    = types.UID("")
	)

	t.Run("bound to service account", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
			},
		}

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		sa, delSvcAcct := createDeleteSvcAcct(t, cs, sa)
		defer delSvcAcct()

		treqWithBadName := treq.DeepCopy()
		treqWithBadName.Name = "invalid-name"
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treqWithBadName, metav1.CreateOptions{}); err == nil || !strings.Contains(err.Error(), "must match the service account name") {
			t.Fatalf("expected err creating token with mismatched name but got: %#v", resp)
		}

		treqWithBadNamespace := treq.DeepCopy()
		treqWithBadNamespace.Namespace = "invalid-namespace"
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treqWithBadNamespace, metav1.CreateOptions{}); err == nil || !strings.Contains(err.Error(), "does not match the namespace") {
			t.Fatalf("expected err creating token with mismatched namespace but got: %#v, %v", resp, err)
		}

		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		if treq.Name != sa.Name {
			t.Errorf("expected name=%s, got %s", sa.Name, treq.Name)
		}
		if treq.Namespace != sa.Namespace {
			t.Errorf("expected namespace=%s, got %s", sa.Namespace, treq.Namespace)
		}
		if treq.CreationTimestamp.IsZero() {
			t.Errorf("expected non-zero creation timestamp")
		}

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		info := doTokenReview(t, cs, treq, false)
		// we are not testing the credential-id feature, so delete this value from the returned extra info map
		if info.Extra != nil {
			delete(info.Extra, user.CredentialIDKey)
		}
		if len(info.Extra) > 0 {
			t.Fatalf("expected Extra to be empty but got: %#v", info.Extra)
		}
		delSvcAcct()
		doTokenReview(t, cs, treq, true)
	})

	t.Run("bound to service account and pod", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Name:       pod.Name,
				},
			},
		}

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant pod but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		pod, delPod := createDeletePod(t, cs, pod)
		defer delPod()

		// right uid
		treq.Spec.BoundObjectRef.UID = pod.UID
		warningHandler.clear()
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to pod with wrong uid but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "node")

		info := doTokenReview(t, cs, treq, false)
		// we are not testing the credential-id feature, so delete this value from the returned extra info map
		delete(info.Extra, user.CredentialIDKey)
		if len(info.Extra) != 2 {
			t.Fatalf("expected Extra have length of 2 but was length %d: %#v", len(info.Extra), info.Extra)
		}
		if expected := map[string]authenticationv1.ExtraValue{
			"authentication.kubernetes.io/pod-name": {pod.ObjectMeta.Name},
			"authentication.kubernetes.io/pod-uid":  {string(pod.ObjectMeta.UID)},
		}; !reflect.DeepEqual(info.Extra, expected) {
			t.Fatalf("unexpected Extra:\ngot:\t%#v\nwant:\t%#v", info.Extra, expected)
		}
		delPod()
		doTokenReview(t, cs, treq, true)
	})

	testPodWithAssignedNode := func(node *v1.Node) func(t *testing.T) {
		return func(t *testing.T) {
			treq := &authenticationv1.TokenRequest{
				Spec: authenticationv1.TokenRequestSpec{
					Audiences: []string{"api"},
					BoundObjectRef: &authenticationv1.BoundObjectReference{
						Kind:       "Pod",
						APIVersion: "v1",
						Name:       scheduledpod.Name,
					},
				},
			}

			warningHandler.clear()
			if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
				t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
			}
			warningHandler.assertEqual(t, nil)
			sa, del := createDeleteSvcAcct(t, cs, sa)
			defer del()

			warningHandler.clear()
			if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
				t.Fatalf("expected err creating token bound to nonexistant pod but got: %#v", resp)
			}
			warningHandler.assertEqual(t, nil)
			pod, delPod := createDeletePod(t, cs, scheduledpod)
			defer delPod()

			if node != nil {
				var delNode func()
				node, delNode = createDeleteNode(t, cs, node)
				defer delNode()
			}

			// right uid
			treq.Spec.BoundObjectRef.UID = pod.UID
			warningHandler.clear()
			if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
				t.Fatalf("err: %v", err)
			}
			warningHandler.assertEqual(t, nil)
			// wrong uid
			treq.Spec.BoundObjectRef.UID = wrongUID
			warningHandler.clear()
			if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
				t.Fatalf("expected err creating token bound to pod with wrong uid but got: %#v", resp)
			}
			warningHandler.assertEqual(t, nil)
			// no uid
			treq.Spec.BoundObjectRef.UID = noUID
			warningHandler.clear()
			treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			warningHandler.assertEqual(t, nil)

			checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
			checkPayload(t, treq.Status.Token, `["api"]`, "aud")
			checkPayload(t, treq.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
			checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
			checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
			checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

			expectedExtraValues := map[string]authenticationv1.ExtraValue{
				"authentication.kubernetes.io/pod-name": {pod.ObjectMeta.Name},
				"authentication.kubernetes.io/pod-uid":  {string(pod.ObjectMeta.UID)},
			}
			// If the NodeName is set at all, expect it to be included in the claims
			if pod.Spec.NodeName != "" {
				checkPayload(t, treq.Status.Token, fmt.Sprintf(`"%s"`, pod.Spec.NodeName), "kubernetes.io", "node", "name")
				expectedExtraValues["authentication.kubernetes.io/node-name"] = authenticationv1.ExtraValue{pod.Spec.NodeName}
			}
			// If the node is non-nil, we expect the UID to be set too
			if node != nil {
				checkPayload(t, treq.Status.Token, fmt.Sprintf(`"%s"`, node.UID), "kubernetes.io", "node", "uid")
				expectedExtraValues["authentication.kubernetes.io/node-uid"] = authenticationv1.ExtraValue{string(node.ObjectMeta.UID)}
			}

			info := doTokenReview(t, cs, treq, false)
			// we are not testing the credential-id feature, so delete this value from the returned extra info map
			delete(info.Extra, user.CredentialIDKey)
			if len(info.Extra) != len(expectedExtraValues) {
				t.Fatalf("expected Extra have length of %d but was length %d: %#v", len(expectedExtraValues), len(info.Extra), info.Extra)
			}
			if !reflect.DeepEqual(info.Extra, expectedExtraValues) {
				t.Fatalf("unexpected Extra:\ngot:\t%#v\nwant:\t%#v", info.Extra, expectedExtraValues)
			}

			delPod()
			doTokenReview(t, cs, treq, true)
		}
	}

	t.Run("bound to service account and a pod with an assigned nodeName that does not exist", testPodWithAssignedNode(nil))
	t.Run("bound to service account and a pod with an assigned nodeName", testPodWithAssignedNode(node))

	t.Run("fails to bind to a Node if the feature gate is disabled", func(t *testing.T) {
		// Disable node binding, emulating 1.32
		featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParseMajorMinor("1.32"))
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAccountTokenNodeBinding, false)

		// Create ServiceAccount and Node objects
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()
		node, delNode := createDeleteNode(t, cs, node)
		defer delNode()

		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Node",
					APIVersion: "v1",
					Name:       node.Name,
					UID:        node.UID,
				},
			},
		}
		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token with featuregate disabled but got: %#v", resp)
		} else if err.Error() != "cannot bind token to a Node object as the \"ServiceAccountTokenNodeBinding\" feature-gate is disabled" {
			t.Fatalf("expected error due to feature gate being disabled, but got: %s", err.Error())
		}
		warningHandler.assertEqual(t, nil)
	})

	t.Run("bound to service account and node", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Node",
					APIVersion: "v1",
					Name:       node.Name,
					UID:        node.UID,
				},
			},
		}

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant node but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		node, delNode := createDeleteNode(t, cs, node)
		defer delNode()

		// right uid
		treq.Spec.BoundObjectRef.UID = node.UID
		warningHandler.clear()
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to node with wrong uid but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `null`, "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, `"test-node"`, "kubernetes.io", "node", "name")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		doTokenReview(t, cs, treq, false)
		delNode()
		doTokenReview(t, cs, treq, true)
	})

	t.Run("bound to service account and secret", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Secret",
					APIVersion: "v1",
					Name:       secret.Name,
					UID:        secret.UID,
				},
			},
		}

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant secret but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		secret, delSecret := createDeleteSecret(t, cs, secret)
		defer delSecret()

		// right uid
		treq.Spec.BoundObjectRef.UID = secret.UID
		warningHandler.clear()
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to secret with wrong uid but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `null`, "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, `"test-secret"`, "kubernetes.io", "secret", "name")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		doTokenReview(t, cs, treq, false)
		delSecret()
		doTokenReview(t, cs, treq, true)
	})

	t.Run("bound to service account and pod running as different service account", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Name:       otherpod.Name,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()
		_, del = createDeletePod(t, cs, otherpod)
		defer del()

		warningHandler.clear()
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err but got: %#v", resp)
		}
		warningHandler.assertEqual(t, nil)
	})

	t.Run("expired token", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		doTokenReview(t, cs, treq, false)

		// backdate the token
		then := time.Now().Add(-2 * time.Hour)
		sc := &jwt.Claims{
			Subject:   apiserverserviceaccount.MakeUsername(sa.Namespace, sa.Name),
			Audience:  jwt.Audience([]string{"api"}),
			IssuedAt:  jwt.NewNumericDate(then),
			NotBefore: jwt.NewNumericDate(then),
			Expiry:    jwt.NewNumericDate(then.Add(time.Duration(60*60) * time.Second)),
		}
		coresa := core.ServiceAccount{
			ObjectMeta: sa.ObjectMeta,
		}
		_, pc, err := serviceaccount.Claims(coresa, nil, nil, nil, 0, 0, nil)
		if err != nil {
			t.Fatalf("err calling Claims: %v", err)
		}
		tok, err := tokenGenerator.GenerateToken(context.TODO(), sc, pc)
		if err != nil {
			t.Fatalf("err signing expired token: %v", err)
		}

		treq.Status.Token = tok
		doTokenReview(t, cs, treq, true)
	})

	t.Run("expiration extended token", func(t *testing.T) {
		var requestExp int64 = tokenExpirationSeconds
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &requestExp,
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Name:       pod.Name,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()
		pod, delPod := createDeletePod(t, cs, pod)
		defer delPod()
		treq.Spec.BoundObjectRef.UID = pod.UID

		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		doTokenReview(t, cs, treq, false)

		// Give some tolerance to avoid flakiness since we are using real time.
		var leeway int64 = 2
		actualExpiry := *jwt.NewNumericDate(time.Now().Add(time.Duration(24*365) * time.Hour))
		assumedExpiry := *jwt.NewNumericDate(time.Now().Add(time.Duration(requestExp) * time.Second))
		exp, err := strconv.ParseInt(getSubObject(t, getPayload(t, treq.Status.Token), "exp"), 10, 64)
		if err != nil {
			t.Fatalf("error parsing exp: %v", err)
		}
		warnafter, err := strconv.ParseInt(getSubObject(t, getPayload(t, treq.Status.Token), "kubernetes.io", "warnafter"), 10, 64)
		if err != nil {
			t.Fatalf("error parsing warnafter: %v", err)
		}

		if exp < int64(actualExpiry)-leeway || exp > int64(actualExpiry)+leeway {
			t.Errorf("unexpected token exp %d, should within range of %d +- %d seconds", exp, actualExpiry, leeway)
		}
		if warnafter < int64(assumedExpiry)-leeway || warnafter > int64(assumedExpiry)+leeway {
			t.Errorf("unexpected token warnafter %d, should within range of %d +- %d seconds", warnafter, assumedExpiry, leeway)
		}

		checkExpiration(t, treq, requestExp)
		expStatus := treq.Status.ExpirationTimestamp.Time.Unix()
		if expStatus < int64(assumedExpiry)-leeway || warnafter > int64(assumedExpiry)+leeway {
			t.Errorf("unexpected expiration returned in tokenrequest status %d, should within range of %d +- %d seconds", expStatus, assumedExpiry, leeway)
		}
	})

	t.Run("extended expiration extended does not apply for other audiences", func(t *testing.T) {
		var requestExp int64 = tokenExpirationSeconds
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"not-the-api", "api"},
				ExpirationSeconds: &requestExp,
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Name:       pod.Name,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()
		pod, delPod := createDeletePod(t, cs, pod)
		defer delPod()
		treq.Spec.BoundObjectRef.UID = pod.UID

		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		// Give some tolerance to avoid flakiness since we are using real time.
		var leeway int64 = 10
		actualExpiry := *jwt.NewNumericDate(time.Now().Add(time.Duration(60*60) * time.Second))
		assumedExpiry := *jwt.NewNumericDate(time.Now().Add(time.Duration(requestExp) * time.Second))

		warnAfter := getSubObject(t, getPayload(t, treq.Status.Token), "kubernetes.io", "warnafter")
		if warnAfter != "null" {
			t.Errorf("warn after should be empty.Found: %s", warnAfter)
		}

		exp, err := strconv.ParseInt(getSubObject(t, getPayload(t, treq.Status.Token), "exp"), 10, 64)
		if err != nil {
			t.Fatalf("error parsing exp: %v", err)
		}
		if exp < int64(actualExpiry)-leeway || exp > int64(actualExpiry)+leeway {
			t.Errorf("unexpected token exp %d, should within range of %d +- %d seconds", exp, actualExpiry, leeway)
		}

		checkExpiration(t, treq, requestExp)
		expStatus := treq.Status.ExpirationTimestamp.Time.Unix()
		if expStatus < int64(assumedExpiry)-leeway {
			t.Errorf("unexpected expiration returned in tokenrequest status %d, should within range of %d +- %d seconds", expStatus, assumedExpiry, leeway)
		}
	})

	t.Run("a token without an api audience is invalid", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"not-the-api"},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		doTokenReview(t, cs, treq, true)
	})

	t.Run("a tokenrequest without an audience is valid against the api", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		warningHandler.clear()
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `["api"]`, "aud")

		doTokenReview(t, cs, treq, false)
	})

	t.Run("a token should be invalid after recreating same name pod", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Name:       pod.Name,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()
		originalPod, originalDelPod := createDeletePod(t, cs, pod)
		defer originalDelPod()

		treq.Spec.BoundObjectRef.UID = originalPod.UID
		warningHandler.clear()
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		doTokenReview(t, cs, treq, false)
		originalDelPod()
		doTokenReview(t, cs, treq, true)

		_, recreateDelPod := createDeletePod(t, cs, pod)
		defer recreateDelPod()

		doTokenReview(t, cs, treq, true)
	})

	t.Run("a token should be invalid after recreating same name secret", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Secret",
					APIVersion: "v1",
					Name:       secret.Name,
					UID:        secret.UID,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		originalSecret, originalDelSecret := createDeleteSecret(t, cs, secret)
		defer originalDelSecret()

		treq.Spec.BoundObjectRef.UID = originalSecret.UID
		warningHandler.clear()
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `null`, "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, `"test-secret"`, "kubernetes.io", "secret", "name")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		doTokenReview(t, cs, treq, false)
		originalDelSecret()
		doTokenReview(t, cs, treq, true)

		_, recreateDelSecret := createDeleteSecret(t, cs, secret)
		defer recreateDelSecret()

		doTokenReview(t, cs, treq, true)
	})

	t.Run("a token request within expiration time", func(t *testing.T) {
		normalExpirationTime := maxExpirationSeconds - 10*60
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &normalExpirationTime,
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Secret",
					APIVersion: "v1",
					Name:       secret.Name,
					UID:        secret.UID,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		originalSecret, originalDelSecret := createDeleteSecret(t, cs, secret)
		defer originalDelSecret()

		treq.Spec.BoundObjectRef.UID = originalSecret.UID
		warningHandler.clear()
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, nil)

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `null`, "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, `"test-secret"`, "kubernetes.io", "secret", "name")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")
		checkExpiration(t, treq, normalExpirationTime)

		doTokenReview(t, cs, treq, false)
		originalDelSecret()
		doTokenReview(t, cs, treq, true)

		_, recreateDelSecret := createDeleteSecret(t, cs, secret)
		defer recreateDelSecret()

		doTokenReview(t, cs, treq, true)
	})

	t.Run("a token request with out-of-range expiration", func(t *testing.T) {
		tooLongExpirationTime := maxExpirationSeconds + 10*60
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &tooLongExpirationTime,
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Secret",
					APIVersion: "v1",
					Name:       secret.Name,
					UID:        secret.UID,
				},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		originalSecret, originalDelSecret := createDeleteSecret(t, cs, secret)
		defer originalDelSecret()

		treq.Spec.BoundObjectRef.UID = originalSecret.UID
		warningHandler.clear()
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		warningHandler.assertEqual(t, []string{fmt.Sprintf("requested expiration of %d seconds shortened to %d seconds", tooLongExpirationTime, maxExpirationSeconds)})

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `null`, "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, `"test-secret"`, "kubernetes.io", "secret", "name")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")
		checkExpiration(t, treq, maxExpirationSeconds)

		doTokenReview(t, cs, treq, false)
		originalDelSecret()
		doTokenReview(t, cs, treq, true)

		_, recreateDelSecret := createDeleteSecret(t, cs, secret)
		defer recreateDelSecret()

		doTokenReview(t, cs, treq, true)
	})

	t.Run("a token is valid against the HTTP-provided service account issuer metadata", func(t *testing.T) {
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		t.Log("get token")
		warningHandler.clear()
		tokenRequest, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(tCtx, sa.Name,
			&authenticationv1.TokenRequest{
				Spec: authenticationv1.TokenRequestSpec{
					Audiences: []string{"api"},
				},
			}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error creating token: %v", err)
		}
		warningHandler.assertEqual(t, nil)
		token := tokenRequest.Status.Token
		if token == "" {
			t.Fatal("no token")
		}

		t.Log("get discovery doc")
		discoveryDoc := struct {
			Issuer string `json:"issuer"`
			JWKS   string `json:"jwks_uri"`
		}{}

		// A little convoluted, but the base path is hidden inside the RESTClient.
		// We can't just use the RESTClient, because it throws away the headers
		// before returning a result, and we need to check the headers.
		discoveryURL := rc.Get().AbsPath("/.well-known/openid-configuration").URL().String()
		resp, err := rc.Client.Get(discoveryURL)
		if err != nil {
			t.Fatalf("error getting metadata: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Errorf("got status: %v, want: %v", resp.StatusCode, http.StatusOK)
		}
		if got, want := resp.Header.Get("Content-Type"), "application/json"; got != want {
			t.Errorf("got Content-Type: %v, want: %v", got, want)
		}
		if got, want := resp.Header.Get("Cache-Control"), "public, max-age=3600"; got != want {
			t.Errorf("got Cache-Control: %v, want: %v", got, want)
		}

		b, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		md := bytes.NewBuffer(b)
		t.Logf("raw discovery doc response:\n---%s\n---", md.String())
		if md.Len() == 0 {
			t.Fatal("empty response for discovery doc")
		}

		if err := json.NewDecoder(md).Decode(&discoveryDoc); err != nil {
			t.Fatalf("could not decode metadata: %v", err)
		}
		if discoveryDoc.Issuer != iss {
			t.Fatalf("invalid issuer in discovery doc: got %s, want %s",
				discoveryDoc.Issuer, iss)
		}
		expectJWKSURI := (&url.URL{
			Scheme: "https",
			Host:   serverAddress,
			Path:   serviceaccount.JWKSPath,
		}).String()
		if discoveryDoc.JWKS != expectJWKSURI {
			t.Fatalf("unexpected jwks_uri in discovery doc: got %s, want %s",
				discoveryDoc.JWKS, expectJWKSURI)
		}

		// Since the test framework hardcodes the host, we combine our client's
		// scheme and host with serviceaccount.JWKSPath. We know that this is what was
		// in the discovery doc because we checked that it matched above.
		jwksURI := rc.Get().AbsPath(serviceaccount.JWKSPath).URL().String()
		t.Log("get jwks from", jwksURI)
		resp, err = rc.Client.Get(jwksURI)
		if err != nil {
			t.Fatalf("error getting jwks: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Errorf("got status: %v, want: %v", resp.StatusCode, http.StatusOK)
		}
		if got, want := resp.Header.Get("Content-Type"), "application/jwk-set+json"; got != want {
			t.Errorf("got Content-Type: %v, want: %v", got, want)
		}
		if got, want := resp.Header.Get("Cache-Control"), "public, max-age=3600"; got != want {
			t.Errorf("got Cache-Control: %v, want: %v", got, want)
		}

		b, err = io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		ks := bytes.NewBuffer(b)
		if ks.Len() == 0 {
			t.Fatal("empty jwks")
		}
		t.Logf("raw JWKS: \n---\n%s\n---", ks.String())

		jwks := jose.JSONWebKeySet{}
		if err := json.NewDecoder(ks).Decode(&jwks); err != nil {
			t.Fatalf("could not decode JWKS: %v", err)
		}
		if len(jwks.Keys) != 1 {
			t.Fatalf("len(jwks.Keys) = %d, want 1", len(jwks.Keys))
		}
		key := jwks.Keys[0]
		tok, err := jwt.ParseSigned(token)
		if err != nil {
			t.Fatalf("could not parse token %q: %v", token, err)
		}
		var claims jwt.Claims
		if err := tok.Claims(key, &claims); err != nil {
			t.Fatalf("could not validate claims on token: %v", err)
		}
		if err := claims.Validate(jwt.Expected{Issuer: discoveryDoc.Issuer}); err != nil {
			t.Fatalf("invalid claims: %v", err)
		}
	})
}

func doTokenReview(t *testing.T, cs clientset.Interface, treq *authenticationv1.TokenRequest, expectErr bool) authenticationv1.UserInfo {
	t.Helper()
	tries := 0
	for {
		trev, err := cs.AuthenticationV1().TokenReviews().Create(context.TODO(), &authenticationv1.TokenReview{
			Spec: authenticationv1.TokenReviewSpec{
				Token: treq.Status.Token,
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		t.Logf("status: %+v", trev.Status)
		if (trev.Status.Error != "") && !expectErr {
			t.Fatalf("expected no error but got: %v", trev.Status.Error)
		}
		if (trev.Status.Error == "") && expectErr {
			// if we expected an error and didn't get one, retry
			// to let changes that invalidate the token percolate through informers
			if tries < 10 {
				tries++
				time.Sleep(100 * time.Millisecond)
				t.Logf("expected error but got: %+v, retrying", trev.Status)
				continue
			}
			t.Fatalf("expected error but got: %+v", trev.Status)
		}
		if !trev.Status.Authenticated && !expectErr {
			t.Fatal("expected token to be authenticated but it wasn't")
		}
		return trev.Status.User
	}
}

func checkPayload(t *testing.T, tok string, want string, parts ...string) {
	t.Helper()
	got := getSubObject(t, getPayload(t, tok), parts...)
	if got != want {
		t.Errorf("unexpected payload.\nsaw:\t%v\nwant:\t%v", got, want)
	}
}

func checkExpiration(t *testing.T, treq *authenticationv1.TokenRequest, expectedExpiration int64) {
	t.Helper()
	if treq.Spec.ExpirationSeconds == nil {
		t.Errorf("unexpected nil expiration seconds.")
	}
	if *treq.Spec.ExpirationSeconds != expectedExpiration {
		t.Errorf("unexpected expiration seconds.\nsaw:\t%d\nwant:\t%d", treq.Spec.ExpirationSeconds, expectedExpiration)
	}
}

func getSubObject(t *testing.T, b string, parts ...string) string {
	t.Helper()
	var obj interface{}
	obj = make(map[string]interface{})
	if err := json.Unmarshal([]byte(b), &obj); err != nil {
		t.Fatalf("err: %v", err)
	}
	for _, part := range parts {
		obj = obj.(map[string]interface{})[part]
	}
	out, err := json.Marshal(obj)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return string(out)
}

func getPayload(t *testing.T, b string) string {
	t.Helper()
	parts := strings.Split(b, ".")
	if len(parts) != 3 {
		t.Fatalf("token did not have three parts: %v", b)
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		t.Fatalf("failed to base64 decode token: %v", err)
	}
	return string(payload)
}

func createDeleteSvcAcct(t *testing.T, cs clientset.Interface, sa *v1.ServiceAccount) (*v1.ServiceAccount, func()) {
	t.Helper()
	sa, err := cs.CoreV1().ServiceAccounts(sa.Namespace).Create(context.TODO(), sa, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	done := false
	return sa, func() {
		t.Helper()
		if done {
			return
		}
		done = true
		if err := cs.CoreV1().ServiceAccounts(sa.Namespace).Delete(context.TODO(), sa.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func createDeletePod(t *testing.T, cs clientset.Interface, pod *v1.Pod) (*v1.Pod, func()) {
	t.Helper()
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	done := false
	return pod, func() {
		t.Helper()
		if done {
			return
		}
		done = true
		if err := cs.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{
			GracePeriodSeconds: ptr.To(int64(0)),
		}); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func createDeleteSecret(t *testing.T, cs clientset.Interface, sec *v1.Secret) (*v1.Secret, func()) {
	t.Helper()
	sec, err := cs.CoreV1().Secrets(sec.Namespace).Create(context.TODO(), sec, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	done := false
	return sec, func() {
		t.Helper()
		if done {
			return
		}
		done = true
		if err := cs.CoreV1().Secrets(sec.Namespace).Delete(context.TODO(), sec.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func createDeleteNode(t *testing.T, cs clientset.Interface, node *v1.Node) (*v1.Node, func()) {
	t.Helper()
	node, err := cs.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	done := false
	return node, func() {
		t.Helper()
		if done {
			return
		}
		done = true
		if err := cs.CoreV1().Nodes().Delete(context.TODO(), node.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

type recordingWarningHandler struct {
	warnings []string

	sync.Mutex
}

func (r *recordingWarningHandler) HandleWarningHeader(code int, agent string, message string) {
	r.Lock()
	defer r.Unlock()
	r.warnings = append(r.warnings, message)
}

func (r *recordingWarningHandler) clear() {
	r.Lock()
	defer r.Unlock()
	r.warnings = nil
}
func (r *recordingWarningHandler) assertEqual(t *testing.T, expected []string) {
	t.Helper()
	r.Lock()
	defer r.Unlock()
	if !reflect.DeepEqual(r.warnings, expected) {
		t.Errorf("expected\n\t%v\ngot\n\t%v", expected, r.warnings)
	}
}
