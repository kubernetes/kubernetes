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
	"crypto/ecdsa"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	jose "gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/jwt"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/keyutil"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	serviceaccountgetter "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/test/integration/framework"
)

const ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEZmTmUhuanLjPA2CLquXivuwBDHTt5XYwgIr/kA1LtRoAoGCCqGSM49
AwEHoUQDQgAEH6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp/C/ASqiIGUeeKQtX0
/IR3qCXyThP/dbCiHrF3v1cuhBOHY8CLVg==
-----END EC PRIVATE KEY-----`

func TestServiceAccountTokenCreate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TokenRequest, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceAccountIssuerDiscovery, true)()

	// Build client config, clientset, and informers
	sk, err := keyutil.ParsePrivateKeyPEM([]byte(ecdsaPrivateKey))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	pk := sk.(*ecdsa.PrivateKey).PublicKey

	const iss = "https://foo.bar.example.com"
	aud := authenticator.Audiences{"api"}

	maxExpirationSeconds := int64(60 * 60 * 2)
	maxExpirationDuration, err := time.ParseDuration(fmt.Sprintf("%ds", maxExpirationSeconds))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	gcs := &clientset.Clientset{}

	// Start the server
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authorization.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()
	masterConfig.GenericConfig.Authentication.APIAudiences = aud
	masterConfig.GenericConfig.Authentication.Authenticator = bearertoken.New(
		serviceaccount.JWTTokenAuthenticator(
			iss,
			[]interface{}{&pk},
			aud,
			serviceaccount.NewValidator(serviceaccountgetter.NewGetterFromClient(
				gcs,
				v1listers.NewSecretLister(newIndexer(func(namespace, name string) (interface{}, error) {
					return gcs.CoreV1().Secrets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
				})),
				v1listers.NewServiceAccountLister(newIndexer(func(namespace, name string) (interface{}, error) {
					return gcs.CoreV1().ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
				})),
				v1listers.NewPodLister(newIndexer(func(namespace, name string) (interface{}, error) {
					return gcs.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{})
				})),
			)),
		),
	)

	tokenGenerator, err := serviceaccount.JWTTokenGenerator(iss, sk)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	masterConfig.ExtraConfig.ServiceAccountIssuer = tokenGenerator
	masterConfig.ExtraConfig.ServiceAccountMaxExpiration = maxExpirationDuration
	masterConfig.GenericConfig.Authentication.APIAudiences = aud
	masterConfig.ExtraConfig.ExtendExpiration = true

	masterConfig.ExtraConfig.ServiceAccountIssuerURL = iss
	masterConfig.ExtraConfig.ServiceAccountJWKSURI = ""
	masterConfig.ExtraConfig.ServiceAccountPublicKeys = []interface{}{&pk}

	master, _, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	cs, err := clientset.NewForConfig(master.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	*gcs = *cs

	rc, err := rest.UnversionedRESTClientFor(master.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	var (
		sa = &v1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-svcacct",
				Namespace: "myns",
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

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		sa, delSvcAcct := createDeleteSvcAcct(t, cs, sa)
		defer delSvcAcct()

		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		info := doTokenReview(t, cs, treq, false)
		if info.Extra != nil {
			t.Fatalf("expected Extra to be nil but got: %#v", info.Extra)
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

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant pod but got: %#v", resp)
		}
		pod, delPod := createDeletePod(t, cs, pod)
		defer delPod()

		// right uid
		treq.Spec.BoundObjectRef.UID = pod.UID
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to pod with wrong uid but got: %#v", resp)
		}
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")

		info := doTokenReview(t, cs, treq, false)
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

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant secret but got: %#v", resp)
		}
		secret, delSecret := createDeleteSecret(t, cs, secret)
		defer delSecret()

		// right uid
		treq.Spec.BoundObjectRef.UID = secret.UID
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err creating token bound to secret with wrong uid but got: %#v", resp)
		}
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

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

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err == nil {
			t.Fatalf("expected err but got: %#v", resp)
		}
	})

	t.Run("expired token", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"api"},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

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
		_, pc := serviceaccount.Claims(coresa, nil, nil, 0, 0, nil)
		tok, err := masterConfig.ExtraConfig.ServiceAccountIssuer.GenerateToken(sc, pc)
		if err != nil {
			t.Fatalf("err signing expired token: %v", err)
		}

		treq.Status.Token = tok
		doTokenReview(t, cs, treq, true)
	})

	t.Run("expiration extended token", func(t *testing.T) {
		var requestExp int64 = 60*60 + 7
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

		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		doTokenReview(t, cs, treq, false)

		// Give some tolerance to avoid flakiness since we are using real time.
		var leeway int64 = 2
		actualExpiry := jwt.NewNumericDate(time.Now().Add(time.Duration(24*365) * time.Hour))
		assumedExpiry := jwt.NewNumericDate(time.Now().Add(time.Duration(requestExp) * time.Second))
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

	t.Run("a token without an api audience is invalid", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences: []string{"not-the-api"},
			},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		doTokenReview(t, cs, treq, true)
	})

	t.Run("a tokenrequest without an audience is valid against the api", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{},
		}

		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("err: %v", err)
		}

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
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}

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
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}

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
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}

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
		if treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(context.TODO(), sa.Name, treq, metav1.CreateOptions{}); err != nil {
			t.Fatalf("err: %v", err)
		}

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
		tokenRequest, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(
			context.TODO(),
			sa.Name,
			&authenticationv1.TokenRequest{
				Spec: authenticationv1.TokenRequestSpec{
					Audiences: []string{"api"},
				},
			}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error creating token: %v", err)
		}
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

		b, err := ioutil.ReadAll(resp.Body)
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
		// Parse the JWKSURI see if the path is what we expect. Since the
		// integration test framework hardcodes 192.168.10.4 as the PublicAddress,
		// which results in the same for ExternalAddress, we expect the JWKS URI
		// to be 192.168.10.4:443, even if that's not necessarily the external
		// IP of the test machine.
		expectJWKSURI := (&url.URL{
			Scheme: "https",
			Host:   "192.168.10.4:443",
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

		b, err = ioutil.ReadAll(resp.Body)
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
		t.Fatalf("expected error but got: %+v", trev.Status)
	}
	if !trev.Status.Authenticated && !expectErr {
		t.Fatal("expected token to be authenticated but it wasn't")
	}
	return trev.Status.User
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
		if err := cs.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{}); err != nil {
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

func newIndexer(get func(namespace, name string) (interface{}, error)) cache.Indexer {
	return &fakeIndexer{get: get}
}

type fakeIndexer struct {
	cache.Indexer
	get func(namespace, name string) (interface{}, error)
}

func (f *fakeIndexer) GetByKey(key string) (interface{}, bool, error) {
	parts := strings.SplitN(key, "/", 2)
	namespace := parts[0]
	name := ""
	if len(parts) == 2 {
		name = parts[1]
	}
	obj, err := f.get(namespace, name)
	return obj, err == nil, err
}
