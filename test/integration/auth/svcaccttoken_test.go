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
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	clientset "k8s.io/client-go/kubernetes"
	certutil "k8s.io/client-go/util/cert"
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
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TokenRequest, true)()

	// Build client config, clientset, and informers
	sk, err := certutil.ParsePrivateKeyPEM([]byte(ecdsaPrivateKey))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Start the server
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authorization.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()
	masterConfig.ExtraConfig.ServiceAccountIssuer = serviceaccount.JWTTokenGenerator("https://foo.bar.example.com", sk)

	master, _, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	cs, err := clientset.NewForConfig(master.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		t.Fatalf("err: %v", err)
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

		one      = int64(1)
		wrongUID = types.UID("wrong")
		noUID    = types.UID("")
	)

	t.Run("bound to service account", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &one,
			},
		}

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")
	})

	t.Run("bound to service account and pod", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &one,
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Name:       pod.Name,
				},
			},
		}

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant pod but got: %#v", resp)
		}
		pod, del := createDeletePod(t, cs, pod)
		defer del()

		// right uid
		treq.Spec.BoundObjectRef.UID = pod.UID
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err != nil {
			t.Fatalf("err: %v", err)
		}
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token bound to pod with wrong uid but got: %#v", resp)
		}
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
		checkPayload(t, treq.Status.Token, "null", "kubernetes.io", "secret")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")
	})

	t.Run("bound to service account and secret", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &one,
				BoundObjectRef: &authenticationv1.BoundObjectReference{
					Kind:       "Secret",
					APIVersion: "v1",
					Name:       secret.Name,
					UID:        secret.UID,
				},
			},
		}

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token for nonexistant svcacct but got: %#v", resp)
		}
		sa, del := createDeleteSvcAcct(t, cs, sa)
		defer del()

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token bound to nonexistant secret but got: %#v", resp)
		}
		secret, del := createDeleteSecret(t, cs, secret)
		defer del()

		// right uid
		treq.Spec.BoundObjectRef.UID = secret.UID
		if _, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err != nil {
			t.Fatalf("err: %v", err)
		}
		// wrong uid
		treq.Spec.BoundObjectRef.UID = wrongUID
		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err creating token bound to secret with wrong uid but got: %#v", resp)
		}
		// no uid
		treq.Spec.BoundObjectRef.UID = noUID
		treq, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		checkPayload(t, treq.Status.Token, `"system:serviceaccount:myns:test-svcacct"`, "sub")
		checkPayload(t, treq.Status.Token, `["api"]`, "aud")
		checkPayload(t, treq.Status.Token, `null`, "kubernetes.io", "pod")
		checkPayload(t, treq.Status.Token, `"test-secret"`, "kubernetes.io", "secret", "name")
		checkPayload(t, treq.Status.Token, `"myns"`, "kubernetes.io", "namespace")
		checkPayload(t, treq.Status.Token, `"test-svcacct"`, "kubernetes.io", "serviceaccount", "name")
	})

	t.Run("bound to service account and pod running as different service account", func(t *testing.T) {
		treq := &authenticationv1.TokenRequest{
			Spec: authenticationv1.TokenRequestSpec{
				Audiences:         []string{"api"},
				ExpirationSeconds: &one,
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

		if resp, err := cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, treq); err == nil {
			t.Fatalf("expected err but got: %#v", resp)
		}
	})
}

func checkPayload(t *testing.T, tok string, want string, parts ...string) {
	t.Helper()
	got := getSubObject(t, getPayload(t, tok), parts...)
	if got != want {
		t.Errorf("unexpected payload.\nsaw:\t%v\nwant:\t%v", got, want)
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
	sa, err := cs.CoreV1().ServiceAccounts(sa.Namespace).Create(sa)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return sa, func() {
		t.Helper()
		if err := cs.CoreV1().ServiceAccounts(sa.Namespace).Delete(sa.Name, nil); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func createDeletePod(t *testing.T, cs clientset.Interface, pod *v1.Pod) (*v1.Pod, func()) {
	t.Helper()
	pod, err := cs.CoreV1().Pods(pod.Namespace).Create(pod)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return pod, func() {
		t.Helper()
		if err := cs.CoreV1().Pods(pod.Namespace).Delete(pod.Name, nil); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func createDeleteSecret(t *testing.T, cs clientset.Interface, sec *v1.Secret) (*v1.Secret, func()) {
	t.Helper()
	sec, err := cs.CoreV1().Secrets(sec.Namespace).Create(sec)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return sec, func() {
		t.Helper()
		if err := cs.CoreV1().Secrets(sec.Namespace).Delete(sec.Name, nil); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}
