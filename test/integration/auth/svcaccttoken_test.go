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

	sa := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
	}

	tr1 := &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: []string{"aud"},
		},
	}

	_, err = cs.CoreV1().ServiceAccounts(sa.Namespace).Create(sa)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	tr1, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, tr1)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	checkPayload(t, tr1.Status.Token, `"system:serviceaccount:default:test"`, "sub")
	checkPayload(t, tr1.Status.Token, `["aud"]`, "aud")
	checkPayload(t, tr1.Status.Token, "null", "kubernetes.io", "pod")
	checkPayload(t, tr1.Status.Token, "null", "kubernetes.io", "secret")
	checkPayload(t, tr1.Status.Token, `"default"`, "kubernetes.io", "namespace")
	checkPayload(t, tr1.Status.Token, `"test"`, "kubernetes.io", "serviceaccount", "name")

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: sa.Namespace,
		},
		Spec: v1.PodSpec{
			ServiceAccountName: sa.Name,
			Containers:         []v1.Container{{Name: "test-container", Image: "nginx"}},
		},
	}
	_, err = cs.CoreV1().Pods(pod.Namespace).Create(pod)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	tr2 := &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: []string{"aud"},
			BoundObjectRef: &authenticationv1.BoundObjectReference{
				Kind:       "Pod",
				APIVersion: "v1",
				Name:       pod.Name,
			},
		},
	}
	tr2, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, tr2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	checkPayload(t, tr2.Status.Token, `"system:serviceaccount:default:test"`, "sub")
	checkPayload(t, tr2.Status.Token, `["aud"]`, "aud")
	checkPayload(t, tr2.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
	checkPayload(t, tr2.Status.Token, "null", "kubernetes.io", "secret")
	checkPayload(t, tr2.Status.Token, `"default"`, "kubernetes.io", "namespace")
	checkPayload(t, tr2.Status.Token, `"test"`, "kubernetes.io", "serviceaccount", "name")

	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: sa.Namespace,
		},
	}
	_, err = cs.CoreV1().Secrets(secret.Namespace).Create(secret)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	tr3 := &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: []string{"aud"},
			BoundObjectRef: &authenticationv1.BoundObjectReference{
				Kind:       "Secret",
				APIVersion: "v1",
				Name:       secret.Name,
			},
		},
	}
	tr3, err = cs.CoreV1().ServiceAccounts(sa.Namespace).CreateToken(sa.Name, tr3)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	checkPayload(t, tr2.Status.Token, `"system:serviceaccount:default:test"`, "sub")
	checkPayload(t, tr2.Status.Token, `["aud"]`, "aud")
	checkPayload(t, tr2.Status.Token, `"test-pod"`, "kubernetes.io", "pod", "name")
	checkPayload(t, tr2.Status.Token, `null`, "kubernetes.io", "secret")
	checkPayload(t, tr2.Status.Token, `"default"`, "kubernetes.io", "namespace")
	checkPayload(t, tr2.Status.Token, `"test"`, "kubernetes.io", "serviceaccount", "name")
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
