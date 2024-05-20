/*
Copyright 2019 The Kubernetes Authors.

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
	"os"
	"testing"
	"time"

	utiltesting "k8s.io/client-go/util/testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/controller-manager/pkg/clientbuilder"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestDynamicClientBuilder(t *testing.T) {
	tmpfile, err := os.CreateTemp("/tmp", "key")
	if err != nil {
		t.Fatalf("create temp file failed: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, tmpfile)

	if err = os.WriteFile(tmpfile.Name(), []byte(ecdsaPrivateKey), 0666); err != nil {
		t.Fatalf("write file %s failed: %v", tmpfile.Name(), err)
	}

	const iss = "https://foo.bar.example.com"
	aud := authenticator.Audiences{"api"}

	maxExpirationDuration := time.Second * 60 * 60
	if err != nil {
		t.Fatalf("parse duration failed: %v", err)
	}

	logger, tCtx := ktesting.NewTestContext(t)
	baseClient, baseConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceAccountSigningKeyFile = tmpfile.Name()
			opts.ServiceAccountTokenMaxExpiration = maxExpirationDuration
			if opts.Authentication == nil {
				opts.Authentication = &kubeoptions.BuiltInAuthenticationOptions{}
			}

			opts.Authentication.APIAudiences = aud
			if opts.Authentication.ServiceAccounts == nil {
				opts.Authentication.ServiceAccounts = &kubeoptions.ServiceAccountAuthenticationOptions{}
			}
			opts.Authentication.ServiceAccounts.Issuers = []string{iss}
			opts.Authentication.ServiceAccounts.KeyFiles = []string{tmpfile.Name()}
			opts.Authorization.Modes = []string{"AlwaysAllow"}
		},
	})
	defer tearDownFn()

	// We want to test if the token rotation works fine here.
	// To minimize the time this test would consume, we use the minimial token expiration.
	// The minimial token expiration is defined in:
	// pkg/apis/authentication/validation/validation.go
	exp := int64(600)
	leeway := 99
	ns := "default"
	clientBuilder := clientbuilder.NewTestDynamicClientBuilder(
		restclient.AnonymousClientConfig(baseConfig),
		baseClient.CoreV1(),
		ns, exp, leeway)

	saName := "dt"
	dymClient, err := clientBuilder.Client(logger, saName)

	if err != nil {
		t.Fatalf("build client via dynamic client builder failed: %v", err)
	}

	if err = testClientBuilder(dymClient, ns, saName); err != nil {
		t.Fatalf("dynamic client get resources failed befroe deleting sa: %v", err)
	}

	// We want to trigger token rotation here by deleting service account
	// the dynamic client was using.
	if err = dymClient.CoreV1().ServiceAccounts(ns).Delete(tCtx, saName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete service account %s failed: %v", saName, err)
	}
	time.Sleep(time.Second * 10)

	if err = testClientBuilder(dymClient, ns, saName); err != nil {
		t.Fatalf("dynamic client get resources failed after deleting sa: %v", err)
	}
}

func testClientBuilder(dymClient clientset.Interface, ns, saName string) error {
	_, err := dymClient.CoreV1().Namespaces().Get(context.TODO(), ns, metav1.GetOptions{})
	if err != nil {
		return err
	}

	_, err = dymClient.CoreV1().ServiceAccounts(ns).Get(context.TODO(), saName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}
