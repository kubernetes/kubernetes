/*
Copyright 2018 The Kubernetes Authors.

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

package kubecontrollermanager

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strings"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	ctrlmgrtesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestStartTestServer(t *testing.T) {
	// Insulate this test from picking up in-cluster config when run inside a pod
	// We can't assume we have permissions to write to /var/run/secrets/... from a unit test to mock in-cluster config for testing
	originalHost := os.Getenv("KUBERNETES_SERVICE_HOST")
	if len(originalHost) > 0 {
		os.Setenv("KUBERNETES_SERVICE_HOST", "")
		defer os.Setenv("KUBERNETES_SERVICE_HOST", originalHost)
	}

	// authenticate to apiserver via bearer token
	token := "flwqkenfjasasdfmwerasd"
	tokenFile, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	tokenFile.WriteString(fmt.Sprintf(`
%s,kube-controller-manager,kube-controller-manager,""
`, token))
	tokenFile.Close()

	// start apiserver
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--token-auth-file", tokenFile.Name(),
		"--authorization-mode", "RBAC",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// allow kube-controller-manager to do SubjectAccessReview
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error creating client config: %v", err)
	}
	_, err = client.RbacV1().ClusterRoleBindings().Create(&rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "kube-controller-manager:system:auth-delegator"},
		Subjects: []rbacv1.Subject{{
			Kind: "User",
			Name: "kube-controller-manager",
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:auth-delegator",
		},
	})
	if err != nil {
		t.Fatalf("failed to create system:auth-delegator rbac cluster role binding: %v", err)
	}

	// allow kube-controller-manager to read kube-system/extension-apiserver-authentication
	_, err = client.RbacV1().RoleBindings("kube-system").Create(&rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "kube-controller-manager:extension-apiserver-authentication-reader"},
		Subjects: []rbacv1.Subject{{
			Kind: "User",
			Name: "kube-controller-manager",
		}},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
			Name:     "extension-apiserver-authentication-reader",
		},
	})
	if err != nil {
		t.Fatalf("failed to create kube-controller-manager:extension-apiserver-authentication-reader rbac role binding: %v", err)
	}

	// create kubeconfig for the apiserver
	apiserverConfig, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	apiserverConfig.WriteString(fmt.Sprintf(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: %s
    certificate-authority: %s
  name: integration
contexts:
- context:
    cluster: integration
    user: kube-controller-manager
  name: default-context
current-context: default-context
users:
- name: kube-controller-manager
  user:
    token: %s
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile, token))
	apiserverConfig.Close()

	// create BROKEN kubeconfig for the apiserver
	brokenApiserverConfig, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	brokenApiserverConfig.WriteString(fmt.Sprintf(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: %s
    certificate-authority: %s
  name: integration
contexts:
- context:
    cluster: integration
    user: kube-controller-manager
  name: default-context
current-context: default-context
users:
- name: kube-controller-manager
  user:
    token: WRONGTOKEN
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile))
	brokenApiserverConfig.Close()

	tests := []struct {
		name                             string
		flags                            []string
		path                             string
		anonymous                        bool // to use the token or not
		wantErr                          bool
		wantSecureCode, wantInsecureCode *int
	}{
		{"no-flags", nil, "/healthz", false, true, nil, nil},
		{"insecurely /healthz", []string{
			"--secure-port=0",
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/healthz", true, false, nil, intPtr(http.StatusOK)},
		{"insecurely /metrics", []string{
			"--secure-port=0",
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/metrics", true, false, nil, intPtr(http.StatusOK)},
		{"/healthz without authn/authz", []string{
			"--port=0",
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/healthz", true, false, intPtr(http.StatusOK), nil},
		{"/metrics without auhn/z", []string{
			"--kubeconfig", apiserverConfig.Name(),
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/metrics", true, false, intPtr(http.StatusForbidden), intPtr(http.StatusOK)},
		{"authorization skipped for /healthz with authn/authz", []string{
			"--port=0",
			"--authentication-kubeconfig", apiserverConfig.Name(),
			"--authorization-kubeconfig", apiserverConfig.Name(),
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/healthz", false, false, intPtr(http.StatusOK), nil},
		{"authorization skipped for /healthz with BROKEN authn/authz", []string{
			"--port=0",
			"--authentication-skip-lookup", // to survive unaccessible extensions-apiserver-authentication configmap
			"--authentication-kubeconfig", brokenApiserverConfig.Name(),
			"--authorization-kubeconfig", brokenApiserverConfig.Name(),
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/healthz", false, false, intPtr(http.StatusOK), nil},
		{"not authorized /metrics", []string{
			"--port=0",
			"--authentication-kubeconfig", apiserverConfig.Name(),
			"--authorization-kubeconfig", apiserverConfig.Name(),
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/metrics", false, false, intPtr(http.StatusForbidden), nil},
		{"not authorized /metrics with BROKEN authn/authz", []string{
			"--authentication-kubeconfig", apiserverConfig.Name(),
			"--authorization-kubeconfig", brokenApiserverConfig.Name(),
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/metrics", false, false, intPtr(http.StatusInternalServerError), intPtr(http.StatusOK)},
		{"always-allowed /metrics with BROKEN authn/authz", []string{
			"--port=0",
			"--authentication-skip-lookup", // to survive unaccessible extensions-apiserver-authentication configmap
			"--authentication-kubeconfig", apiserverConfig.Name(),
			"--authorization-kubeconfig", apiserverConfig.Name(),
			"--authorization-always-allow-paths", "/healthz,/metrics",
			"--kubeconfig", apiserverConfig.Name(),
			"--leader-elect=false",
		}, "/metrics", false, false, intPtr(http.StatusOK), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotResult, err := ctrlmgrtesting.StartTestServer(t, tt.flags)
			if gotResult.TearDownFn != nil {
				defer gotResult.TearDownFn()
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("StartTestServer() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			if want, got := tt.wantSecureCode != nil, gotResult.Config.SecureServing != nil; want != got {
				t.Errorf("SecureServing enabled: expected=%v got=%v", want, got)
			} else if want {
				url := fmt.Sprintf("https://%s%s", gotResult.Config.SecureServing.Listener.Addr().String(), tt.path)
				url = strings.Replace(url, "[::]", "127.0.0.1", -1) // switch to IPv4 because the self-signed cert does not support [::]

				// read self-signed server cert disk
				pool := x509.NewCertPool()
				serverCertPath := path.Join(gotResult.Options.SecureServing.ServerCert.CertDirectory, gotResult.Options.SecureServing.ServerCert.PairName+".crt")
				serverCert, err := ioutil.ReadFile(serverCertPath)
				if err != nil {
					t.Fatalf("Failed to read controller-manager server cert %q: %v", serverCertPath, err)
				}
				pool.AppendCertsFromPEM(serverCert)
				tr := &http.Transport{
					TLSClientConfig: &tls.Config{
						RootCAs: pool,
					},
				}

				client := &http.Client{Transport: tr}
				req, err := http.NewRequest("GET", url, nil)
				if err != nil {
					t.Fatal(err)
				}
				if !tt.anonymous {
					req.Header.Add("Authorization", fmt.Sprintf("Token %s", token))
				}
				r, err := client.Do(req)
				if err != nil {
					t.Fatalf("failed to GET %s from controller-manager: %v", tt.path, err)
				}

				body, err := ioutil.ReadAll(r.Body)
				defer r.Body.Close()
				if got, expected := r.StatusCode, *tt.wantSecureCode; got != expected {
					t.Fatalf("expected http %d at %s of controller-manager, got: %d %q", expected, tt.path, got, string(body))
				}
			}

			if want, got := tt.wantInsecureCode != nil, gotResult.Config.InsecureServing != nil; want != got {
				t.Errorf("InsecureServing enabled: expected=%v got=%v", want, got)
			} else if want {
				url := fmt.Sprintf("http://%s%s", gotResult.Config.InsecureServing.Listener.Addr().String(), tt.path)
				r, err := http.Get(url)
				if err != nil {
					t.Fatalf("failed to GET %s from controller-manager: %v", tt.path, err)
				}
				body, err := ioutil.ReadAll(r.Body)
				defer r.Body.Close()
				if got, expected := r.StatusCode, *tt.wantInsecureCode; got != expected {
					t.Fatalf("expected http %d at %s of controller-manager, got: %d %q", expected, tt.path, got, string(body))
				}
			}
		})
	}
}

func intPtr(x int) *int {
	return &x
}
