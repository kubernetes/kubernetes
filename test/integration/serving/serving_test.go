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

package serving

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	flagzv1alpha1 "k8s.io/apiserver/pkg/server/flagz/api/v1alpha1"
	"k8s.io/apiserver/pkg/server/statusz/api/v1alpha1"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudprovider "k8s.io/cloud-provider"
	cloudctrlmgrtesting "k8s.io/cloud-provider/app/testing"
	"k8s.io/cloud-provider/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	zpagesfeatures "k8s.io/component-base/zpages/features"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubectrlmgrtesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	kubeschedulertesting "k8s.io/kubernetes/cmd/kube-scheduler/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

type componentTester interface {
	StartTestServer(t *testing.T, ctx context.Context, customFlags []string) (*options.SecureServingOptions, *server.SecureServingInfo, func(), error)
}

type kubeControllerManagerTester struct{}

func (kubeControllerManagerTester) StartTestServer(t *testing.T, ctx context.Context, customFlags []string) (*options.SecureServingOptions, *server.SecureServingInfo, func(), error) {
	// avoid starting any controller loops, we're just testing serving
	customFlags = append([]string{"--controllers="}, customFlags...)
	gotResult, err := kubectrlmgrtesting.StartTestServer(t, ctx, customFlags)
	if err != nil {
		return nil, nil, nil, err
	}
	return gotResult.Options.SecureServing, gotResult.Config.SecureServing, gotResult.TearDownFn, err
}

type cloudControllerManagerTester struct{}

func (cloudControllerManagerTester) StartTestServer(t *testing.T, ctx context.Context, customFlags []string) (*options.SecureServingOptions, *server.SecureServingInfo, func(), error) {
	gotResult, err := cloudctrlmgrtesting.StartTestServer(t, ctx, customFlags)
	if err != nil {
		return nil, nil, nil, err
	}
	return gotResult.Options.SecureServing.SecureServingOptions, gotResult.Config.SecureServing, gotResult.TearDownFn, err
}

type kubeSchedulerTester struct{}

func (kubeSchedulerTester) StartTestServer(t *testing.T, ctx context.Context, customFlags []string) (*options.SecureServingOptions, *server.SecureServingInfo, func(), error) {
	gotResult, err := kubeschedulertesting.StartTestServer(t, ctx, customFlags)
	if err != nil {
		return nil, nil, nil, err
	}
	return gotResult.Options.SecureServing, gotResult.Config.SecureServing, gotResult.TearDownFn, err
}

func TestComponentSecureServingAndAuth(t *testing.T) {
	if !cloudprovider.IsCloudProvider("fake") {
		cloudprovider.RegisterCloudProvider("fake", fakeCloudProviderFactory)
	}

	// Insulate this test from picking up in-cluster config when run inside a pod
	// We can't assume we have permissions to write to /var/run/secrets/... from a unit test to mock in-cluster config for testing
	if len(os.Getenv("KUBERNETES_SERVICE_HOST")) > 0 {
		t.Setenv("KUBERNETES_SERVICE_HOST", "")
	}

	// authenticate to apiserver via bearer token
	token := "flwqkenfjasasdfmwerasd" // Fake token for testing.
	tokenFile, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	tokenFile.WriteString(fmt.Sprintf(`
%s,system:kube-controller-manager,system:kube-controller-manager,""
`, token))
	tokenFile.Close()

	// start apiserver
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--token-auth-file", tokenFile.Name(),
		"--authorization-mode", "RBAC",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// create kubeconfig for the apiserver
	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
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
    user: controller-manager
  name: default-context
current-context: default-context
users:
- name: controller-manager
  user:
    token: %s
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile, token))
	apiserverConfig.Close()

	// create BROKEN kubeconfig for the apiserver

	brokenApiserverConfig, err := os.CreateTemp("", "kubeconfig")
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
    user: controller-manager
  name: default-context
current-context: default-context
users:
- name: controller-manager
  user:
    token: WRONGTOKEN
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile))
	brokenApiserverConfig.Close()

	tests := []struct {
		name       string
		tester     componentTester
		extraFlags []string
	}{
		{"kube-controller-manager", kubeControllerManagerTester{}, nil},
		{"cloud-controller-manager", cloudControllerManagerTester{}, []string{"--cloud-provider=fake", "--webhook-secure-port=0"}},
		{"kube-scheduler", kubeSchedulerTester{}, nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testComponentWithSecureServing(t, tt.tester, apiserverConfig.Name(), brokenApiserverConfig.Name(), token, tt.extraFlags)
		})
	}
}

func testComponentWithSecureServing(t *testing.T, tester componentTester, kubeconfig, brokenKubeconfig, token string, extraFlags []string) {
	tests := []struct {
		name           string
		flags          []string
		path           string
		anonymous      bool // to use the token or not
		wantErr        bool
		wantSecureCode *int
	}{
		{"no-flags", nil, "/healthz", false, true, nil},
		{"/healthz without authn/authz", []string{
			"--kubeconfig", kubeconfig,
			"--leader-elect=false",
		}, "/healthz", true, false, ptr.To(http.StatusOK)},
		{"/metrics without authn/authz", []string{
			"--kubeconfig", kubeconfig,
			"--leader-elect=false",
		}, "/metrics", true, false, ptr.To(http.StatusForbidden)},
		{"authorization skipped for /healthz with authn/authz", []string{
			"--authentication-kubeconfig", kubeconfig,
			"--authorization-kubeconfig", kubeconfig,
			"--kubeconfig", kubeconfig,
			"--leader-elect=false",
		}, "/healthz", false, false, ptr.To(http.StatusOK)},
		{"authorization skipped for /healthz with BROKEN authn/authz", []string{
			"--authentication-skip-lookup", // to survive inaccessible extensions-apiserver-authentication configmap
			"--authentication-kubeconfig", brokenKubeconfig,
			"--authorization-kubeconfig", brokenKubeconfig,
			"--kubeconfig", kubeconfig,
			"--leader-elect=false",
		}, "/healthz", false, false, ptr.To(http.StatusOK)},
		{"not authorized /metrics with BROKEN authn/authz", []string{
			"--authentication-kubeconfig", kubeconfig,
			"--authorization-kubeconfig", brokenKubeconfig,
			"--kubeconfig", kubeconfig,
			"--leader-elect=false",
		}, "/metrics", false, false, ptr.To(http.StatusInternalServerError)},
		{"always-allowed /metrics with BROKEN authn/authz", []string{
			"--authentication-skip-lookup", // to survive inaccessible extensions-apiserver-authentication configmap
			"--authentication-kubeconfig", brokenKubeconfig,
			"--authorization-kubeconfig", brokenKubeconfig,
			"--authorization-always-allow-paths", "/healthz,/metrics",
			"--kubeconfig", kubeconfig,
			"--leader-elect=false",
		}, "/metrics", false, false, ptr.To(http.StatusOK)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			secureOptions, secureInfo, tearDownFn, err := tester.StartTestServer(t, ctx, slices.Concat(tt.flags, extraFlags))
			if tearDownFn != nil {
				defer tearDownFn()
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("StartTestServer() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			if want, got := tt.wantSecureCode != nil, secureInfo != nil; want != got {
				t.Errorf("SecureServing enabled: expected=%v got=%v", want, got)
			} else if want {
				url := fmt.Sprintf("https://%s%s", secureInfo.Listener.Addr().String(), tt.path)
				url = strings.Replace(url, "[::]", "127.0.0.1", -1) // switch to IPv4 because the self-signed cert does not support [::]

				// read self-signed server cert disk
				pool := x509.NewCertPool()
				serverCertPath := path.Join(secureOptions.ServerCert.CertDirectory, secureOptions.ServerCert.PairName+".crt")
				serverCert, err := os.ReadFile(serverCertPath)
				if err != nil {
					t.Fatalf("Failed to read component server cert %q: %v", serverCertPath, err)
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
					t.Fatalf("failed to GET %s from component: %v", tt.path, err)
				}

				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}
				defer r.Body.Close()
				if got, expected := r.StatusCode, *tt.wantSecureCode; got != expected {
					t.Fatalf("expected http %d at %s of component, got: %d %q", expected, tt.path, got, string(body))
				}
			}
		})
	}
}

func fakeCloudProviderFactory(io.Reader) (cloudprovider.Interface, error) {
	return &fake.Cloud{
		DisableRoutes: true, // disable routes for server tests, otherwise --cluster-cidr is required
	}, nil
}

func TestKubeControllerManagerServingZPages(t *testing.T) {
	// authenticate to apiserver via bearer token
	token := "flwqkenfjasasdfmwerasd" // Fake token for testing.
	tokenFile, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tokenFile.WriteString(fmt.Sprintf(`
%s,system:kube-controller-manager,system:kube-controller-manager,""
`, token)); err != nil {
		t.Fatal(err)
	}
	if err = tokenFile.Close(); err != nil {
		t.Fatal(err)
	}

	// start apiserver
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--token-auth-file", tokenFile.Name(),
		"--authorization-mode", "RBAC",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	// create kubeconfig for the apiserver
	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = apiserverConfig.WriteString(fmt.Sprintf(`
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
    user: controller-manager
  name: default-context
current-context: default-context
users:
- name: controller-manager
  user:
    token: %s
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile, token)); err != nil {
		t.Fatal(err)
	}
	if err = apiserverConfig.Close(); err != nil {
		t.Fatal(err)
	}

	statuszWantBodyStr := "kube-controller-manager statusz\nWarning: This endpoint is not meant to be machine parseable"
	statuszWantBodyJSON := &v1alpha1.Statusz{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Statusz",
			APIVersion: "config.k8s.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-controller-manager",
		},
		Paths: []string{"/configz", "/flagz", "/healthz", "/metrics"},
	}

	flagzWantBodyStr := "kube-controller-manager flagz\nWarning: This endpoint is not meant to be machine parseable"
	flagzWantBodyJSON := &flagzv1alpha1.Flagz{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Flagz",
			APIVersion: "config.k8s.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-controller-manager",
		},
	}

	statuszTestCases := []struct {
		name         string
		acceptHeader string
		wantStatus   int
		wantBodySub  string            // for text/plain
		wantJSON     *v1alpha1.Statusz // for structured json
	}{
		{
			name:         "text plain response",
			acceptHeader: "text/plain",
			wantStatus:   http.StatusOK,
			wantBodySub:  statuszWantBodyStr,
		},
		{
			name:         "structured json response",
			acceptHeader: "application/json;v=v1alpha1;g=config.k8s.io;as=Statusz",
			wantStatus:   http.StatusOK,
			wantJSON:     statuszWantBodyJSON,
		},
		{
			name:         "no accept header (defaults to text)",
			acceptHeader: "",
			wantStatus:   http.StatusOK,
			wantBodySub:  statuszWantBodyStr,
		},
		{
			name:         "invalid accept header",
			acceptHeader: "application/xml",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "application/json without params",
			acceptHeader: "application/json",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "application/json with missing as",
			acceptHeader: "application/json;v=v1alpha1;g=config.k8s.io",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "wildcard accept header",
			acceptHeader: "*/*",
			wantStatus:   http.StatusOK,
			wantBodySub:  statuszWantBodyStr,
		},
		{
			name:         "bad json header fall back wildcard",
			acceptHeader: "application/json;v=foo;g=config.k8s.io;as=Statusz,*/*",
			wantStatus:   http.StatusOK,
			wantBodySub:  statuszWantBodyStr,
		},
	}

	flagzTestCases := []struct {
		name         string
		acceptHeader string
		wantStatus   int
		wantBodySub  string               // for text/plain
		wantJSON     *flagzv1alpha1.Flagz // for structured json
	}{
		{
			name:         "text plain response",
			acceptHeader: "text/plain",
			wantStatus:   http.StatusOK,
			wantBodySub:  flagzWantBodyStr,
		},
		{
			name:         "structured json response",
			acceptHeader: "application/json;v=v1alpha1;g=config.k8s.io;as=Flagz",
			wantStatus:   http.StatusOK,
			wantJSON:     flagzWantBodyJSON,
		},
		{
			name:         "no accept header (defaults to text)",
			acceptHeader: "",
			wantStatus:   http.StatusOK,
			wantBodySub:  flagzWantBodyStr,
		},
		{
			name:         "invalid accept header",
			acceptHeader: "application/xml",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "application/json without params",
			acceptHeader: "application/json",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "application/json with missing as",
			acceptHeader: "application/json;v=v1alpha1;g=config.k8s.io",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "wildcard accept header",
			acceptHeader: "*/*",
			wantStatus:   http.StatusOK,
			wantBodySub:  flagzWantBodyStr,
		},
		{
			name:         "bad json header fall back wildcard",
			acceptHeader: "application/json;v=foo;g=config.k8s.io;as=Flagz,*/*",
			wantStatus:   http.StatusOK,
			wantBodySub:  flagzWantBodyStr,
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, zpagesfeatures.ComponentStatusz, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, zpagesfeatures.ComponentFlagz, true)
	_, ctx := ktesting.NewTestContext(t)
	flags := []string{
		"--authentication-skip-lookup",
		"--authentication-kubeconfig", apiserverConfig.Name(),
		"--authorization-kubeconfig", apiserverConfig.Name(),
		"--authorization-always-allow-paths", "/statusz,/flagz",
		"--kubeconfig", apiserverConfig.Name(),
		"--leader-elect=false",
	}
	secureOptions, secureInfo, tearDownFn, err := kubeControllerManagerTester{}.StartTestServer(t, ctx, flags)
	if tearDownFn != nil {
		defer tearDownFn()
	}
	if err != nil {
		t.Fatalf("StartTestServer() error = %v", err)
	}
	if secureInfo == nil {
		t.Fatalf("SecureServing not enabled")
	}
	_, port, err := net.SplitHostPort(secureInfo.Listener.Addr().String())
	if err != nil {
		t.Fatalf("could not get host and port from %s : %v", secureInfo.Listener.Addr().String(), err)
	}
	statuszURL := fmt.Sprintf("https://127.0.0.1:%s/statusz", port)
	flagzURL := fmt.Sprintf("https://127.0.0.1:%s/flagz", port)

	// read self-signed server cert disk
	pool := x509.NewCertPool()
	serverCertPath := path.Join(secureOptions.ServerCert.CertDirectory, secureOptions.ServerCert.PairName+".crt")
	serverCert, err := os.ReadFile(serverCertPath)
	if err != nil {
		t.Fatalf("Failed to read component server cert %q: %v", serverCertPath, err)
	}
	pool.AppendCertsFromPEM(serverCert)
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs: pool,
		},
	}
	client := &http.Client{Transport: tr}

	for _, tc := range statuszTestCases {
		t.Run("statusz_"+tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, statuszURL, nil)
			if err != nil {
				t.Fatal(err)
			}

			req.Header.Set("Accept", tc.acceptHeader)
			req.Header.Add("Authorization", fmt.Sprintf("Token %s", token))
			r, err := client.Do(req)
			if err != nil {
				t.Fatalf("failed to GET /statusz: %v", err)
			}

			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("failed to read response body: %v", err)
			}

			if err = r.Body.Close(); err != nil {
				t.Fatalf("failed to close response body: %v", err)
			}

			if r.StatusCode != tc.wantStatus {
				t.Fatalf("want status %d, got %d", tc.wantStatus, r.StatusCode)
			}

			if tc.wantStatus == http.StatusOK {
				if tc.wantBodySub != "" {
					if !strings.Contains(string(body), tc.wantBodySub) {
						t.Errorf("body missing expected substring: %q\nGot:\n%s", tc.wantBodySub, string(body))
					}
				}
				if tc.wantJSON != nil {
					var got v1alpha1.Statusz
					if err := json.Unmarshal(body, &got); err != nil {
						t.Fatalf("error unmarshalling JSON: %v", err)
					}
					// Only check static fields, since others are dynamic
					if got.TypeMeta != tc.wantJSON.TypeMeta {
						t.Errorf("TypeMeta mismatch: want %+v, got %+v", tc.wantJSON.TypeMeta, got.TypeMeta)
					}
					if got.ObjectMeta.Name != tc.wantJSON.ObjectMeta.Name {
						t.Errorf("ObjectMeta.Name mismatch: want %q, got %q", tc.wantJSON.ObjectMeta.Name, got.ObjectMeta.Name)
					}
					if diff := cmp.Diff(tc.wantJSON.Paths, got.Paths); diff != "" {
						t.Errorf("Paths mismatch (-want,+got):\n%s", diff)
					}
				}
			}
		})
	}

	for _, tc := range flagzTestCases {
		t.Run("flagz_"+tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, flagzURL, nil)
			if err != nil {
				t.Fatal(err)
			}

			req.Header.Set("Accept", tc.acceptHeader)
			req.Header.Add("Authorization", fmt.Sprintf("Token %s", token))
			r, err := client.Do(req)
			if err != nil {
				t.Fatalf("failed to GET /flagz: %v", err)
			}

			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("failed to read response body: %v", err)
			}

			if err = r.Body.Close(); err != nil {
				t.Fatalf("failed to close response body: %v", err)
			}

			if r.StatusCode != tc.wantStatus {
				t.Fatalf("want status %d, got %d", tc.wantStatus, r.StatusCode)
			}

			if tc.wantStatus == http.StatusOK {
				if tc.wantBodySub != "" {
					if !strings.Contains(string(body), tc.wantBodySub) {
						t.Errorf("body missing expected substring: %q\nGot:\n%s", tc.wantBodySub, string(body))
					}
				}
				if tc.wantJSON != nil {
					var got flagzv1alpha1.Flagz
					if err := json.Unmarshal(body, &got); err != nil {
						t.Fatalf("error unmarshalling JSON: %v", err)
					}
					// Only check static fields, since others are dynamic
					if got.TypeMeta != tc.wantJSON.TypeMeta {
						t.Errorf("TypeMeta mismatch: want %+v, got %+v", tc.wantJSON.TypeMeta, got.TypeMeta)
					}
					if got.ObjectMeta.Name != tc.wantJSON.ObjectMeta.Name {
						t.Errorf("ObjectMeta.Name mismatch: want %q, got %q", tc.wantJSON.ObjectMeta.Name, got.ObjectMeta.Name)
					}
				}
			}
		})
	}
}
