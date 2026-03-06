/*
Copyright 2024 The Kubernetes Authors.

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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"regexp"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserverfeat "k8s.io/apiserver/pkg/features"
	flagzv1alpha1 "k8s.io/apiserver/pkg/server/flagz/api/v1alpha1"
	flagzv1beta1 "k8s.io/apiserver/pkg/server/flagz/api/v1beta1"
	flagztesting "k8s.io/apiserver/pkg/server/flagz/testing"
	statuszv1alpha1 "k8s.io/apiserver/pkg/server/statusz/api/v1alpha1"
	statuszv1beta1 "k8s.io/apiserver/pkg/server/statusz/api/v1beta1"
	statusztesting "k8s.io/apiserver/pkg/server/statusz/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/zpages/features"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubeschedulertesting "k8s.io/kubernetes/cmd/kube-scheduler/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestEndpointHandlers(t *testing.T) {
	server, configStr, _, err := startTestAPIServer(t)
	if err != nil {
		t.Fatalf("Failed to start kube-apiserver server: %v", err)
	}
	defer server.TearDownFn()

	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create config file: %v", err)
	}
	defer func() {
		_ = os.Remove(apiserverConfig.Name())
	}()
	if _, err = apiserverConfig.WriteString(configStr); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	brokenConfigStr := strings.ReplaceAll(configStr, "127.0.0.1", "127.0.0.2")
	brokenConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create config file: %v", err)
	}
	if _, err := brokenConfig.WriteString(brokenConfigStr); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}
	defer func() {
		_ = os.Remove(brokenConfig.Name())
	}()

	tests := []struct {
		name                 string
		path                 string
		useBrokenConfig      bool
		requestHeader        map[string]string
		wantResponseCode     int
		wantResponseBodyRegx string
	}{
		{
			name:             "/healthz",
			path:             "/healthz",
			useBrokenConfig:  false,
			wantResponseCode: http.StatusOK,
		},
		{
			name:             "/livez",
			path:             "/livez",
			useBrokenConfig:  false,
			wantResponseCode: http.StatusOK,
		},
		{
			name:             "/livez with ping check",
			path:             "/livez/ping",
			useBrokenConfig:  false,
			wantResponseCode: http.StatusOK,
		},
		{
			name:             "/readyz",
			path:             "/readyz",
			useBrokenConfig:  false,
			wantResponseCode: http.StatusOK,
		},
		{
			name:             "/readyz with sched-handler-sync",
			path:             "/readyz/sched-handler-sync",
			useBrokenConfig:  false,
			wantResponseCode: http.StatusOK,
		},
		{
			name:             "/readyz with shutdown",
			path:             "/readyz/shutdown",
			useBrokenConfig:  false,
			wantResponseCode: http.StatusOK,
		},
		{
			name:             "/readyz with broken apiserver",
			path:             "/readyz",
			useBrokenConfig:  true,
			wantResponseCode: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt := tt
			_, ctx := ktesting.NewTestContext(t)

			configFile := apiserverConfig.Name()
			if tt.useBrokenConfig {
				configFile = brokenConfig.Name()
			}
			result, err := kubeschedulertesting.StartTestServer(
				t, ctx,
				[]string{"--kubeconfig", configFile, "--leader-elect=false", "--authorization-always-allow-paths", tt.path})

			if err != nil {
				t.Fatalf("Failed to start kube-scheduler server: %v", err)
			}
			if result.TearDownFn != nil {
				defer result.TearDownFn()
			}

			client, base, err := clientAndURLFromTestServer(result)
			if err != nil {
				t.Fatalf("Failed to get client from test server: %v", err)
			}
			req, err := http.NewRequest(http.MethodGet, base+tt.path, nil)
			if err != nil {
				t.Fatalf("failed to request: %v", err)
			}
			if tt.requestHeader != nil {
				for k, v := range tt.requestHeader {
					req.Header.Set(k, v)
				}
			}

			err = retry.OnError(
				retry.DefaultBackoff,
				func(err error) bool {
					// the endpoint /readyz/sched-handler-sync needs some time to finish
					// the initialization, so we need to retry
					return !tt.useBrokenConfig &&
						// The message differ depending on endpoint
						(tt.path == "/readyz" && strings.Contains(err.Error(), "sched-handler-sync failed: reason withheld") ||
							tt.path == "/readyz/sched-handler-sync" && strings.Contains(err.Error(), "handlers are not fully synchronized"))
				},
				func() error {
					r, err := client.Do(req)
					if err != nil {
						return fmt.Errorf("failed to GET %s from component: %w", tt.path, err)
					}

					body, err := io.ReadAll(r.Body)
					if err != nil {
						return fmt.Errorf("failed to read response body: %w", err)
					}
					if err = r.Body.Close(); err != nil {
						return fmt.Errorf("failed to close response body: %w", err)
					}
					if got, expected := r.StatusCode, tt.wantResponseCode; got != expected {
						return fmt.Errorf("expected http %d at %s of component, got: %d %q", expected, tt.path, got, string(body))
					}
					if tt.wantResponseBodyRegx != "" {
						matched, err := regexp.MatchString(tt.wantResponseBodyRegx, string(body))
						if err != nil {
							return fmt.Errorf("failed to compile regex: %w", err)
						}
						if !matched {
							return fmt.Errorf("response body does not match regex.\nExpected:\n%s\n\nGot:\n%s", tt.wantResponseBodyRegx, string(body))
						}
					}
					return nil
				},
			)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSchedulerZPages(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.ComponentStatusz:           true,
		features.ComponentFlagz:             true,
		apiserverfeat.CBORServingAndStorage: true,
	})

	server, configStr, _, err := startTestAPIServer(t)
	if err != nil {
		t.Fatalf("Failed to start kube-apiserver server: %v", err)
	}
	defer server.TearDownFn()

	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create config file: %v", err)
	}
	defer func() {
		_ = os.Remove(apiserverConfig.Name())
	}()
	if _, err = apiserverConfig.WriteString(configStr); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	_, ctx := ktesting.NewTestContext(t)
	result, err := kubeschedulertesting.StartTestServer(
		t, ctx,
		[]string{"--kubeconfig", apiserverConfig.Name(), "--leader-elect=false", "--authorization-always-allow-paths=/statusz,/flagz"},
	)
	if err != nil {
		t.Fatalf("Failed to start kube-scheduler server: %v", err)
	}
	if result.TearDownFn != nil {
		defer result.TearDownFn()
	}

	client, base, err := clientAndURLFromTestServer(result)
	if err != nil {
		t.Fatalf("Failed to get client from test server: %v", err)
	}

	statuszWantBodyStr := "kube-scheduler statusz\nWarning: This endpoint is not meant to be machine parseable"
	statuszWantBodyStructuredAlpha := &statuszv1alpha1.Statusz{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Statusz",
			APIVersion: "config.k8s.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-scheduler",
		},
		Paths: []string{"/configz", "/flagz", "/healthz", "/livez", "/metrics", "/readyz"},
	}
	statuszWantBodyStructuredBeta := &statuszv1beta1.Statusz{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Statusz",
			APIVersion: "config.k8s.io/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-scheduler",
		},
		Paths: []string{"/configz", "/flagz", "/healthz", "/livez", "/metrics", "/readyz"},
	}

	flagzWantBodyStr := "kube-scheduler flagz\nWarning: This endpoint is not meant to be machine parseable"
	flagzWantBodyStructuredAlpha := &flagzv1alpha1.Flagz{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Flagz",
			APIVersion: "config.k8s.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-scheduler",
		},
		Flags: map[string]string{
			"leader-elect-resource-name": "kube-scheduler",
		},
	}
	flagzWantBodyStructuredBeta := &flagzv1beta1.Flagz{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Flagz",
			APIVersion: "config.k8s.io/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "kube-scheduler",
		},
		Flags: map[string]string{
			"leader-elect-resource-name": "kube-scheduler",
		},
	}

	statuszTestCases := []struct {
		name                  string
		acceptHeader          string
		wantStatus            int
		wantBodyText          string
		wantBodyStructured    interface{}
		wantDeprecationHeader bool
	}{
		{
			name:         "text plain response",
			acceptHeader: "text/plain",
			wantStatus:   http.StatusOK,
			wantBodyText: statuszWantBodyStr,
		},
		{
			name:               "structured json response",
			acceptHeader:       "application/json;v=v1beta1;g=config.k8s.io;as=Statusz",
			wantStatus:         http.StatusOK,
			wantBodyStructured: statuszWantBodyStructuredBeta,
		},
		{
			name:         "no accept header (defaults to text)",
			acceptHeader: "",
			wantStatus:   http.StatusOK,
			wantBodyText: statuszWantBodyStr,
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
			acceptHeader: "application/json;v=v1beta1;g=config.k8s.io",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "wildcard accept header",
			acceptHeader: "*/*",
			wantStatus:   http.StatusOK,
			wantBodyText: statuszWantBodyStr,
		},
		{
			name:         "bad json header fall back wildcard",
			acceptHeader: "application/json;v=foo;g=config.k8s.io;as=Statusz,*/*",
			wantStatus:   http.StatusOK,
			wantBodyText: statuszWantBodyStr,
		},
		{
			name:               "structured yaml response",
			acceptHeader:       "application/yaml;v=v1beta1;g=config.k8s.io;as=Statusz",
			wantStatus:         http.StatusOK,
			wantBodyStructured: statuszWantBodyStructuredBeta,
		},
		{
			name:               "structured cbor response",
			acceptHeader:       "application/cbor;v=v1beta1;g=config.k8s.io;as=Statusz",
			wantStatus:         http.StatusOK,
			wantBodyStructured: statuszWantBodyStructuredBeta,
		},
		{
			name:                  "alpha specified before beta, should show warning",
			acceptHeader:          "application/json;g=config.k8s.io;v=v1alpha1;as=Statusz,application/json;g=config.k8s.io;v=v1beta1;as=Statusz",
			wantStatus:            http.StatusOK,
			wantBodyStructured:    statuszWantBodyStructuredAlpha,
			wantDeprecationHeader: true,
		},
		{
			name:                  "beta specified before alpha, no warning",
			acceptHeader:          "application/json;g=config.k8s.io;v=v1beta1;as=Statusz,application/json;g=config.k8s.io;v=v1alpha1;as=Statusz",
			wantStatus:            http.StatusOK,
			wantBodyStructured:    statuszWantBodyStructuredBeta,
			wantDeprecationHeader: false,
		},
	}

	flagzTestCases := []struct {
		name                  string
		acceptHeader          string
		wantStatus            int
		wantBodyText          string
		wantBodyStructured    interface{}
		wantDeprecationHeader bool
	}{
		{
			name:         "text plain response",
			acceptHeader: "text/plain",
			wantStatus:   http.StatusOK,
			wantBodyText: flagzWantBodyStr,
		},
		{
			name:               "structured json response",
			acceptHeader:       "application/json;v=v1beta1;g=config.k8s.io;as=Flagz",
			wantStatus:         http.StatusOK,
			wantBodyStructured: flagzWantBodyStructuredBeta,
		},
		{
			name:         "no accept header (defaults to text)",
			acceptHeader: "",
			wantStatus:   http.StatusOK,
			wantBodyText: flagzWantBodyStr,
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
			acceptHeader: "application/json;v=v1beta1;g=config.k8s.io",
			wantStatus:   http.StatusNotAcceptable,
		},
		{
			name:         "wildcard accept header",
			acceptHeader: "*/*",
			wantStatus:   http.StatusOK,
			wantBodyText: flagzWantBodyStr,
		},
		{
			name:         "bad json header fall back wildcard",
			acceptHeader: "application/json;v=foo;g=config.k8s.io;as=Flagz,*/*",
			wantStatus:   http.StatusOK,
			wantBodyText: flagzWantBodyStr,
		},
		{
			name:               "structured cbor response",
			acceptHeader:       "application/cbor;v=v1beta1;g=config.k8s.io;as=Flagz",
			wantStatus:         http.StatusOK,
			wantBodyStructured: flagzWantBodyStructuredBeta,
		},
		{
			name:               "structured yaml response",
			acceptHeader:       "application/yaml;v=v1beta1;g=config.k8s.io;as=Flagz",
			wantStatus:         http.StatusOK,
			wantBodyStructured: flagzWantBodyStructuredBeta,
		},
		{
			name:                  "alpha specified before beta, should show warning",
			acceptHeader:          "application/json;v=v1alpha1;g=config.k8s.io;as=Flagz,application/json;v=v1beta1;g=config.k8s.io;as=Flagz",
			wantStatus:            http.StatusOK,
			wantBodyStructured:    flagzWantBodyStructuredAlpha,
			wantDeprecationHeader: true,
		},
		{
			name:                  "beta specified before alpha, no warning",
			acceptHeader:          "application/json;v=v1beta1;g=config.k8s.io;as=Flagz,application/json;v=v1alpha1;g=config.k8s.io;as=Flagz",
			wantStatus:            http.StatusOK,
			wantBodyStructured:    flagzWantBodyStructuredBeta,
			wantDeprecationHeader: false,
		},
	}

	for _, tc := range statuszTestCases {
		t.Run("statusz_"+tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, base+"/statusz", nil)
			if err != nil {
				t.Fatalf("failed to request: %v", err)
			}

			req.Header.Set("Accept", tc.acceptHeader)
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
				if tc.wantBodyText != "" {
					if !strings.Contains(string(body), tc.wantBodyText) {
						t.Errorf("body missing expected substring: %q\nGot:\n%s", tc.wantBodyText, string(body))
					}
				}
				if tc.wantBodyStructured != nil {
					warnings := append([]string{}, r.Header.Values("Warning")...)
					statusztesting.VerifyStructuredResponse(t, tc.acceptHeader, body, warnings, tc.wantBodyStructured, tc.wantDeprecationHeader)
				}
			}
		})
	}

	for _, tc := range flagzTestCases {
		t.Run("flagz_"+tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, base+"/flagz", nil)
			if err != nil {
				t.Fatalf("failed to request: %v", err)
			}

			req.Header.Set("Accept", tc.acceptHeader)
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
				if tc.wantBodyText != "" {
					if !strings.Contains(string(body), tc.wantBodyText) {
						t.Errorf("body missing expected substring: %q\nGot:\n%s", tc.wantBodyText, string(body))
					}
				}
				if tc.wantBodyStructured != nil {
					warnings := append([]string{}, r.Header.Values("Warning")...)
					flagztesting.VerifyStructuredResponse(t, tc.acceptHeader, body, warnings, tc.wantBodyStructured, tc.wantDeprecationHeader)
				}
			}
		})
	}
}

// TODO: Make this a util function once there is a unified way to start a testing apiserver so that we can reuse it.
func startTestAPIServer(t *testing.T) (server *kubeapiservertesting.TestServer, apiserverConfig, token string, err error) {
	// Insulate this test from picking up in-cluster config when run inside a pod
	// We can't assume we have permissions to write to /var/run/secrets/... from a unit test to mock in-cluster config for testing
	originalHost := os.Getenv("KUBERNETES_SERVICE_HOST")
	if len(originalHost) > 0 {
		if err = os.Setenv("KUBERNETES_SERVICE_HOST", ""); err != nil {
			return
		}
		defer func() {
			err = os.Setenv("KUBERNETES_SERVICE_HOST", originalHost)
		}()
	}

	// authenticate to apiserver via bearer token
	token = "flwqkenfjasasdfmwerasd" // Fake token for testing.
	var tokenFile *os.File
	tokenFile, err = os.CreateTemp("", "kubeconfig")
	if err != nil {
		return
	}
	if _, err = tokenFile.WriteString(fmt.Sprintf(`%s,system:kube-scheduler,system:kube-scheduler,""`, token)); err != nil {
		return
	}
	if err = tokenFile.Close(); err != nil {
		return
	}

	// start apiserver
	server = kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--token-auth-file", tokenFile.Name(),
		"--authorization-mode", "AlwaysAllow",
	}, framework.SharedEtcd())

	apiserverConfig = fmt.Sprintf(`
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
    user: kube-scheduler
  name: default-context
current-context: default-context
users:
- name: kube-scheduler
  user:
    token: %s
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile, token)
	return server, apiserverConfig, token, nil
}

func clientAndURLFromTestServer(s kubeschedulertesting.TestServer) (*http.Client, string, error) {
	secureInfo := s.Config.SecureServing
	secureOptions := s.Options.SecureServing
	url := fmt.Sprintf("https://%s", secureInfo.Listener.Addr().String())
	url = strings.ReplaceAll(url, "[::]", "127.0.0.1") // switch to IPv4 because the self-signed cert does not support [::]

	// read self-signed server cert disk
	pool := x509.NewCertPool()
	serverCertPath := path.Join(secureOptions.ServerCert.CertDirectory, secureOptions.ServerCert.PairName+".crt")
	serverCert, err := os.ReadFile(serverCertPath)
	if err != nil {
		return nil, "", fmt.Errorf("Failed to read component server cert %q: %w", serverCertPath, err)
	}
	pool.AppendCertsFromPEM(serverCert)
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs: pool,
		},
	}
	client := &http.Client{Transport: tr}
	return client, url, nil
}
