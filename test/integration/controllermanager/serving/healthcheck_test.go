/*
Copyright The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubecontrollermanageresting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestHealthEndpoints(t *testing.T) {
	server, configStr, _, err := startTestAPIServer(t)
	if err != nil {
		t.Fatalf("Failed to start kube-apiserver server: %v", err)
	}
	defer server.TearDownFn()

	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create kubeconfig file: %v", err)
	}
	defer func() {
		_ = os.Remove(apiserverConfig.Name())
	}()
	if _, err = apiserverConfig.WriteString(configStr); err != nil {
		t.Fatalf("Failed to write kubeconfig file: %v", err)
	}

	brokenConfigStr := strings.ReplaceAll(configStr, "127.0.0.1", "127.0.0.2")
	brokenConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create kubeconfig file: %v", err)
	}
	if _, err := brokenConfig.WriteString(brokenConfigStr); err != nil {
		t.Fatalf("Failed to write kubeconfig file: %v", err)
	}
	defer func() {
		_ = os.Remove(brokenConfig.Name())
	}()

	tests := []struct {
		name             string
		path             string
		useBrokenConfig  bool
		wantResponseCode int
	}{
		{
			"/healthz",
			"/healthz",
			false,
			http.StatusOK,
		},
		{
			"/livez",
			"/livez",
			false,
			http.StatusOK,
		},
		{
			"/livez with ping check",
			"/livez/ping",
			false,
			http.StatusOK,
		},
		{
			"/readyz",
			"/readyz",
			false,
			http.StatusOK,
		},
		{
			"/readyz with sched-handler-sync",
			"/readyz/kcm-handler-sync",
			false,
			http.StatusOK,
		},
		{
			"/readyz with shutdown",
			"/readyz/shutdown",
			false,
			http.StatusOK,
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
			result, err := kubecontrollermanageresting.StartTestServer(
				t,
				ctx,
				[]string{"--controllers=", "--kubeconfig", configFile, "--leader-elect=false", "--authorization-always-allow-paths", tt.path},
			)

			if err != nil {
				t.Fatalf("Failed to start kube-controller-manager server: %v", err)
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
				t.Fatalf("Failed to request: %v", err)
			}
			r, err := client.Do(req)
			if err != nil {
				t.Fatalf("Failed to GET %s from component: %v", tt.path, err)
			}

			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("Failed to read response body: %v", err)
			}
			if err = r.Body.Close(); err != nil {
				t.Fatalf("Failed to close response body: %v", err)
			}
			if got, expected := r.StatusCode, tt.wantResponseCode; got != expected {
				t.Fatalf("Expected http %d at %s of component, got: %d %q", expected, tt.path, got, string(body))
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

	// Authenticate to apiserver via bearer token
	token = "flwqkenfjasasdfmwerasd" // Fake token for testing.
	var tokenFile *os.File
	tokenFile, err = os.CreateTemp("", "kubeconfig")
	if err != nil {
		return
	}
	if _, err = fmt.Fprintf(tokenFile, `%s,system:kube-controller-manager,system:kube-controller-manager,""`, token); err != nil {
		return
	}
	if err = tokenFile.Close(); err != nil {
		return
	}

	// Start apiserver
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
    user: kube-controller-manager
  name: default-context
current-context: default-context
users:
- name: kube-controller-manager
  user:
    token: %s
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile, token)

	return server, apiserverConfig, token, nil
}

func clientAndURLFromTestServer(s kubecontrollermanageresting.TestServer) (*http.Client, string, error) {
	secureInfo := s.Config.SecureServing
	secureOptions := s.Options.SecureServing
	url := fmt.Sprintf("https://%s", secureInfo.Listener.Addr().String())
	url = strings.ReplaceAll(url, "[::]", "127.0.0.1") // Switch to IPv4 because the self-signed cert does not support [::]

	// Read the self-signed server cert from disk
	pool := x509.NewCertPool()
	serverCertPath := path.Join(secureOptions.ServerCert.CertDirectory, secureOptions.ServerCert.PairName+".crt")
	serverCert, err := os.ReadFile(serverCertPath)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read the server cert %q: %w", serverCertPath, err)
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
