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
	"testing"

	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"path"
	"strings"

	"k8s.io/kubernetes/test/integration/framework"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubeschedulertesting "k8s.io/kubernetes/cmd/kube-scheduler/app/testing"
)

func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run)
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
