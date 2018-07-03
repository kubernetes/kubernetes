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

package options

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestSchedulerOptions(t *testing.T) {
	// temp dir
	tmpDir, err := ioutil.TempDir("", "scheduler-options")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// record the username requests were made with
	username := ""
	// https server
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		username, _, _ = req.BasicAuth()
		if username == "" {
			username = "none, tls"
		}
		w.WriteHeader(200)
		w.Write([]byte(`ok`))
	}))
	defer server.Close()
	// http server
	insecureserver := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		username, _, _ = req.BasicAuth()
		if username == "" {
			username = "none, http"
		}
		w.WriteHeader(200)
		w.Write([]byte(`ok`))
	}))
	defer insecureserver.Close()

	// config file and kubeconfig
	configFile := filepath.Join(tmpDir, "scheduler.yaml")
	configKubeconfig := filepath.Join(tmpDir, "config.kubeconfig")
	if err := ioutil.WriteFile(configFile, []byte(fmt.Sprintf(`
apiVersion: componentconfig/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(configKubeconfig, []byte(fmt.Sprintf(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
    server: %s
  name: default
contexts:
- context:
    cluster: default
    user: default
  name: default
current-context: default
users:
- name: default
  user:
    username: config
`, server.URL)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	// flag-specified kubeconfig
	flagKubeconfig := filepath.Join(tmpDir, "flag.kubeconfig")
	if err := ioutil.WriteFile(flagKubeconfig, []byte(fmt.Sprintf(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
    server: %s
  name: default
contexts:
- context:
    cluster: default
    user: default
  name: default
current-context: default
users:
- name: default
  user:
    username: flag
`, server.URL)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	// Insulate this test from picking up in-cluster config when run inside a pod
	// We can't assume we have permissions to write to /var/run/secrets/... from a unit test to mock in-cluster config for testing
	originalHost := os.Getenv("KUBERNETES_SERVICE_HOST")
	if len(originalHost) > 0 {
		os.Setenv("KUBERNETES_SERVICE_HOST", "")
		defer os.Setenv("KUBERNETES_SERVICE_HOST", originalHost)
	}

	testcases := []struct {
		name             string
		options          *Options
		expectedUsername string
		expectedError    string
	}{
		{
			name:             "config file",
			options:          &Options{ConfigFile: configFile},
			expectedUsername: "config",
		},
		{
			name: "kubeconfig flag",
			options: &Options{
				ComponentConfig: componentconfig.KubeSchedulerConfiguration{
					ClientConnection: componentconfig.ClientConnectionConfiguration{
						KubeConfigFile: flagKubeconfig}}},
			expectedUsername: "flag",
		},
		{
			name:             "overridden master",
			options:          &Options{Master: insecureserver.URL},
			expectedUsername: "none, http",
		},
		{
			name:          "no config",
			options:       &Options{},
			expectedError: "no configuration has been provided",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			// create the config
			config, err := tc.options.Config()

			// handle errors
			if err != nil {
				if tc.expectedError == "" {
					t.Error(err)
				} else if !strings.Contains(err.Error(), tc.expectedError) {
					t.Errorf("expected %q, got %q", tc.expectedError, err.Error())
				}
				return
			}

			// ensure we have a client
			if config.Client == nil {
				t.Error("unexpected nil client")
				return
			}

			// test the client talks to the endpoint we expect with the credentials we expect
			username = ""
			_, err = config.Client.Discovery().RESTClient().Get().AbsPath("/").DoRaw()
			if err != nil {
				t.Error(err)
				return
			}
			if username != tc.expectedUsername {
				t.Errorf("expected server call with user %s, got %s", tc.expectedUsername, username)
			}
		})
	}
}
