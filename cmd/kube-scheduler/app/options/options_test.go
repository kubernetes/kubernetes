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
	"reflect"
	"strings"
	"testing"
	"time"

	apimachineryconfig "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
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
apiVersion: kubescheduler.config.k8s.io/v1alpha1
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

	oldconfigFile := filepath.Join(tmpDir, "scheduler_old.yaml")
	if err := ioutil.WriteFile(oldconfigFile, []byte(fmt.Sprintf(`
apiVersion: componentconfig/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	invalidconfigFile := filepath.Join(tmpDir, "scheduler_invalid.yaml")
	if err := ioutil.WriteFile(invalidconfigFile, []byte(fmt.Sprintf(`
apiVersion: componentconfig/v1alpha2
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
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

	defaultSource := "DefaultProvider"
	defaultBindTimeoutSeconds := int64(600)

	testcases := []struct {
		name             string
		options          *Options
		expectedUsername string
		expectedError    string
		expectedConfig   kubeschedulerconfig.KubeSchedulerConfiguration
	}{
		{
			name: "config file",
			options: &Options{
				ConfigFile: configFile,
				ComponentConfig: func() kubeschedulerconfig.KubeSchedulerConfiguration {
					cfg, err := newDefaultComponentConfig()
					if err != nil {
						t.Fatal(err)
					}
					return *cfg
				}(),
			},
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "0.0.0.0:10251",
				MetricsBindAddress:             "0.0.0.0:10251",
				FailureDomains:                 "kubernetes.io/hostname,failure-domain.beta.kubernetes.io/zone,failure-domain.beta.kubernetes.io/region",
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: apiserverconfig.LeaderElectionConfiguration{
						LeaderElect:   true,
						LeaseDuration: metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline: metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:   metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:  "endpoints",
					},
					LockObjectNamespace: "kube-system",
					LockObjectName:      "kube-scheduler",
				},
				ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
					Kubeconfig:  configKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: 50,
				BindTimeoutSeconds:       &defaultBindTimeoutSeconds,
			},
		},
		{
			name: "config file in componentconfig/v1alpha1",
			options: &Options{
				ConfigFile: oldconfigFile,
				ComponentConfig: func() kubeschedulerconfig.KubeSchedulerConfiguration {
					cfg, err := newDefaultComponentConfig()
					if err != nil {
						t.Fatal(err)
					}
					return *cfg
				}(),
			},
			// TODO: switch this to expect an error in 1.13 when the special-case coercion is removed from loadConfig
			// expectedError: "no kind \"KubeSchedulerConfiguration\" is registered for version \"componentconfig/v1alpha1\"",
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "0.0.0.0:10251",
				MetricsBindAddress:             "0.0.0.0:10251",
				FailureDomains:                 "kubernetes.io/hostname,failure-domain.beta.kubernetes.io/zone,failure-domain.beta.kubernetes.io/region",
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: apiserverconfig.LeaderElectionConfiguration{
						LeaderElect:   true,
						LeaseDuration: metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline: metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:   metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:  "endpoints",
					},
					LockObjectNamespace: "kube-system",
					LockObjectName:      "kube-scheduler",
				},
				ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
					Kubeconfig:  configKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: 50,
				BindTimeoutSeconds:       &defaultBindTimeoutSeconds,
			},
		},

		{
			name:          "invalid config file in componentconfig/v1alpha2",
			options:       &Options{ConfigFile: invalidconfigFile},
			expectedError: "no kind \"KubeSchedulerConfiguration\" is registered for version \"componentconfig/v1alpha2\"",
		},
		{
			name: "kubeconfig flag",
			options: &Options{
				ComponentConfig: func() kubeschedulerconfig.KubeSchedulerConfiguration {
					cfg, _ := newDefaultComponentConfig()
					cfg.ClientConnection.Kubeconfig = flagKubeconfig
					return *cfg
				}(),
			},
			expectedUsername: "flag",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "", // defaults empty when not running from config file
				MetricsBindAddress:             "", // defaults empty when not running from config file
				FailureDomains:                 "kubernetes.io/hostname,failure-domain.beta.kubernetes.io/zone,failure-domain.beta.kubernetes.io/region",
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: apiserverconfig.LeaderElectionConfiguration{
						LeaderElect:   true,
						LeaseDuration: metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline: metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:   metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:  "endpoints",
					},
					LockObjectNamespace: "kube-system",
					LockObjectName:      "kube-scheduler",
				},
				ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
					Kubeconfig:  flagKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: 50,
				BindTimeoutSeconds:       &defaultBindTimeoutSeconds,
			},
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

			if !reflect.DeepEqual(config.ComponentConfig, tc.expectedConfig) {
				t.Errorf("config.diff:\n%s", diff.ObjectReflectDiff(tc.expectedConfig, config.ComponentConfig))
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
