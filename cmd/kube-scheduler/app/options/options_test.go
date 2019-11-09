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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	componentbaseconfig "k8s.io/component-base/config"
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

	invalidconfigFile := filepath.Join(tmpDir, "scheduler_invalid_wrong_api_version.yaml")
	if err := ioutil.WriteFile(invalidconfigFile, []byte(fmt.Sprintf(`
apiVersion: componentconfig/v1alpha2
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	unknownFieldConfig := filepath.Join(tmpDir, "scheduler_invalid_unknown_field.yaml")
	if err := ioutil.WriteFile(unknownFieldConfig, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true
foo: bar`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	duplicateFieldConfig := filepath.Join(tmpDir, "scheduler_invalid_duplicate_fields.yaml")
	if err := ioutil.WriteFile(duplicateFieldConfig, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true
  leaderElect: false`, configKubeconfig)), os.FileMode(0600)); err != nil {
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

	// plugin config
	pluginconfigFile := filepath.Join(tmpDir, "plugin.yaml")
	if err := ioutil.WriteFile(pluginconfigFile, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
plugins:
  reserve:
    enabled:
    - name: foo
    - name: bar
    disabled:
    - name: baz
  preBind:
    enabled:
    - name: foo
    disabled:
    - name: baz
pluginConfig:
- name: foo
`, configKubeconfig)), os.FileMode(0600)); err != nil {
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
	defaultPodInitialBackoffSeconds := int64(1)
	defaultPodMaxBackoffSeconds := int64(10)
	defaultPercentageOfNodesToScore := int32(0)

	testcases := []struct {
		name             string
		options          *Options
		expectedUsername string
		expectedError    string
		expectedConfig   kubeschedulerconfig.KubeSchedulerConfiguration
		checkErrFn       func(err error) bool
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
				SecureServing: (&apiserveroptions.SecureServingOptions{
					ServerCert: apiserveroptions.GeneratableKeyCert{
						CertDirectory: "/a/b/c",
						PairName:      "kube-scheduler",
					},
					HTTP2MaxStreamsPerConnection: 47,
				}).WithLoopback(),
				Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
					CacheTTL:   10 * time.Second,
					ClientCert: apiserveroptions.ClientCertAuthenticationOptions{},
					RequestHeader: apiserveroptions.RequestHeaderAuthenticationOptions{
						UsernameHeaders:     []string{"x-remote-user"},
						GroupHeaders:        []string{"x-remote-group"},
						ExtraHeaderPrefixes: []string{"x-remote-extra-"},
					},
					RemoteKubeConfigFileOptional: true,
				},
				Authorization: &apiserveroptions.DelegatingAuthorizationOptions{
					AllowCacheTTL:                10 * time.Second,
					DenyCacheTTL:                 10 * time.Second,
					RemoteKubeConfigFileOptional: true,
					AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or /healthz/*
				},
			},
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "0.0.0.0:10251",
				MetricsBindAddress:             "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       true,
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "kube-system",
						ResourceName:      "kube-scheduler",
					},
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  configKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: defaultPercentageOfNodesToScore,
				BindTimeoutSeconds:       defaultBindTimeoutSeconds,
				PodInitialBackoffSeconds: defaultPodInitialBackoffSeconds,
				PodMaxBackoffSeconds:     defaultPodMaxBackoffSeconds,
				Plugins:                  nil,
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
			expectedError: "no kind \"KubeSchedulerConfiguration\" is registered for version \"componentconfig/v1alpha1\"",
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
				SecureServing: (&apiserveroptions.SecureServingOptions{
					ServerCert: apiserveroptions.GeneratableKeyCert{
						CertDirectory: "/a/b/c",
						PairName:      "kube-scheduler",
					},
					HTTP2MaxStreamsPerConnection: 47,
				}).WithLoopback(),
				Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
					CacheTTL:   10 * time.Second,
					ClientCert: apiserveroptions.ClientCertAuthenticationOptions{},
					RequestHeader: apiserveroptions.RequestHeaderAuthenticationOptions{
						UsernameHeaders:     []string{"x-remote-user"},
						GroupHeaders:        []string{"x-remote-group"},
						ExtraHeaderPrefixes: []string{"x-remote-extra-"},
					},
					RemoteKubeConfigFileOptional: true,
				},
				Authorization: &apiserveroptions.DelegatingAuthorizationOptions{
					AllowCacheTTL:                10 * time.Second,
					DenyCacheTTL:                 10 * time.Second,
					RemoteKubeConfigFileOptional: true,
					AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or /healthz/*
				},
			},
			expectedUsername: "flag",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "", // defaults empty when not running from config file
				MetricsBindAddress:             "", // defaults empty when not running from config file
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       true,
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "kube-system",
						ResourceName:      "kube-scheduler",
					},
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  flagKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: defaultPercentageOfNodesToScore,
				BindTimeoutSeconds:       defaultBindTimeoutSeconds,
				PodInitialBackoffSeconds: defaultPodInitialBackoffSeconds,
				PodMaxBackoffSeconds:     defaultPodMaxBackoffSeconds,
			},
		},
		{
			name: "overridden master",
			options: &Options{
				ComponentConfig: func() kubeschedulerconfig.KubeSchedulerConfiguration {
					cfg, _ := newDefaultComponentConfig()
					cfg.ClientConnection.Kubeconfig = flagKubeconfig
					return *cfg
				}(),
				Master: insecureserver.URL,
				SecureServing: (&apiserveroptions.SecureServingOptions{
					ServerCert: apiserveroptions.GeneratableKeyCert{
						CertDirectory: "/a/b/c",
						PairName:      "kube-scheduler",
					},
					HTTP2MaxStreamsPerConnection: 47,
				}).WithLoopback(),
				Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
					CacheTTL: 10 * time.Second,
					RequestHeader: apiserveroptions.RequestHeaderAuthenticationOptions{
						UsernameHeaders:     []string{"x-remote-user"},
						GroupHeaders:        []string{"x-remote-group"},
						ExtraHeaderPrefixes: []string{"x-remote-extra-"},
					},
					RemoteKubeConfigFileOptional: true,
				},
				Authorization: &apiserveroptions.DelegatingAuthorizationOptions{
					AllowCacheTTL:                10 * time.Second,
					DenyCacheTTL:                 10 * time.Second,
					RemoteKubeConfigFileOptional: true,
					AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or /healthz/*
				},
			},
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "", // defaults empty when not running from config file
				MetricsBindAddress:             "", // defaults empty when not running from config file
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       true,
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "kube-system",
						ResourceName:      "kube-scheduler",
					},
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  flagKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: defaultPercentageOfNodesToScore,
				BindTimeoutSeconds:       defaultBindTimeoutSeconds,
				PodInitialBackoffSeconds: defaultPodInitialBackoffSeconds,
				PodMaxBackoffSeconds:     defaultPodMaxBackoffSeconds,
			},
			expectedUsername: "none, http",
		},
		{
			name: "plugin config",
			options: &Options{
				ConfigFile: pluginconfigFile,
			},
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "0.0.0.0:10251",
				MetricsBindAddress:             "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       true,
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "kube-system",
						ResourceName:      "kube-scheduler",
					},
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  configKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: defaultPercentageOfNodesToScore,
				BindTimeoutSeconds:       defaultBindTimeoutSeconds,
				PodInitialBackoffSeconds: defaultPodInitialBackoffSeconds,
				PodMaxBackoffSeconds:     defaultPodMaxBackoffSeconds,
				Plugins: &kubeschedulerconfig.Plugins{
					Reserve: &kubeschedulerconfig.PluginSet{
						Enabled: []kubeschedulerconfig.Plugin{
							{
								Name: "foo",
							},
							{
								Name: "bar",
							},
						},
						Disabled: []kubeschedulerconfig.Plugin{
							{
								Name: "baz",
							},
						},
					},
					PreBind: &kubeschedulerconfig.PluginSet{
						Enabled: []kubeschedulerconfig.Plugin{
							{
								Name: "foo",
							},
						},
						Disabled: []kubeschedulerconfig.Plugin{
							{
								Name: "baz",
							},
						},
					},
				},
				PluginConfig: []kubeschedulerconfig.PluginConfig{
					{
						Name: "foo",
						Args: runtime.Unknown{},
					},
				},
			},
		},
		{
			name:          "no config",
			options:       &Options{},
			expectedError: "no configuration has been provided",
		},
		{
			name: "unknown field",
			options: &Options{
				ConfigFile: unknownFieldConfig,
			},
			// TODO (obitech): Remove this comment and add a new test for v1alpha2, when it's available, as this should fail then.
			// expectedError: "found unknown field: foo",
			// checkErrFn:    runtime.IsStrictDecodingError,
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "0.0.0.0:10251",
				MetricsBindAddress:             "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       true,
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "kube-system",
						ResourceName:      "kube-scheduler",
					},
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  configKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: defaultPercentageOfNodesToScore,
				BindTimeoutSeconds:       defaultBindTimeoutSeconds,
				PodInitialBackoffSeconds: defaultPodInitialBackoffSeconds,
				PodMaxBackoffSeconds:     defaultPodMaxBackoffSeconds,
				Plugins:                  nil,
			},
		},
		{
			name: "duplicate fields",
			options: &Options{
				ConfigFile: duplicateFieldConfig,
			},
			// TODO (obitech): Remove this comment and add a new test for v1alpha2, when it's available, as this should fail then.
			// expectedError: `key "leaderElect" already set`,
			// checkErrFn:    runtime.IsStrictDecodingError,
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				SchedulerName:                  "default-scheduler",
				AlgorithmSource:                kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HardPodAffinitySymmetricWeight: 1,
				HealthzBindAddress:             "0.0.0.0:10251",
				MetricsBindAddress:             "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: kubeschedulerconfig.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       false,
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "kube-system",
						ResourceName:      "kube-scheduler",
					},
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  configKubeconfig,
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: defaultPercentageOfNodesToScore,
				BindTimeoutSeconds:       defaultBindTimeoutSeconds,
				PodInitialBackoffSeconds: defaultPodInitialBackoffSeconds,
				PodMaxBackoffSeconds:     defaultPodMaxBackoffSeconds,
				Plugins:                  nil,
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			// create the config
			config, err := tc.options.Config()

			// handle errors
			if err != nil {
				if tc.expectedError != "" || tc.checkErrFn != nil {
					if tc.expectedError != "" {
						assert.Contains(t, err.Error(), tc.expectedError, tc.name)
					}
					if tc.checkErrFn != nil {
						assert.True(t, tc.checkErrFn(err), "got error: %v", err)
					}
					return
				}
				assert.NoError(t, err)
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
				t.Errorf("expected server call with user %q, got %q", tc.expectedUsername, username)
			}
		})
	}
}
