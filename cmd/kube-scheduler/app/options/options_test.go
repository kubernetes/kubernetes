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
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
apiVersion: kubescheduler.config.k8s.io/v1alpha2
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

	oldConfigFile := filepath.Join(tmpDir, "scheduler_old.yaml")
	if err := ioutil.WriteFile(oldConfigFile, []byte(fmt.Sprintf(`
apiVersion: componentconfig/v1alpha1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	unknownVersionConfig := filepath.Join(tmpDir, "scheduler_invalid_wrong_api_version.yaml")
	if err := ioutil.WriteFile(unknownVersionConfig, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/unknown
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	noVersionConfig := filepath.Join(tmpDir, "scheduler_invalid_no_version.yaml")
	if err := ioutil.WriteFile(noVersionConfig, []byte(fmt.Sprintf(`
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
leaderElection:
  leaderElect: true`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	unknownFieldConfig := filepath.Join(tmpDir, "scheduler_invalid_unknown_field.yaml")
	if err := ioutil.WriteFile(unknownFieldConfig, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
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
apiVersion: kubescheduler.config.k8s.io/v1alpha2
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
	pluginConfigFile := filepath.Join(tmpDir, "plugin.yaml")
	if err := ioutil.WriteFile(pluginConfigFile, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
profiles:
- plugins:
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
  - name: InterPodAffinity
    args:
      hardPodAffinityWeight: 2
  - name: foo
    args:
      bar: baz
`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	// multiple profiles config
	multiProfilesConfig := filepath.Join(tmpDir, "multi-profiles.yaml")
	if err := ioutil.WriteFile(multiProfilesConfig, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
profiles:
- schedulerName: "foo-profile"
  plugins:
    reserve:
      enabled:
      - name: foo
- schedulerName: "bar-profile"
  plugins:
    preBind:
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
				AlgorithmSource:    kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HealthzBindAddress: "0.0.0.0:10251",
				MetricsBindAddress: "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{SchedulerName: "default-scheduler"},
				},
			},
		},
		{
			name: "config file in componentconfig/v1alpha1",
			options: &Options{
				ConfigFile: oldConfigFile,
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
			name:          "unknown version kubescheduler.config.k8s.io/unknown",
			options:       &Options{ConfigFile: unknownVersionConfig},
			expectedError: "no kind \"KubeSchedulerConfiguration\" is registered for version \"kubescheduler.config.k8s.io/unknown\"",
		},
		{
			name:          "config file with no version",
			options:       &Options{ConfigFile: noVersionConfig},
			expectedError: "Object 'apiVersion' is missing",
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
				AlgorithmSource:    kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HealthzBindAddress: "", // defaults empty when not running from config file
				MetricsBindAddress: "", // defaults empty when not running from config file
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{SchedulerName: "default-scheduler"},
				},
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
				AlgorithmSource:    kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HealthzBindAddress: "", // defaults empty when not running from config file
				MetricsBindAddress: "", // defaults empty when not running from config file
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{SchedulerName: "default-scheduler"},
				},
			},
			expectedUsername: "none, http",
		},
		{
			name: "plugin config",
			options: &Options{
				ConfigFile: pluginConfigFile,
			},
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				AlgorithmSource:    kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HealthzBindAddress: "0.0.0.0:10251",
				MetricsBindAddress: "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{
						SchedulerName: "default-scheduler",
						Plugins: &kubeschedulerconfig.Plugins{
							Reserve: &kubeschedulerconfig.PluginSet{
								Enabled: []kubeschedulerconfig.Plugin{
									{Name: "foo"},
									{Name: "bar"},
								},
								Disabled: []kubeschedulerconfig.Plugin{
									{Name: "baz"},
								},
							},
							PreBind: &kubeschedulerconfig.PluginSet{
								Enabled: []kubeschedulerconfig.Plugin{
									{Name: "foo"},
								},
								Disabled: []kubeschedulerconfig.Plugin{
									{Name: "baz"},
								},
							},
						},
						PluginConfig: []kubeschedulerconfig.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: &kubeschedulerconfig.InterPodAffinityArgs{
									HardPodAffinityWeight: 2,
								},
							},
							{
								Name: "foo",
								Args: &runtime.Unknown{
									Raw:         []byte(`{"bar":"baz"}`),
									ContentType: "application/json",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "multiple profiles",
			options: &Options{
				ConfigFile: multiProfilesConfig,
			},
			expectedUsername: "config",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				AlgorithmSource:    kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				HealthzBindAddress: "0.0.0.0:10251",
				MetricsBindAddress: "0.0.0.0:10251",
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{
						SchedulerName: "foo-profile",
						Plugins: &kubeschedulerconfig.Plugins{
							Reserve: &kubeschedulerconfig.PluginSet{
								Enabled: []kubeschedulerconfig.Plugin{
									{Name: "foo"},
								},
							},
						},
					},
					{
						SchedulerName: "bar-profile",
						Plugins: &kubeschedulerconfig.Plugins{
							PreBind: &kubeschedulerconfig.PluginSet{
								Disabled: []kubeschedulerconfig.Plugin{
									{Name: "baz"},
								},
							},
						},
						PluginConfig: []kubeschedulerconfig.PluginConfig{
							{
								Name: "foo",
							},
						},
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
			name: "Deprecated HardPodAffinitySymmetricWeight",
			options: &Options{
				ComponentConfig: func() kubeschedulerconfig.KubeSchedulerConfiguration {
					cfg, _ := newDefaultComponentConfig()
					cfg.ClientConnection.Kubeconfig = flagKubeconfig
					return *cfg
				}(),
				Deprecated: &DeprecatedOptions{
					HardPodAffinitySymmetricWeight: 5,
				},
			},
			expectedUsername: "flag",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				AlgorithmSource: kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{
						SchedulerName: "default-scheduler",
						PluginConfig: []kubeschedulerconfig.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: &kubeschedulerconfig.InterPodAffinityArgs{HardPodAffinityWeight: 5},
							},
						},
					},
				},
			},
		},
		{
			name: "Deprecated SchedulerName flag",
			options: &Options{
				ComponentConfig: func() kubeschedulerconfig.KubeSchedulerConfiguration {
					cfg, _ := newDefaultComponentConfig()
					cfg.ClientConnection.Kubeconfig = flagKubeconfig
					return *cfg
				}(),
				Deprecated: &DeprecatedOptions{
					SchedulerName:                  "my-nice-scheduler",
					HardPodAffinitySymmetricWeight: 1,
				},
			},
			expectedUsername: "flag",
			expectedConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
				AlgorithmSource: kubeschedulerconfig.SchedulerAlgorithmSource{Provider: &defaultSource},
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           true,
					EnableContentionProfiling: true,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       true,
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
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
				Profiles: []kubeschedulerconfig.KubeSchedulerProfile{
					{
						SchedulerName: "my-nice-scheduler",
						PluginConfig: []kubeschedulerconfig.PluginConfig{
							{
								Name: interpodaffinity.Name,
								Args: &kubeschedulerconfig.InterPodAffinityArgs{
									HardPodAffinityWeight: 1,
								},
							},
						},
					},
				},
			},
		},
		{
			name: "unknown field",
			options: &Options{
				ConfigFile: unknownFieldConfig,
			},
			expectedError: "found unknown field: foo",
			checkErrFn:    runtime.IsStrictDecodingError,
		},
		{
			name: "duplicate fields",
			options: &Options{
				ConfigFile: duplicateFieldConfig,
			},
			expectedError: `key "leaderElect" already set`,
			checkErrFn:    runtime.IsStrictDecodingError,
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
						assert.Contains(t, err.Error(), tc.expectedError)
					}
					if tc.checkErrFn != nil {
						assert.True(t, tc.checkErrFn(err), "got error: %v", err)
					}
					return
				}
				assert.NoError(t, err)
				return
			}

			if diff := cmp.Diff(tc.expectedConfig, config.ComponentConfig); diff != "" {
				t.Errorf("incorrect config (-want,+got):\n%s", diff)
			}

			// ensure we have a client
			if config.Client == nil {
				t.Error("unexpected nil client")
				return
			}

			// test the client talks to the endpoint we expect with the credentials we expect
			username = ""
			_, err = config.Client.Discovery().RESTClient().Get().AbsPath("/").DoRaw(context.TODO())
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
