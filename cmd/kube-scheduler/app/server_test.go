/*
Copyright 2020 The Kubernetes Authors.

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

package app

import (
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/kubernetes/fake"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/kubernetes/cmd/kube-scheduler/app/options"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/testing/defaults"
)

const (
	pprofPath = "/debug/pprof"
	logPath   = "/debug/flags"
)

func TestSetup(t *testing.T) {
	// temp dir
	tmpDir, err := ioutil.TempDir("", "scheduler-options")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// https server
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`ok`))
	}))
	defer server.Close()

	configKubeconfig := filepath.Join(tmpDir, "config.kubeconfig")
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

	// plugin config
	pluginConfigFile := filepath.Join(tmpDir, "plugin.yaml")
	if err := ioutil.WriteFile(pluginConfigFile, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1beta2
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
profiles:
- plugins:
    preFilter:
      enabled:
      - name: NodeResourcesFit
      - name: NodePorts
      disabled:
      - name: "*"
    filter:
      enabled:
      - name: NodeResourcesFit
      - name: NodePorts
      disabled:
      - name: "*"
    preScore:
      enabled:
      - name: InterPodAffinity
      - name: TaintToleration
      disabled:
      - name: "*"
    score:
      enabled:
      - name: InterPodAffinity
      - name: TaintToleration
      disabled:
      - name: "*"
`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	// multiple profiles config
	multiProfilesConfig := filepath.Join(tmpDir, "multi-profiles.yaml")
	if err := ioutil.WriteFile(multiProfilesConfig, []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1beta2
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"
profiles:
- schedulerName: "profile-default-plugins"
- schedulerName: "profile-disable-all-filter-and-score-plugins"
  plugins:
    preFilter:
      disabled:
      - name: "*"
    filter:
      disabled:
      - name: "*"
    postFilter:
      disabled:
      - name: "*"
    preScore:
      disabled:
      - name: "*"
    score:
      disabled:
      - name: "*"
`, configKubeconfig)), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	// policy config file
	policyConfigFile := filepath.Join(tmpDir, "policy-config.yaml")
	if err := ioutil.WriteFile(policyConfigFile, []byte(`{
		"kind": "Policy",
		"apiVersion": "v1",
		"predicates": [
		  {"name": "MatchInterPodAffinity"}
		],"priorities": [
		  {"name": "InterPodAffinityPriority",   "weight": 2}
		]}`), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	testcases := []struct {
		name        string
		flags       []string
		wantPlugins map[string]*config.Plugins
	}{
		{
			name: "default config",
			flags: []string{
				"--kubeconfig", configKubeconfig,
			},
			wantPlugins: map[string]*config.Plugins{
				"default-scheduler": defaults.PluginsV1beta3,
			},
		},
		{
			name: "component configuration",
			flags: []string{
				"--config", pluginConfigFile,
				"--kubeconfig", configKubeconfig,
			},
			wantPlugins: map[string]*config.Plugins{
				"default-scheduler": {
					Bind: config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultBinder"}}},
					Filter: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "NodeResourcesFit"},
							{Name: "NodePorts"},
						},
					},
					PreFilter: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "NodeResourcesFit"},
							{Name: "NodePorts"},
						},
					},
					PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultPreemption"}}},
					PreScore: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "InterPodAffinity"},
							{Name: "TaintToleration"},
						},
					},
					QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "PrioritySort"}}},
					Score: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "InterPodAffinity", Weight: 1},
							{Name: "TaintToleration", Weight: 1},
						},
					},
					Reserve: config.PluginSet{Enabled: []config.Plugin{{Name: "VolumeBinding"}}},
					PreBind: config.PluginSet{Enabled: []config.Plugin{{Name: "VolumeBinding"}}},
				},
			},
		},
		{
			name: "policy config file",
			flags: []string{
				"--kubeconfig", configKubeconfig,
				"--policy-config-file", policyConfigFile,
			},
			wantPlugins: map[string]*config.Plugins{
				"default-scheduler": {
					QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "PrioritySort"}}},
					PreFilter: config.PluginSet{Enabled: []config.Plugin{{Name: "InterPodAffinity"}}},
					Filter: config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "NodeUnschedulable"},
							{Name: "TaintToleration"},
							{Name: "InterPodAffinity"},
						},
					},
					PostFilter: config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultPreemption"}}},
					PreScore:   config.PluginSet{Enabled: []config.Plugin{{Name: "InterPodAffinity"}}},
					Score:      config.PluginSet{Enabled: []config.Plugin{{Name: "InterPodAffinity", Weight: 2}}},
					Bind:       config.PluginSet{Enabled: []config.Plugin{{Name: "DefaultBinder"}}},
				},
			},
		},
	}

	makeListener := func(t *testing.T) net.Listener {
		t.Helper()
		l, err := net.Listen("tcp", ":0")
		if err != nil {
			t.Fatal(err)
		}
		return l
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			fs := pflag.NewFlagSet("test", pflag.PanicOnError)
			opts, err := options.NewOptions()
			if err != nil {
				t.Fatal(err)
			}

			nfs := opts.Flags()
			for _, f := range nfs.FlagSets {
				fs.AddFlagSet(f)
			}
			if err := fs.Parse(tc.flags); err != nil {
				t.Fatal(err)
			}

			if err := opts.Complete(&nfs); err != nil {
				t.Fatal(err)
			}

			// use listeners instead of static ports so parallel test runs don't conflict
			opts.SecureServing.Listener = makeListener(t)
			defer opts.SecureServing.Listener.Close()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			_, sched, err := Setup(ctx, opts)
			if err != nil {
				t.Fatal(err)
			}

			gotPlugins := make(map[string]*config.Plugins)
			for n, p := range sched.Profiles {
				gotPlugins[n] = p.ListPlugins()
			}

			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}
}

func TestNewHealthzAndMetricsHandler(t *testing.T) {
	// Setup any healthz checks we will want to use.
	var checks []healthz.HealthChecker

	// Test EnableProfiling
	tests := []struct {
		enableProfiling           bool
		enableContentionProfiling bool
		status                    int
	}{
		{
			enableProfiling:           true,
			enableContentionProfiling: true,
			status:                    200,
		},
		{
			enableProfiling:           false,
			enableContentionProfiling: false,
			status:                    404,
		},
	}

	for _, test := range tests {
		c := &config.KubeSchedulerConfiguration{
			DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
				EnableProfiling:           test.enableProfiling,
				EnableContentionProfiling: test.enableContentionProfiling,
			},
		}

		client := fake.NewSimpleClientset()
		informers := scheduler.NewInformerFactory(client, 0)

		isLeader := func() bool {
			return true
		}

		mux := newHealthzAndMetricsHandler(c, informers, isLeader, checks...)

		s := httptest.NewServer(mux)
		defer s.Close()
		resp, _ := http.Get(s.URL + pprofPath)
		assert.Equal(t, test.status, resp.StatusCode)

		resp, _ = http.Get(s.URL + logPath)
		assert.Equal(t, test.status, resp.StatusCode)
	}
}
