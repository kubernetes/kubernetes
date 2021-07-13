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
	"k8s.io/kubernetes/cmd/kube-scheduler/app/options"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/testing/defaults"
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
apiVersion: kubescheduler.config.k8s.io/v1beta1
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
apiVersion: kubescheduler.config.k8s.io/v1beta1
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
				"default-scheduler": defaults.PluginsV1beta2,
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
			opts.CombinedInsecureServing.Metrics.Listener = makeListener(t)
			defer opts.CombinedInsecureServing.Metrics.Listener.Close()
			opts.CombinedInsecureServing.Healthz.Listener = makeListener(t)
			defer opts.CombinedInsecureServing.Healthz.Listener.Close()

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

func TestBackwardCompatibilityForDefaultConfiguration(t *testing.T) {
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

	kubeConfig := fmt.Sprintf(`
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
`, server.URL)
	configKubeconfig := filepath.Join(tmpDir, "config.kubeconfig")
	if err := ioutil.WriteFile(configKubeconfig, []byte(kubeConfig), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	compatibilityWithDefaultConfig := filepath.Join(tmpDir, "default-config.yaml")

	// defaultSchedulerConfig generated by following cmds and fixed by explicitly disabled then
	// enabled plugins to avoid warnings about duplicates
	// ``git checkout upstream/release-1.21``
	// ``make kube-scheduler``
	// ``kube-scheduler --master=http://localhost:8080 --write-config-to 1.21_default.yaml``
	defaultSchedulerConfig := `
apiVersion: kubescheduler.config.k8s.io/v1beta1
clientConnection:
  acceptContentTypes: ""
  burst: 100
  contentType: application/vnd.kubernetes.protobuf
  kubeconfig: ""
  qps: 50
enableContentionProfiling: true
enableProfiling: true
healthzBindAddress: 0.0.0.0:10251
kind: KubeSchedulerConfiguration
leaderElection:
  leaderElect: true
  leaseDuration: 15s
  renewDeadline: 10s
  resourceLock: leases
  resourceName: kube-scheduler
  resourceNamespace: kube-system
  retryPeriod: 2s
metricsBindAddress: 0.0.0.0:10251
parallelism: 16
percentageOfNodesToScore: 0
podInitialBackoffSeconds: 1
podMaxBackoffSeconds: 10
profiles:
- pluginConfig:
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: DefaultPreemptionArgs
      minCandidateNodesAbsolute: 100
      minCandidateNodesPercentage: 10
    name: DefaultPreemption
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      hardPodAffinityWeight: 1
      kind: InterPodAffinityArgs
    name: InterPodAffinity
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: NodeAffinityArgs
    name: NodeAffinity
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: NodeResourcesFitArgs
    name: NodeResourcesFit
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: NodeResourcesLeastAllocatedArgs
      resources:
      - name: cpu
        weight: 1
      - name: memory
        weight: 1
    name: NodeResourcesLeastAllocated
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      defaultingType: System
      kind: PodTopologySpreadArgs
    name: PodTopologySpread
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      bindTimeoutSeconds: 600
      kind: VolumeBindingArgs
    name: VolumeBinding
  plugins:
    bind:
      disabled:
      - name: DefaultBinder
      enabled:
      - name: DefaultBinder
        weight: 0
    filter:
      disabled:
      - name: NodeUnschedulable
      - name: NodeName
      - name: TaintToleration
      - name: NodeAffinity
      - name: NodePorts
      - name: NodeResourcesFit
      - name: VolumeRestrictions
      - name: EBSLimits
      - name: GCEPDLimits
      - name: NodeVolumeLimits
      - name: AzureDiskLimits
      - name: VolumeBinding
      - name: VolumeZone
      - name: PodTopologySpread
      - name: InterPodAffinity
      enabled:
      - name: NodeUnschedulable
        weight: 0
      - name: NodeName
        weight: 0
      - name: TaintToleration
        weight: 0
      - name: NodeAffinity
        weight: 0
      - name: NodePorts
        weight: 0
      - name: NodeResourcesFit
        weight: 0
      - name: VolumeRestrictions
        weight: 0
      - name: EBSLimits
        weight: 0
      - name: GCEPDLimits
        weight: 0
      - name: NodeVolumeLimits
        weight: 0
      - name: AzureDiskLimits
        weight: 0
      - name: VolumeBinding
        weight: 0
      - name: VolumeZone
        weight: 0
      - name: PodTopologySpread
        weight: 0
      - name: InterPodAffinity
        weight: 0
    permit: {}
    postBind: {}
    postFilter:
      disabled:
      - name: DefaultPreemption
      enabled:
      - name: DefaultPreemption
        weight: 0
    preBind:
      disabled:
      - name: VolumeBinding
      enabled:
      - name: VolumeBinding
        weight: 0
    preFilter:
      disabled:
      - name: NodeResourcesFit
      - name: NodePorts
      - name: PodTopologySpread
      - name: InterPodAffinity
      - name: VolumeBinding
      - name: NodeAffinity
      enabled:
      - name: NodeResourcesFit
        weight: 0
      - name: NodePorts
        weight: 0
      - name: PodTopologySpread
        weight: 0
      - name: InterPodAffinity
        weight: 0
      - name: VolumeBinding
        weight: 0
      - name: NodeAffinity
        weight: 0
    preScore:
      disabled:
      - name: InterPodAffinity
      - name: PodTopologySpread
      - name: TaintToleration
      - name: NodeAffinity
      enabled:
      - name: InterPodAffinity
        weight: 0
      - name: PodTopologySpread
        weight: 0
      - name: TaintToleration
        weight: 0
      - name: NodeAffinity
        weight: 0
    queueSort:
      disabled:
      - name: PrioritySort
      enabled:
      - name: PrioritySort
        weight: 0
    reserve:
      disabled:
      - name: VolumeBinding
      enabled:
      - name: VolumeBinding
        weight: 0
    score:
      disabled:
      - name: NodeResourcesBalancedAllocation
      - name: ImageLocality
      - name: InterPodAffinity
      - name: NodeResourcesLeastAllocated
      - name: NodeAffinity
      - name: NodePreferAvoidPods
      - name: PodTopologySpread
      - name: TaintToleration
      enabled:
      - name: NodeResourcesBalancedAllocation
        weight: 1
      - name: ImageLocality
        weight: 1
      - name: InterPodAffinity
        weight: 1
      - name: NodeResourcesLeastAllocated
        weight: 1
      - name: NodeAffinity
        weight: 1
      - name: NodePreferAvoidPods
        weight: 10000
      - name: PodTopologySpread
        weight: 2
      - name: TaintToleration
        weight: 1
  schedulerName: default-scheduler
`

	if err := ioutil.WriteFile(compatibilityWithDefaultConfig, []byte(defaultSchedulerConfig), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	testcases := []struct {
		name                string
		flags               []string
		pluginsValidateFunc func(map[string]*config.Plugins) error
	}{
		{
			name: "backward compatibility with default config",
			flags: []string{
				"--config", compatibilityWithDefaultConfig,
				"--kubeconfig", configKubeconfig,
				"--master", "http://localhost:8080",
			},
			pluginsValidateFunc: func(plugins map[string]*config.Plugins) error {
				defaultProfile := plugins["default-scheduler"]
				notExistsPluginSet := make(map[string][]*config.Plugin)
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.QueueSort.Enabled, []config.Plugin{
					{Name: "PrioritySort"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["QueueSort"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.PreFilter.Enabled, []config.Plugin{
					{Name: "NodeResourcesFit"},
					{Name: "NodePorts"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
					{Name: "VolumeBinding"},
					{Name: "NodeAffinity"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["PreFilter"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.Filter.Enabled, []config.Plugin{
					{Name: "NodeUnschedulable"},
					{Name: "NodeName"},
					{Name: "TaintToleration"},
					{Name: "NodeAffinity"},
					{Name: "NodePorts"},
					{Name: "NodeResourcesFit"},
					{Name: "VolumeRestrictions"},
					{Name: "EBSLimits"},
					{Name: "GCEPDLimits"},
					{Name: "NodeVolumeLimits"},
					{Name: "AzureDiskLimits"},
					{Name: "VolumeBinding"},
					{Name: "VolumeZone"},
					{Name: "PodTopologySpread"},
					{Name: "InterPodAffinity"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["Filter"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.PostFilter.Enabled, []config.Plugin{
					{Name: "DefaultPreemption"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["PostFilter"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.PreScore.Enabled, []config.Plugin{
					{Name: "InterPodAffinity"},
					{Name: "PodTopologySpread"},
					{Name: "TaintToleration"},
					{Name: "NodeAffinity"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["PreScore"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.Score.Enabled, []config.Plugin{
					{Name: "NodeResourcesBalancedAllocation", Weight: 1},
					{Name: "ImageLocality", Weight: 1},
					{Name: "InterPodAffinity", Weight: 1},
					{Name: "NodeResourcesLeastAllocated", Weight: 1},
					{Name: "NodeAffinity", Weight: 1},
					{Name: "NodePreferAvoidPods", Weight: 10000},
					{Name: "PodTopologySpread", Weight: 2},
					{Name: "TaintToleration", Weight: 1},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["Score"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.Reserve.Enabled, []config.Plugin{
					{Name: "VolumeBinding"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["Reserve"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.PreBind.Enabled, []config.Plugin{
					{Name: "VolumeBinding"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["PreBind"] = notExistsPlugins
				}
				if notExistsPlugins := findNotExistsPlugins(defaultProfile.Bind.Enabled, []config.Plugin{
					{Name: "DefaultBinder"},
				}); len(notExistsPlugins) != 0 {
					notExistsPluginSet["Bind"] = notExistsPlugins
				}

				if len(notExistsPluginSet) == 0 {
					return nil
				}

				var noneExistsPluginsErrorDescription string
				for extensionPoint, plugins := range notExistsPluginSet {
					for _, plugin := range plugins {
						noneExistsPluginsErrorDescription = fmt.Sprintf("%s%s->%s ", noneExistsPluginsErrorDescription, extensionPoint, plugin.Name)
					}
				}

				return fmt.Errorf(noneExistsPluginsErrorDescription)
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
			opts.CombinedInsecureServing.Metrics.Listener = makeListener(t)
			defer opts.CombinedInsecureServing.Metrics.Listener.Close()
			opts.CombinedInsecureServing.Healthz.Listener = makeListener(t)
			defer opts.CombinedInsecureServing.Healthz.Listener.Close()

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

			if err := tc.pluginsValidateFunc(gotPlugins); err != nil {
				t.Errorf("expected plugins: %s", err)
			}
		})
	}
}

// findNotExistsPlugins find and return plugins that was not exists in given plugins
func findNotExistsPlugins(gotPlugins []config.Plugin, wantedPlugins []config.Plugin) []*config.Plugin {
	pluginNotExists := make([]*config.Plugin, 0, len(wantedPlugins))
	for _, wantedPlugin := range wantedPlugins {
		pluginFound := false
		for _, plugin := range gotPlugins {
			if cmp.Equal(plugin, wantedPlugin) {
				pluginFound = true
				break
			}
		}
		if !pluginFound {
			pluginNotExists = append(pluginNotExists, &wantedPlugin)
		}
	}
	return pluginNotExists
}
