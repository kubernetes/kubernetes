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
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
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

	defaultPlugins := map[string][]kubeschedulerconfig.Plugin{
		"QueueSortPlugin": {
			{Name: "PrioritySort"},
		},
		"PreFilterPlugin": {
			{Name: "NodeResourcesFit"},
			{Name: "NodePorts"},
			{Name: "PodTopologySpread"},
			{Name: "InterPodAffinity"},
			{Name: "VolumeBinding"},
		},
		"FilterPlugin": {
			{Name: "NodeUnschedulable"},
			{Name: "NodeResourcesFit"},
			{Name: "NodeName"},
			{Name: "NodePorts"},
			{Name: "NodeAffinity"},
			{Name: "VolumeRestrictions"},
			{Name: "TaintToleration"},
			{Name: "EBSLimits"},
			{Name: "GCEPDLimits"},
			{Name: "NodeVolumeLimits"},
			{Name: "AzureDiskLimits"},
			{Name: "VolumeBinding"},
			{Name: "VolumeZone"},
			{Name: "PodTopologySpread"},
			{Name: "InterPodAffinity"},
		},
		"PostFilterPlugin": {
			{Name: "DefaultPreemption"},
		},
		"PreScorePlugin": {
			{Name: "InterPodAffinity"},
			{Name: "PodTopologySpread"},
			{Name: "TaintToleration"},
			{Name: "SelectorSpread"},
		},
		"ScorePlugin": {
			{Name: "NodeResourcesBalancedAllocation", Weight: 1},
			{Name: "ImageLocality", Weight: 1},
			{Name: "InterPodAffinity", Weight: 1},
			{Name: "NodeResourcesLeastAllocated", Weight: 1},
			{Name: "NodeAffinity", Weight: 1},
			{Name: "NodePreferAvoidPods", Weight: 10000},
			{Name: "PodTopologySpread", Weight: 2},
			{Name: "TaintToleration", Weight: 1},
			{Name: "SelectorSpread", Weight: 1},
		},
		"BindPlugin":    {{Name: "DefaultBinder"}},
		"ReservePlugin": {{Name: "VolumeBinding"}},
		"PreBindPlugin": {{Name: "VolumeBinding"}},
	}

	testcases := []struct {
		name        string
		flags       []string
		wantPlugins map[string]map[string][]kubeschedulerconfig.Plugin
	}{
		{
			name: "default config",
			flags: []string{
				"--kubeconfig", configKubeconfig,
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"default-scheduler": defaultPlugins,
			},
		},
		{
			name: "plugin config with single profile",
			flags: []string{
				"--config", pluginConfigFile,
				"--kubeconfig", configKubeconfig,
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"default-scheduler": {
					"BindPlugin":       {{Name: "DefaultBinder"}},
					"FilterPlugin":     {{Name: "NodeResourcesFit"}, {Name: "NodePorts"}},
					"PreFilterPlugin":  {{Name: "NodeResourcesFit"}, {Name: "NodePorts"}},
					"PostFilterPlugin": {{Name: "DefaultPreemption"}},
					"PreScorePlugin":   {{Name: "InterPodAffinity"}, {Name: "TaintToleration"}},
					"QueueSortPlugin":  {{Name: "PrioritySort"}},
					"ScorePlugin":      {{Name: "InterPodAffinity", Weight: 1}, {Name: "TaintToleration", Weight: 1}},
					"ReservePlugin":    {{Name: "VolumeBinding"}},
					"PreBindPlugin":    {{Name: "VolumeBinding"}},
				},
			},
		},
		{
			name: "plugin config with multiple profiles",
			flags: []string{
				"--config", multiProfilesConfig,
				"--kubeconfig", configKubeconfig,
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"profile-default-plugins": defaultPlugins,
				"profile-disable-all-filter-and-score-plugins": {
					"BindPlugin":      {{Name: "DefaultBinder"}},
					"QueueSortPlugin": {{Name: "PrioritySort"}},
					"ReservePlugin":   {{Name: "VolumeBinding"}},
					"PreBindPlugin":   {{Name: "VolumeBinding"}},
				},
			},
		},
		{
			name: "Deprecated SchedulerName flag",
			flags: []string{
				"--kubeconfig", configKubeconfig,
				"--scheduler-name", "my-scheduler",
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"my-scheduler": defaultPlugins,
			},
		},
		{
			name: "default algorithm provider",
			flags: []string{
				"--kubeconfig", configKubeconfig,
				"--algorithm-provider", "DefaultProvider",
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"default-scheduler": defaultPlugins,
			},
		},
		{
			name: "cluster autoscaler provider",
			flags: []string{
				"--kubeconfig", configKubeconfig,
				"--algorithm-provider", "ClusterAutoscalerProvider",
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"default-scheduler": {
					"QueueSortPlugin": {
						{Name: "PrioritySort"},
					},
					"PreFilterPlugin": {
						{Name: "NodeResourcesFit"},
						{Name: "NodePorts"},
						{Name: "PodTopologySpread"},
						{Name: "InterPodAffinity"},
						{Name: "VolumeBinding"},
					},
					"FilterPlugin": {
						{Name: "NodeUnschedulable"},
						{Name: "NodeResourcesFit"},
						{Name: "NodeName"},
						{Name: "NodePorts"},
						{Name: "NodeAffinity"},
						{Name: "VolumeRestrictions"},
						{Name: "TaintToleration"},
						{Name: "EBSLimits"},
						{Name: "GCEPDLimits"},
						{Name: "NodeVolumeLimits"},
						{Name: "AzureDiskLimits"},
						{Name: "VolumeBinding"},
						{Name: "VolumeZone"},
						{Name: "PodTopologySpread"},
						{Name: "InterPodAffinity"},
					},
					"PostFilterPlugin": {
						{Name: "DefaultPreemption"},
					},
					"PreScorePlugin": {
						{Name: "InterPodAffinity"},
						{Name: "PodTopologySpread"},
						{Name: "TaintToleration"},
						{Name: "SelectorSpread"},
					},
					"ScorePlugin": {
						{Name: "NodeResourcesBalancedAllocation", Weight: 1},
						{Name: "ImageLocality", Weight: 1},
						{Name: "InterPodAffinity", Weight: 1},
						{Name: "NodeResourcesMostAllocated", Weight: 1},
						{Name: "NodeAffinity", Weight: 1},
						{Name: "NodePreferAvoidPods", Weight: 10000},
						{Name: "PodTopologySpread", Weight: 2},
						{Name: "TaintToleration", Weight: 1},
						{Name: "SelectorSpread", Weight: 1},
					},
					"BindPlugin":    {{Name: "DefaultBinder"}},
					"ReservePlugin": {{Name: "VolumeBinding"}},
					"PreBindPlugin": {{Name: "VolumeBinding"}},
				},
			},
		},
		{
			name: "policy config file",
			flags: []string{
				"--kubeconfig", configKubeconfig,
				"--policy-config-file", policyConfigFile,
			},
			wantPlugins: map[string]map[string][]kubeschedulerconfig.Plugin{
				"default-scheduler": {
					"QueueSortPlugin": {{Name: "PrioritySort"}},
					"PreFilterPlugin": {
						{Name: "InterPodAffinity"},
					},
					"FilterPlugin": {
						{Name: "NodeUnschedulable"},
						{Name: "TaintToleration"},
						{Name: "InterPodAffinity"},
					},
					"PreScorePlugin": {
						{Name: "InterPodAffinity"},
					},
					"ScorePlugin": {
						{Name: "InterPodAffinity", Weight: 2},
					},
					"BindPlugin": {{Name: "DefaultBinder"}},
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

			// use listeners instead of static ports so parallel test runs don't conflict
			opts.SecureServing.Listener = makeListener(t)
			defer opts.SecureServing.Listener.Close()
			opts.CombinedInsecureServing.Metrics.Listener = makeListener(t)
			defer opts.CombinedInsecureServing.Metrics.Listener.Close()
			opts.CombinedInsecureServing.Healthz.Listener = makeListener(t)
			defer opts.CombinedInsecureServing.Healthz.Listener.Close()

			for _, f := range opts.Flags().FlagSets {
				fs.AddFlagSet(f)
			}
			if err := fs.Parse(tc.flags); err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			_, sched, err := Setup(ctx, opts)
			if err != nil {
				t.Fatal(err)
			}

			gotPlugins := make(map[string]map[string][]kubeschedulerconfig.Plugin)
			for n, p := range sched.Profiles {
				gotPlugins[n] = p.ListPlugins()
			}

			if diff := cmp.Diff(tc.wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}
}
