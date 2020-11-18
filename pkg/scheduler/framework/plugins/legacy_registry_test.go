/*
Copyright 2019 The Kubernetes Authors.

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

package plugins

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/imagelocality"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodepreferavoidpods"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
)

func TestRegisterConfigProducers(t *testing.T) {
	registry := NewLegacyRegistry()
	testPredicateName1 := "testPredicate1"
	testFilterName1 := "testFilter1"
	registry.registerPredicateConfigProducer(testPredicateName1,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, testFilterName1, nil)
		})

	testPredicateName2 := "testPredicate2"
	testFilterName2 := "testFilter2"
	registry.registerPredicateConfigProducer(testPredicateName2,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, testFilterName2, nil)
		})

	testPriorityName1 := "testPriority1"
	testScoreName1 := "testScore1"
	registry.registerPriorityConfigProducer(testPriorityName1,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, testScoreName1, &args.Weight)
		})

	testPriorityName2 := "testPriority2"
	testScoreName2 := "testScore2"
	registry.registerPriorityConfigProducer(testPriorityName2,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, testScoreName2, &args.Weight)
		})

	args := ConfigProducerArgs{Weight: 1}
	var gotPlugins config.Plugins
	gotPlugins, _, err := registry.AppendPredicateConfigs(sets.NewString(testPredicateName1, testPredicateName2), &args, gotPlugins, nil)
	if err != nil {
		t.Fatalf("producing predicate framework configs: %v.", err)
	}

	priorities := map[string]int64{
		testPriorityName1: 1,
		testPriorityName2: 1,
	}
	gotPlugins, _, err = registry.AppendPriorityConfigs(priorities, &args, gotPlugins, nil)
	if err != nil {
		t.Fatalf("producing priority framework configs: %v.", err)
	}

	wantPlugins := config.Plugins{
		Filter: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: nodeunschedulable.Name},
				{Name: tainttoleration.Name},
				{Name: testFilterName1},
				{Name: testFilterName2},
			},
		},
		Score: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: testScoreName1, Weight: 1},
				{Name: testScoreName2, Weight: 1},
			},
		},
	}

	if diff := cmp.Diff(wantPlugins, gotPlugins); diff != "" {
		t.Errorf("unexpected plugin configuration (-want, +got): %s", diff)
	}
}

func TestAppendPriorityConfigs(t *testing.T) {
	cases := []struct {
		name             string
		features         map[featuregate.Feature]bool
		keys             map[string]int64
		args             ConfigProducerArgs
		wantPlugins      config.Plugins
		wantPluginConfig []config.PluginConfig
	}{
		{
			name: "default priorities",
			wantPlugins: config.Plugins{
				PreScore: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name},
						{Name: interpodaffinity.Name},
						{Name: tainttoleration.Name},
					},
				},
				Score: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: noderesources.BalancedAllocationName, Weight: 1},
						{Name: podtopologyspread.Name, Weight: 2},
						{Name: imagelocality.Name, Weight: 1},
						{Name: interpodaffinity.Name, Weight: 1},
						{Name: noderesources.LeastAllocatedName, Weight: 1},
						{Name: nodeaffinity.Name, Weight: 1},
						{Name: nodepreferavoidpods.Name, Weight: 10000},
						{Name: tainttoleration.Name, Weight: 1},
					},
				},
			},
			wantPluginConfig: []config.PluginConfig{
				{
					Name: podtopologyspread.Name,
					Args: &config.PodTopologySpreadArgs{
						DefaultingType: config.SystemDefaulting,
					},
				},
			},
		},
		{
			name: "DefaultPodTopologySpread enabled, SelectorSpreadPriority only",
			keys: map[string]int64{
				SelectorSpreadPriority: 3,
			},
			wantPlugins: config.Plugins{
				PreScore: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name},
					},
				},
				Score: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name, Weight: 3},
					},
				},
			},
			wantPluginConfig: []config.PluginConfig{
				{
					Name: podtopologyspread.Name,
					Args: &config.PodTopologySpreadArgs{
						DefaultingType: config.SystemDefaulting,
					},
				},
			},
		},
		{
			name: "DefaultPodTopologySpread enabled, EvenPodsSpreadPriority only",
			keys: map[string]int64{
				EvenPodsSpreadPriority: 4,
			},
			wantPlugins: config.Plugins{
				PreScore: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name},
					},
				},
				Score: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name, Weight: 4},
					},
				},
			},
			wantPluginConfig: []config.PluginConfig{
				{
					Name: podtopologyspread.Name,
					Args: &config.PodTopologySpreadArgs{
						DefaultingType: config.ListDefaulting,
					},
				},
			},
		},
		{
			name: "DefaultPodTopologySpread disabled, SelectorSpreadPriority+EvenPodsSpreadPriority",
			features: map[featuregate.Feature]bool{
				features.DefaultPodTopologySpread: false,
			},
			keys: map[string]int64{
				SelectorSpreadPriority: 1,
				EvenPodsSpreadPriority: 2,
			},
			wantPlugins: config.Plugins{
				PreScore: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name},
						{Name: selectorspread.Name},
					},
				},
				Score: &config.PluginSet{
					Enabled: []config.Plugin{
						{Name: podtopologyspread.Name, Weight: 2},
						{Name: selectorspread.Name, Weight: 1},
					},
				},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for k, v := range tc.features {
				defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, k, v)()
			}

			r := NewLegacyRegistry()
			keys := tc.keys
			if keys == nil {
				keys = r.DefaultPriorities
			}
			plugins, pluginConfig, err := r.AppendPriorityConfigs(keys, &tc.args, config.Plugins{}, nil)
			if err != nil {
				t.Fatalf("Appending Priority Configs: %v", err)
			}
			if diff := cmp.Diff(tc.wantPlugins, plugins); diff != "" {
				t.Errorf("Unexpected Plugin (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantPluginConfig, pluginConfig); diff != "" {
				t.Errorf("Unexpected PluginConfig (-want,+got):\n%s", diff)
			}
		})
	}
}
