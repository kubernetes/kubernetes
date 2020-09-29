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
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
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
