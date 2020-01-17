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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func produceConfig(keys []string, producersMap map[string]ConfigProducer, args ConfigProducerArgs) (*config.Plugins, []config.PluginConfig, error) {
	var plugins config.Plugins
	var pluginConfig []config.PluginConfig
	for _, k := range keys {
		p, exist := producersMap[k]
		if !exist {
			return nil, nil, fmt.Errorf("finding key %q", k)
		}
		pl, plc := p(args)
		plugins.Append(&pl)
		pluginConfig = append(pluginConfig, plc...)
	}
	return &plugins, pluginConfig, nil
}

func TestRegisterConfigProducers(t *testing.T) {
	registry := NewLegacyRegistry()
	testPredicateName1 := "testPredicate1"
	testFilterName1 := "testFilter1"
	registry.registerPredicateConfigProducer(testPredicateName1,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, testFilterName1, nil)
			return
		})

	testPredicateName2 := "testPredicate2"
	testFilterName2 := "testFilter2"
	registry.registerPredicateConfigProducer(testPredicateName2,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, testFilterName2, nil)
			return
		})

	testPriorityName1 := "testPriority1"
	testScoreName1 := "testScore1"
	registry.registerPriorityConfigProducer(testPriorityName1,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, testScoreName1, &args.Weight)
			return
		})

	testPriorityName2 := "testPriority2"
	testScoreName2 := "testScore2"
	registry.registerPriorityConfigProducer(testPriorityName2,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, testScoreName2, &args.Weight)
			return
		})

	args := ConfigProducerArgs{Weight: 1}
	predicatePlugins, _, err := produceConfig(
		[]string{testPredicateName1, testPredicateName2}, registry.PredicateToConfigProducer, args)
	if err != nil {
		t.Fatalf("producing predicate framework configs: %v.", err)
	}

	priorityPlugins, _, err := produceConfig(
		[]string{testPriorityName1, testPriorityName2}, registry.PriorityToConfigProducer, args)
	if err != nil {
		t.Fatalf("producing predicate framework configs: %v.", err)
	}

	// Verify that predicates and priorities are in the map and produce the expected score configurations.
	var gotPlugins config.Plugins
	gotPlugins.Append(predicatePlugins)
	gotPlugins.Append(priorityPlugins)

	// Verify the aggregated configuration.
	wantPlugins := config.Plugins{
		QueueSort: &config.PluginSet{},
		PreFilter: &config.PluginSet{},
		Filter: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: testFilterName1},
				{Name: testFilterName2},
			},
		},
		PostFilter: &config.PluginSet{},
		Score: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: testScoreName1, Weight: 1},
				{Name: testScoreName2, Weight: 1},
			},
		},
		Reserve:   &config.PluginSet{},
		Permit:    &config.PluginSet{},
		PreBind:   &config.PluginSet{},
		Bind:      &config.PluginSet{},
		PostBind:  &config.PluginSet{},
		Unreserve: &config.PluginSet{},
	}

	if diff := cmp.Diff(wantPlugins, gotPlugins); diff != "" {
		t.Errorf("unexpected plugin configuration (-want, +got): %s", diff)
	}
}
