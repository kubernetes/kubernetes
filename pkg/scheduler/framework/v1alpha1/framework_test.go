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

package v1alpha1

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

const (
	scoreWithNormalizePlugin1  = "score-with-normalize-plugin-1"
	scoreWithNormalizePlugin2  = "score-with-normalize-plugin-2"
	scorePlugin1               = "score-plugin-1"
	pluginNotImplementingScore = "plugin-not-implementing-score"
)

// TestScoreWithNormalizePlugin implements ScoreWithNormalizePlugin interface.
// TestScorePlugin only implements ScorePlugin interface.
var _ = ScoreWithNormalizePlugin(&TestScoreWithNormalizePlugin{})
var _ = ScorePlugin(&TestScorePlugin{})

func newScoreWithNormalizePlugin1(inj injectedResult) Plugin {
	return &TestScoreWithNormalizePlugin{scoreWithNormalizePlugin1, inj}
}
func newScoreWithNormalizePlugin2(inj injectedResult) Plugin {
	return &TestScoreWithNormalizePlugin{scoreWithNormalizePlugin2, inj}
}
func newScorePlugin1(inj injectedResult) Plugin {
	return &TestScorePlugin{scorePlugin1, inj}
}
func newPluginNotImplementingScore(injectedResult) Plugin {
	return &PluginNotImplementingScore{}
}

type TestScoreWithNormalizePlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestScoreWithNormalizePlugin) Name() string {
	return pl.name
}

func (pl *TestScoreWithNormalizePlugin) NormalizeScore(pc *PluginContext, pod *v1.Pod, scores NodeScoreList) *Status {
	return injectNormalizeRes(pl.inj, scores)
}

func (pl *TestScoreWithNormalizePlugin) Score(pc *PluginContext, p *v1.Pod, nodeName string) (int, *Status) {
	return setScoreRes(pl.inj)
}

// TestScorePlugin only implements ScorePlugin interface.
type TestScorePlugin struct {
	name string
	inj  injectedResult
}

func (pl *TestScorePlugin) Name() string {
	return pl.name
}

func (pl *TestScorePlugin) Score(pc *PluginContext, p *v1.Pod, nodeName string) (int, *Status) {
	return setScoreRes(pl.inj)
}

// PluginNotImplementingScore doesn't implement the ScorePlugin interface.
type PluginNotImplementingScore struct{}

func (pl *PluginNotImplementingScore) Name() string {
	return pluginNotImplementingScore
}

var defaultConstructors = map[string]func(injectedResult) Plugin{
	scoreWithNormalizePlugin1:  newScoreWithNormalizePlugin1,
	scoreWithNormalizePlugin2:  newScoreWithNormalizePlugin2,
	scorePlugin1:               newScorePlugin1,
	pluginNotImplementingScore: newPluginNotImplementingScore,
}

var defaultWeights = map[string]int32{
	scoreWithNormalizePlugin1: 1,
	scoreWithNormalizePlugin2: 2,
	scorePlugin1:              1,
}

// No specific config required.
var args []config.PluginConfig
var pc = &PluginContext{}

// Pod is only used for logging errors.
var pod = &v1.Pod{}
var nodes = []*v1.Node{
	{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
	{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
}

func TestInitFrameworkWithScorePlugins(t *testing.T) {
	tests := []struct {
		name    string
		plugins *config.Plugins
		// If initErr is true, we expect framework initialization to fail.
		initErr bool
	}{
		{
			name:    "enabled Score plugin doesn't exist in registry",
			plugins: buildConfigDefaultWeights("notExist"),
			initErr: true,
		},
		{
			name:    "enabled Score plugin doesn't extend the ScorePlugin interface",
			plugins: buildConfigDefaultWeights(pluginNotImplementingScore),
			initErr: true,
		},
		{
			name:    "Score plugins are nil",
			plugins: &config.Plugins{Score: nil},
		},
		{
			name:    "enabled Score plugin list is empty",
			plugins: buildConfigDefaultWeights(),
		},
		{
			name:    "enabled plugin only implements ScorePlugin interface",
			plugins: buildConfigDefaultWeights(scorePlugin1),
		},
		{
			name:    "enabled plugin implements ScoreWithNormalizePlugin interface",
			plugins: buildConfigDefaultWeights(scoreWithNormalizePlugin1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewFramework(toRegistry(defaultConstructors, make(map[string]injectedResult)), tt.plugins, args)
			if tt.initErr && err == nil {
				t.Fatal("Framework initialization should fail")
			}
			if !tt.initErr && err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}
		})
	}
}

func TestRunScorePlugins(t *testing.T) {
	tests := []struct {
		name        string
		registry    Registry
		plugins     *config.Plugins
		injectedRes map[string]injectedResult
		want        PluginToNodeScores
		// If err is true, we expect RunScorePlugin to fail.
		err bool
	}{
		{
			name:    "no Score plugins",
			plugins: buildConfigDefaultWeights(),
			want:    PluginToNodeScores{},
		},
		{
			name:    "single Score plugin",
			plugins: buildConfigDefaultWeights(scorePlugin1),
			injectedRes: map[string]injectedResult{
				scorePlugin1: {scoreRes: 1},
			},
			// scorePlugin1 Score returns 1, weight=1, so want=1.
			want: PluginToNodeScores{
				scorePlugin1: {{Name: "node1", Score: 1}, {Name: "node2", Score: 1}},
			},
		},
		{
			name: "single ScoreWithNormalize plugin",
			//registry: registry,
			plugins: buildConfigDefaultWeights(scoreWithNormalizePlugin1),
			injectedRes: map[string]injectedResult{
				scoreWithNormalizePlugin1: {scoreRes: 10, normalizeRes: 5},
			},
			// scoreWithNormalizePlugin1 Score returns 10, but NormalizeScore overrides to 5, weight=1, so want=5
			want: PluginToNodeScores{
				scoreWithNormalizePlugin1: {{Name: "node1", Score: 5}, {Name: "node2", Score: 5}},
			},
		},
		{
			name:    "2 Score plugins, 2 NormalizeScore plugins",
			plugins: buildConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1, scoreWithNormalizePlugin2),
			injectedRes: map[string]injectedResult{
				scorePlugin1:              {scoreRes: 1},
				scoreWithNormalizePlugin1: {scoreRes: 3, normalizeRes: 4},
				scoreWithNormalizePlugin2: {scoreRes: 4, normalizeRes: 5},
			},
			// scorePlugin1 Score returns 1, weight =1, so want=1.
			// scoreWithNormalizePlugin1 Score returns 3, but NormalizeScore overrides to 4, weight=1, so want=4.
			// scoreWithNormalizePlugin2 Score returns 4, but NormalizeScore overrides to 5, weight=2, so want=10.
			want: PluginToNodeScores{
				scorePlugin1:              {{Name: "node1", Score: 1}, {Name: "node2", Score: 1}},
				scoreWithNormalizePlugin1: {{Name: "node1", Score: 4}, {Name: "node2", Score: 4}},
				scoreWithNormalizePlugin2: {{Name: "node1", Score: 10}, {Name: "node2", Score: 10}},
			},
		},
		{
			name: "score fails",
			injectedRes: map[string]injectedResult{
				scoreWithNormalizePlugin1: {scoreErr: true},
			},
			plugins: buildConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1),
			err:     true,
		},
		{
			name: "normalize fails",
			injectedRes: map[string]injectedResult{
				scoreWithNormalizePlugin1: {normalizeErr: true},
			},
			plugins: buildConfigDefaultWeights(scorePlugin1, scoreWithNormalizePlugin1),
			err:     true,
		},
		{
			name:    "Score plugin return score greater than MaxNodeScore",
			plugins: buildConfigDefaultWeights(scorePlugin1),
			injectedRes: map[string]injectedResult{
				scorePlugin1: {scoreRes: MaxNodeScore + 1},
			},
			err: true,
		},
		{
			name:    "Score plugin return score less than MinNodeScore",
			plugins: buildConfigDefaultWeights(scorePlugin1),
			injectedRes: map[string]injectedResult{
				scorePlugin1: {scoreRes: MinNodeScore - 1},
			},
			err: true,
		},
		{
			name:    "ScoreWithNormalize plugin return score greater than MaxNodeScore",
			plugins: buildConfigDefaultWeights(scoreWithNormalizePlugin1),
			injectedRes: map[string]injectedResult{
				scoreWithNormalizePlugin1: {normalizeRes: MaxNodeScore + 1},
			},
			err: true,
		},
		{
			name:    "ScoreWithNormalize plugin return score less than MinNodeScore",
			plugins: buildConfigDefaultWeights(scoreWithNormalizePlugin1),
			injectedRes: map[string]injectedResult{
				scoreWithNormalizePlugin1: {normalizeRes: MinNodeScore - 1},
			},
			err: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Inject the results for each plugin.
			registry := toRegistry(defaultConstructors, tt.injectedRes)
			f, err := NewFramework(registry, tt.plugins, args)
			if err != nil {
				t.Fatalf("Failed to create framework for testing: %v", err)
			}

			res, status := f.RunScorePlugins(pc, pod, nodes)

			if tt.err {
				if status.IsSuccess() {
					t.Error("Expected status to be non-success.")
				}
				return
			}

			if !status.IsSuccess() {
				t.Errorf("Expected status to be success.")
			}
			if !reflect.DeepEqual(res, tt.want) {
				t.Errorf("Score map after RunScorePlugin: %+v, want: %+v.", res, tt.want)
			}
		})
	}
}

func toRegistry(constructors map[string]func(injectedResult) Plugin, injectedRes map[string]injectedResult) Registry {
	registry := make(Registry)
	for pl, f := range constructors {
		npl := pl
		nf := f
		registry[pl] = func(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
			return nf(injectedRes[npl]), nil
		}
	}
	return registry
}

func buildConfigDefaultWeights(ps ...string) *config.Plugins {
	return buildConfigWithWeights(defaultWeights, ps...)
}

func buildConfigWithWeights(weights map[string]int32, ps ...string) *config.Plugins {
	var plugins []config.Plugin
	for _, p := range ps {
		plugins = append(plugins, config.Plugin{Name: p, Weight: weights[p]})
	}
	return &config.Plugins{Score: &config.PluginSet{Enabled: plugins}}
}

type injectedResult struct {
	scoreRes     int
	normalizeRes int
	scoreErr     bool
	normalizeErr bool
}

func setScoreRes(inj injectedResult) (int, *Status) {
	if inj.scoreErr {
		return 0, NewStatus(Error, "injecting failure.")
	}
	return inj.scoreRes, nil
}

func injectNormalizeRes(inj injectedResult, scores NodeScoreList) *Status {
	if inj.normalizeErr {
		return NewStatus(Error, "injecting failure.")
	}
	for i := range scores {
		scores[i].Score = inj.normalizeRes
	}
	return nil
}
