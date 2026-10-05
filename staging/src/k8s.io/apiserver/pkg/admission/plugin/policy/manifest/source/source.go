/*
Copyright The Kubernetes Authors.

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

// Package source provides a Source implementation that loads policy configurations from manifest files.
package source

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/util/filesystem"
	"k8s.io/klog/v2"
)

// defaultReloadInterval is the default interval at which the manifest directory is checked for changes.
var defaultReloadInterval = 1 * time.Minute

// SetReloadIntervalForTests sets the reload interval for testing and returns a function to restore the original value.
func SetReloadIntervalForTests(interval time.Duration) func() {
	original := defaultReloadInterval
	defaultReloadInterval = interval
	return func() {
		defaultReloadInterval = original
	}
}

// PolicyLoadFunc loads policy and binding manifests from a directory.
type PolicyLoadFunc[P, B runtime.Object] func(dir string) ([]P, []B, string, error)

// PolicyCompiler compiles a policy into an Evaluator.
type PolicyCompiler[P runtime.Object, E generic.Evaluator] func(P) (E, error)

// BindingPolicyName extracts the policy name referenced by a binding.
type BindingPolicyName[B runtime.Object] func(B) string

// StaticPolicySource provides policy configurations loaded from manifest files.
type StaticPolicySource[P, B runtime.Object, E generic.Evaluator] struct {
	manifestsDir         string
	apiServerID          string
	reloadInterval       time.Duration
	compiler             PolicyCompiler[P, E]
	loadFunc             PolicyLoadFunc[P, B]
	getBindingPolicyName BindingPolicyName[B]
	manifestType         metrics.ManifestType

	current      atomic.Pointer[[]generic.PolicyHook[P, B, E]]
	lastReadHash atomic.Pointer[string] // hash of last file content read (for short-circuiting)
	hasSynced    atomic.Bool
}

// NewStaticPolicySource creates a new static policy source that loads configurations from the specified directory.
func NewStaticPolicySource[P, B runtime.Object, E generic.Evaluator](
	manifestsDir, apiServerID string,
	compiler PolicyCompiler[P, E],
	loadFunc PolicyLoadFunc[P, B],
	getBindingPolicyName BindingPolicyName[B],
	manifestType metrics.ManifestType,
) *StaticPolicySource[P, B, E] {
	metrics.RegisterMetrics()
	return &StaticPolicySource[P, B, E]{
		manifestsDir:         manifestsDir,
		apiServerID:          apiServerID,
		reloadInterval:       defaultReloadInterval,
		compiler:             compiler,
		loadFunc:             loadFunc,
		getBindingPolicyName: getBindingPolicyName,
		manifestType:         manifestType,
	}
}

// LoadInitial performs the initial load of manifests.
// This should be called during API server startup and will fail if the manifests cannot be loaded.
func (s *StaticPolicySource[P, B, E]) LoadInitial() error {
	policies, bindings, hash, err := s.loadFunc(s.manifestsDir)
	if err != nil {
		return err
	}

	hooks, err := s.compile(policies, bindings)
	if err != nil {
		return err
	}
	s.current.Store(&hooks)
	s.lastReadHash.Store(&hash)
	s.hasSynced.Store(true)

	klog.InfoS("Loaded manifest-based admission policy configurations", "plugin", string(s.manifestType), "count", len(hooks))
	metrics.RecordAutomaticReloadSuccess(s.manifestType, s.apiServerID, hash)
	return nil
}

// RunReloadLoop watches for configuration changes and reloads when detected.
// It blocks until ctx is canceled.
func (s *StaticPolicySource[P, B, E]) RunReloadLoop(ctx context.Context) {
	filesystem.WatchUntil(
		ctx,
		s.reloadInterval,
		s.manifestsDir,
		func() {
			s.checkAndReload()
		},
		func(err error) {
			klog.ErrorS(err, "watching manifest directory", "plugin", string(s.manifestType), "dir", s.manifestsDir)
		},
	)
}

func (s *StaticPolicySource[P, B, E]) compile(policies []P, bindings []B) ([]generic.PolicyHook[P, B, E], error) {
	bindingsByPolicy := make(map[string][]B)
	for _, binding := range bindings {
		policyName := s.getBindingPolicyName(binding)
		bindingsByPolicy[policyName] = append(bindingsByPolicy[policyName], binding)
	}

	var hooks []generic.PolicyHook[P, B, E]
	for _, policy := range policies {
		nameGetter, ok := any(policy).(interface{ GetName() string })
		if !ok {
			return nil, fmt.Errorf("policy type %T does not implement GetName()", policy)
		}
		evaluator, err := s.compiler(policy)
		if err != nil {
			return nil, fmt.Errorf("failed to compile policy %q: %w", nameGetter.GetName(), err)
		}
		hook := generic.PolicyHook[P, B, E]{
			Policy:    policy,
			Bindings:  bindingsByPolicy[nameGetter.GetName()],
			Evaluator: evaluator,
		}
		hooks = append(hooks, hook)
	}

	return hooks, nil
}

func (s *StaticPolicySource[P, B, E]) checkAndReload() {
	policies, bindings, hash, err := s.loadFunc(s.manifestsDir)
	if err != nil {
		klog.ErrorS(err, "reloading manifest config", "plugin", string(s.manifestType), "dir", s.manifestsDir)
		metrics.RecordAutomaticReloadFailure(s.manifestType, s.apiServerID)
		return
	}

	// Short-circuit if file content hasn't changed since last read.
	if last := s.lastReadHash.Load(); last != nil && hash == *last {
		return
	}
	s.lastReadHash.Store(&hash)

	hooks, err := s.compile(policies, bindings)
	if err != nil {
		klog.ErrorS(err, "compiling manifest config", "plugin", string(s.manifestType), "dir", s.manifestsDir)
		metrics.RecordAutomaticReloadFailure(s.manifestType, s.apiServerID)
		return
	}
	s.current.Store(&hooks)

	klog.InfoS("reloaded manifest config", "plugin", string(s.manifestType), "dir", s.manifestsDir)
	metrics.RecordAutomaticReloadSuccess(s.manifestType, s.apiServerID, hash)
}

// HasSynced returns true if the initial load has completed.
func (s *StaticPolicySource[P, B, E]) HasSynced() bool {
	return s.hasSynced.Load()
}

// Hooks returns the list of policy hooks.
func (s *StaticPolicySource[P, B, E]) Hooks() []generic.PolicyHook[P, B, E] {
	current := s.current.Load()
	if current == nil {
		return nil
	}
	return *current
}
