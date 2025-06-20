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

// Package profile holds the definition of a scheduling Profile.
package profile

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

// RecorderFactory builds an EventRecorder for a given scheduler name.
type RecorderFactory func(string) events.EventRecorder

// newProfile builds a Profile for the given configuration.
func newProfile(ctx context.Context, cfg config.KubeSchedulerProfile, r frameworkruntime.Registry, recorderFact RecorderFactory,
	opts ...frameworkruntime.Option) (framework.Framework, error) {
	recorder := recorderFact(cfg.SchedulerName)
	opts = append(opts, frameworkruntime.WithEventRecorder(recorder))
	return frameworkruntime.NewFramework(ctx, r, &cfg, opts...)
}

// Map holds frameworks indexed by scheduler name.
type Map map[string]framework.Framework

// NewMap builds the frameworks given by the configuration, indexed by name.
func NewMap(ctx context.Context, cfgs []config.KubeSchedulerProfile, r frameworkruntime.Registry, recorderFact RecorderFactory,
	opts ...frameworkruntime.Option) (Map, error) {
	m := make(Map)
	v := cfgValidator{m: m}

	for _, cfg := range cfgs {
		p, err := newProfile(ctx, cfg, r, recorderFact, opts...)
		if err != nil {
			return nil, fmt.Errorf("creating profile for scheduler name %s: %v", cfg.SchedulerName, err)
		}
		if err := v.validate(cfg, p); err != nil {
			return nil, err
		}
		m[cfg.SchedulerName] = p
	}
	return m, nil
}

// HandlesSchedulerName returns whether a profile handles the given scheduler name.
func (m Map) HandlesSchedulerName(name string) bool {
	_, ok := m[name]
	return ok
}

// Close closes all frameworks registered in this map.
func (m Map) Close() error {
	var errs []error
	for name, f := range m {
		err := f.Close()
		if err != nil {
			errs = append(errs, fmt.Errorf("framework %s failed to close: %w", name, err))
		}
	}
	return errors.Join(errs...)
}

// NewRecorderFactory returns a RecorderFactory for the broadcaster.
func NewRecorderFactory(b events.EventBroadcaster) RecorderFactory {
	return func(name string) events.EventRecorder {
		return b.NewRecorder(scheme.Scheme, name)
	}
}

type cfgValidator struct {
	m             Map
	queueSort     string
	queueSortArgs runtime.Object
}

func (v *cfgValidator) validate(cfg config.KubeSchedulerProfile, f framework.Framework) error {
	if len(f.ProfileName()) == 0 {
		return errors.New("scheduler name is needed")
	}
	if cfg.Plugins == nil {
		return fmt.Errorf("plugins required for profile with scheduler name %q", f.ProfileName())
	}
	if v.m[f.ProfileName()] != nil {
		return fmt.Errorf("duplicate profile with scheduler name %q", f.ProfileName())
	}

	queueSort := f.ListPlugins().QueueSort.Enabled[0].Name
	var queueSortArgs runtime.Object
	for _, plCfg := range cfg.PluginConfig {
		if plCfg.Name == queueSort {
			queueSortArgs = plCfg.Args
			break
		}
	}
	if len(v.queueSort) == 0 {
		v.queueSort = queueSort
		v.queueSortArgs = queueSortArgs
		return nil
	}
	if v.queueSort != queueSort {
		return fmt.Errorf("different queue sort plugins for profile %q: %q, first: %q", cfg.SchedulerName, queueSort, v.queueSort)
	}
	if diff.Diff(v.queueSortArgs, queueSortArgs) != "" {
		return fmt.Errorf("different queue sort plugin args for profile %q", cfg.SchedulerName)
	}
	return nil
}
