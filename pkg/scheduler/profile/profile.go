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
	"errors"
	"fmt"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// RecorderFactory builds an EventRecorder for a given scheduler name.
type RecorderFactory func(string) events.EventRecorder

// FrameworkFactory builds a Framework for a given profile configuration.
type FrameworkFactory func(config.KubeSchedulerProfile, ...frameworkruntime.Option) (framework.Framework, error)

// Profile is a scheduling profile.
type Profile struct {
	framework.Framework
	Recorder events.EventRecorder
	Name     string
}

// NewProfile builds a Profile for the given configuration.
func NewProfile(cfg config.KubeSchedulerProfile, frameworkFact FrameworkFactory, recorderFact RecorderFactory,
	opts ...frameworkruntime.Option) (*Profile, error) {
	recorder := recorderFact(cfg.SchedulerName)
	opts = append(opts, frameworkruntime.WithEventRecorder(recorder), frameworkruntime.WithProfileName(cfg.SchedulerName))
	fwk, err := frameworkFact(cfg, opts...)
	if err != nil {
		return nil, err
	}
	return &Profile{
		Name:      cfg.SchedulerName,
		Framework: fwk,
		Recorder:  recorder,
	}, nil
}

// Map holds profiles indexed by scheduler name.
type Map map[string]*Profile

// NewMap builds the profiles given by the configuration, indexed by name.
func NewMap(cfgs []config.KubeSchedulerProfile, frameworkFact FrameworkFactory, recorderFact RecorderFactory,
	opts ...frameworkruntime.Option) (Map, error) {
	m := make(Map)
	v := cfgValidator{m: m}

	for _, cfg := range cfgs {
		if err := v.validate(cfg); err != nil {
			return nil, err
		}
		p, err := NewProfile(cfg, frameworkFact, recorderFact, opts...)
		if err != nil {
			return nil, fmt.Errorf("creating profile for scheduler name %s: %v", cfg.SchedulerName, err)
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

func (v *cfgValidator) validate(cfg config.KubeSchedulerProfile) error {
	if len(cfg.SchedulerName) == 0 {
		return errors.New("scheduler name is needed")
	}
	if cfg.Plugins == nil {
		return fmt.Errorf("plugins required for profile with scheduler name %q", cfg.SchedulerName)
	}
	if v.m[cfg.SchedulerName] != nil {
		return fmt.Errorf("duplicate profile with scheduler name %q", cfg.SchedulerName)
	}
	if cfg.Plugins.QueueSort == nil || len(cfg.Plugins.QueueSort.Enabled) != 1 {
		return fmt.Errorf("one queue sort plugin required for profile with scheduler name %q", cfg.SchedulerName)
	}
	queueSort := cfg.Plugins.QueueSort.Enabled[0].Name
	var queueSortArgs runtime.Object
	for _, plCfg := range cfg.PluginConfig {
		if plCfg.Name == queueSort {
			queueSortArgs = plCfg.Args
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
	if !cmp.Equal(v.queueSortArgs, queueSortArgs) {
		return fmt.Errorf("different queue sort plugin args for profile %q", cfg.SchedulerName)
	}
	return nil
}
