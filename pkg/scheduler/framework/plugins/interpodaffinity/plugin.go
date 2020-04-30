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

package interpodaffinity

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "InterPodAffinity"

	// DefaultHardPodAffinityWeight is the default HardPodAffinityWeight.
	DefaultHardPodAffinityWeight int32 = 1
	// MinHardPodAffinityWeight is the minimum HardPodAffinityWeight.
	MinHardPodAffinityWeight int32 = 0
	// MaxHardPodAffinityWeight is the maximum HardPodAffinityWeight.
	MaxHardPodAffinityWeight int32 = 100
)

var _ framework.PreFilterPlugin = &InterPodAffinity{}
var _ framework.FilterPlugin = &InterPodAffinity{}
var _ framework.PreScorePlugin = &InterPodAffinity{}
var _ framework.ScorePlugin = &InterPodAffinity{}

// InterPodAffinity is a plugin that checks inter pod affinity
type InterPodAffinity struct {
	args         config.InterPodAffinityArgs
	sharedLister framework.SharedLister
	sync.Mutex
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *InterPodAffinity) Name() string {
	return Name
}

// BuildArgs returns the args that were used to build the plugin.
func (pl *InterPodAffinity) BuildArgs() interface{} {
	return pl.args
}

// New initializes a new plugin and returns it.
func New(plArgs runtime.Object, h framework.FrameworkHandle) (framework.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	args, err := getArgs(plArgs)
	if err != nil {
		return nil, err
	}
	if err := ValidateHardPodAffinityWeight(field.NewPath("hardPodAffinityWeight"), args.HardPodAffinityWeight); err != nil {
		return nil, err
	}
	return &InterPodAffinity{
		args:         args,
		sharedLister: h.SnapshotSharedLister(),
	}, nil
}

func getArgs(obj runtime.Object) (config.InterPodAffinityArgs, error) {
	if obj == nil {
		return config.InterPodAffinityArgs{
			HardPodAffinityWeight: DefaultHardPodAffinityWeight,
		}, nil
	}
	ptr, ok := obj.(*config.InterPodAffinityArgs)
	if !ok {
		return config.InterPodAffinityArgs{}, fmt.Errorf("want args to be of type InterPodAffinityArgs, got %T", obj)
	}
	return *ptr, nil
}

// ValidateHardPodAffinityWeight validates that weight is within allowed range.
func ValidateHardPodAffinityWeight(path *field.Path, w int32) error {
	if w < MinHardPodAffinityWeight || w > MaxHardPodAffinityWeight {
		msg := fmt.Sprintf("not in valid range [%d-%d]", MinHardPodAffinityWeight, MaxHardPodAffinityWeight)
		return field.Invalid(path, w, msg)
	}
	return nil
}
