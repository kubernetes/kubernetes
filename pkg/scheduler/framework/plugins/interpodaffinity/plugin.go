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
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	"k8s.io/utils/pointer"
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

// Args holds the args that are used to configure the plugin.
type Args struct {
	// HardPodAffinityWeight is the scoring weight for existing pods with a
	// matching hard affinity to the incoming pod.
	HardPodAffinityWeight *int32 `json:"hardPodAffinityWeight,omitempty"`
}

var _ framework.PreFilterPlugin = &InterPodAffinity{}
var _ framework.FilterPlugin = &InterPodAffinity{}
var _ framework.PreScorePlugin = &InterPodAffinity{}
var _ framework.ScorePlugin = &InterPodAffinity{}

// InterPodAffinity is a plugin that checks inter pod affinity
type InterPodAffinity struct {
	Args
	sharedLister schedulerlisters.SharedLister
	sync.Mutex
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *InterPodAffinity) Name() string {
	return Name
}

// BuildArgs returns the args that were used to build the plugin.
func (pl *InterPodAffinity) BuildArgs() interface{} {
	return pl.Args
}

// New initializes a new plugin and returns it.
func New(plArgs *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	pl := &InterPodAffinity{
		sharedLister: h.SnapshotSharedLister(),
	}
	if err := framework.DecodeInto(plArgs, &pl.Args); err != nil {
		return nil, err
	}
	if err := validateArgs(&pl.Args); err != nil {
		return nil, err
	}
	if pl.HardPodAffinityWeight == nil {
		pl.HardPodAffinityWeight = pointer.Int32Ptr(DefaultHardPodAffinityWeight)
	}
	return pl, nil
}

func validateArgs(args *Args) error {
	if args.HardPodAffinityWeight == nil {
		return nil
	}
	return ValidateHardPodAffinityWeight(field.NewPath("hardPodAffinityWeight"), *args.HardPodAffinityWeight)
}

// ValidateHardPodAffinityWeight validates that weight is within allowed range.
func ValidateHardPodAffinityWeight(path *field.Path, w int32) error {
	if w < MinHardPodAffinityWeight || w > MaxHardPodAffinityWeight {
		msg := fmt.Sprintf("not in valid range [%d-%d]", MinHardPodAffinityWeight, MaxHardPodAffinityWeight)
		return field.Invalid(path, w, msg)
	}
	return nil
}
