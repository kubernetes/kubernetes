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

package podtopologyspread

import (
	"fmt"

	"k8s.io/api/core/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/informers"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
)

const (
	// ErrReasonConstraintsNotMatch is used for PodTopologySpread filter error.
	ErrReasonConstraintsNotMatch = "node(s) didn't match pod topology spread constraints"
)

var (
	supportedScheduleActions = sets.NewString(string(v1.DoNotSchedule), string(v1.ScheduleAnyway))
)

// Args holds the arguments to configure the plugin.
type Args struct {
	// DefaultConstraints defines topology spread constraints to be applied to
	// pods that don't define any in `pod.spec.topologySpreadConstraints`.
	// `topologySpreadConstraint.labelSelectors` must be empty, as they are
	// deduced the pods' membership to Services, Replication Controllers, Replica
	// Sets or Stateful Sets.
	// Empty by default.
	// +optional
	// +listType=atomic
	DefaultConstraints []v1.TopologySpreadConstraint `json:"defaultConstraints"`
}

// PodTopologySpread is a plugin that ensures pod's topologySpreadConstraints is satisfied.
type PodTopologySpread struct {
	Args
	sharedLister     schedulerlisters.SharedLister
	services         corelisters.ServiceLister
	replicationCtrls corelisters.ReplicationControllerLister
	replicaSets      appslisters.ReplicaSetLister
	statefulSets     appslisters.StatefulSetLister
}

var _ framework.PreFilterPlugin = &PodTopologySpread{}
var _ framework.FilterPlugin = &PodTopologySpread{}
var _ framework.PreScorePlugin = &PodTopologySpread{}
var _ framework.ScorePlugin = &PodTopologySpread{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "PodTopologySpread"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *PodTopologySpread) Name() string {
	return Name
}

// BuildArgs returns the arguments used to build the plugin.
func (pl *PodTopologySpread) BuildArgs() interface{} {
	return pl.Args
}

// New initializes a new plugin and returns it.
func New(args *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	pl := &PodTopologySpread{sharedLister: h.SnapshotSharedLister()}
	if err := framework.DecodeInto(args, &pl.Args); err != nil {
		return nil, err
	}
	if err := validateArgs(&pl.Args); err != nil {
		return nil, err
	}
	if len(pl.DefaultConstraints) != 0 {
		if h.SharedInformerFactory() == nil {
			return nil, fmt.Errorf("SharedInformerFactory is nil")
		}
		pl.setListers(h.SharedInformerFactory())
	}
	return pl, nil
}

func (pl *PodTopologySpread) setListers(factory informers.SharedInformerFactory) {
	pl.services = factory.Core().V1().Services().Lister()
	pl.replicationCtrls = factory.Core().V1().ReplicationControllers().Lister()
	pl.replicaSets = factory.Apps().V1().ReplicaSets().Lister()
	pl.statefulSets = factory.Apps().V1().StatefulSets().Lister()
}

// validateArgs replicates the validation from
// pkg/apis/core/validation.validateTopologySpreadConstraints.
// This has the additional check for .labelSelector to be nil.
func validateArgs(args *Args) error {
	var allErrs field.ErrorList
	path := field.NewPath("defaultConstraints")
	for i, c := range args.DefaultConstraints {
		p := path.Index(i)
		if c.MaxSkew <= 0 {
			f := p.Child("maxSkew")
			allErrs = append(allErrs, field.Invalid(f, c.MaxSkew, "must be greater than zero"))
		}
		allErrs = append(allErrs, validateTopologyKey(p.Child("topologyKey"), c.TopologyKey)...)
		if err := validateWhenUnsatisfiable(p.Child("whenUnsatisfiable"), c.WhenUnsatisfiable); err != nil {
			allErrs = append(allErrs, err)
		}
		if c.LabelSelector != nil {
			f := field.Forbidden(p.Child("labelSelector"), "constraint must not define a selector, as they deduced for each pod")
			allErrs = append(allErrs, f)
		}
		if err := validateConstraintNotRepeat(path, args.DefaultConstraints, i); err != nil {
			allErrs = append(allErrs, err)
		}
	}
	if len(allErrs) == 0 {
		return nil
	}
	return allErrs.ToAggregate()
}

func validateTopologyKey(p *field.Path, v string) field.ErrorList {
	var allErrs field.ErrorList
	if len(v) == 0 {
		allErrs = append(allErrs, field.Required(p, "can not be empty"))
	} else {
		allErrs = append(allErrs, metav1validation.ValidateLabelName(v, p)...)
	}
	return allErrs
}

func validateWhenUnsatisfiable(p *field.Path, v v1.UnsatisfiableConstraintAction) *field.Error {
	if len(v) == 0 {
		return field.Required(p, "can not be empty")
	}
	if !supportedScheduleActions.Has(string(v)) {
		return field.NotSupported(p, v, supportedScheduleActions.List())
	}
	return nil
}

func validateConstraintNotRepeat(path *field.Path, constraints []v1.TopologySpreadConstraint, idx int) *field.Error {
	c := &constraints[idx]
	for i := range constraints[:idx] {
		other := &constraints[i]
		if c.TopologyKey == other.TopologyKey && c.WhenUnsatisfiable == other.WhenUnsatisfiable {
			return field.Duplicate(path.Index(idx), fmt.Sprintf("{%v, %v}", c.TopologyKey, c.WhenUnsatisfiable))
		}
	}
	return nil
}
