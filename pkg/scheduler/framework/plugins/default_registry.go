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
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpodtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/imagelocality"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodepreferavoidpods"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodevolumelimits"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/requestedtocapacityratio"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

// RegistryArgs arguments needed to create default plugin factories.
type RegistryArgs struct {
	VolumeBinder *volumebinder.VolumeBinder
}

// NewDefaultRegistry builds the default registry with all the in-tree plugins.
// This is the registry that Kubernetes default scheduler uses. A scheduler that runs out of tree
// plugins can register additional plugins through the WithFrameworkOutOfTreeRegistry option.
func NewDefaultRegistry(args *RegistryArgs) framework.Registry {
	return framework.Registry{
		defaultpodtopologyspread.Name:        defaultpodtopologyspread.New,
		imagelocality.Name:                   imagelocality.New,
		tainttoleration.Name:                 tainttoleration.New,
		nodename.Name:                        nodename.New,
		nodeports.Name:                       nodeports.New,
		nodepreferavoidpods.Name:             nodepreferavoidpods.New,
		nodeaffinity.Name:                    nodeaffinity.New,
		podtopologyspread.Name:               podtopologyspread.New,
		nodeunschedulable.Name:               nodeunschedulable.New,
		noderesources.FitName:                noderesources.NewFit,
		noderesources.BalancedAllocationName: noderesources.NewBalancedAllocation,
		noderesources.MostAllocatedName:      noderesources.NewMostAllocated,
		noderesources.LeastAllocatedName:     noderesources.NewLeastAllocated,
		volumebinding.Name: func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
			return volumebinding.NewFromVolumeBinder(args.VolumeBinder), nil
		},
		volumerestrictions.Name:        volumerestrictions.New,
		volumezone.Name:                volumezone.New,
		nodevolumelimits.CSIName:       nodevolumelimits.NewCSI,
		nodevolumelimits.EBSName:       nodevolumelimits.NewEBS,
		nodevolumelimits.GCEPDName:     nodevolumelimits.NewGCEPD,
		nodevolumelimits.AzureDiskName: nodevolumelimits.NewAzureDisk,
		nodevolumelimits.CinderName:    nodevolumelimits.NewCinder,
		interpodaffinity.Name:          interpodaffinity.New,
		nodelabel.Name:                 nodelabel.New,
		requestedtocapacityratio.Name:  requestedtocapacityratio.New,
		serviceaffinity.Name:           serviceaffinity.New,
	}
}

// ConfigProducerArgs contains arguments that are passed to the producer.
// As we add more predicates/priorities to framework plugins mappings, more arguments
// may be added here.
type ConfigProducerArgs struct {
	// Weight used for priority functions.
	Weight int32
	// NodeLabelArgs is the args for the NodeLabel plugin.
	NodeLabelArgs *nodelabel.Args
	// RequestedToCapacityRatioArgs is the args for the RequestedToCapacityRatio plugin.
	RequestedToCapacityRatioArgs *requestedtocapacityratio.Args
	// ServiceAffinityArgs is the args for the ServiceAffinity plugin.
	ServiceAffinityArgs *serviceaffinity.Args
}

// ConfigProducer produces a framework's configuration.
type ConfigProducer func(args ConfigProducerArgs) (config.Plugins, []config.PluginConfig)

// ConfigProducerRegistry tracks mappings from predicates/priorities to framework config producers.
type ConfigProducerRegistry struct {
	// maps that associate predicates/priorities with framework plugin configurations.
	PredicateToConfigProducer map[string]ConfigProducer
	PriorityToConfigProducer  map[string]ConfigProducer
}

// NewDefaultConfigProducerRegistry creates a new producer registry.
func NewDefaultConfigProducerRegistry() *ConfigProducerRegistry {
	registry := &ConfigProducerRegistry{
		PredicateToConfigProducer: make(map[string]ConfigProducer),
		PriorityToConfigProducer:  make(map[string]ConfigProducer),
	}
	// Register Predicates.
	registry.RegisterPredicate(predicates.GeneralPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			// GeneralPredicate is a combination of predicates.
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodename.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeports.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeaffinity.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.PodToleratesNodeTaintsPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, tainttoleration.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.PodFitsResourcesPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			return
		})
	registry.RegisterPredicate(predicates.HostNamePred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodename.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.PodFitsHostPortsPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeports.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.MatchNodeSelectorPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeaffinity.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.CheckNodeUnschedulablePred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeunschedulable.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.CheckVolumeBindingPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumebinding.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.NoDiskConflictPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumerestrictions.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.NoVolumeZoneConflictPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumezone.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.MaxCSIVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.CSIName, nil)
			return
		})
	registry.RegisterPredicate(predicates.MaxEBSVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.EBSName, nil)
			return
		})
	registry.RegisterPredicate(predicates.MaxGCEPDVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.GCEPDName, nil)
			return
		})
	registry.RegisterPredicate(predicates.MaxAzureDiskVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.AzureDiskName, nil)
			return
		})
	registry.RegisterPredicate(predicates.MaxCinderVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.CinderName, nil)
			return
		})
	registry.RegisterPredicate(predicates.MatchInterPodAffinityPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, interpodaffinity.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.EvenPodsSpreadPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, podtopologyspread.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.CheckNodeLabelPresencePred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodelabel.Name, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(nodelabel.Name, args.NodeLabelArgs))
			return
		})
	registry.RegisterPredicate(predicates.CheckServiceAffinityPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, serviceaffinity.Name, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(serviceaffinity.Name, args.ServiceAffinityArgs))
			return
		})

	// Register Priorities.
	registry.RegisterPriority(priorities.SelectorSpreadPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, defaultpodtopologyspread.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.TaintTolerationPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, tainttoleration.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.NodeAffinityPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, nodeaffinity.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.ImageLocalityPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, imagelocality.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.InterPodAffinityPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, interpodaffinity.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.NodePreferAvoidPodsPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, nodepreferavoidpods.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.MostRequestedPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.MostAllocatedName, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.BalancedResourceAllocation,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.BalancedAllocationName, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.LeastRequestedPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.LeastAllocatedName, &args.Weight)
			return
		})
	registry.RegisterPriority(priorities.EvenPodsSpreadPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, podtopologyspread.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(requestedtocapacityratio.Name,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, requestedtocapacityratio.Name, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(requestedtocapacityratio.Name, args.RequestedToCapacityRatioArgs))
			return
		})

	registry.RegisterPriority(nodelabel.Name,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, nodelabel.Name, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(nodelabel.Name, args.NodeLabelArgs))
			return
		})
	registry.RegisterPriority(serviceaffinity.Name,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, serviceaffinity.Name, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(serviceaffinity.Name, args.ServiceAffinityArgs))
			return
		})
	return registry
}

func registerProducer(name string, producer ConfigProducer, producersMap map[string]ConfigProducer) error {
	if _, exist := producersMap[name]; exist {
		return fmt.Errorf("already registered %q", name)
	}
	producersMap[name] = producer
	return nil
}

// RegisterPredicate registers a config producer for a predicate.
func (f *ConfigProducerRegistry) RegisterPredicate(name string, producer ConfigProducer) error {
	return registerProducer(name, producer, f.PredicateToConfigProducer)
}

// RegisterPriority registers a framework config producer for a priority.
func (f *ConfigProducerRegistry) RegisterPriority(name string, producer ConfigProducer) error {
	return registerProducer(name, producer, f.PriorityToConfigProducer)
}

func appendToPluginSet(set *config.PluginSet, name string, weight *int32) *config.PluginSet {
	if set == nil {
		set = &config.PluginSet{}
	}
	cfg := config.Plugin{Name: name}
	if weight != nil {
		cfg.Weight = *weight
	}
	set.Enabled = append(set.Enabled, cfg)
	return set
}

func makePluginConfig(pluginName string, args interface{}) config.PluginConfig {
	encoding, err := json.Marshal(args)
	if err != nil {
		klog.Fatal(fmt.Errorf("Failed to marshal %+v: %v", args, err))
		return config.PluginConfig{}
	}
	config := config.PluginConfig{
		Name: pluginName,
		Args: runtime.Unknown{Raw: encoding},
	}
	return config
}
