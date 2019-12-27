/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// LegacyRegistry is used to store current state of registered predicates and priorities.
type LegacyRegistry struct {
	Predicates             sets.String
	Priorities             map[string]int64
	MandatoryPredicates    sets.String
	DefaultPredicates      sets.String
	DefaultPriorities      sets.String
	ConfigProducerRegistry *ConfigProducerRegistry
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
	RequestedToCapacityRatioArgs *noderesources.RequestedToCapacityRatioArgs
	// ServiceAffinityArgs is the args for the ServiceAffinity plugin.
	ServiceAffinityArgs *serviceaffinity.Args
	// NodeResourcesFitArgs is the args for the NodeResources fit filter.
	NodeResourcesFitArgs *noderesources.FitArgs
	// InterPodAffinityArgs is the args for InterPodAffinity plugin
	InterPodAffinityArgs *interpodaffinity.Args
}

// ConfigProducer produces a framework's configuration.
type ConfigProducer func(args ConfigProducerArgs) (config.Plugins, []config.PluginConfig)

// ConfigProducerRegistry tracks mappings from predicates/priorities to framework config producers.
type ConfigProducerRegistry struct {
	// maps that associate predicates/priorities with framework plugin configurations.
	PredicateToConfigProducer map[string]ConfigProducer
	PriorityToConfigProducer  map[string]ConfigProducer
}

// newConfigProducerRegistry creates a new producer registry.
func newConfigProducerRegistry() *ConfigProducerRegistry {
	registry := &ConfigProducerRegistry{
		PredicateToConfigProducer: make(map[string]ConfigProducer),
		PriorityToConfigProducer:  make(map[string]ConfigProducer),
	}
	// Register Predicates.
	registry.RegisterPredicate(predicates.GeneralPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			// GeneralPredicate is a combination of predicates.
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, noderesources.FitName, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(noderesources.FitName, args.NodeResourcesFitArgs))
			plugins.Filter = appendToPluginSet(plugins.Filter, nodename.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeports.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeports.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeaffinity.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.PodToleratesNodeTaintsPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, tainttoleration.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.PodFitsResourcesPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, noderesources.FitName, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(noderesources.FitName, args.NodeResourcesFitArgs))
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
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeports.Name, nil)
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
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, interpodaffinity.Name, nil)
			return
		})
	registry.RegisterPredicate(predicates.EvenPodsSpreadPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, podtopologyspread.Name, nil)
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
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, serviceaffinity.Name, nil)
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
			plugins.PostFilter = appendToPluginSet(plugins.PostFilter, tainttoleration.Name, nil)
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
			plugins.PostFilter = appendToPluginSet(plugins.PostFilter, interpodaffinity.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, interpodaffinity.Name, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(interpodaffinity.Name, args.InterPodAffinityArgs))
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
			plugins.PostFilter = appendToPluginSet(plugins.PostFilter, podtopologyspread.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, podtopologyspread.Name, &args.Weight)
			return
		})
	registry.RegisterPriority(noderesources.RequestedToCapacityRatioName,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.RequestedToCapacityRatioName, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(noderesources.RequestedToCapacityRatioName, args.RequestedToCapacityRatioArgs))
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
	registry.RegisterPriority(priorities.ResourceLimitsPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.PostFilter = appendToPluginSet(plugins.PostFilter, noderesources.ResourceLimitsName, nil)
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.ResourceLimitsName, &args.Weight)
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

// NewLegacyRegistry returns a legacy algorithm registry of predicates and priorities.
func NewLegacyRegistry() *LegacyRegistry {
	registry := &LegacyRegistry{
		// predicate keys supported for backward compatibility with v1.Policy.
		Predicates: sets.NewString(
			"PodFitsPorts", // This exists for compatibility reasons.
			predicates.PodFitsHostPortsPred,
			predicates.PodFitsResourcesPred,
			predicates.HostNamePred,
			predicates.MatchNodeSelectorPred,
			predicates.NoVolumeZoneConflictPred,
			predicates.MaxEBSVolumeCountPred,
			predicates.MaxGCEPDVolumeCountPred,
			predicates.MaxAzureDiskVolumeCountPred,
			predicates.MaxCSIVolumeCountPred,
			predicates.MaxCinderVolumeCountPred,
			predicates.MatchInterPodAffinityPred,
			predicates.NoDiskConflictPred,
			predicates.GeneralPred,
			predicates.PodToleratesNodeTaintsPred,
			predicates.CheckNodeUnschedulablePred,
			predicates.CheckVolumeBindingPred,
		),

		// priority keys to weights, this exist for backward compatibility with v1.Policy.
		Priorities: map[string]int64{
			priorities.LeastRequestedPriority:      1,
			priorities.BalancedResourceAllocation:  1,
			priorities.MostRequestedPriority:       1,
			priorities.ImageLocalityPriority:       1,
			priorities.NodeAffinityPriority:        1,
			priorities.SelectorSpreadPriority:      1,
			priorities.ServiceSpreadingPriority:    1,
			priorities.TaintTolerationPriority:     1,
			priorities.InterPodAffinityPriority:    1,
			priorities.NodePreferAvoidPodsPriority: 10000,
		},

		// MandatoryPredicates the set of keys for predicates that the scheduler will
		// be configured with all the time.
		MandatoryPredicates: sets.NewString(
			predicates.PodToleratesNodeTaintsPred,
			predicates.CheckNodeUnschedulablePred,
		),

		DefaultPredicates: sets.NewString(
			predicates.NoVolumeZoneConflictPred,
			predicates.MaxEBSVolumeCountPred,
			predicates.MaxGCEPDVolumeCountPred,
			predicates.MaxAzureDiskVolumeCountPred,
			predicates.MaxCSIVolumeCountPred,
			predicates.MatchInterPodAffinityPred,
			predicates.NoDiskConflictPred,
			predicates.GeneralPred,
			predicates.PodToleratesNodeTaintsPred,
			predicates.CheckVolumeBindingPred,
			predicates.CheckNodeUnschedulablePred,
		),

		DefaultPriorities: sets.NewString(
			priorities.SelectorSpreadPriority,
			priorities.InterPodAffinityPriority,
			priorities.LeastRequestedPriority,
			priorities.BalancedResourceAllocation,
			priorities.NodePreferAvoidPodsPriority,
			priorities.NodeAffinityPriority,
			priorities.TaintTolerationPriority,
			priorities.ImageLocalityPriority,
		),

		ConfigProducerRegistry: newConfigProducerRegistry(),
	}

	// The following two features are the last ones to be supported as predicate/priority.
	// Once they graduate to GA, there will be no more checking for featue gates here.
	// Only register EvenPodsSpread predicate & priority if the feature is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.EvenPodsSpread) {
		klog.Infof("Registering EvenPodsSpread predicate and priority function")

		registry.Predicates.Insert(predicates.EvenPodsSpreadPred)
		registry.DefaultPredicates.Insert(predicates.EvenPodsSpreadPred)

		registry.Priorities[priorities.EvenPodsSpreadPriority] = 1
		registry.DefaultPriorities.Insert(priorities.EvenPodsSpreadPriority)
	}

	// Prioritizes nodes that satisfy pod's resource limits
	if utilfeature.DefaultFeatureGate.Enabled(features.ResourceLimitsPriorityFunction) {
		klog.Infof("Registering resourcelimits priority function")

		registry.Priorities[priorities.ResourceLimitsPriority] = 1
		registry.DefaultPriorities.Insert(priorities.ResourceLimitsPriority)
	}

	return registry
}

// RegisterCustomPredicate registers a custom fit predicate with the algorithm registry.
// Returns the name, with which the predicate was registered.
func (a *LegacyRegistry) RegisterCustomPredicate(policy config.PredicatePolicy, pluginArgs *ConfigProducerArgs) string {
	var ok bool
	var predicate string

	validatePredicateOrDie(policy)

	// generate the predicate function, if a custom type is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAffinity != nil {
			// We use the ServiceAffinity predicate name for all ServiceAffinity custom predicates.
			// It may get called multiple times but we essentially only register one instance of ServiceAffinity predicate.
			// This name is then used to find the registered plugin and run the plugin instead of the predicate.
			predicate = predicates.CheckServiceAffinityPred

			// map LabelsPresence policy to ConfigProducerArgs that's used to configure the ServiceAffinity plugin.
			if pluginArgs.ServiceAffinityArgs == nil {
				pluginArgs.ServiceAffinityArgs = &serviceaffinity.Args{}
			}

			pluginArgs.ServiceAffinityArgs.AffinityLabels = append(pluginArgs.ServiceAffinityArgs.AffinityLabels, policy.Argument.ServiceAffinity.Labels...)
		} else if policy.Argument.LabelsPresence != nil {
			// We use the CheckNodeLabelPresencePred predicate name for all kNodeLabel custom predicates.
			// It may get called multiple times but we essentially only register one instance of NodeLabel predicate.
			// This name is then used to find the registered plugin and run the plugin instead of the predicate.
			predicate = predicates.CheckNodeLabelPresencePred

			// Map LabelPresence policy to ConfigProducerArgs that's used to configure the NodeLabel plugin.
			if pluginArgs.NodeLabelArgs == nil {
				pluginArgs.NodeLabelArgs = &nodelabel.Args{}
			}
			if policy.Argument.LabelsPresence.Presence {
				pluginArgs.NodeLabelArgs.PresentLabels = append(pluginArgs.NodeLabelArgs.PresentLabels, policy.Argument.LabelsPresence.Labels...)
			} else {
				pluginArgs.NodeLabelArgs.AbsentLabels = append(pluginArgs.NodeLabelArgs.AbsentLabels, policy.Argument.LabelsPresence.Labels...)
			}
		}
	} else if _, ok = a.Predicates[policy.Name]; ok {
		// checking to see if a pre-defined predicate is requested
		klog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
		return policy.Name
	}

	if len(predicate) == 0 {
		klog.Fatalf("Invalid configuration: Predicate type not found for %s", policy.Name)
	}

	return predicate
}

// RegisterCustomPriority registers a custom priority with the algorithm registry.
// Returns the name, with which the priority function was registered.
func (a *LegacyRegistry) RegisterCustomPriority(policy config.PriorityPolicy, configProducerArgs *ConfigProducerArgs) string {
	var priority string
	var weight int64

	validatePriorityOrDie(policy)

	// generate the priority function, if a custom priority is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAntiAffinity != nil {
			// We use the ServiceAffinity plugin name for all ServiceAffinity custom priorities.
			// It may get called multiple times but we essentially only register one instance of
			// ServiceAffinity priority.
			// This name is then used to find the registered plugin and run the plugin instead of the priority.
			priority = serviceaffinity.Name

			if configProducerArgs.ServiceAffinityArgs == nil {
				configProducerArgs.ServiceAffinityArgs = &serviceaffinity.Args{}
			}
			configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference = append(configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference, policy.Argument.ServiceAntiAffinity.Label)

			weight = policy.Weight
			if existingWeight, ok := a.Priorities[priority]; ok {
				// If there are n ServiceAffinity priorities in the policy, the weight for the corresponding
				// score plugin is n*(weight of each priority).
				weight += existingWeight
			}
		} else if policy.Argument.LabelPreference != nil {
			// We use the NodeLabel plugin name for all NodeLabel custom priorities.
			// It may get called multiple times but we essentially only register one instance of NodeLabel priority.
			// This name is then used to find the registered plugin and run the plugin instead of the priority.
			priority = nodelabel.Name
			if configProducerArgs.NodeLabelArgs == nil {
				configProducerArgs.NodeLabelArgs = &nodelabel.Args{}
			}
			if policy.Argument.LabelPreference.Presence {
				configProducerArgs.NodeLabelArgs.PresentLabelsPreference = append(configProducerArgs.NodeLabelArgs.PresentLabelsPreference, policy.Argument.LabelPreference.Label)
			} else {
				configProducerArgs.NodeLabelArgs.AbsentLabelsPreference = append(configProducerArgs.NodeLabelArgs.AbsentLabelsPreference, policy.Argument.LabelPreference.Label)
			}
			weight = policy.Weight
			if existingWeight, ok := a.Priorities[priority]; ok {
				// If there are n NodeLabel priority configured in the policy, the weight for the corresponding
				// priority is n*(weight of each priority in policy).
				weight += existingWeight
			}
		} else if policy.Argument.RequestedToCapacityRatioArguments != nil {
			scoringFunctionShape, resources := buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(policy.Argument.RequestedToCapacityRatioArguments)
			configProducerArgs.RequestedToCapacityRatioArgs = &noderesources.RequestedToCapacityRatioArgs{
				FunctionShape:       scoringFunctionShape,
				ResourceToWeightMap: resources,
			}
			// We do not allow specifying the name for custom plugins, see #83472
			priority = noderesources.RequestedToCapacityRatioName
			weight = policy.Weight
		}
	} else if _, ok := a.Priorities[policy.Name]; ok {
		klog.V(2).Infof("Priority type %s already registered, reusing.", policy.Name)
		// set/update the weight based on the policy
		priority = policy.Name
		weight = policy.Weight
	}

	if len(priority) == 0 {
		klog.Fatalf("Invalid configuration: Priority type not found for %s", policy.Name)
	}

	a.Priorities[priority] = weight
	return priority
}

func buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(arguments *config.RequestedToCapacityRatioArguments) (noderesources.FunctionShape, noderesources.ResourceToWeightMap) {
	n := len(arguments.Shape)
	points := make([]noderesources.FunctionShapePoint, 0, n)
	for _, point := range arguments.Shape {
		points = append(points, noderesources.FunctionShapePoint{
			Utilization: int64(point.Utilization),
			// MaxCustomPriorityScore may diverge from the max score used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by requested to capacity ratio to the score range
			// used by the scheduler.
			Score: int64(point.Score) * (framework.MaxNodeScore / config.MaxCustomPriorityScore),
		})
	}
	shape, err := noderesources.NewFunctionShape(points)
	if err != nil {
		klog.Fatalf("invalid RequestedToCapacityRatioPriority arguments: %s", err.Error())
	}
	resourceToWeightMap := make(noderesources.ResourceToWeightMap, 0)
	if len(arguments.Resources) == 0 {
		resourceToWeightMap = noderesources.DefaultRequestedRatioResources
		return shape, resourceToWeightMap
	}
	for _, resource := range arguments.Resources {
		resourceToWeightMap[v1.ResourceName(resource.Name)] = resource.Weight
		if resource.Weight == 0 {
			resourceToWeightMap[v1.ResourceName(resource.Name)] = 1
		}
	}
	return shape, resourceToWeightMap
}

func validatePredicateOrDie(predicate config.PredicatePolicy) {
	if predicate.Argument != nil {
		numArgs := 0
		if predicate.Argument.ServiceAffinity != nil {
			numArgs++
		}
		if predicate.Argument.LabelsPresence != nil {
			numArgs++
		}
		if numArgs != 1 {
			klog.Fatalf("Exactly 1 predicate argument is required, numArgs: %v, Predicate: %s", numArgs, predicate.Name)
		}
	}
}

func validatePriorityOrDie(priority config.PriorityPolicy) {
	if priority.Argument != nil {
		numArgs := 0
		if priority.Argument.ServiceAntiAffinity != nil {
			numArgs++
		}
		if priority.Argument.LabelPreference != nil {
			numArgs++
		}
		if priority.Argument.RequestedToCapacityRatioArguments != nil {
			numArgs++
		}
		if numArgs != 1 {
			klog.Fatalf("Exactly 1 priority argument is required, numArgs: %v, Priority: %s", numArgs, priority.Name)
		}
	}
}
