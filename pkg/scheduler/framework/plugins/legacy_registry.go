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
	// maps that associate predicates/priorities with framework plugin configurations.
	PredicateToConfigProducer map[string]ConfigProducer
	PriorityToConfigProducer  map[string]ConfigProducer
	// predicates that will always be configured.
	MandatoryPredicates sets.String
	// predicates and priorities that will be used if either was set to nil in a
	// given v1.Policy configuration.
	DefaultPredicates sets.String
	DefaultPriorities map[string]int64
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

// ConfigProducer returns the set of plugins and their configuration for a
// predicate/priority given the args.
type ConfigProducer func(args ConfigProducerArgs) (config.Plugins, []config.PluginConfig)

// NewLegacyRegistry returns a legacy algorithm registry of predicates and priorities.
func NewLegacyRegistry() *LegacyRegistry {
	registry := &LegacyRegistry{
		// MandatoryPredicates the set of keys for predicates that the scheduler will
		// be configured with all the time.
		MandatoryPredicates: sets.NewString(
			predicates.PodToleratesNodeTaintsPred,
			predicates.CheckNodeUnschedulablePred,
		),

		// Used as the default set of predicates if Policy was specified, but predicates was nil.
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

		// Used as the default set of predicates if Policy was specified, but priorities was nil.
		DefaultPriorities: map[string]int64{
			priorities.SelectorSpreadPriority:      1,
			priorities.InterPodAffinityPriority:    1,
			priorities.LeastRequestedPriority:      1,
			priorities.BalancedResourceAllocation:  1,
			priorities.NodePreferAvoidPodsPriority: 10000,
			priorities.NodeAffinityPriority:        1,
			priorities.TaintTolerationPriority:     1,
			priorities.ImageLocalityPriority:       1,
		},

		PredicateToConfigProducer: make(map[string]ConfigProducer),
		PriorityToConfigProducer:  make(map[string]ConfigProducer),
	}

	registry.registerPredicateConfigProducer(predicates.GeneralPred,
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
	registry.registerPredicateConfigProducer(predicates.PodToleratesNodeTaintsPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, tainttoleration.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.PodFitsResourcesPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, noderesources.FitName, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(noderesources.FitName, args.NodeResourcesFitArgs))
			return
		})
	registry.registerPredicateConfigProducer(predicates.HostNamePred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodename.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.PodFitsHostPortsPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeports.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeports.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MatchNodeSelectorPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeaffinity.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.CheckNodeUnschedulablePred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeunschedulable.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.CheckVolumeBindingPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumebinding.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.NoDiskConflictPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumerestrictions.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.NoVolumeZoneConflictPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumezone.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MaxCSIVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.CSIName, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MaxEBSVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.EBSName, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MaxGCEPDVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.GCEPDName, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MaxAzureDiskVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.AzureDiskName, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MaxCinderVolumeCountPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.CinderName, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.MatchInterPodAffinityPred,
		func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, interpodaffinity.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, interpodaffinity.Name, nil)
			return
		})
	registry.registerPredicateConfigProducer(predicates.CheckNodeLabelPresencePred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodelabel.Name, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(nodelabel.Name, args.NodeLabelArgs))
			return
		})
	registry.registerPredicateConfigProducer(predicates.CheckServiceAffinityPred,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, serviceaffinity.Name, nil)
			pluginConfig = append(pluginConfig, makePluginConfig(serviceaffinity.Name, args.ServiceAffinityArgs))
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, serviceaffinity.Name, nil)
			return
		})

	// Register Priorities.
	registry.registerPriorityConfigProducer(priorities.SelectorSpreadPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, defaultpodtopologyspread.Name, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.TaintTolerationPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.PostFilter = appendToPluginSet(plugins.PostFilter, tainttoleration.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, tainttoleration.Name, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.NodeAffinityPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, nodeaffinity.Name, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.ImageLocalityPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, imagelocality.Name, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.InterPodAffinityPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.PostFilter = appendToPluginSet(plugins.PostFilter, interpodaffinity.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, interpodaffinity.Name, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(interpodaffinity.Name, args.InterPodAffinityArgs))
			return
		})
	registry.registerPriorityConfigProducer(priorities.NodePreferAvoidPodsPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, nodepreferavoidpods.Name, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.MostRequestedPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.MostAllocatedName, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.BalancedResourceAllocation,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.BalancedAllocationName, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(priorities.LeastRequestedPriority,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.LeastAllocatedName, &args.Weight)
			return
		})
	registry.registerPriorityConfigProducer(noderesources.RequestedToCapacityRatioName,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.RequestedToCapacityRatioName, &args.Weight)
			pluginConfig = append(pluginConfig, makePluginConfig(noderesources.RequestedToCapacityRatioName, args.RequestedToCapacityRatioArgs))
			return
		})

	registry.registerPriorityConfigProducer(nodelabel.Name,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			// If there are n LabelPreference priorities in the policy, the weight for the corresponding
			// score plugin is n*weight (note that the validation logic verifies that all LabelPreference
			// priorities specified in Policy have the same weight).
			weight := args.Weight * int32(len(args.NodeLabelArgs.PresentLabelsPreference)+len(args.NodeLabelArgs.AbsentLabelsPreference))
			plugins.Score = appendToPluginSet(plugins.Score, nodelabel.Name, &weight)
			pluginConfig = append(pluginConfig, makePluginConfig(nodelabel.Name, args.NodeLabelArgs))
			return
		})
	registry.registerPriorityConfigProducer(serviceaffinity.Name,
		func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
			// If there are n ServiceAffinity priorities in the policy, the weight for the corresponding
			// score plugin is n*weight (note that the validation logic verifies that all ServiceAffinity
			// priorities specified in Policy have the same weight).
			weight := args.Weight * int32(len(args.ServiceAffinityArgs.AntiAffinityLabelsPreference))
			plugins.Score = appendToPluginSet(plugins.Score, serviceaffinity.Name, &weight)
			pluginConfig = append(pluginConfig, makePluginConfig(serviceaffinity.Name, args.ServiceAffinityArgs))
			return
		})

	// The following two features are the last ones to be supported as predicate/priority.
	// Once they graduate to GA, there will be no more checking for featue gates here.
	// Only register EvenPodsSpread predicate & priority if the feature is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.EvenPodsSpread) {
		klog.Infof("Registering EvenPodsSpread predicate and priority function")

		registry.registerPredicateConfigProducer(predicates.EvenPodsSpreadPred,
			func(_ ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
				plugins.PreFilter = appendToPluginSet(plugins.PreFilter, podtopologyspread.Name, nil)
				plugins.Filter = appendToPluginSet(plugins.Filter, podtopologyspread.Name, nil)
				return
			})
		registry.DefaultPredicates.Insert(predicates.EvenPodsSpreadPred)

		registry.registerPriorityConfigProducer(priorities.EvenPodsSpreadPriority,
			func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
				plugins.PostFilter = appendToPluginSet(plugins.PostFilter, podtopologyspread.Name, nil)
				plugins.Score = appendToPluginSet(plugins.Score, podtopologyspread.Name, &args.Weight)
				return
			})
		registry.DefaultPriorities[priorities.EvenPodsSpreadPriority] = 1
	}

	// Prioritizes nodes that satisfy pod's resource limits
	if utilfeature.DefaultFeatureGate.Enabled(features.ResourceLimitsPriorityFunction) {
		klog.Infof("Registering resourcelimits priority function")

		registry.registerPriorityConfigProducer(priorities.ResourceLimitsPriority,
			func(args ConfigProducerArgs) (plugins config.Plugins, pluginConfig []config.PluginConfig) {
				plugins.PostFilter = appendToPluginSet(plugins.PostFilter, noderesources.ResourceLimitsName, nil)
				plugins.Score = appendToPluginSet(plugins.Score, noderesources.ResourceLimitsName, &args.Weight)
				return
			})
		registry.DefaultPriorities[priorities.ResourceLimitsPriority] = 1
	}

	return registry
}

// registers a config producer for a predicate.
func (lr *LegacyRegistry) registerPredicateConfigProducer(name string, producer ConfigProducer) {
	if _, exist := lr.PredicateToConfigProducer[name]; exist {
		klog.Fatalf("already registered %q", name)
	}
	lr.PredicateToConfigProducer[name] = producer
}

// registers a framework config producer for a priority.
func (lr *LegacyRegistry) registerPriorityConfigProducer(name string, producer ConfigProducer) {
	if _, exist := lr.PriorityToConfigProducer[name]; exist {
		klog.Fatalf("already registered %q", name)
	}
	lr.PriorityToConfigProducer[name] = producer
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
		klog.Fatal(fmt.Errorf("failed to marshal %+v: %v", args, err))
		return config.PluginConfig{}
	}
	config := config.PluginConfig{
		Name: pluginName,
		Args: runtime.Unknown{Raw: encoding},
	}
	return config
}

// ProcessPredicatePolicy given a PredicatePolicy, return the plugin name implementing the predicate and update
// the ConfigProducerArgs if necessary.
func (lr *LegacyRegistry) ProcessPredicatePolicy(policy config.PredicatePolicy, pluginArgs *ConfigProducerArgs) string {
	validatePredicateOrDie(policy)

	predicateName := policy.Name
	if policy.Name == "PodFitsPorts" {
		// For compatibility reasons, "PodFitsPorts" as a key is still supported.
		predicateName = predicates.PodFitsHostPortsPred
	}

	if _, ok := lr.PredicateToConfigProducer[predicateName]; ok {
		// checking to see if a pre-defined predicate is requested
		klog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
		return predicateName
	}

	if policy.Argument == nil || (policy.Argument.ServiceAffinity == nil &&
		policy.Argument.LabelsPresence == nil) {
		klog.Fatalf("Invalid configuration: Predicate type not found for %q", policy.Name)
	}

	// generate the predicate function, if a custom type is requested
	if policy.Argument.ServiceAffinity != nil {
		// map LabelsPresence policy to ConfigProducerArgs that's used to configure the ServiceAffinity plugin.
		if pluginArgs.ServiceAffinityArgs == nil {
			pluginArgs.ServiceAffinityArgs = &serviceaffinity.Args{}
		}
		pluginArgs.ServiceAffinityArgs.AffinityLabels = append(pluginArgs.ServiceAffinityArgs.AffinityLabels, policy.Argument.ServiceAffinity.Labels...)

		// We use the ServiceAffinity predicate name for all ServiceAffinity custom predicates.
		// It may get called multiple times but we essentially only register one instance of ServiceAffinity predicate.
		// This name is then used to find the registered plugin and run the plugin instead of the predicate.
		predicateName = predicates.CheckServiceAffinityPred
	}

	if policy.Argument.LabelsPresence != nil {
		// Map LabelPresence policy to ConfigProducerArgs that's used to configure the NodeLabel plugin.
		if pluginArgs.NodeLabelArgs == nil {
			pluginArgs.NodeLabelArgs = &nodelabel.Args{}
		}
		if policy.Argument.LabelsPresence.Presence {
			pluginArgs.NodeLabelArgs.PresentLabels = append(pluginArgs.NodeLabelArgs.PresentLabels, policy.Argument.LabelsPresence.Labels...)
		} else {
			pluginArgs.NodeLabelArgs.AbsentLabels = append(pluginArgs.NodeLabelArgs.AbsentLabels, policy.Argument.LabelsPresence.Labels...)
		}

		// We use the CheckNodeLabelPresencePred predicate name for all kNodeLabel custom predicates.
		// It may get called multiple times but we essentially only register one instance of NodeLabel predicate.
		// This name is then used to find the registered plugin and run the plugin instead of the predicate.
		predicateName = predicates.CheckNodeLabelPresencePred

	}
	return predicateName
}

// ProcessPriorityPolicy given a PriorityPolicy, return the plugin name implementing the priority and update
// the ConfigProducerArgs if necessary.
func (lr *LegacyRegistry) ProcessPriorityPolicy(policy config.PriorityPolicy, configProducerArgs *ConfigProducerArgs) string {
	validatePriorityOrDie(policy)

	priorityName := policy.Name
	if policy.Name == priorities.ServiceSpreadingPriority {
		// For compatibility reasons, "ServiceSpreadingPriority" as a key is still supported.
		priorityName = priorities.SelectorSpreadPriority
	}

	if _, ok := lr.PriorityToConfigProducer[priorityName]; ok {
		klog.V(2).Infof("Priority type %s already registered, reusing.", priorityName)
		return priorityName
	}

	// generate the priority function, if a custom priority is requested
	if policy.Argument == nil ||
		(policy.Argument.ServiceAntiAffinity == nil &&
			policy.Argument.RequestedToCapacityRatioArguments == nil &&
			policy.Argument.LabelPreference == nil) {
		klog.Fatalf("Invalid configuration: Priority type not found for %q", priorityName)
	}

	if policy.Argument.ServiceAntiAffinity != nil {
		// We use the ServiceAffinity plugin name for all ServiceAffinity custom priorities.
		// It may get called multiple times but we essentially only register one instance of
		// ServiceAffinity priority.
		// This name is then used to find the registered plugin and run the plugin instead of the priority.
		priorityName = serviceaffinity.Name
		if configProducerArgs.ServiceAffinityArgs == nil {
			configProducerArgs.ServiceAffinityArgs = &serviceaffinity.Args{}
		}
		configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference = append(
			configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference,
			policy.Argument.ServiceAntiAffinity.Label,
		)
	}

	if policy.Argument.LabelPreference != nil {
		// We use the NodeLabel plugin name for all NodeLabel custom priorities.
		// It may get called multiple times but we essentially only register one instance of NodeLabel priority.
		// This name is then used to find the registered plugin and run the plugin instead of the priority.
		priorityName = nodelabel.Name
		if configProducerArgs.NodeLabelArgs == nil {
			configProducerArgs.NodeLabelArgs = &nodelabel.Args{}
		}
		if policy.Argument.LabelPreference.Presence {
			configProducerArgs.NodeLabelArgs.PresentLabelsPreference = append(
				configProducerArgs.NodeLabelArgs.PresentLabelsPreference,
				policy.Argument.LabelPreference.Label,
			)
		} else {
			configProducerArgs.NodeLabelArgs.AbsentLabelsPreference = append(
				configProducerArgs.NodeLabelArgs.AbsentLabelsPreference,
				policy.Argument.LabelPreference.Label,
			)
		}
	}

	if policy.Argument.RequestedToCapacityRatioArguments != nil {
		scoringFunctionShape, resources := buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(policy.Argument.RequestedToCapacityRatioArguments)
		configProducerArgs.RequestedToCapacityRatioArgs = &noderesources.RequestedToCapacityRatioArgs{
			FunctionShape:       scoringFunctionShape,
			ResourceToWeightMap: resources,
		}
		// We do not allow specifying the name for custom plugins, see #83472
		priorityName = noderesources.RequestedToCapacityRatioName
	}

	return priorityName
}

// TODO(ahg-g): move to RequestedToCapacityRatio plugin.
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
	resourceToWeightMap := make(noderesources.ResourceToWeightMap)
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
