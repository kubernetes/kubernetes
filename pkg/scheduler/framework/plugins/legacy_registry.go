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
	"fmt"
	"sort"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
)

const (
	// EqualPriority defines the name of prioritizer function that gives an equal weight of one to all nodes.
	EqualPriority = "EqualPriority"
	// MostRequestedPriority defines the name of prioritizer function that gives used nodes higher priority.
	MostRequestedPriority = "MostRequestedPriority"
	// RequestedToCapacityRatioPriority defines the name of RequestedToCapacityRatioPriority.
	RequestedToCapacityRatioPriority = "RequestedToCapacityRatioPriority"
	// SelectorSpreadPriority defines the name of prioritizer function that spreads pods by minimizing
	// the number of pods (belonging to the same service or replication controller) on the same node.
	SelectorSpreadPriority = "SelectorSpreadPriority"
	// ServiceSpreadingPriority is largely replaced by "SelectorSpreadPriority".
	ServiceSpreadingPriority = "ServiceSpreadingPriority"
	// InterPodAffinityPriority defines the name of prioritizer function that decides which pods should or
	// should not be placed in the same topological domain as some other pods.
	InterPodAffinityPriority = "InterPodAffinityPriority"
	// LeastRequestedPriority defines the name of prioritizer function that prioritize nodes by least
	// requested utilization.
	LeastRequestedPriority = "LeastRequestedPriority"
	// BalancedResourceAllocation defines the name of prioritizer function that prioritizes nodes
	// to help achieve balanced resource usage.
	BalancedResourceAllocation = "BalancedResourceAllocation"
	// NodePreferAvoidPodsPriority defines the name of prioritizer function that priorities nodes according to
	// the node annotation "scheduler.alpha.kubernetes.io/preferAvoidPods".
	NodePreferAvoidPodsPriority = "NodePreferAvoidPodsPriority"
	// NodeAffinityPriority defines the name of prioritizer function that prioritizes nodes which have labels
	// matching NodeAffinity.
	NodeAffinityPriority = "NodeAffinityPriority"
	// TaintTolerationPriority defines the name of prioritizer function that prioritizes nodes that marked
	// with taint which pod can tolerate.
	TaintTolerationPriority = "TaintTolerationPriority"
	// ImageLocalityPriority defines the name of prioritizer function that prioritizes nodes that have images
	// requested by the pod present.
	ImageLocalityPriority = "ImageLocalityPriority"
	// EvenPodsSpreadPriority defines the name of prioritizer function that prioritizes nodes
	// which have pods and labels matching the incoming pod's topologySpreadConstraints.
	EvenPodsSpreadPriority = "EvenPodsSpreadPriority"
)

const (
	// MatchInterPodAffinityPred defines the name of predicate MatchInterPodAffinity.
	MatchInterPodAffinityPred = "MatchInterPodAffinity"
	// CheckVolumeBindingPred defines the name of predicate CheckVolumeBinding.
	CheckVolumeBindingPred = "CheckVolumeBinding"
	// GeneralPred defines the name of predicate GeneralPredicates.
	GeneralPred = "GeneralPredicates"
	// HostNamePred defines the name of predicate HostName.
	HostNamePred = "HostName"
	// PodFitsHostPortsPred defines the name of predicate PodFitsHostPorts.
	PodFitsHostPortsPred = "PodFitsHostPorts"
	// MatchNodeSelectorPred defines the name of predicate MatchNodeSelector.
	MatchNodeSelectorPred = "MatchNodeSelector"
	// PodFitsResourcesPred defines the name of predicate PodFitsResources.
	PodFitsResourcesPred = "PodFitsResources"
	// NoDiskConflictPred defines the name of predicate NoDiskConflict.
	NoDiskConflictPred = "NoDiskConflict"
	// PodToleratesNodeTaintsPred defines the name of predicate PodToleratesNodeTaints.
	PodToleratesNodeTaintsPred = "PodToleratesNodeTaints"
	// CheckNodeUnschedulablePred defines the name of predicate CheckNodeUnschedulablePredicate.
	CheckNodeUnschedulablePred = "CheckNodeUnschedulable"
	// CheckNodeLabelPresencePred defines the name of predicate CheckNodeLabelPresence.
	CheckNodeLabelPresencePred = "CheckNodeLabelPresence"
	// CheckServiceAffinityPred defines the name of predicate checkServiceAffinity.
	CheckServiceAffinityPred = "CheckServiceAffinity"
	// MaxEBSVolumeCountPred defines the name of predicate MaxEBSVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxEBSVolumeCountPred = "MaxEBSVolumeCount"
	// MaxGCEPDVolumeCountPred defines the name of predicate MaxGCEPDVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxGCEPDVolumeCountPred = "MaxGCEPDVolumeCount"
	// MaxAzureDiskVolumeCountPred defines the name of predicate MaxAzureDiskVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxAzureDiskVolumeCountPred = "MaxAzureDiskVolumeCount"
	// MaxCinderVolumeCountPred defines the name of predicate MaxCinderDiskVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxCinderVolumeCountPred = "MaxCinderVolumeCount"
	// MaxCSIVolumeCountPred defines the predicate that decides how many CSI volumes should be attached.
	MaxCSIVolumeCountPred = "MaxCSIVolumeCountPred"
	// NoVolumeZoneConflictPred defines the name of predicate NoVolumeZoneConflict.
	NoVolumeZoneConflictPred = "NoVolumeZoneConflict"
	// EvenPodsSpreadPred defines the name of predicate EvenPodsSpread.
	EvenPodsSpreadPred = "EvenPodsSpread"
)

// predicateOrdering is the ordering of predicate execution.
var predicateOrdering = []string{
	CheckNodeUnschedulablePred,
	GeneralPred, HostNamePred, PodFitsHostPortsPred,
	MatchNodeSelectorPred, PodFitsResourcesPred, NoDiskConflictPred,
	PodToleratesNodeTaintsPred, CheckNodeLabelPresencePred,
	CheckServiceAffinityPred, MaxEBSVolumeCountPred, MaxGCEPDVolumeCountPred, MaxCSIVolumeCountPred,
	MaxAzureDiskVolumeCountPred, MaxCinderVolumeCountPred, CheckVolumeBindingPred, NoVolumeZoneConflictPred,
	EvenPodsSpreadPred, MatchInterPodAffinityPred,
}

// LegacyRegistry is used to store current state of registered predicates and priorities.
type LegacyRegistry struct {
	// maps that associate predicates/priorities with framework plugin configurations.
	predicateToConfigProducer map[string]configProducer
	priorityToConfigProducer  map[string]configProducer
	// predicates that will always be configured.
	mandatoryPredicates sets.String
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
	NodeLabelArgs *config.NodeLabelArgs
	// RequestedToCapacityRatioArgs is the args for the RequestedToCapacityRatio plugin.
	RequestedToCapacityRatioArgs *config.RequestedToCapacityRatioArgs
	// ServiceAffinityArgs is the args for the ServiceAffinity plugin.
	ServiceAffinityArgs *config.ServiceAffinityArgs
	// NodeResourcesFitArgs is the args for the NodeResources fit filter.
	NodeResourcesFitArgs *config.NodeResourcesFitArgs
	// InterPodAffinityArgs is the args for InterPodAffinity plugin
	InterPodAffinityArgs *config.InterPodAffinityArgs
}

// configProducer appends the set of plugins and their configuration for a
// predicate/priority given the args.
type configProducer func(ConfigProducerArgs, *config.Plugins, *[]config.PluginConfig)

// NewLegacyRegistry returns a legacy algorithm registry of predicates and priorities.
func NewLegacyRegistry() *LegacyRegistry {
	registry := &LegacyRegistry{
		// mandatoryPredicates the set of keys for predicates that the scheduler will
		// be configured with all the time.
		mandatoryPredicates: sets.NewString(
			PodToleratesNodeTaintsPred,
			CheckNodeUnschedulablePred,
		),

		// Used as the default set of predicates if Policy was specified, but predicates was nil.
		DefaultPredicates: sets.NewString(
			NoVolumeZoneConflictPred,
			MaxEBSVolumeCountPred,
			MaxGCEPDVolumeCountPred,
			MaxAzureDiskVolumeCountPred,
			MaxCSIVolumeCountPred,
			MatchInterPodAffinityPred,
			NoDiskConflictPred,
			GeneralPred,
			PodToleratesNodeTaintsPred,
			CheckVolumeBindingPred,
			CheckNodeUnschedulablePred,
			EvenPodsSpreadPred,
		),

		// Used as the default set of predicates if Policy was specified, but priorities was nil.
		DefaultPriorities: map[string]int64{
			SelectorSpreadPriority:      1,
			InterPodAffinityPriority:    1,
			LeastRequestedPriority:      1,
			BalancedResourceAllocation:  1,
			NodePreferAvoidPodsPriority: 10000,
			NodeAffinityPriority:        1,
			TaintTolerationPriority:     1,
			ImageLocalityPriority:       1,
			EvenPodsSpreadPriority:      2,
		},

		predicateToConfigProducer: make(map[string]configProducer),
		priorityToConfigProducer:  make(map[string]configProducer),
	}

	registry.registerPredicateConfigProducer(GeneralPred,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			// GeneralPredicate is a combination of predicates.
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, noderesources.FitName, nil)
			if args.NodeResourcesFitArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: noderesources.FitName, Args: args.NodeResourcesFitArgs})
			}
			plugins.Filter = appendToPluginSet(plugins.Filter, nodename.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeports.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeports.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeaffinity.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeaffinity.Name, nil)
		})
	registry.registerPredicateConfigProducer(PodToleratesNodeTaintsPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, tainttoleration.Name, nil)
		})
	registry.registerPredicateConfigProducer(PodFitsResourcesPred,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, noderesources.FitName, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, noderesources.FitName, nil)
			if args.NodeResourcesFitArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: noderesources.FitName, Args: args.NodeResourcesFitArgs})
			}
		})
	registry.registerPredicateConfigProducer(HostNamePred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodename.Name, nil)
		})
	registry.registerPredicateConfigProducer(PodFitsHostPortsPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeports.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeports.Name, nil)
		})
	registry.registerPredicateConfigProducer(MatchNodeSelectorPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeaffinity.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, nodeaffinity.Name, nil)
		})
	registry.registerPredicateConfigProducer(CheckNodeUnschedulablePred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodeunschedulable.Name, nil)
		})
	registry.registerPredicateConfigProducer(CheckVolumeBindingPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, volumebinding.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, volumebinding.Name, nil)
			plugins.Reserve = appendToPluginSet(plugins.Reserve, volumebinding.Name, nil)
			plugins.PreBind = appendToPluginSet(plugins.PreBind, volumebinding.Name, nil)
		})
	registry.registerPredicateConfigProducer(NoDiskConflictPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumerestrictions.Name, nil)
		})
	registry.registerPredicateConfigProducer(NoVolumeZoneConflictPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, volumezone.Name, nil)
		})
	registry.registerPredicateConfigProducer(MaxCSIVolumeCountPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.CSIName, nil)
		})
	registry.registerPredicateConfigProducer(MaxEBSVolumeCountPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.EBSName, nil)
		})
	registry.registerPredicateConfigProducer(MaxGCEPDVolumeCountPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.GCEPDName, nil)
		})
	registry.registerPredicateConfigProducer(MaxAzureDiskVolumeCountPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.AzureDiskName, nil)
		})
	registry.registerPredicateConfigProducer(MaxCinderVolumeCountPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodevolumelimits.CinderName, nil)
		})
	registry.registerPredicateConfigProducer(MatchInterPodAffinityPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, interpodaffinity.Name, nil)
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, interpodaffinity.Name, nil)
		})
	registry.registerPredicateConfigProducer(CheckNodeLabelPresencePred,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, nodelabel.Name, nil)
			if args.NodeLabelArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: nodelabel.Name, Args: args.NodeLabelArgs})
			}
		})
	registry.registerPredicateConfigProducer(CheckServiceAffinityPred,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			plugins.Filter = appendToPluginSet(plugins.Filter, serviceaffinity.Name, nil)
			if args.ServiceAffinityArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: serviceaffinity.Name, Args: args.ServiceAffinityArgs})
			}
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, serviceaffinity.Name, nil)
		})
	registry.registerPredicateConfigProducer(EvenPodsSpreadPred,
		func(_ ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.PreFilter = appendToPluginSet(plugins.PreFilter, podtopologyspread.Name, nil)
			plugins.Filter = appendToPluginSet(plugins.Filter, podtopologyspread.Name, nil)
		})

	// Register Priorities.
	registry.registerPriorityConfigProducer(SelectorSpreadPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			if !feature.DefaultFeatureGate.Enabled(features.DefaultPodTopologySpread) {
				plugins.Score = appendToPluginSet(plugins.Score, selectorspread.Name, &args.Weight)
				plugins.PreScore = appendToPluginSet(plugins.PreScore, selectorspread.Name, nil)
				return
			}
			plugins.Score = appendToPluginSet(plugins.Score, podtopologyspread.Name, &args.Weight)
			plugins.PreScore = appendToPluginSet(plugins.PreScore, podtopologyspread.Name, nil)
			plArgs := config.PodTopologySpreadArgs{
				DefaultingType: config.SystemDefaulting,
			}
			// The order in which SelectorSpreadPriority or EvenPodsSpreadPriority producers
			// are called is not guaranteed. Override or append configuration.
			for i, e := range *pluginConfig {
				if e.Name == podtopologyspread.Name {
					(*pluginConfig)[i].Args = &plArgs
					return
				}
			}
			*pluginConfig = append(*pluginConfig, config.PluginConfig{
				Name: podtopologyspread.Name,
				Args: &plArgs,
			})
		})
	registry.registerPriorityConfigProducer(TaintTolerationPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.PreScore = appendToPluginSet(plugins.PreScore, tainttoleration.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, tainttoleration.Name, &args.Weight)
		})
	registry.registerPriorityConfigProducer(NodeAffinityPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.PreScore = appendToPluginSet(plugins.PreScore, nodeaffinity.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, nodeaffinity.Name, &args.Weight)
		})
	registry.registerPriorityConfigProducer(ImageLocalityPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, imagelocality.Name, &args.Weight)
		})
	registry.registerPriorityConfigProducer(InterPodAffinityPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			plugins.PreScore = appendToPluginSet(plugins.PreScore, interpodaffinity.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, interpodaffinity.Name, &args.Weight)
			if args.InterPodAffinityArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: interpodaffinity.Name, Args: args.InterPodAffinityArgs})
			}
		})
	registry.registerPriorityConfigProducer(NodePreferAvoidPodsPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, nodepreferavoidpods.Name, &args.Weight)
		})
	registry.registerPriorityConfigProducer(MostRequestedPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.MostAllocatedName, &args.Weight)
		})
	registry.registerPriorityConfigProducer(BalancedResourceAllocation,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.BalancedAllocationName, &args.Weight)
		})
	registry.registerPriorityConfigProducer(LeastRequestedPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, _ *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.LeastAllocatedName, &args.Weight)
		})
	registry.registerPriorityConfigProducer(noderesources.RequestedToCapacityRatioName,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			plugins.Score = appendToPluginSet(plugins.Score, noderesources.RequestedToCapacityRatioName, &args.Weight)
			if args.RequestedToCapacityRatioArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: noderesources.RequestedToCapacityRatioName, Args: args.RequestedToCapacityRatioArgs})
			}
		})
	registry.registerPriorityConfigProducer(nodelabel.Name,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			// If there are n LabelPreference priorities in the policy, the weight for the corresponding
			// score plugin is n*weight (note that the validation logic verifies that all LabelPreference
			// priorities specified in Policy have the same weight).
			weight := args.Weight * int32(len(args.NodeLabelArgs.PresentLabelsPreference)+len(args.NodeLabelArgs.AbsentLabelsPreference))
			plugins.Score = appendToPluginSet(plugins.Score, nodelabel.Name, &weight)
			if args.NodeLabelArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: nodelabel.Name, Args: args.NodeLabelArgs})
			}
		})
	registry.registerPriorityConfigProducer(serviceaffinity.Name,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			// If there are n ServiceAffinity priorities in the policy, the weight for the corresponding
			// score plugin is n*weight (note that the validation logic verifies that all ServiceAffinity
			// priorities specified in Policy have the same weight).
			weight := args.Weight * int32(len(args.ServiceAffinityArgs.AntiAffinityLabelsPreference))
			plugins.Score = appendToPluginSet(plugins.Score, serviceaffinity.Name, &weight)
			if args.ServiceAffinityArgs != nil {
				*pluginConfig = append(*pluginConfig,
					config.PluginConfig{Name: serviceaffinity.Name, Args: args.ServiceAffinityArgs})
			}
		})
	registry.registerPriorityConfigProducer(EvenPodsSpreadPriority,
		func(args ConfigProducerArgs, plugins *config.Plugins, pluginConfig *[]config.PluginConfig) {
			plugins.PreScore = appendToPluginSet(plugins.PreScore, podtopologyspread.Name, nil)
			plugins.Score = appendToPluginSet(plugins.Score, podtopologyspread.Name, &args.Weight)
			if feature.DefaultFeatureGate.Enabled(features.DefaultPodTopologySpread) {
				// The order in which SelectorSpreadPriority or EvenPodsSpreadPriority producers
				// are called is not guaranteed. If plugin was not configured yet, append
				// configuration where system default constraints are disabled.
				for _, e := range *pluginConfig {
					if e.Name == podtopologyspread.Name {
						return
					}
				}
				*pluginConfig = append(*pluginConfig, config.PluginConfig{
					Name: podtopologyspread.Name,
					Args: &config.PodTopologySpreadArgs{
						DefaultingType: config.ListDefaulting,
					},
				})
			}
		})

	return registry
}

// AppendPredicateConfigs returns predicates configuration that will run as framework plugins.
// Note that the framework executes plugins according to their order in the Plugins list, and so predicates run as plugins
// are added to the Plugins list according to the order specified in predicateOrdering.
func (lr *LegacyRegistry) AppendPredicateConfigs(keys sets.String, args *ConfigProducerArgs, plugins config.Plugins, pluginConfig []config.PluginConfig) (config.Plugins, []config.PluginConfig, error) {
	allPredicates := keys.Union(lr.mandatoryPredicates)

	// Create the framework plugin configurations, and place them in the order
	// that the corresponding predicates were supposed to run.
	for _, predicateKey := range predicateOrdering {
		if allPredicates.Has(predicateKey) {
			producer, exist := lr.predicateToConfigProducer[predicateKey]
			if !exist {
				return config.Plugins{}, nil, fmt.Errorf("no framework config producer registered for %q", predicateKey)
			}
			producer(*args, &plugins, &pluginConfig)
			allPredicates.Delete(predicateKey)
		}
	}

	// Sort the keys so that it is easier for unit tests to do compare.
	sortedKeys := make([]string, 0, len(allPredicates))
	for k := range allPredicates {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)

	for _, predicateKey := range sortedKeys {
		producer, exist := lr.predicateToConfigProducer[predicateKey]
		if !exist {
			return config.Plugins{}, nil, fmt.Errorf("no framework config producer registered for %q", predicateKey)
		}
		producer(*args, &plugins, &pluginConfig)
	}

	return plugins, pluginConfig, nil
}

// AppendPriorityConfigs returns priorities configuration that will run as framework plugins.
func (lr *LegacyRegistry) AppendPriorityConfigs(keys map[string]int64, args *ConfigProducerArgs, plugins config.Plugins, pluginConfig []config.PluginConfig) (config.Plugins, []config.PluginConfig, error) {
	// Sort the keys so that it is easier for unit tests to do compare.
	sortedKeys := make([]string, 0, len(keys))
	for k := range keys {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)

	for _, priority := range sortedKeys {
		weight := keys[priority]
		producer, exist := lr.priorityToConfigProducer[priority]
		if !exist {
			return config.Plugins{}, nil, fmt.Errorf("no config producer registered for %q", priority)
		}
		a := *args
		a.Weight = int32(weight)
		producer(a, &plugins, &pluginConfig)
	}
	return plugins, pluginConfig, nil
}

// registers a config producer for a predicate.
func (lr *LegacyRegistry) registerPredicateConfigProducer(name string, producer configProducer) {
	if _, exist := lr.predicateToConfigProducer[name]; exist {
		klog.Fatalf("already registered %q", name)
	}
	lr.predicateToConfigProducer[name] = producer
}

// registers a framework config producer for a priority.
func (lr *LegacyRegistry) registerPriorityConfigProducer(name string, producer configProducer) {
	if _, exist := lr.priorityToConfigProducer[name]; exist {
		klog.Fatalf("already registered %q", name)
	}
	lr.priorityToConfigProducer[name] = producer
}

func appendToPluginSet(set config.PluginSet, name string, weight *int32) config.PluginSet {
	for _, e := range set.Enabled {
		if e.Name == name {
			// Keep the max weight.
			if weight != nil && *weight > e.Weight {
				e.Weight = *weight
			}
			return set
		}
	}
	cfg := config.Plugin{Name: name}
	if weight != nil {
		cfg.Weight = *weight
	}
	set.Enabled = append(set.Enabled, cfg)
	return set
}

// ProcessPredicatePolicy given a PredicatePolicy, return the plugin name implementing the predicate and update
// the ConfigProducerArgs if necessary.
func (lr *LegacyRegistry) ProcessPredicatePolicy(policy config.PredicatePolicy, pluginArgs *ConfigProducerArgs) (string, error) {
	if err := validatePredicate(policy); err != nil {
		return "", err
	}

	predicateName := policy.Name
	if policy.Name == "PodFitsPorts" {
		// For compatibility reasons, "PodFitsPorts" as a key is still supported.
		predicateName = PodFitsHostPortsPred
	}

	if _, ok := lr.predicateToConfigProducer[predicateName]; ok {
		// checking to see if a pre-defined predicate is requested
		klog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
		return predicateName, nil
	}

	if policy.Argument == nil || (policy.Argument.ServiceAffinity == nil &&
		policy.Argument.LabelsPresence == nil) {
		return "", fmt.Errorf("predicate type not found for %q", predicateName)
	}

	// generate the predicate function, if a custom type is requested
	if policy.Argument.ServiceAffinity != nil {
		// map LabelsPresence policy to ConfigProducerArgs that's used to configure the ServiceAffinity plugin.
		if pluginArgs.ServiceAffinityArgs == nil {
			pluginArgs.ServiceAffinityArgs = &config.ServiceAffinityArgs{}
		}
		pluginArgs.ServiceAffinityArgs.AffinityLabels = append(pluginArgs.ServiceAffinityArgs.AffinityLabels, policy.Argument.ServiceAffinity.Labels...)

		// We use the ServiceAffinity predicate name for all ServiceAffinity custom predicates.
		// It may get called multiple times but we essentially only register one instance of ServiceAffinity predicate.
		// This name is then used to find the registered plugin and run the plugin instead of the predicate.
		predicateName = CheckServiceAffinityPred
	}

	if policy.Argument.LabelsPresence != nil {
		// Map LabelPresence policy to ConfigProducerArgs that's used to configure the NodeLabel plugin.
		if pluginArgs.NodeLabelArgs == nil {
			pluginArgs.NodeLabelArgs = &config.NodeLabelArgs{}
		}
		if policy.Argument.LabelsPresence.Presence {
			pluginArgs.NodeLabelArgs.PresentLabels = append(pluginArgs.NodeLabelArgs.PresentLabels, policy.Argument.LabelsPresence.Labels...)
		} else {
			pluginArgs.NodeLabelArgs.AbsentLabels = append(pluginArgs.NodeLabelArgs.AbsentLabels, policy.Argument.LabelsPresence.Labels...)
		}

		// We use the CheckNodeLabelPresencePred predicate name for all kNodeLabel custom predicates.
		// It may get called multiple times but we essentially only register one instance of NodeLabel predicate.
		// This name is then used to find the registered plugin and run the plugin instead of the predicate.
		predicateName = CheckNodeLabelPresencePred

	}
	return predicateName, nil
}

// ProcessPriorityPolicy given a PriorityPolicy, return the plugin name implementing the priority and update
// the ConfigProducerArgs if necessary.
func (lr *LegacyRegistry) ProcessPriorityPolicy(policy config.PriorityPolicy, configProducerArgs *ConfigProducerArgs) (string, error) {
	if err := validatePriority(policy); err != nil {
		return "", err
	}

	priorityName := policy.Name
	if policy.Name == ServiceSpreadingPriority {
		// For compatibility reasons, "ServiceSpreadingPriority" as a key is still supported.
		priorityName = SelectorSpreadPriority
	}

	if _, ok := lr.priorityToConfigProducer[priorityName]; ok {
		klog.V(2).Infof("Priority type %s already registered, reusing.", priorityName)
		return priorityName, nil
	}

	// generate the priority function, if a custom priority is requested
	if policy.Argument == nil ||
		(policy.Argument.ServiceAntiAffinity == nil &&
			policy.Argument.RequestedToCapacityRatioArguments == nil &&
			policy.Argument.LabelPreference == nil) {
		return "", fmt.Errorf("priority type not found for %q", priorityName)
	}

	if policy.Argument.ServiceAntiAffinity != nil {
		// We use the ServiceAffinity plugin name for all ServiceAffinity custom priorities.
		// It may get called multiple times but we essentially only register one instance of
		// ServiceAffinity priority.
		// This name is then used to find the registered plugin and run the plugin instead of the priority.
		priorityName = serviceaffinity.Name
		if configProducerArgs.ServiceAffinityArgs == nil {
			configProducerArgs.ServiceAffinityArgs = &config.ServiceAffinityArgs{}
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
			configProducerArgs.NodeLabelArgs = &config.NodeLabelArgs{}
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
		policyArgs := policy.Argument.RequestedToCapacityRatioArguments
		args := &config.RequestedToCapacityRatioArgs{}

		args.Shape = make([]config.UtilizationShapePoint, len(policyArgs.Shape))
		for i, s := range policyArgs.Shape {
			args.Shape[i] = config.UtilizationShapePoint{
				Utilization: s.Utilization,
				Score:       s.Score,
			}
		}

		args.Resources = make([]config.ResourceSpec, len(policyArgs.Resources))
		for i, r := range policyArgs.Resources {
			args.Resources[i] = config.ResourceSpec{
				Name:   r.Name,
				Weight: r.Weight,
			}
		}

		configProducerArgs.RequestedToCapacityRatioArgs = args

		// We do not allow specifying the name for custom plugins, see #83472
		priorityName = noderesources.RequestedToCapacityRatioName
	}

	return priorityName, nil
}

func validatePredicate(predicate config.PredicatePolicy) error {
	if predicate.Argument != nil {
		numArgs := 0
		if predicate.Argument.ServiceAffinity != nil {
			numArgs++
		}
		if predicate.Argument.LabelsPresence != nil {
			numArgs++
		}
		if numArgs != 1 {
			return fmt.Errorf("exactly 1 predicate argument is required, numArgs: %v, predicate %v", numArgs, predicate)
		}
	}
	return nil
}

func validatePriority(priority config.PriorityPolicy) error {
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
			return fmt.Errorf("exactly 1 priority argument is required, numArgs: %v, priority %v", numArgs, priority)
		}
	}
	return nil
}
