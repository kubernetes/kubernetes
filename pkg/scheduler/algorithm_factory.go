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

package scheduler

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"

	"k8s.io/klog"
)

// AlgorithmFactoryArgs are passed to all factory functions.
type AlgorithmFactoryArgs struct {
	SharedLister                   schedulerlisters.SharedLister
	InformerFactory                informers.SharedInformerFactory
	VolumeBinder                   *volumebinder.VolumeBinder
	HardPodAffinitySymmetricWeight int32
}

// PriorityMetadataProducerFactory produces MetadataProducer from the given args.
type PriorityMetadataProducerFactory func(AlgorithmFactoryArgs) priorities.MetadataProducer

var (
	algorithmRegistry = &AlgorithmRegistry{
		// predicate keys supported for backward compatibility with v1.Policy.
		predicateKeys: sets.NewString(
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
		priorityKeys: map[string]int64{
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
		mandatoryPredicateKeys: sets.NewString(
			predicates.PodToleratesNodeTaintsPred,
			predicates.CheckNodeUnschedulablePred,
		),

		algorithmProviders: make(map[string]AlgorithmProviderConfig),
	}
	// Registered metadata producers
	priorityMetadataProducerFactory PriorityMetadataProducerFactory

	schedulerFactoryMutex sync.RWMutex
)

// AlgorithmProviderConfig is used to store the configuration of algorithm providers.
type AlgorithmProviderConfig struct {
	PredicateKeys sets.String
	PriorityKeys  sets.String
}

// AlgorithmRegistry is used to store current state of registered predicates and priorities.
type AlgorithmRegistry struct {
	predicateKeys          sets.String
	priorityKeys           map[string]int64
	mandatoryPredicateKeys sets.String
	algorithmProviders     map[string]AlgorithmProviderConfig
}

// RegisteredPredicatesAndPrioritiesSnapshot returns a snapshot of current registered predicates and priorities.
func RegisteredPredicatesAndPrioritiesSnapshot() *AlgorithmRegistry {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	copy := AlgorithmRegistry{
		predicateKeys:          sets.NewString(),
		mandatoryPredicateKeys: sets.NewString(),
		priorityKeys:           make(map[string]int64),
		algorithmProviders:     make(map[string]AlgorithmProviderConfig),
	}
	for k := range algorithmRegistry.predicateKeys {
		copy.predicateKeys.Insert(k)
	}
	for k := range algorithmRegistry.mandatoryPredicateKeys {
		copy.mandatoryPredicateKeys.Insert(k)
	}
	for k, v := range algorithmRegistry.priorityKeys {
		copy.priorityKeys[k] = v
	}
	for provider, config := range algorithmRegistry.algorithmProviders {
		copyPredKeys, copyPrioKeys := sets.NewString(), sets.NewString()
		for k := range config.PredicateKeys {
			copyPredKeys.Insert(k)
		}
		for k := range config.PriorityKeys {
			copyPrioKeys.Insert(k)
		}
		copy.algorithmProviders[provider] = AlgorithmProviderConfig{
			PredicateKeys: copyPredKeys,
			PriorityKeys:  copyPrioKeys,
		}
	}
	return &copy
}

// ApplyPredicatesAndPriorities sets state of predicates and priorities to `s`.
func ApplyPredicatesAndPriorities(s *AlgorithmRegistry) {
	schedulerFactoryMutex.Lock()
	algorithmRegistry = s
	schedulerFactoryMutex.Unlock()
}

// RegisterPredicate registers a fit predicate with the algorithm
// registry. Returns the name with which the predicate was registered.
// TODO(Huang-Wei): remove this.
func RegisterPredicate(name string) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	algorithmRegistry.predicateKeys.Insert(name)
	return name
}

// RegisterMandatoryPredicate registers a mandatory predicate.
func RegisterMandatoryPredicate(name string) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	algorithmRegistry.predicateKeys.Insert(name)
	algorithmRegistry.mandatoryPredicateKeys.Insert(name)
	return name
}

// AddPredicateToAlgorithmProviders adds a predicate key to all algorithm providers.
func AddPredicateToAlgorithmProviders(key string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	for _, provider := range algorithmRegistry.algorithmProviders {
		provider.PredicateKeys.Insert(key)
	}
}

// AddPriorityToAlgorithmProviders adds a priority key to all algorithm providers.
func AddPriorityToAlgorithmProviders(key string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	for _, provider := range algorithmRegistry.algorithmProviders {
		provider.PriorityKeys.Insert(key)
	}
}

// RegisterCustomPredicate registers a custom fit predicate with the algorithm registry.
// Returns the name, with which the predicate was registered.
func RegisterCustomPredicate(policy schedulerapi.PredicatePolicy, pluginArgs *plugins.ConfigProducerArgs) string {
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
	} else if _, ok = algorithmRegistry.predicateKeys[policy.Name]; ok {
		// checking to see if a pre-defined predicate is requested
		klog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
		return policy.Name
	}

	if len(predicate) == 0 {
		klog.Fatalf("Invalid configuration: Predicate type not found for %s", policy.Name)
	}

	return predicate
}

// RegisterPriorityMetadataProducerFactory registers a PriorityMetadataProducerFactory.
func RegisterPriorityMetadataProducerFactory(f PriorityMetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	priorityMetadataProducerFactory = f
}

// RegisterPriority registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
func RegisterPriority(name string, weight int64) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	algorithmRegistry.priorityKeys[name] = weight
	return name
}

// RegisterCustomPriority registers a custom priority with the algorithm registry.
// Returns the name, with which the priority function was registered.
func RegisterCustomPriority(policy schedulerapi.PriorityPolicy, configProducerArgs *plugins.ConfigProducerArgs) string {
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
			schedulerFactoryMutex.RLock()
			if existingWeight, ok := algorithmRegistry.priorityKeys[priority]; ok {
				// If there are n ServiceAffinity priorities in the policy, the weight for the corresponding
				// score plugin is n*(weight of each priority).
				weight += existingWeight
			}
			schedulerFactoryMutex.RUnlock()
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
			schedulerFactoryMutex.RLock()
			if existingWeight, ok := algorithmRegistry.priorityKeys[priority]; ok {
				// If there are n NodeLabel priority configured in the policy, the weight for the corresponding
				// priority is n*(weight of each priority in policy).
				weight += existingWeight
			}
			schedulerFactoryMutex.RUnlock()
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
	} else if _, ok := algorithmRegistry.priorityKeys[policy.Name]; ok {
		klog.V(2).Infof("Priority type %s already registered, reusing.", policy.Name)
		// set/update the weight based on the policy
		priority = policy.Name
		weight = policy.Weight
	}

	if len(priority) == 0 {
		klog.Fatalf("Invalid configuration: Priority type not found for %s", policy.Name)
	}

	return RegisterPriority(priority, weight)
}

func buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(arguments *schedulerapi.RequestedToCapacityRatioArguments) (priorities.FunctionShape, priorities.ResourceToWeightMap) {
	n := len(arguments.Shape)
	points := make([]priorities.FunctionShapePoint, 0, n)
	for _, point := range arguments.Shape {
		points = append(points, priorities.FunctionShapePoint{
			Utilization: int64(point.Utilization),
			// MaxCustomPriorityScore may diverge from the max score used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by requested to capacity ratio to the score range
			// used by the scheduler.
			Score: int64(point.Score) * (framework.MaxNodeScore / schedulerapi.MaxCustomPriorityScore),
		})
	}
	shape, err := priorities.NewFunctionShape(points)
	if err != nil {
		klog.Fatalf("invalid RequestedToCapacityRatioPriority arguments: %s", err.Error())
	}
	resourceToWeightMap := make(priorities.ResourceToWeightMap, 0)
	if len(arguments.Resources) == 0 {
		resourceToWeightMap = priorities.DefaultRequestedRatioResources
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

// RegisterAlgorithmProvider registers a new algorithm provider with the algorithm registry.
func RegisterAlgorithmProvider(name string, predicateKeys, priorityKeys sets.String) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	algorithmRegistry.algorithmProviders[name] = AlgorithmProviderConfig{
		PredicateKeys: predicateKeys,
		PriorityKeys:  priorityKeys,
	}
	return name
}

// GetAlgorithmProvider should not be used to modify providers. It is publicly visible for testing.
func GetAlgorithmProvider(name string) (*AlgorithmProviderConfig, error) {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	provider, ok := algorithmRegistry.algorithmProviders[name]
	if !ok {
		return nil, fmt.Errorf("provider %q is not registered", name)
	}

	return &provider, nil
}

func getPriorityMetadataProducer(args AlgorithmFactoryArgs) (priorities.MetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if priorityMetadataProducerFactory == nil {
		return priorities.EmptyMetadataProducer, nil
	}
	return priorityMetadataProducerFactory(args), nil
}

var validName = regexp.MustCompile("^[a-zA-Z0-9]([-a-zA-Z0-9]*[a-zA-Z0-9])$")

func validateAlgorithmNameOrDie(name string) {
	if !validName.MatchString(name) {
		klog.Fatalf("Algorithm name %v does not match the name validation regexp \"%v\".", name, validName)
	}
}

func validatePredicateOrDie(predicate schedulerapi.PredicatePolicy) {
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

func validatePriorityOrDie(priority schedulerapi.PriorityPolicy) {
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

// ListAlgorithmProviders is called when listing all available algorithm providers in `kube-scheduler --help`
func ListAlgorithmProviders() string {
	var availableAlgorithmProviders []string
	for name := range algorithmRegistry.algorithmProviders {
		availableAlgorithmProviders = append(availableAlgorithmProviders, name)
	}
	sort.Strings(availableAlgorithmProviders)
	return strings.Join(availableAlgorithmProviders, " | ")
}
