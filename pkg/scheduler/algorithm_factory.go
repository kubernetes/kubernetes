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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/requestedtocapacityratio"
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

// PredicateMetadataProducerFactory produces MetadataProducer from the given args.
type PredicateMetadataProducerFactory func(AlgorithmFactoryArgs) predicates.MetadataProducer

// FitPredicateFactory produces a FitPredicate from the given args.
type FitPredicateFactory func(AlgorithmFactoryArgs) predicates.FitPredicate

// PriorityFunctionFactory produces map & reduce priority functions
// from a given args.
type PriorityFunctionFactory func(AlgorithmFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction)

// PriorityConfigFactory produces a PriorityConfig from the given function and weight
type PriorityConfigFactory struct {
	MapReduceFunction PriorityFunctionFactory
	Weight            int64
}

var (
	schedulerFactoryMutex sync.RWMutex

	// maps that hold registered algorithm types
	fitPredicateMap        = make(map[string]FitPredicateFactory)
	mandatoryFitPredicates = sets.NewString()
	priorityFunctionMap    = make(map[string]PriorityConfigFactory)
	algorithmProviderMap   = make(map[string]AlgorithmProviderConfig)

	// Registered metadata producers
	priorityMetadataProducerFactory  PriorityMetadataProducerFactory
	predicateMetadataProducerFactory PredicateMetadataProducerFactory
)

const (
	// DefaultProvider defines the default algorithm provider name.
	DefaultProvider = "DefaultProvider"
)

// AlgorithmProviderConfig is used to store the configuration of algorithm providers.
type AlgorithmProviderConfig struct {
	FitPredicateKeys     sets.String
	PriorityFunctionKeys sets.String
}

// Snapshot is used to store current state of registered predicates and priorities.
type Snapshot struct {
	fitPredicateMap        map[string]FitPredicateFactory
	mandatoryFitPredicates sets.String
	priorityFunctionMap    map[string]PriorityConfigFactory
	algorithmProviderMap   map[string]AlgorithmProviderConfig
}

// RegisteredPredicatesAndPrioritiesSnapshot returns a snapshot of current registered predicates and priorities.
func RegisteredPredicatesAndPrioritiesSnapshot() *Snapshot {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	copy := Snapshot{
		fitPredicateMap:        make(map[string]FitPredicateFactory),
		mandatoryFitPredicates: sets.NewString(),
		priorityFunctionMap:    make(map[string]PriorityConfigFactory),
		algorithmProviderMap:   make(map[string]AlgorithmProviderConfig),
	}
	for k, v := range fitPredicateMap {
		copy.fitPredicateMap[k] = v
	}
	for k := range mandatoryFitPredicates {
		copy.mandatoryFitPredicates[k] = struct{}{}
	}
	for k, v := range priorityFunctionMap {
		copy.priorityFunctionMap[k] = v
	}
	for provider, config := range algorithmProviderMap {
		copyPredKeys, copyPrioKeys := sets.NewString(), sets.NewString()
		for k := range config.FitPredicateKeys {
			copyPredKeys[k] = struct{}{}
		}
		for k := range config.PriorityFunctionKeys {
			copyPrioKeys[k] = struct{}{}
		}
		copy.algorithmProviderMap[provider] = AlgorithmProviderConfig{
			FitPredicateKeys:     copyPredKeys,
			PriorityFunctionKeys: copyPrioKeys,
		}
	}
	return &copy
}

// ApplyPredicatesAndPriorities sets state of predicates and priorities to `s`.
func ApplyPredicatesAndPriorities(s *Snapshot) {
	schedulerFactoryMutex.Lock()
	fitPredicateMap = s.fitPredicateMap
	mandatoryFitPredicates = s.mandatoryFitPredicates
	priorityFunctionMap = s.priorityFunctionMap
	algorithmProviderMap = s.algorithmProviderMap
	schedulerFactoryMutex.Unlock()
}

// RegisterFitPredicate registers a fit predicate with the algorithm
// registry. Returns the name with which the predicate was registered.
func RegisterFitPredicate(name string, predicate predicates.FitPredicate) string {
	return RegisterFitPredicateFactory(name, func(AlgorithmFactoryArgs) predicates.FitPredicate { return predicate })
}

// RemoveFitPredicate removes a fit predicate from factory.
func RemoveFitPredicate(name string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	validateAlgorithmNameOrDie(name)
	delete(fitPredicateMap, name)
	mandatoryFitPredicates.Delete(name)
}

// RemovePredicateKeyFromAlgoProvider removes a fit predicate key from algorithmProvider.
func RemovePredicateKeyFromAlgoProvider(providerName, key string) error {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	validateAlgorithmNameOrDie(providerName)
	provider, ok := algorithmProviderMap[providerName]
	if !ok {
		return fmt.Errorf("provider %v is not registered", providerName)
	}
	provider.FitPredicateKeys.Delete(key)
	return nil
}

// RemovePredicateKeyFromAlgorithmProviderMap removes a fit predicate key from all algorithmProviders which in algorithmProviderMap.
func RemovePredicateKeyFromAlgorithmProviderMap(key string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	for _, provider := range algorithmProviderMap {
		provider.FitPredicateKeys.Delete(key)
	}
}

// InsertPredicateKeyToAlgoProvider insert a fit predicate key to algorithmProvider.
func InsertPredicateKeyToAlgoProvider(providerName, key string) error {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	validateAlgorithmNameOrDie(providerName)
	provider, ok := algorithmProviderMap[providerName]
	if !ok {
		return fmt.Errorf("provider %v is not registered", providerName)
	}
	provider.FitPredicateKeys.Insert(key)
	return nil
}

// InsertPredicateKeyToAlgorithmProviderMap insert a fit predicate key to all algorithmProviders which in algorithmProviderMap.
func InsertPredicateKeyToAlgorithmProviderMap(key string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	for _, provider := range algorithmProviderMap {
		provider.FitPredicateKeys.Insert(key)
	}
}

// InsertPriorityKeyToAlgorithmProviderMap inserts a priority function to all algorithmProviders which are in algorithmProviderMap.
func InsertPriorityKeyToAlgorithmProviderMap(key string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	for _, provider := range algorithmProviderMap {
		provider.PriorityFunctionKeys.Insert(key)
	}
}

// RegisterMandatoryFitPredicate registers a fit predicate with the algorithm registry, the predicate is used by
// kubelet, DaemonSet; it is always included in configuration. Returns the name with which the predicate was
// registered.
func RegisterMandatoryFitPredicate(name string, predicate predicates.FitPredicate) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = func(AlgorithmFactoryArgs) predicates.FitPredicate { return predicate }
	mandatoryFitPredicates.Insert(name)
	return name
}

// RegisterFitPredicateFactory registers a fit predicate factory with the
// algorithm registry. Returns the name with which the predicate was registered.
func RegisterFitPredicateFactory(name string, predicateFactory FitPredicateFactory) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = predicateFactory
	return name
}

// RegisterCustomFitPredicate registers a custom fit predicate with the algorithm registry.
// Returns the name, with which the predicate was registered.
func RegisterCustomFitPredicate(policy schedulerapi.PredicatePolicy, pluginArgs *plugins.ConfigProducerArgs) string {
	var predicateFactory FitPredicateFactory
	var ok bool
	policyName := policy.Name

	validatePredicateOrDie(policy)

	// generate the predicate function, if a custom type is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAffinity != nil {
			// We use the ServiceAffinity predicate name for all ServiceAffinity custom predicates.
			// It may get called multiple times but we essentially only register one instance of ServiceAffinity predicate.
			// This name is then used to find the registered plugin and run the plugin instead of the predicate.
			policyName = predicates.CheckServiceAffinityPred

			// map LabelsPresence policy to ConfigProducerArgs that's used to configure the ServiceAffinity plugin.
			if pluginArgs.ServiceAffinityArgs == nil {
				pluginArgs.ServiceAffinityArgs = &serviceaffinity.Args{}
			}

			pluginArgs.ServiceAffinityArgs.AffinityLabels = append(pluginArgs.ServiceAffinityArgs.AffinityLabels, policy.Argument.ServiceAffinity.Labels...)

			predicateFactory = func(args AlgorithmFactoryArgs) predicates.FitPredicate {
				predicate, precomputationFunction := predicates.NewServiceAffinityPredicate(
					args.SharedLister.NodeInfos(),
					args.SharedLister.Pods(),
					args.InformerFactory.Core().V1().Services().Lister(),
					pluginArgs.ServiceAffinityArgs.AffinityLabels,
				)

				// Once we generate the predicate we should also Register the Precomputation
				predicates.RegisterPredicateMetadataProducer(policyName, precomputationFunction)
				return predicate
			}
		} else if policy.Argument.LabelsPresence != nil {
			// We use the CheckNodeLabelPresencePred predicate name for all kNodeLabel custom predicates.
			// It may get called multiple times but we essentially only register one instance of NodeLabel predicate.
			// This name is then used to find the registered plugin and run the plugin instead of the predicate.
			policyName = predicates.CheckNodeLabelPresencePred

			// Map LabelPresence policy to ConfigProducerArgs that's used to configure the NodeLabel plugin.
			if pluginArgs.NodeLabelArgs == nil {
				pluginArgs.NodeLabelArgs = &nodelabel.Args{}
			}
			if policy.Argument.LabelsPresence.Presence {
				pluginArgs.NodeLabelArgs.PresentLabels = append(pluginArgs.NodeLabelArgs.PresentLabels, policy.Argument.LabelsPresence.Labels...)
			} else {
				pluginArgs.NodeLabelArgs.AbsentLabels = append(pluginArgs.NodeLabelArgs.AbsentLabels, policy.Argument.LabelsPresence.Labels...)
			}
			predicateFactory = func(_ AlgorithmFactoryArgs) predicates.FitPredicate {
				return predicates.NewNodeLabelPredicate(
					pluginArgs.NodeLabelArgs.PresentLabels,
					pluginArgs.NodeLabelArgs.AbsentLabels,
				)
			}
		}
	} else if predicateFactory, ok = fitPredicateMap[policyName]; ok {
		// checking to see if a pre-defined predicate is requested
		klog.V(2).Infof("Predicate type %s already registered, reusing.", policyName)
		return policyName
	}

	if predicateFactory == nil {
		klog.Fatalf("Invalid configuration: Predicate type not found for %s", policyName)
	}

	return RegisterFitPredicateFactory(policyName, predicateFactory)
}

// IsFitPredicateRegistered is useful for testing providers.
func IsFitPredicateRegistered(name string) bool {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()
	_, ok := fitPredicateMap[name]
	return ok
}

// RegisterPriorityMetadataProducerFactory registers a PriorityMetadataProducerFactory.
func RegisterPriorityMetadataProducerFactory(f PriorityMetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	priorityMetadataProducerFactory = f
}

// RegisterPredicateMetadataProducerFactory registers a MetadataProducer.
func RegisterPredicateMetadataProducerFactory(f PredicateMetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	predicateMetadataProducerFactory = f
}

// RegisterPriorityMapReduceFunction registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
func RegisterPriorityMapReduceFunction(
	name string,
	mapFunction priorities.PriorityMapFunction,
	reduceFunction priorities.PriorityReduceFunction,
	weight int) string {
	return RegisterPriorityConfigFactory(name, PriorityConfigFactory{
		MapReduceFunction: func(AlgorithmFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
			return mapFunction, reduceFunction
		},
		Weight: int64(weight),
	})
}

// RegisterPriorityConfigFactory registers a priority config factory with its name.
func RegisterPriorityConfigFactory(name string, pcf PriorityConfigFactory) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	priorityFunctionMap[name] = pcf
	return name
}

// RegisterCustomPriorityFunction registers a custom priority function with the algorithm registry.
// Returns the name, with which the priority function was registered.
func RegisterCustomPriorityFunction(policy schedulerapi.PriorityPolicy, configProducerArgs *plugins.ConfigProducerArgs) string {
	var pcf *PriorityConfigFactory
	name := policy.Name

	validatePriorityOrDie(policy)

	// generate the priority function, if a custom priority is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAntiAffinity != nil {
			// We use the ServiceAffinity plugin name for all ServiceAffinity custom priorities.
			// It may get called multiple times but we essentially only register one instance of
			// ServiceAffinity priority.
			// This name is then used to find the registered plugin and run the plugin instead of the priority.
			name = serviceaffinity.Name

			if configProducerArgs.ServiceAffinityArgs == nil {
				configProducerArgs.ServiceAffinityArgs = &serviceaffinity.Args{}
			}
			configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference = append(configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference, policy.Argument.ServiceAntiAffinity.Label)

			weight := policy.Weight
			schedulerFactoryMutex.RLock()
			if existing, ok := priorityFunctionMap[name]; ok {
				// If there are n ServiceAffinity priorities in the policy, the weight for the corresponding
				// score plugin is n*(weight of each priority).
				weight += existing.Weight
			}
			schedulerFactoryMutex.RUnlock()
			pcf = &PriorityConfigFactory{
				MapReduceFunction: func(args AlgorithmFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
					return priorities.NewServiceAntiAffinityPriority(
						args.SharedLister.Pods(),
						args.InformerFactory.Core().V1().Services().Lister(),
						configProducerArgs.ServiceAffinityArgs.AntiAffinityLabelsPreference,
					)
				},
				Weight: weight,
			}
		} else if policy.Argument.LabelPreference != nil {
			// We use the NodeLabel plugin name for all NodeLabel custom priorities.
			// It may get called multiple times but we essentially only register one instance of NodeLabel priority.
			// This name is then used to find the registered plugin and run the plugin instead of the priority.
			name = nodelabel.Name
			if configProducerArgs.NodeLabelArgs == nil {
				configProducerArgs.NodeLabelArgs = &nodelabel.Args{}
			}
			if policy.Argument.LabelPreference.Presence {
				configProducerArgs.NodeLabelArgs.PresentLabelsPreference = append(configProducerArgs.NodeLabelArgs.PresentLabelsPreference, policy.Argument.LabelPreference.Label)
			} else {
				configProducerArgs.NodeLabelArgs.AbsentLabelsPreference = append(configProducerArgs.NodeLabelArgs.AbsentLabelsPreference, policy.Argument.LabelPreference.Label)
			}
			weight := policy.Weight
			schedulerFactoryMutex.RLock()
			if existing, ok := priorityFunctionMap[name]; ok {
				// If there are n NodeLabel priority configured in the policy, the weight for the corresponding
				// priority is n*(weight of each priority in policy).
				weight += existing.Weight
			}
			schedulerFactoryMutex.RUnlock()
			pcf = &PriorityConfigFactory{
				MapReduceFunction: func(_ AlgorithmFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
					return priorities.NewNodeLabelPriority(
						configProducerArgs.NodeLabelArgs.PresentLabelsPreference,
						configProducerArgs.NodeLabelArgs.AbsentLabelsPreference,
					)
				},
				Weight: weight,
			}
		} else if policy.Argument.RequestedToCapacityRatioArguments != nil {
			scoringFunctionShape, resources := buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(policy.Argument.RequestedToCapacityRatioArguments)
			configProducerArgs.RequestedToCapacityRatioArgs = &requestedtocapacityratio.Args{
				FunctionShape:       scoringFunctionShape,
				ResourceToWeightMap: resources,
			}
			pcf = &PriorityConfigFactory{
				MapReduceFunction: func(args AlgorithmFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
					p := priorities.RequestedToCapacityRatioResourceAllocationPriority(scoringFunctionShape, resources)
					return p.PriorityMap, nil
				},
				Weight: policy.Weight,
			}
			// We do not allow specifying the name for custom plugins, see #83472
			name = requestedtocapacityratio.Name
		}
	} else if existingPcf, ok := priorityFunctionMap[name]; ok {
		klog.V(2).Infof("Priority type %s already registered, reusing.", name)
		// set/update the weight based on the policy
		pcf = &PriorityConfigFactory{
			MapReduceFunction: existingPcf.MapReduceFunction,
			Weight:            policy.Weight,
		}
	}

	if pcf == nil {
		klog.Fatalf("Invalid configuration: Priority type not found for %s", name)
	}

	return RegisterPriorityConfigFactory(name, *pcf)
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

// IsPriorityFunctionRegistered is useful for testing providers.
func IsPriorityFunctionRegistered(name string) bool {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()
	_, ok := priorityFunctionMap[name]
	return ok
}

// RegisterAlgorithmProvider registers a new algorithm provider with the algorithm registry.
func RegisterAlgorithmProvider(name string, predicateKeys, priorityKeys sets.String) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	algorithmProviderMap[name] = AlgorithmProviderConfig{
		FitPredicateKeys:     predicateKeys,
		PriorityFunctionKeys: priorityKeys,
	}
	return name
}

// GetAlgorithmProvider should not be used to modify providers. It is publicly visible for testing.
func GetAlgorithmProvider(name string) (*AlgorithmProviderConfig, error) {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	provider, ok := algorithmProviderMap[name]
	if !ok {
		return nil, fmt.Errorf("provider %q is not registered", name)
	}

	return &provider, nil
}

func getFitPredicateFunctions(names sets.String, args AlgorithmFactoryArgs) (map[string]predicates.FitPredicate, error) {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	fitPredicates := map[string]predicates.FitPredicate{}
	for _, name := range names.List() {
		factory, ok := fitPredicateMap[name]
		if !ok {
			return nil, fmt.Errorf("invalid predicate name %q specified - no corresponding function found", name)
		}
		fitPredicates[name] = factory(args)
	}

	// Always include mandatory fit predicates.
	for name := range mandatoryFitPredicates {
		if factory, found := fitPredicateMap[name]; found {
			fitPredicates[name] = factory(args)
		}
	}

	return fitPredicates, nil
}

func getPriorityMetadataProducer(args AlgorithmFactoryArgs) (priorities.MetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if priorityMetadataProducerFactory == nil {
		return priorities.EmptyMetadataProducer, nil
	}
	return priorityMetadataProducerFactory(args), nil
}

func getPredicateMetadataProducer(args AlgorithmFactoryArgs) (predicates.MetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if predicateMetadataProducerFactory == nil {
		return predicates.EmptyMetadataProducer, nil
	}
	return predicateMetadataProducerFactory(args), nil
}

func getPriorityFunctionConfigs(names sets.String, args AlgorithmFactoryArgs) ([]priorities.PriorityConfig, error) {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	var configs []priorities.PriorityConfig
	for _, name := range names.List() {
		factory, ok := priorityFunctionMap[name]
		if !ok {
			return nil, fmt.Errorf("invalid priority name %s specified - no corresponding function found", name)
		}
		mapFunction, reduceFunction := factory.MapReduceFunction(args)
		configs = append(configs, priorities.PriorityConfig{
			Name:   name,
			Map:    mapFunction,
			Reduce: reduceFunction,
			Weight: factory.Weight,
		})
	}
	if err := validateSelectedConfigs(configs); err != nil {
		return nil, err
	}
	return configs, nil
}

// validateSelectedConfigs validates the config weights to avoid the overflow.
func validateSelectedConfigs(configs []priorities.PriorityConfig) error {
	var totalPriority int64
	for _, config := range configs {
		// Checks totalPriority against MaxTotalScore to avoid overflow
		if config.Weight*framework.MaxNodeScore > framework.MaxTotalScore-totalPriority {
			return fmt.Errorf("total priority of priority functions has overflown")
		}
		totalPriority += config.Weight * framework.MaxNodeScore
	}
	return nil
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

// ListRegisteredFitPredicates returns the registered fit predicates.
func ListRegisteredFitPredicates() []string {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	var names []string
	for name := range fitPredicateMap {
		names = append(names, name)
	}
	return names
}

// ListRegisteredPriorityFunctions returns the registered priority functions.
func ListRegisteredPriorityFunctions() []string {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	var names []string
	for name := range priorityFunctionMap {
		names = append(names, name)
	}
	return names
}

// ListAlgorithmProviders is called when listing all available algorithm providers in `kube-scheduler --help`
func ListAlgorithmProviders() string {
	var availableAlgorithmProviders []string
	for name := range algorithmProviderMap {
		availableAlgorithmProviders = append(availableAlgorithmProviders, name)
	}
	sort.Strings(availableAlgorithmProviders)
	return strings.Join(availableAlgorithmProviders, " | ")
}
