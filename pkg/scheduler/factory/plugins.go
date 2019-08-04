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

package factory

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"

	"k8s.io/klog"
)

// PluginFactoryArgs are passed to all plugin factory functions.
type PluginFactoryArgs struct {
	PodLister                      algorithm.PodLister
	ServiceLister                  algorithm.ServiceLister
	ControllerLister               algorithm.ControllerLister
	ReplicaSetLister               algorithm.ReplicaSetLister
	StatefulSetLister              algorithm.StatefulSetLister
	NodeLister                     algorithm.NodeLister
	PDBLister                      algorithm.PDBLister
	NodeInfo                       predicates.NodeInfo
	CSINodeInfo                    predicates.CSINodeInfo
	PVInfo                         predicates.PersistentVolumeInfo
	PVCInfo                        predicates.PersistentVolumeClaimInfo
	StorageClassInfo               predicates.StorageClassInfo
	VolumeBinder                   *volumebinder.VolumeBinder
	HardPodAffinitySymmetricWeight int32
}

// PriorityMetadataProducerFactory produces PriorityMetadataProducer from the given args.
type PriorityMetadataProducerFactory func(PluginFactoryArgs) priorities.PriorityMetadataProducer

// PredicateMetadataProducerFactory produces PredicateMetadataProducer from the given args.
type PredicateMetadataProducerFactory func(PluginFactoryArgs) predicates.PredicateMetadataProducer

// FitPredicateFactory produces a FitPredicate from the given args.
type FitPredicateFactory func(PluginFactoryArgs) predicates.FitPredicate

// PriorityFunctionFactory produces a PriorityConfig from the given args.
// DEPRECATED
// Use Map-Reduce pattern for priority functions.
type PriorityFunctionFactory func(PluginFactoryArgs) priorities.PriorityFunction

// PriorityFunctionFactory2 produces map & reduce priority functions
// from a given args.
// FIXME: Rename to PriorityFunctionFactory.
type PriorityFunctionFactory2 func(PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction)

// PriorityConfigFactory produces a PriorityConfig from the given function and weight
type PriorityConfigFactory struct {
	Function          PriorityFunctionFactory
	MapReduceFunction PriorityFunctionFactory2
	Weight            int
}

var (
	schedulerFactoryMutex sync.RWMutex

	// maps that hold registered algorithm types
	fitPredicateMap        = make(map[string]FitPredicateFactory)
	mandatoryFitPredicates = sets.NewString()
	priorityFunctionMap    = make(map[string]PriorityConfigFactory)
	algorithmProviderMap   = make(map[string]AlgorithmProviderConfig)

	// Registered metadata producers
	priorityMetadataProducer  PriorityMetadataProducerFactory
	predicateMetadataProducer PredicateMetadataProducerFactory
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

// Copy returns a snapshot of current registered predicates and priorities.
func Copy() *Snapshot {
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

// Apply sets state of predicates and priorities to `s`.
func Apply(s *Snapshot) {
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
	return RegisterFitPredicateFactory(name, func(PluginFactoryArgs) predicates.FitPredicate { return predicate })
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
		return fmt.Errorf("plugin %v has not been registered", providerName)
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
		return fmt.Errorf("plugin %v has not been registered", providerName)
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
	return
}

// InsertPriorityKeyToAlgorithmProviderMap inserts a priority function to all algorithmProviders which are in algorithmProviderMap.
func InsertPriorityKeyToAlgorithmProviderMap(key string) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	for _, provider := range algorithmProviderMap {
		provider.PriorityFunctionKeys.Insert(key)
	}
	return
}

// RegisterMandatoryFitPredicate registers a fit predicate with the algorithm registry, the predicate is used by
// kubelet, DaemonSet; it is always included in configuration. Returns the name with which the predicate was
// registered.
func RegisterMandatoryFitPredicate(name string, predicate predicates.FitPredicate) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = func(PluginFactoryArgs) predicates.FitPredicate { return predicate }
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
func RegisterCustomFitPredicate(policy schedulerapi.PredicatePolicy) string {
	var predicateFactory FitPredicateFactory
	var ok bool

	validatePredicateOrDie(policy)

	// generate the predicate function, if a custom type is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAffinity != nil {
			predicateFactory = func(args PluginFactoryArgs) predicates.FitPredicate {
				predicate, precomputationFunction := predicates.NewServiceAffinityPredicate(
					args.PodLister,
					args.ServiceLister,
					args.NodeInfo,
					policy.Argument.ServiceAffinity.Labels,
				)

				// Once we generate the predicate we should also Register the Precomputation
				predicates.RegisterPredicateMetadataProducer(policy.Name, precomputationFunction)
				return predicate
			}
		} else if policy.Argument.LabelsPresence != nil {
			predicateFactory = func(args PluginFactoryArgs) predicates.FitPredicate {
				return predicates.NewNodeLabelPredicate(
					policy.Argument.LabelsPresence.Labels,
					policy.Argument.LabelsPresence.Presence,
				)
			}
		}
	} else if predicateFactory, ok = fitPredicateMap[policy.Name]; ok {
		// checking to see if a pre-defined predicate is requested
		klog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
		return policy.Name
	}

	if predicateFactory == nil {
		klog.Fatalf("Invalid configuration: Predicate type not found for %s", policy.Name)
	}

	return RegisterFitPredicateFactory(policy.Name, predicateFactory)
}

// IsFitPredicateRegistered is useful for testing providers.
func IsFitPredicateRegistered(name string) bool {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()
	_, ok := fitPredicateMap[name]
	return ok
}

// RegisterPriorityMetadataProducerFactory registers a PriorityMetadataProducerFactory.
func RegisterPriorityMetadataProducerFactory(factory PriorityMetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	priorityMetadataProducer = factory
}

// RegisterPredicateMetadataProducerFactory registers a PredicateMetadataProducerFactory.
func RegisterPredicateMetadataProducerFactory(factory PredicateMetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	predicateMetadataProducer = factory
}

// RegisterPriorityFunction registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
// DEPRECATED
// Use Map-Reduce pattern for priority functions.
func RegisterPriorityFunction(name string, function priorities.PriorityFunction, weight int) string {
	return RegisterPriorityConfigFactory(name, PriorityConfigFactory{
		Function: func(PluginFactoryArgs) priorities.PriorityFunction {
			return function
		},
		Weight: weight,
	})
}

// RegisterPriorityMapReduceFunction registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
func RegisterPriorityMapReduceFunction(
	name string,
	mapFunction priorities.PriorityMapFunction,
	reduceFunction priorities.PriorityReduceFunction,
	weight int) string {
	return RegisterPriorityConfigFactory(name, PriorityConfigFactory{
		MapReduceFunction: func(PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
			return mapFunction, reduceFunction
		},
		Weight: weight,
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
func RegisterCustomPriorityFunction(policy schedulerapi.PriorityPolicy) string {
	var pcf *PriorityConfigFactory

	validatePriorityOrDie(policy)

	// generate the priority function, if a custom priority is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAntiAffinity != nil {
			pcf = &PriorityConfigFactory{
				MapReduceFunction: func(args PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
					return priorities.NewServiceAntiAffinityPriority(
						args.PodLister,
						args.ServiceLister,
						policy.Argument.ServiceAntiAffinity.Label,
					)
				},
				Weight: policy.Weight,
			}
		} else if policy.Argument.LabelPreference != nil {
			pcf = &PriorityConfigFactory{
				MapReduceFunction: func(args PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
					return priorities.NewNodeLabelPriority(
						policy.Argument.LabelPreference.Label,
						policy.Argument.LabelPreference.Presence,
					)
				},
				Weight: policy.Weight,
			}
		} else if policy.Argument.RequestedToCapacityRatioArguments != nil {
			pcf = &PriorityConfigFactory{
				MapReduceFunction: func(args PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
					scoringFunctionShape := buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(policy.Argument.RequestedToCapacityRatioArguments)
					p := priorities.RequestedToCapacityRatioResourceAllocationPriority(scoringFunctionShape)
					return p.PriorityMap, nil
				},
				Weight: policy.Weight,
			}
		}
	} else if existingPcf, ok := priorityFunctionMap[policy.Name]; ok {
		klog.V(2).Infof("Priority type %s already registered, reusing.", policy.Name)
		// set/update the weight based on the policy
		pcf = &PriorityConfigFactory{
			Function:          existingPcf.Function,
			MapReduceFunction: existingPcf.MapReduceFunction,
			Weight:            policy.Weight,
		}
	}

	if pcf == nil {
		klog.Fatalf("Invalid configuration: Priority type not found for %s", policy.Name)
	}

	return RegisterPriorityConfigFactory(policy.Name, *pcf)
}

func buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(arguments *schedulerapi.RequestedToCapacityRatioArguments) priorities.FunctionShape {
	n := len(arguments.UtilizationShape)
	points := make([]priorities.FunctionShapePoint, 0, n)
	for _, point := range arguments.UtilizationShape {
		points = append(points, priorities.FunctionShapePoint{Utilization: int64(point.Utilization), Score: int64(point.Score)})
	}
	shape, err := priorities.NewFunctionShape(points)
	if err != nil {
		klog.Fatalf("invalid RequestedToCapacityRatioPriority arguments: %s", err.Error())
	}
	return shape
}

// IsPriorityFunctionRegistered is useful for testing providers.
func IsPriorityFunctionRegistered(name string) bool {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()
	_, ok := priorityFunctionMap[name]
	return ok
}

// RegisterAlgorithmProvider registers a new algorithm provider with the algorithm registry. This should
// be called from the init function in a provider plugin.
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
		return nil, fmt.Errorf("plugin %q has not been registered", name)
	}

	return &provider, nil
}

func getFitPredicateFunctions(names sets.String, args PluginFactoryArgs) (map[string]predicates.FitPredicate, error) {
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

func getPriorityMetadataProducer(args PluginFactoryArgs) (priorities.PriorityMetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if priorityMetadataProducer == nil {
		return priorities.EmptyPriorityMetadataProducer, nil
	}
	return priorityMetadataProducer(args), nil
}

func getPredicateMetadataProducer(args PluginFactoryArgs) (predicates.PredicateMetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if predicateMetadataProducer == nil {
		return predicates.EmptyPredicateMetadataProducer, nil
	}
	return predicateMetadataProducer(args), nil
}

func getPriorityFunctionConfigs(names sets.String, args PluginFactoryArgs) ([]priorities.PriorityConfig, error) {
	schedulerFactoryMutex.RLock()
	defer schedulerFactoryMutex.RUnlock()

	var configs []priorities.PriorityConfig
	for _, name := range names.List() {
		factory, ok := priorityFunctionMap[name]
		if !ok {
			return nil, fmt.Errorf("invalid priority name %s specified - no corresponding function found", name)
		}
		if factory.Function != nil {
			configs = append(configs, priorities.PriorityConfig{
				Name:     name,
				Function: factory.Function(args),
				Weight:   factory.Weight,
			})
		} else {
			mapFunction, reduceFunction := factory.MapReduceFunction(args)
			configs = append(configs, priorities.PriorityConfig{
				Name:   name,
				Map:    mapFunction,
				Reduce: reduceFunction,
				Weight: factory.Weight,
			})
		}
	}
	if err := validateSelectedConfigs(configs); err != nil {
		return nil, err
	}
	return configs, nil
}

// validateSelectedConfigs validates the config weights to avoid the overflow.
func validateSelectedConfigs(configs []priorities.PriorityConfig) error {
	var totalPriority int
	for _, config := range configs {
		// Checks totalPriority against MaxTotalPriority to avoid overflow
		if config.Weight*schedulerapi.MaxPriority > schedulerapi.MaxTotalPriority-totalPriority {
			return fmt.Errorf("total priority of priority functions has overflown")
		}
		totalPriority += config.Weight * schedulerapi.MaxPriority
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
