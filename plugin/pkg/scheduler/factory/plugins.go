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
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"

	"github.com/golang/glog"
)

// PluginFactoryArgs are passed to all plugin factory functions.
type PluginFactoryArgs struct {
	PodLister                      algorithm.PodLister
	ServiceLister                  algorithm.ServiceLister
	ControllerLister               algorithm.ControllerLister
	ReplicaSetLister               algorithm.ReplicaSetLister
	StatefulSetLister              algorithm.StatefulSetLister
	NodeLister                     algorithm.NodeLister
	NodeInfo                       predicates.NodeInfo
	PVInfo                         predicates.PersistentVolumeInfo
	PVCInfo                        predicates.PersistentVolumeClaimInfo
	HardPodAffinitySymmetricWeight int
}

// MetadataProducerFactory produces MetadataProducer from the given args.
// TODO: Rename this to PriorityMetadataProducerFactory.
type MetadataProducerFactory func(PluginFactoryArgs) algorithm.MetadataProducer

// PredicateMetadataProducerFactory produces PredicateMetadataProducer from the given args.
type PredicateMetadataProducerFactory func(PluginFactoryArgs) algorithm.PredicateMetadataProducer

// A FitPredicateFactory produces a FitPredicate from the given args.
type FitPredicateFactory func(PluginFactoryArgs) algorithm.FitPredicate

// DEPRECATED
// Use Map-Reduce pattern for priority functions.
// A PriorityFunctionFactory produces a PriorityConfig from the given args.
type PriorityFunctionFactory func(PluginFactoryArgs) algorithm.PriorityFunction

// A PriorityFunctionFactory produces map & reduce priority functions
// from a given args.
// FIXME: Rename to PriorityFunctionFactory.
type PriorityFunctionFactory2 func(PluginFactoryArgs) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction)

// A PriorityConfigFactory produces a PriorityConfig from the given function and weight
type PriorityConfigFactory struct {
	Function          PriorityFunctionFactory
	MapReduceFunction PriorityFunctionFactory2
	Weight            int
}

var (
	schedulerFactoryMutex sync.Mutex

	// maps that hold registered algorithm types
	fitPredicateMap        = make(map[string]FitPredicateFactory)
	mandatoryFitPredicates = sets.NewString()
	priorityFunctionMap    = make(map[string]PriorityConfigFactory)
	algorithmProviderMap   = make(map[string]AlgorithmProviderConfig)

	// Registered metadata producers
	priorityMetadataProducer  MetadataProducerFactory
	predicateMetadataProducer PredicateMetadataProducerFactory

	// get equivalence pod function
	getEquivalencePodFunc algorithm.GetEquivalencePodFunc
)

const (
	DefaultProvider = "DefaultProvider"
)

type AlgorithmProviderConfig struct {
	FitPredicateKeys     sets.String
	PriorityFunctionKeys sets.String
}

// RegisterFitPredicate registers a fit predicate with the algorithm
// registry. Returns the name with which the predicate was registered.
func RegisterFitPredicate(name string, predicate algorithm.FitPredicate) string {
	return RegisterFitPredicateFactory(name, func(PluginFactoryArgs) algorithm.FitPredicate { return predicate })
}

// RegisterMandatoryFitPredicate registers a fit predicate with the algorithm registry, the predicate is used by
// kubelet, DaemonSet; it is always included in configuration. Returns the name with which the predicate was
// registered.
func RegisterMandatoryFitPredicate(name string, predicate algorithm.FitPredicate) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = func(PluginFactoryArgs) algorithm.FitPredicate { return predicate }
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
			predicateFactory = func(args PluginFactoryArgs) algorithm.FitPredicate {
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
			predicateFactory = func(args PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewNodeLabelPredicate(
					policy.Argument.LabelsPresence.Labels,
					policy.Argument.LabelsPresence.Presence,
				)
			}
		}
	} else if predicateFactory, ok = fitPredicateMap[policy.Name]; ok {
		// checking to see if a pre-defined predicate is requested
		glog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
		return policy.Name
	}

	if predicateFactory == nil {
		glog.Fatalf("Invalid configuration: Predicate type not found for %s", policy.Name)
	}

	return RegisterFitPredicateFactory(policy.Name, predicateFactory)
}

// IsFitPredicateRegistered is useful for testing providers.
func IsFitPredicateRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := fitPredicateMap[name]
	return ok
}

func RegisterPriorityMetadataProducerFactory(factory MetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	priorityMetadataProducer = factory
}

func RegisterPredicateMetadataProducerFactory(factory PredicateMetadataProducerFactory) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	predicateMetadataProducer = factory
}

// DEPRECATED
// Use Map-Reduce pattern for priority functions.
// Registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
func RegisterPriorityFunction(name string, function algorithm.PriorityFunction, weight int) string {
	return RegisterPriorityConfigFactory(name, PriorityConfigFactory{
		Function: func(PluginFactoryArgs) algorithm.PriorityFunction {
			return function
		},
		Weight: weight,
	})
}

// RegisterPriorityFunction2 registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
// FIXME: Rename to PriorityFunctionFactory.
func RegisterPriorityFunction2(
	name string,
	mapFunction algorithm.PriorityMapFunction,
	reduceFunction algorithm.PriorityReduceFunction,
	weight int) string {
	return RegisterPriorityConfigFactory(name, PriorityConfigFactory{
		MapReduceFunction: func(PluginFactoryArgs) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
			return mapFunction, reduceFunction
		},
		Weight: weight,
	})
}

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
				Function: func(args PluginFactoryArgs) algorithm.PriorityFunction {
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
				MapReduceFunction: func(args PluginFactoryArgs) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
					return priorities.NewNodeLabelPriority(
						policy.Argument.LabelPreference.Label,
						policy.Argument.LabelPreference.Presence,
					)
				},
				Weight: policy.Weight,
			}
		}
	} else if existingPcf, ok := priorityFunctionMap[policy.Name]; ok {
		glog.V(2).Infof("Priority type %s already registered, reusing.", policy.Name)
		// set/update the weight based on the policy
		pcf = &PriorityConfigFactory{
			Function:          existingPcf.Function,
			MapReduceFunction: existingPcf.MapReduceFunction,
			Weight:            policy.Weight,
		}
	}

	if pcf == nil {
		glog.Fatalf("Invalid configuration: Priority type not found for %s", policy.Name)
	}

	return RegisterPriorityConfigFactory(policy.Name, *pcf)
}

func RegisterGetEquivalencePodFunction(equivalenceFunc algorithm.GetEquivalencePodFunc) {
	getEquivalencePodFunc = equivalenceFunc
}

// IsPriorityFunctionRegistered is useful for testing providers.
func IsPriorityFunctionRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
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
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	provider, ok := algorithmProviderMap[name]
	if !ok {
		return nil, fmt.Errorf("plugin %q has not been registered", name)
	}

	return &provider, nil
}

func getFitPredicateFunctions(names sets.String, args PluginFactoryArgs) (map[string]algorithm.FitPredicate, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	predicates := map[string]algorithm.FitPredicate{}
	for _, name := range names.List() {
		factory, ok := fitPredicateMap[name]
		if !ok {
			return nil, fmt.Errorf("Invalid predicate name %q specified - no corresponding function found", name)
		}
		predicates[name] = factory(args)
	}

	// Always include mandatory fit predicates.
	for name := range mandatoryFitPredicates {
		if factory, found := fitPredicateMap[name]; found {
			predicates[name] = factory(args)
		}
	}

	return predicates, nil
}

func getPriorityMetadataProducer(args PluginFactoryArgs) (algorithm.MetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if priorityMetadataProducer == nil {
		return algorithm.EmptyMetadataProducer, nil
	}
	return priorityMetadataProducer(args), nil
}

func getPredicateMetadataProducer(args PluginFactoryArgs) (algorithm.PredicateMetadataProducer, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	if predicateMetadataProducer == nil {
		return algorithm.EmptyPredicateMetadataProducer, nil
	}
	return predicateMetadataProducer(args), nil
}

func getPriorityFunctionConfigs(names sets.String, args PluginFactoryArgs) ([]algorithm.PriorityConfig, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	configs := []algorithm.PriorityConfig{}
	for _, name := range names.List() {
		factory, ok := priorityFunctionMap[name]
		if !ok {
			return nil, fmt.Errorf("Invalid priority name %s specified - no corresponding function found", name)
		}
		if factory.Function != nil {
			configs = append(configs, algorithm.PriorityConfig{
				Function: factory.Function(args),
				Weight:   factory.Weight,
			})
		} else {
			mapFunction, reduceFunction := factory.MapReduceFunction(args)
			configs = append(configs, algorithm.PriorityConfig{
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
func validateSelectedConfigs(configs []algorithm.PriorityConfig) error {
	var totalPriority int
	for _, config := range configs {
		// Checks totalPriority against MaxTotalPriority to avoid overflow
		if config.Weight*schedulerapi.MaxPriority > schedulerapi.MaxTotalPriority-totalPriority {
			return fmt.Errorf("Total priority of priority functions has overflown")
		}
		totalPriority += config.Weight * schedulerapi.MaxPriority
	}
	return nil
}

var validName = regexp.MustCompile("^[a-zA-Z0-9]([-a-zA-Z0-9]*[a-zA-Z0-9])$")

func validateAlgorithmNameOrDie(name string) {
	if !validName.MatchString(name) {
		glog.Fatalf("Algorithm name %v does not match the name validation regexp \"%v\".", name, validName)
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
			glog.Fatalf("Exactly 1 predicate argument is required, numArgs: %v, Predicate: %s", numArgs, predicate.Name)
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
		if numArgs != 1 {
			glog.Fatalf("Exactly 1 priority argument is required, numArgs: %v, Priority: %s", numArgs, priority.Name)
		}
	}
}

func ListRegisteredFitPredicates() []string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	names := []string{}
	for name := range fitPredicateMap {
		names = append(names, name)
	}
	return names
}

func ListRegisteredPriorityFunctions() []string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	names := []string{}
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
