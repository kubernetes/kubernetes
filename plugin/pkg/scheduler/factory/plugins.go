/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"sync"

	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"

	"github.com/golang/glog"
)

// PluginFactoryArgs are passed to all plugin factory functions.
type PluginFactoryArgs struct {
	algorithm.PodLister
	algorithm.ServiceLister
	algorithm.ControllerLister
	NodeLister algorithm.NodeLister
	NodeInfo   predicates.NodeInfo
}

// A FitPredicateFactory produces a FitPredicate from the given args.
type FitPredicateFactory func(PluginFactoryArgs) algorithm.FitPredicate

// A PriorityFunctionFactory produces a PriorityConfig from the given args.
type PriorityFunctionFactory func(PluginFactoryArgs) algorithm.PriorityFunction

// A PriorityConfigFactory produces a PriorityConfig from the given function and weight
type PriorityConfigFactory struct {
	Function PriorityFunctionFactory
	Weight   int
}

var (
	schedulerFactoryMutex sync.Mutex

	// maps that hold registered algorithm types
	fitPredicateMap      = make(map[string]FitPredicateFactory)
	priorityFunctionMap  = make(map[string]PriorityConfigFactory)
	algorithmProviderMap = make(map[string]AlgorithmProviderConfig)
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

// RegisterFitPredicateFactory registers a fit predicate factory with the
// algorithm registry. Returns the name with which the predicate was registered.
func RegisterFitPredicateFactory(name string, predicateFactory FitPredicateFactory) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = predicateFactory
	return name
}

// Registers a custom fit predicate with the algorithm registry.
// Returns the name, with which the predicate was registered.
func RegisterCustomFitPredicate(policy schedulerapi.PredicatePolicy) string {
	var predicateFactory FitPredicateFactory
	var ok bool

	validatePredicateOrDie(policy)

	// generate the predicate function, if a custom type is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAffinity != nil {
			predicateFactory = func(args PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewServiceAffinityPredicate(
					args.PodLister,
					args.ServiceLister,
					args.NodeInfo,
					policy.Argument.ServiceAffinity.Labels,
				)
			}
		} else if policy.Argument.LabelsPresence != nil {
			predicateFactory = func(args PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewNodeLabelPredicate(
					args.NodeInfo,
					policy.Argument.LabelsPresence.Labels,
					policy.Argument.LabelsPresence.Presence,
				)
			}
		}
	} else if predicateFactory, ok = fitPredicateMap[policy.Name]; ok {
		// checking to see if a pre-defined predicate is requested
		glog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
	}

	if predicateFactory == nil {
		glog.Fatalf("Invalid configuration: Predicate type not found for %s", policy.Name)
	}

	return RegisterFitPredicateFactory(policy.Name, predicateFactory)
}

// This check is useful for testing providers.
func IsFitPredicateRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := fitPredicateMap[name]
	return ok
}

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

func RegisterPriorityConfigFactory(name string, pcf PriorityConfigFactory) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	priorityFunctionMap[name] = pcf
	return name
}

// Registers a custom priority function with the algorithm registry.
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
						args.ServiceLister,
						policy.Argument.ServiceAntiAffinity.Label,
					)
				},
				Weight: policy.Weight,
			}
		} else if policy.Argument.LabelPreference != nil {
			pcf = &PriorityConfigFactory{
				Function: func(args PluginFactoryArgs) algorithm.PriorityFunction {
					return priorities.NewNodeLabelPriority(
						policy.Argument.LabelPreference.Label,
						policy.Argument.LabelPreference.Presence,
					)
				},
				Weight: policy.Weight,
			}
		}
	} else if existing_pcf, ok := priorityFunctionMap[policy.Name]; ok {
		glog.V(2).Infof("Priority type %s already registered, reusing.", policy.Name)
		// set/update the weight based on the policy
		pcf = &PriorityConfigFactory{
			Function: existing_pcf.Function,
			Weight:   policy.Weight,
		}
	}

	if pcf == nil {
		glog.Fatalf("Invalid configuration: Priority type not found for %s", policy.Name)
	}

	return RegisterPriorityConfigFactory(policy.Name, *pcf)
}

// This check is useful for testing providers.
func IsPriorityFunctionRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := priorityFunctionMap[name]
	return ok
}

// Registers a new algorithm provider with the algorithm registry. This should
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

// This function should not be used to modify providers. It is publicly visible for testing.
func GetAlgorithmProvider(name string) (*AlgorithmProviderConfig, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	var provider AlgorithmProviderConfig
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
	return predicates, nil
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
		configs = append(configs, algorithm.PriorityConfig{
			Function: factory.Function(args),
			Weight:   factory.Weight,
		})
	}
	return configs, nil
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
			glog.Fatalf("Exactly 1 predicate argument is required, numArgs: %v", numArgs)
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
			glog.Fatalf("Exactly 1 priority argument is required")
		}
	}
}

// ListAlgorithmProviders is called when listing all available algortihm providers in `kube-scheduler --help`
func ListAlgorithmProviders() string {
	var availableAlgorithmProviders []string
	for name := range algorithmProviderMap {
		availableAlgorithmProviders = append(availableAlgorithmProviders, name)
	}
	return strings.Join(availableAlgorithmProviders, " | ")
}
