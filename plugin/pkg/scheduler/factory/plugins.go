/*
Copyright 2014 Google Inc. All rights reserved.

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
	"sync"

	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	schedulerapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/api"

	"github.com/golang/glog"
)

var (
	schedulerFactoryMutex sync.Mutex

	// maps that hold registered algorithm types
	fitPredicateMap      = make(map[string]algorithm.FitPredicate)
	priorityFunctionMap  = make(map[string]algorithm.PriorityConfig)
	algorithmProviderMap = make(map[string]AlgorithmProviderConfig)
)

const (
	DefaultProvider = "DefaultProvider"
)

type AlgorithmProviderConfig struct {
	FitPredicateKeys     util.StringSet
	PriorityFunctionKeys util.StringSet
}

// Registers a fit predicate with the algorithm registry. Returns the name,
// with which the predicate was registered.
func RegisterFitPredicate(name string, predicate algorithm.FitPredicate) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = predicate
	return name
}

// Registers a custom fit predicate with the algorithm registry.
// Returns the name, with which the predicate was registered.
func RegisterCustomFitPredicate(policy schedulerapi.PredicatePolicy) string {
	var predicate algorithm.FitPredicate
	var ok bool

	validatePredicateOrDie(policy)

	// generate the predicate function, if a custom type is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAffinity != nil {
			predicate = algorithm.NewServiceAffinityPredicate(PodLister, ServiceLister, MinionLister, policy.Argument.ServiceAffinity.Labels)
		} else if policy.Argument.LabelsPresence != nil {
			predicate = algorithm.NewNodeLabelPredicate(MinionLister, policy.Argument.LabelsPresence.Labels, policy.Argument.LabelsPresence.Presence)
		}
		// check to see if a pre-defined predicate is requested
	} else if predicate, ok = fitPredicateMap[policy.Name]; ok {
		glog.V(2).Infof("Predicate type %s already registered, reusing.", policy.Name)
	}

	if predicate == nil {
		glog.Fatalf("Invalid configuration: Predicate type not found for %s", policy.Name)
	}

	return RegisterFitPredicate(policy.Name, predicate)
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
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	priorityFunctionMap[name] = algorithm.PriorityConfig{Function: function, Weight: weight}
	return name
}

// Registers a custom priority function with the algorithm registry.
// Returns the name, with which the priority function was registered.
func RegisterCustomPriorityFunction(policy schedulerapi.PriorityPolicy) string {
	var priority algorithm.PriorityFunction

	validatePriorityOrDie(policy)

	// generate the priority function, if a custom priority is requested
	if policy.Argument != nil {
		if policy.Argument.ServiceAntiAffinity != nil {
			priority = algorithm.NewServiceAntiAffinityPriority(ServiceLister, policy.Argument.ServiceAntiAffinity.Label)
		} else if policy.Argument.LabelPreference != nil {
			priority = algorithm.NewNodeLabelPriority(policy.Argument.LabelPreference.Label, policy.Argument.LabelPreference.Presence)
		}
	} else if priorityConfig, ok := priorityFunctionMap[policy.Name]; ok {
		glog.V(2).Infof("Priority type %s already registered, reusing.", policy.Name)
		priority = priorityConfig.Function
	}

	if priority == nil {
		glog.Fatalf("Invalid configuration: Priority type not found for %s", policy.Name)
	}

	return RegisterPriorityFunction(policy.Name, priority, policy.Weight)
}

// This check is useful for testing providers.
func IsPriorityFunctionRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := priorityFunctionMap[name]
	return ok
}

// Sets the weight of an already registered priority function.
func SetPriorityFunctionWeight(name string, weight int) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	config, ok := priorityFunctionMap[name]
	if !ok {
		glog.Errorf("Invalid priority name %s specified - no corresponding function found", name)
		return
	}
	config.Weight = weight
	priorityFunctionMap[name] = config
}

// Registers a new algorithm provider with the algorithm registry. This should
// be called from the init function in a provider plugin.
func RegisterAlgorithmProvider(name string, predicateKeys, priorityKeys util.StringSet) string {
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

func getFitPredicateFunctions(names util.StringSet) (map[string]algorithm.FitPredicate, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	predicates := map[string]algorithm.FitPredicate{}
	for _, name := range names.List() {
		function, ok := fitPredicateMap[name]
		if !ok {
			return nil, fmt.Errorf("Invalid predicate name %q specified - no corresponding function found", name)
		}
		predicates[name] = function
	}
	return predicates, nil
}

func getPriorityFunctionConfigs(names util.StringSet) ([]algorithm.PriorityConfig, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	configs := []algorithm.PriorityConfig{}
	for _, name := range names.List() {
		config, ok := priorityFunctionMap[name]
		if !ok {
			return nil, fmt.Errorf("Invalid priority name %s specified - no corresponding function found", name)
		}
		configs = append(configs, config)
	}
	return configs, nil
}

var validName = regexp.MustCompile("^[a-zA-Z0-9]([-a-zA-Z0-9]*[a-zA-Z0-9])$")

func validateAlgorithmNameOrDie(name string) {
	if !validName.MatchString(name) {
		glog.Fatalf("algorithm name %v does not match the name validation regexp \"%v\".", name, validName)
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
			glog.Fatalf("Exactly 1 predicate argument is required")
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
