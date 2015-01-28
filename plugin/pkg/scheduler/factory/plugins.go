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

// RegisterFitPredicate registers a fit predicate with the algorithm registry. Returns the name,
// with which the predicate was registered.
func RegisterFitPredicate(name string, predicate algorithm.FitPredicate) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	fitPredicateMap[name] = predicate
	return name
}

// IsFitPredicateRegistered check is useful for testing providers.
func IsFitPredicateRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := fitPredicateMap[name]
	return ok
}

// RegisterFitPredicate registers a priority function with the algorithm registry. Returns the name,
// with which the function was registered.
func RegisterPriorityFunction(name string, function algorithm.PriorityFunction, weight int) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	validateAlgorithmNameOrDie(name)
	priorityFunctionMap[name] = algorithm.PriorityConfig{Function: function, Weight: weight}
	return name
}

// IsPriorityFunctionRegistered check is useful for testing providers.
func IsPriorityFunctionRegistered(name string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := priorityFunctionMap[name]
	return ok
}

// SetPriorityFunctionWeight sets the weight of an already registered priority function.
func SetPriorityFunctionWeight(name string, weight int) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	config, ok := priorityFunctionMap[name]
	if !ok {
		glog.Errorf("Invalid priority name %s specified - no corresponding function found", name)
		return
	}
	config.Weight = weight
}

// RegisterAlgorithmProvider registers a new algorithm provider with the algorithm registry. This should
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

// GetAlgorithmProvider should not be used to modify providers. It is publicly visible for testing.
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

func getFitPredicateFunctions(names util.StringSet) ([]algorithm.FitPredicate, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	predicates := []algorithm.FitPredicate{}
	for _, name := range names.List() {
		function, ok := fitPredicateMap[name]
		if !ok {
			return nil, fmt.Errorf("Invalid predicate name %q specified - no corresponding function found", name)
		}
		predicates = append(predicates, function)
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
