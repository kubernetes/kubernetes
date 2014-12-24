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
	DefaultProvider = "default"
)

type AlgorithmProviderConfig struct {
	FitPredicateKeys     util.StringSet
	PriorityFunctionKeys util.StringSet
}

// RegisterFitPredicate registers a fit predicate with the algorithm registry. Returns the key,
// with which the predicate was registered.
func RegisterFitPredicate(key string, predicate algorithm.FitPredicate) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	fitPredicateMap[key] = predicate
	return key
}

// IsFitPredicateRegistered check is useful for testing providers.
func IsFitPredicateRegistered(key string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := fitPredicateMap[key]
	return ok
}

// RegisterFitPredicate registers a priority function with the algorithm registry. Returns the key,
// with which the function was registered.
func RegisterPriorityFunction(key string, function algorithm.PriorityFunction, weight int) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	priorityFunctionMap[key] = algorithm.PriorityConfig{Function: function, Weight: weight}
	return key
}

// IsPriorityFunctionRegistered check is useful for testing providers.
func IsPriorityFunctionRegistered(key string) bool {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	_, ok := priorityFunctionMap[key]
	return ok
}

// SetPriorityFunctionWeight sets the weight of an already registered priority function.
func SetPriorityFunctionWeight(key string, weight int) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
	config, ok := priorityFunctionMap[key]
	if !ok {
		glog.Errorf("Invalid priority key %s specified - no corresponding function found", key)
		return
	}
	config.Weight = weight
}

// RegisterAlgorithmProvider registers a new algorithm provider with the algorithm registry. This should
// be called from the init function in a provider plugin.
func RegisterAlgorithmProvider(name string, predicateKeys, priorityKeys util.StringSet) string {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()
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

func getFitPredicateFunctions(keys util.StringSet) ([]algorithm.FitPredicate, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	predicates := []algorithm.FitPredicate{}
	for _, key := range keys.List() {
		function, ok := fitPredicateMap[key]
		if !ok {
			return nil, fmt.Errorf("Invalid predicate key %q specified - no corresponding function found", key)
		}
		predicates = append(predicates, function)
	}
	return predicates, nil
}

func getPriorityFunctionConfigs(keys util.StringSet) ([]algorithm.PriorityConfig, error) {
	schedulerFactoryMutex.Lock()
	defer schedulerFactoryMutex.Unlock()

	configs := []algorithm.PriorityConfig{}
	for _, key := range keys.List() {
		config, ok := priorityFunctionMap[key]
		if !ok {
			return nil, fmt.Errorf("Invalid priority key %s specified - no corresponding function found", key)
		}
		configs = append(configs, config)
	}
	return configs, nil
}
