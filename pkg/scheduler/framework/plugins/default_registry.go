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

	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	noop "k8s.io/kubernetes/pkg/scheduler/framework/plugins/noop"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// NewDefaultRegistry builds a default registry with all the default plugins.
// This is the registry that Kubernetes default scheduler uses. A scheduler that
// runs custom plugins, can pass a different Registry when initializing the scheduler.
func NewDefaultRegistry() framework.Registry {
	return framework.Registry{
		// This is just a test plugin to showcase the setup, it should be deleted once
		// we have at least one legitimate plugin here.
		noop.Name: noop.New,
	}
}

// ConfigProducerArgs contains arguments that are passed to the producer.
// As we add more predicates/priorities to framework plugins mappings, more arguments
// may be added here.
type ConfigProducerArgs struct {
	// Weight used for priority functions.
	Weight int32
}

// ConfigProducer produces a framework's configuration.
type ConfigProducer func(args ConfigProducerArgs) (config.Plugins, []config.PluginConfig)

// ConfigProducerRegistry tracks mappings from predicates/priorities to framework config producers.
type ConfigProducerRegistry struct {
	// maps that associate predicates/priorities with framework plugin configurations.
	PredicateToConfigProducer map[string]ConfigProducer
	PriorityToConfigProducer  map[string]ConfigProducer
}

// NewConfigProducerRegistry creates a new producer registry.
func NewConfigProducerRegistry() *ConfigProducerRegistry {
	return &ConfigProducerRegistry{
		PredicateToConfigProducer: make(map[string]ConfigProducer),
		PriorityToConfigProducer:  make(map[string]ConfigProducer),
	}
}

func registerProducer(name string, producer ConfigProducer, producersMap map[string]ConfigProducer) error {
	if _, exist := producersMap[name]; exist {
		return fmt.Errorf("already registered %q", name)
	}
	producersMap[name] = producer
	return nil
}

// RegisterPredicate registers a config producer for a predicate.
func (f *ConfigProducerRegistry) RegisterPredicate(name string, producer ConfigProducer) error {
	return registerProducer(name, producer, f.PredicateToConfigProducer)
}

// RegisterPriority registers a framework config producer for a priority.
func (f *ConfigProducerRegistry) RegisterPriority(name string, producer ConfigProducer) error {
	return registerProducer(name, producer, f.PriorityToConfigProducer)
}
