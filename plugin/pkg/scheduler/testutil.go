/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"
)

// FakeConfigurator is an implementation for test.
type FakeConfigurator struct {
	Config *Config
}

// GetPriorityFunctionConfigs is not implemented yet.
func (fc *FakeConfigurator) GetPriorityFunctionConfigs(priorityKeys sets.String) ([]algorithm.PriorityConfig, error) {
	return nil, fmt.Errorf("not implemented")
}

// GetPriorityMetadataProducer is not implemented yet.
func (fc *FakeConfigurator) GetPriorityMetadataProducer() (algorithm.MetadataProducer, error) {
	return nil, fmt.Errorf("not implemented")
}

// GetPredicateMetadataProducer is not implemented yet.
func (fc *FakeConfigurator) GetPredicateMetadataProducer() (algorithm.PredicateMetadataProducer, error) {
	return nil, fmt.Errorf("not implemented")
}

// GetPredicates is not implemented yet.
func (fc *FakeConfigurator) GetPredicates(predicateKeys sets.String) (map[string]algorithm.FitPredicate, error) {
	return nil, fmt.Errorf("not implemented")
}

// GetHardPodAffinitySymmetricWeight is not implemented yet.
func (fc *FakeConfigurator) GetHardPodAffinitySymmetricWeight() int {
	panic("not implemented")
}

// GetSchedulerName is not implemented yet.
func (fc *FakeConfigurator) GetSchedulerName() string {
	panic("not implemented")
}

// MakeDefaultErrorFunc is not implemented yet.
func (fc *FakeConfigurator) MakeDefaultErrorFunc(backoff *util.PodBackoff, podQueue *cache.FIFO) func(pod *v1.Pod, err error) {
	return nil
}

// ResponsibleForPod is not implemented yet.
func (fc *FakeConfigurator) ResponsibleForPod(pod *v1.Pod) bool {
	panic("not implemented")
}

// GetNodeLister is not implemented yet.
func (fc *FakeConfigurator) GetNodeLister() corelisters.NodeLister {
	return nil
}

// GetClient is not implemented yet.
func (fc *FakeConfigurator) GetClient() clientset.Interface {
	return nil
}

// GetScheduledPodLister is not implemented yet.
func (fc *FakeConfigurator) GetScheduledPodLister() corelisters.PodLister {
	return nil
}

// Run is not implemented yet.
func (fc *FakeConfigurator) Run() {
	panic("not implemented")
}

// Create returns FakeConfigurator.Config
func (fc *FakeConfigurator) Create() (*Config, error) {
	return fc.Config, nil
}

// CreateFromProvider returns FakeConfigurator.Config
func (fc *FakeConfigurator) CreateFromProvider(providerName string) (*Config, error) {
	return fc.Config, nil
}

// CreateFromConfig returns FakeConfigurator.Config
func (fc *FakeConfigurator) CreateFromConfig(policy schedulerapi.Policy) (*Config, error) {
	return fc.Config, nil
}

// CreateFromKeys returns FakeConfigurator.Config
func (fc *FakeConfigurator) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*Config, error) {
	return fc.Config, nil
}
