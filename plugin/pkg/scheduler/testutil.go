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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"
)

type FakeConfigurator struct {
	Config                         *Config
	PriorityConfigs                []algorithm.PriorityConfig
	PriorityMetadataProducer       algorithm.MetadataProducer
	PredicateMetadataProducer      algorithm.MetadataProducer
	Predicates                     map[string]algorithm.FitPredicate
	HardPodAffinitySymmetricWeight int
	SchedulerName                  string
	DefaultErrorFunc               func(pod *v1.Pod, err error)
	ResponsibleForPodVal           bool
	NodeLister                     corelisters.NodeLister
	Client                         clientset.Interface
	ScheduledPodLister             corelisters.PodLister
}

func (fc *FakeConfigurator) GetPriorityFunctionConfigs(priorityKeys sets.String) ([]algorithm.PriorityConfig, error) {
	return fc.PriorityConfigs, nil
}

func (fc *FakeConfigurator) GetPriorityMetadataProducer() (algorithm.MetadataProducer, error) {
	return fc.PriorityMetadataProducer, nil
}

func (fc *FakeConfigurator) GetPredicateMetadataProducer() (algorithm.MetadataProducer, error) {
	return fc.PredicateMetadataProducer, nil
}

func (fc *FakeConfigurator) GetPredicates(predicateKeys sets.String) (map[string]algorithm.FitPredicate, error) {
	return fc.Predicates, nil
}

func (fc *FakeConfigurator) GetHardPodAffinitySymmetricWeight() int {
	return fc.HardPodAffinitySymmetricWeight
}

func (fc *FakeConfigurator) GetSchedulerName() string {
	return fc.SchedulerName
}

func (fc *FakeConfigurator) MakeDefaultErrorFunc(backoff *util.PodBackoff, podQueue *cache.FIFO) func(pod *v1.Pod, err error) {
	return fc.DefaultErrorFunc
}

func (fc *FakeConfigurator) ResponsibleForPod(pod *v1.Pod) bool {
	return fc.ResponsibleForPodVal
}

func (fc *FakeConfigurator) GetNodeLister() corelisters.NodeLister {
	return fc.NodeLister
}

func (fc *FakeConfigurator) GetClient() clientset.Interface {
	return fc.Client
}

func (fc *FakeConfigurator) GetScheduledPodLister() corelisters.PodLister {
	return fc.ScheduledPodLister
}

func (fc *FakeConfigurator) Run() {

}

func (fc *FakeConfigurator) Create() (*Config, error) {
	return fc.Config, nil
}

func (fc *FakeConfigurator) CreateFromProvider(providerName string) (*Config, error) {
	return fc.Config, nil
}

func (fc *FakeConfigurator) CreateFromConfig(policy schedulerapi.Policy) (*Config, error) {
	return fc.Config, nil
}

func (fc *FakeConfigurator) CreateFromKeys(predicateKeys, priorityKeys sets.String, extenders []algorithm.SchedulerExtender) (*Config, error) {
	return fc.Config, nil
}
