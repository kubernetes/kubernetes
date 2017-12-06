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

package configuration

import (
	"fmt"
	"sort"
	"sync/atomic"

	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	admissionregistrationinformers "k8s.io/client-go/informers/admissionregistration/v1beta1"
	admissionregistrationlisters "k8s.io/client-go/listers/admissionregistration/v1beta1"
	"k8s.io/client-go/tools/cache"
)

// ValidatingWebhookConfigurationManager collects the validating webhook objects so that they can be called.
type ValidatingWebhookConfigurationManager struct {
	configuration *atomic.Value
	lister        admissionregistrationlisters.ValidatingWebhookConfigurationLister
}

func NewValidatingWebhookConfigurationManager(informer admissionregistrationinformers.ValidatingWebhookConfigurationInformer) *ValidatingWebhookConfigurationManager {
	manager := &ValidatingWebhookConfigurationManager{
		configuration: &atomic.Value{},
		lister:        informer.Lister(),
	}

	// Start with an empty list
	manager.configuration.Store(&v1beta1.ValidatingWebhookConfiguration{})

	// On any change, rebuild the config
	informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(_ interface{}) { manager.updateConfiguration() },
		UpdateFunc: func(_, _ interface{}) { manager.updateConfiguration() },
		DeleteFunc: func(_ interface{}) { manager.updateConfiguration() },
	})

	return manager
}

// Webhooks returns the merged ValidatingWebhookConfiguration.
func (v *ValidatingWebhookConfigurationManager) Webhooks() *v1beta1.ValidatingWebhookConfiguration {
	return v.configuration.Load().(*v1beta1.ValidatingWebhookConfiguration)
}

func (v *ValidatingWebhookConfigurationManager) updateConfiguration() {
	configurations, err := v.lister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error updating configuration: %v", err))
		return
	}
	v.configuration.Store(mergeValidatingWebhookConfigurations(configurations))
}

func mergeValidatingWebhookConfigurations(
	configurations []*v1beta1.ValidatingWebhookConfiguration,
) *v1beta1.ValidatingWebhookConfiguration {
	sort.SliceStable(configurations, ValidatingWebhookConfigurationSorter(configurations).ByName)
	var ret v1beta1.ValidatingWebhookConfiguration
	for _, c := range configurations {
		ret.Webhooks = append(ret.Webhooks, c.Webhooks...)
	}
	return &ret
}

type ValidatingWebhookConfigurationSorter []*v1beta1.ValidatingWebhookConfiguration

func (a ValidatingWebhookConfigurationSorter) ByName(i, j int) bool {
	return a[i].Name < a[j].Name
}
