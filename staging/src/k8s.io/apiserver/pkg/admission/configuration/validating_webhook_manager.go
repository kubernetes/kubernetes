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
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	"k8s.io/client-go/informers"
	admissionregistrationlisters "k8s.io/client-go/listers/admissionregistration/v1beta1"
	"k8s.io/client-go/tools/cache"
)

// validatingWebhookConfigurationManager collects the validating webhook objects so that they can be called.
type validatingWebhookConfigurationManager struct {
	configuration *atomic.Value
	lister        admissionregistrationlisters.ValidatingWebhookConfigurationLister
	hasSynced     func() bool
}

var _ generic.Source = &validatingWebhookConfigurationManager{}

func NewValidatingWebhookConfigurationManager(f informers.SharedInformerFactory) generic.Source {
	informer := f.Admissionregistration().V1beta1().ValidatingWebhookConfigurations()
	manager := &validatingWebhookConfigurationManager{
		configuration: &atomic.Value{},
		lister:        informer.Lister(),
		hasSynced:     informer.Informer().HasSynced,
	}

	// Start with an empty list
	manager.configuration.Store([]webhook.WebhookAccessor{})

	// On any change, rebuild the config
	informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(_ interface{}) { manager.updateConfiguration() },
		UpdateFunc: func(_, _ interface{}) { manager.updateConfiguration() },
		DeleteFunc: func(_ interface{}) { manager.updateConfiguration() },
	})

	return manager
}

// Webhooks returns the merged ValidatingWebhookConfiguration.
func (v *validatingWebhookConfigurationManager) Webhooks() []webhook.WebhookAccessor {
	return v.configuration.Load().([]webhook.WebhookAccessor)
}

// HasSynced returns true if the shared informers have synced.
func (v *validatingWebhookConfigurationManager) HasSynced() bool {
	return v.hasSynced()
}

func (v *validatingWebhookConfigurationManager) updateConfiguration() {
	configurations, err := v.lister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error updating configuration: %v", err))
		return
	}
	v.configuration.Store(mergeValidatingWebhookConfigurations(configurations))
}

func mergeValidatingWebhookConfigurations(configurations []*v1beta1.ValidatingWebhookConfiguration) []webhook.WebhookAccessor {
	sort.SliceStable(configurations, ValidatingWebhookConfigurationSorter(configurations).ByName)
	accessors := []webhook.WebhookAccessor{}
	for _, c := range configurations {
		// webhook names are not validated for uniqueness, so we check for duplicates and
		// add a int suffix to distinguish between them
		names := map[string]int{}
		for i := range c.Webhooks {
			n := c.Webhooks[i].Name
			uid := fmt.Sprintf("%s/%s/%d", c.Name, n, names[n])
			names[n]++
			accessors = append(accessors, webhook.NewValidatingWebhookAccessor(uid, c.Name, &c.Webhooks[i]))
		}
	}
	return accessors
}

type ValidatingWebhookConfigurationSorter []*v1beta1.ValidatingWebhookConfiguration

func (a ValidatingWebhookConfigurationSorter) ByName(i, j int) bool {
	return a[i].Name < a[j].Name
}
