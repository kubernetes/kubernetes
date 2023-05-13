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
	"sync"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	"k8s.io/client-go/informers"
	admissionregistrationlisters "k8s.io/client-go/listers/admissionregistration/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/cache/synctrack"
)

type webhookAccessorCreator func(uid string, configurationName string, h *v1.ValidatingWebhook) webhook.WebhookAccessor

// validatingWebhookConfigurationManager collects the validating webhook objects so that they can be called.
type validatingWebhookConfigurationManager struct {
	lister              admissionregistrationlisters.ValidatingWebhookConfigurationLister
	hasSynced           func() bool
	lazy                synctrack.Lazy[[]webhook.WebhookAccessor]
	configurationsCache sync.Map

	// createValidatingWebhookAccessor is used to instantiate webhook accessors.
	// This function is defined as field instead of a struct method to allow us
	// to replace it with a mock during unit tests.
	createValidatingWebhookAccessor webhookAccessorCreator
}

var _ generic.Source = &validatingWebhookConfigurationManager{}

func NewValidatingWebhookConfigurationManager(f informers.SharedInformerFactory) generic.Source {
	informer := f.Admissionregistration().V1().ValidatingWebhookConfigurations()
	manager := &validatingWebhookConfigurationManager{
		lister:                          informer.Lister(),
		createValidatingWebhookAccessor: webhook.NewValidatingWebhookAccessor,
	}
	manager.lazy.Evaluate = manager.getConfiguration

	handle, _ := informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(_ interface{}) { manager.lazy.Notify() },
		UpdateFunc: func(old, new interface{}) {
			//TODO: we can still dig deeper and figure out which ones of
			// the webhooks have changed + whether CEL expressions changed
			// etc... thinking about using reflect..
			obj := new.(*v1.ValidatingWebhookConfiguration)
			// lock R+W
			manager.configurationsCache.Delete(obj.GetName())
			manager.lazy.Notify()
		},
		DeleteFunc: func(vwc interface{}) {
			obj := vwc.(*v1.ValidatingWebhookConfiguration)
			manager.configurationsCache.Delete(obj.Name)
			manager.lazy.Notify()
		},
	})
	manager.hasSynced = handle.HasSynced

	return manager
}

// Webhooks returns the merged ValidatingWebhookConfiguration.
func (v *validatingWebhookConfigurationManager) Webhooks() []webhook.WebhookAccessor {
	out, err := v.lazy.Get()
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error getting webhook configuration: %v", err))
	}
	return out
}

// HasSynced returns true if the initial set of mutating webhook configurations
// has been loaded.
func (v *validatingWebhookConfigurationManager) HasSynced() bool { return v.hasSynced() }

func (v *validatingWebhookConfigurationManager) getConfiguration() ([]webhook.WebhookAccessor, error) {
	configurations, err := v.lister.List(labels.Everything())
	if err != nil {
		return []webhook.WebhookAccessor{}, err
	}
	return v.smartReloadValidatingWebhookConfigurations(configurations), nil
}

func (v *validatingWebhookConfigurationManager) smartReloadValidatingWebhookConfigurations(configurations []*v1.ValidatingWebhookConfiguration) []webhook.WebhookAccessor {
	sort.SliceStable(configurations, ValidatingWebhookConfigurationSorter(configurations).ByName)
	accessors := []webhook.WebhookAccessor{}
cfgLoop:
	for _, c := range configurations {
		cachedConfigurationAccessors, ok := v.configurationsCache.Load(c.Name)
		if ok {
			// Pick an already cached webhookAccessor
			accessors = append(accessors, cachedConfigurationAccessors.([]webhook.WebhookAccessor)...)
			continue cfgLoop
		}

		// webhook names are not validated for uniqueness, so we check for duplicates and
		// add a int suffix to distinguish between them
		//
		// NOTE: In `pkg/apis/admissionregistration/validation.go` webhook names are checked
		// for uniqueness now. Is it safe to remove this?
		configurationAccessors := []webhook.WebhookAccessor{}
		names := map[string]int{}
		for i := range c.Webhooks {
			n := c.Webhooks[i].Name
			uid := fmt.Sprintf("%s/%s/%d", c.Name, n, names[n])
			names[n]++
			configurationAccessors = append(configurationAccessors, v.createValidatingWebhookAccessor(uid, c.Name, &c.Webhooks[i]))
			// Cache the new accessors
			v.configurationsCache.Store(c.Name, configurationAccessors)
			accessors = append(accessors, configurationAccessors...)
		}
	}

	return accessors
}

type ValidatingWebhookConfigurationSorter []*v1.ValidatingWebhookConfiguration

func (a ValidatingWebhookConfigurationSorter) ByName(i, j int) bool {
	return a[i].Name < a[j].Name
}
