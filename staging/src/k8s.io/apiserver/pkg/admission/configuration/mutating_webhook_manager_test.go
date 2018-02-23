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
	"reflect"
	"testing"
	"time"

	"k8s.io/api/admissionregistration/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	admissionregistrationlisters "k8s.io/client-go/listers/admissionregistration/v1beta1"
	"k8s.io/client-go/tools/cache"
)

type fakeMutatingWebhookConfigSharedInformer struct {
	informer *fakeMutatingWebhookConfigInformer
	lister   *fakeMutatingWebhookConfigLister
}

func (f *fakeMutatingWebhookConfigSharedInformer) Informer() cache.SharedIndexInformer {
	return f.informer
}
func (f *fakeMutatingWebhookConfigSharedInformer) Lister() admissionregistrationlisters.MutatingWebhookConfigurationLister {
	return f.lister
}

type fakeMutatingWebhookConfigInformer struct {
	eventHandler cache.ResourceEventHandler
}

func (f *fakeMutatingWebhookConfigInformer) AddEventHandler(handler cache.ResourceEventHandler) {
	fmt.Println("added handler")
	f.eventHandler = handler
}
func (f *fakeMutatingWebhookConfigInformer) AddEventHandlerWithResyncPeriod(handler cache.ResourceEventHandler, resyncPeriod time.Duration) {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) GetStore() cache.Store {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) GetController() cache.Controller {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) Run(stopCh <-chan struct{}) {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) HasSynced() bool {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) LastSyncResourceVersion() string {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) AddIndexers(indexers cache.Indexers) error {
	panic("unsupported")
}
func (f *fakeMutatingWebhookConfigInformer) GetIndexer() cache.Indexer { panic("unsupported") }

type fakeMutatingWebhookConfigLister struct {
	list []*v1beta1.MutatingWebhookConfiguration
	err  error
}

func (f *fakeMutatingWebhookConfigLister) List(selector labels.Selector) (ret []*v1beta1.MutatingWebhookConfiguration, err error) {
	return f.list, f.err
}

func (f *fakeMutatingWebhookConfigLister) Get(name string) (*v1beta1.MutatingWebhookConfiguration, error) {
	panic("unsupported")
}

func TestGetMutatingWebhookConfig(t *testing.T) {
	informer := &fakeMutatingWebhookConfigSharedInformer{
		informer: &fakeMutatingWebhookConfigInformer{},
		lister:   &fakeMutatingWebhookConfigLister{},
	}

	// no configurations
	informer.lister.list = nil
	manager := NewMutatingWebhookConfigurationManager(informer)
	if configurations := manager.Webhooks(); len(configurations.Webhooks) != 0 {
		t.Errorf("expected empty webhooks, but got %v", configurations.Webhooks)
	}

	// list err
	webhookConfiguration := &v1beta1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "webhook1"},
		Webhooks:   []v1beta1.Webhook{{Name: "webhook1.1"}},
	}
	informer.lister.list = []*v1beta1.MutatingWebhookConfiguration{webhookConfiguration.DeepCopy()}
	informer.lister.err = fmt.Errorf("mutating webhook configuration list error")
	informer.informer.eventHandler.OnAdd(webhookConfiguration.DeepCopy())
	if configurations := manager.Webhooks(); len(configurations.Webhooks) != 0 {
		t.Errorf("expected empty webhooks, but got %v", configurations.Webhooks)
	}

	// configuration populated
	informer.lister.err = nil
	informer.informer.eventHandler.OnAdd(webhookConfiguration.DeepCopy())
	configurations := manager.Webhooks()
	if len(configurations.Webhooks) == 0 {
		t.Errorf("expected non empty webhooks")
	}
	if !reflect.DeepEqual(configurations.Webhooks, webhookConfiguration.Webhooks) {
		t.Errorf("Expected\n%#v\ngot\n%#v", webhookConfiguration.Webhooks, configurations.Webhooks)
	}
}
