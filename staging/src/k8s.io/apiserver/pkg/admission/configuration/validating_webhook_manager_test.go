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

type fakeValidatingWebhookConfigSharedInformer struct {
	informer *fakeValidatingWebhookConfigInformer
	lister   *fakeValidatingWebhookConfigLister
}

func (f *fakeValidatingWebhookConfigSharedInformer) Informer() cache.SharedIndexInformer {
	return f.informer
}
func (f *fakeValidatingWebhookConfigSharedInformer) Lister() admissionregistrationlisters.ValidatingWebhookConfigurationLister {
	return f.lister
}

type fakeValidatingWebhookConfigInformer struct {
	eventHandler cache.ResourceEventHandler
}

func (f *fakeValidatingWebhookConfigInformer) AddEventHandler(handler cache.ResourceEventHandler) {
	fmt.Println("added handler")
	f.eventHandler = handler
}
func (f *fakeValidatingWebhookConfigInformer) AddEventHandlerWithResyncPeriod(handler cache.ResourceEventHandler, resyncPeriod time.Duration) {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) GetStore() cache.Store {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) GetController() cache.Controller {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) Run(stopCh <-chan struct{}) {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) HasSynced() bool {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) LastSyncResourceVersion() string {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) AddIndexers(indexers cache.Indexers) error {
	panic("unsupported")
}
func (f *fakeValidatingWebhookConfigInformer) GetIndexer() cache.Indexer { panic("unsupported") }

type fakeValidatingWebhookConfigLister struct {
	list []*v1beta1.ValidatingWebhookConfiguration
	err  error
}

func (f *fakeValidatingWebhookConfigLister) List(selector labels.Selector) (ret []*v1beta1.ValidatingWebhookConfiguration, err error) {
	return f.list, f.err
}

func (f *fakeValidatingWebhookConfigLister) Get(name string) (*v1beta1.ValidatingWebhookConfiguration, error) {
	panic("unsupported")
}

func TestGettValidatingWebhookConfig(t *testing.T) {
	informer := &fakeValidatingWebhookConfigSharedInformer{
		informer: &fakeValidatingWebhookConfigInformer{},
		lister:   &fakeValidatingWebhookConfigLister{},
	}

	// no configurations
	informer.lister.list = nil
	manager := NewValidatingWebhookConfigurationManager(informer)
	if configurations := manager.Webhooks(); len(configurations.Webhooks) != 0 {
		t.Errorf("expected empty webhooks, but got %v", configurations.Webhooks)
	}

	// list error
	webhookConfiguration := &v1beta1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "webhook1"},
		Webhooks:   []v1beta1.Webhook{{Name: "webhook1.1"}},
	}
	informer.lister.list = []*v1beta1.ValidatingWebhookConfiguration{webhookConfiguration.DeepCopy()}
	informer.lister.err = fmt.Errorf("validating webhook configuration list error")
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
