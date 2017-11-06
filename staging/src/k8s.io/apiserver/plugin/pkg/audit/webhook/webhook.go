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

// Package webhook implements the audit.Backend interface using HTTP webhooks.
package webhook

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/apis/audit/install"
	auditv1alpha1 "k8s.io/apiserver/pkg/apis/audit/v1alpha1"
	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
)

const (
	// The plugin name reported in error metrics.
	pluginName            = "webhook"
	defaultInitialBackoff = 10 * time.Second // Wait at least 10 seconds before retrying.
)

// NewBackend returns an audit backend that sends events over HTTP to an external service.
// The mode indicates the caching behavior of the webhook. Either blocking (ModeBlocking)
// or buffered with batch POSTs (ModeBatch).
func NewBackend(kubeConfigFile string, groupVersion schema.GroupVersion) (audit.Backend, error) {
	w, err := loadWebhook(kubeConfigFile, groupVersion)
	if err != nil {
		return nil, err
	}
	return &backend{w}, nil
}

var (
	// NOTE: Copied from other webhook implementations
	//
	// Can we make these passable to NewGenericWebhook?
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	// TODO(audit): figure out a general way to let the client choose their preferred version
	registry = registered.NewOrDie("")
)

func init() {
	allGVs := []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion, auditv1beta1.SchemeGroupVersion}
	registry.RegisterVersions(allGVs)
	if err := registry.EnableVersions(allGVs...); err != nil {
		panic(fmt.Sprintf("failed to enable version %v", allGVs))
	}
	install.Install(groupFactoryRegistry, registry, audit.Scheme)
}

func loadWebhook(configFile string, groupVersion schema.GroupVersion) (*webhook.GenericWebhook, error) {
	return webhook.NewGenericWebhook(registry, audit.Codecs, configFile,
		[]schema.GroupVersion{groupVersion}, defaultInitialBackoff)
}

type backend struct {
	w *webhook.GenericWebhook
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (b *backend) Shutdown() {
	// nothing to do here
}

func (b *backend) ProcessEvents(ev ...*auditinternal.Event) {
	if err := b.processEvents(ev...); err != nil {
		audit.HandlePluginError(pluginName, err, ev...)
	}
}

func (b *backend) processEvents(ev ...*auditinternal.Event) error {
	var list auditinternal.EventList
	for _, e := range ev {
		list.Items = append(list.Items, *e)
	}
	// Retry with exponential backoff
	err := b.w.WithExponentialBackoff(func() rest.Result {
		return b.w.RestClient.Post().Body(&list).Do()
	}).Error()

	return err
}
