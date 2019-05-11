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

	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/apis/audit/install"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
)

const (
	// PluginName is the name of this plugin, to be used in help and logs.
	PluginName = "webhook"

	// DefaultInitialBackoff is the default amount of time to wait before
	// retrying sending audit events through a webhook.
	DefaultInitialBackoff = 10 * time.Second
)

func init() {
	install.Install(audit.Scheme)
}

func loadWebhook(configFile string, groupVersion schema.GroupVersion, initialBackoff time.Duration) (*webhook.GenericWebhook, error) {
	return webhook.NewGenericWebhook(audit.Scheme, audit.Codecs, configFile,
		[]schema.GroupVersion{groupVersion}, initialBackoff)
}

type backend struct {
	w    *webhook.GenericWebhook
	name string
}

// NewDynamicBackend returns an audit backend configured from a REST client that
// sends events over HTTP to an external service.
func NewDynamicBackend(rc *rest.RESTClient, initialBackoff time.Duration) audit.Backend {
	return &backend{
		w: &webhook.GenericWebhook{
			RestClient:     rc,
			InitialBackoff: initialBackoff,
		},
		name: fmt.Sprintf("dynamic_%s", PluginName),
	}
}

// NewBackend returns an audit backend that sends events over HTTP to an external service.
func NewBackend(kubeConfigFile string, groupVersion schema.GroupVersion, initialBackoff time.Duration) (audit.Backend, error) {
	w, err := loadWebhook(kubeConfigFile, groupVersion, initialBackoff)
	if err != nil {
		return nil, err
	}
	return &backend{w: w, name: PluginName}, nil
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (b *backend) Shutdown() {
	// nothing to do here
}

func (b *backend) ProcessEvents(ev ...*auditinternal.Event) bool {
	if err := b.processEvents(ev...); err != nil {
		audit.HandlePluginError(b.String(), err, ev...)
		return false
	}
	return true
}

func (b *backend) processEvents(ev ...*auditinternal.Event) error {
	var list auditinternal.EventList
	for _, e := range ev {
		list.Items = append(list.Items, *e)
	}
	return b.w.WithExponentialBackoff(func() rest.Result {
		return b.w.RestClient.Post().Body(&list).Do()
	}).Error()
}

func (b *backend) String() string {
	return b.name
}
