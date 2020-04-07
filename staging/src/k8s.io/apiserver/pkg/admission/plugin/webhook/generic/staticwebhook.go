/*
Copyright 2020 The Kubernetes Authors.

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

package generic

import (
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/client-go/informers"
)

type wrappedSource struct {
	factory  staticWebhookFactory
	embedded Source
}

func (s *wrappedSource) Webhooks() []webhook.WebhookAccessor {
	if s.embedded != nil {
		return append(s.factory(), s.embedded.Webhooks()...)
	}
	return s.factory()
}

func (s *wrappedSource) HasSynced() bool {
	if s.embedded != nil {
		return s.embedded.HasSynced()
	}
	return true
}

func NewWrappedSource(embedded Source, f staticWebhookFactory) Source {
	return &wrappedSource{embedded: embedded, factory: f}
}

func wrapSourceFactory(s sourceFactory, factory staticWebhookFactory) sourceFactory {
	return func(f informers.SharedInformerFactory) Source {
		return NewWrappedSource(s(f), factory)
	}
}

type staticWebhookFactory func() []webhook.WebhookAccessor

func wrapHookSourceFactory(s sourceFactory, staticConfigFile string, defaultor WebhookDefaultor, validator WebhookValidator) sourceFactory {
	staticConfigWatcher := NewStaticConfigWatcher(staticConfigFile, defaultor, validator)
	staticConfigWatcher.Init()
	return wrapSourceFactory(s, staticConfigWatcher.getWebhookAccessors)
}
