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
	"sync/atomic"

	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

type wrappedSource struct {
	watcher                        *ManifestWatcher
	embedded                       Source
	webhookType                    WebhookType
	generation                     uint64
	lastObservedGenerationEmbedded uint64
	lastObservedGenerationManifest uint64
	hooks                          *atomic.Value
}

func (s *wrappedSource) Webhooks() []webhook.WebhookAccessor {
	if s.embedded == nil {
		return s.watcher.getWebhookAccessors()
	}
	curManifestGen := s.watcher.generation()
	curEmbeddedGen := s.embedded.Generation()
	if curEmbeddedGen > s.lastObservedGenerationEmbedded ||
		curManifestGen > s.lastObservedGenerationManifest {
		atomic.AddUint64(&s.generation, 1)
		atomic.StoreUint64(&s.lastObservedGenerationManifest, curManifestGen)
		atomic.StoreUint64(&s.lastObservedGenerationEmbedded, curEmbeddedGen)
		s.hooks.Store(append(s.watcher.getWebhookAccessors(), s.embedded.Webhooks()...))
	}

	hooks := s.hooks.Load()
	if hooks == nil {
		return []webhook.WebhookAccessor{}
	}
	return hooks.([]webhook.WebhookAccessor)
}

func (s *wrappedSource) HasSynced() bool {
	if s.embedded != nil {
		return s.embedded.HasSynced()
	}
	return true
}

func (s *wrappedSource) WebhookType() WebhookType {
	return s.webhookType
}

func (s *wrappedSource) Generation() uint64 {
	return atomic.LoadUint64(&s.generation)
}

// NewWrappedSource returns a webhook Source that adds in webhooks from the
// specified ManifestWatcher to the specified Source.
func NewWrappedSource(embedded Source, m *ManifestWatcher) Source {
	return &wrappedSource{embedded: embedded, watcher: m, hooks: &atomic.Value{}}
}

type manifestHookWrapper struct {
	manifestFile string
	defaulter    WebhookDefaulter
	validator    WebhookValidator
	watcher      *ManifestWatcher
}

// NewManifestHookWrapper returns a new manifestHookWrapper which implements
// ManifestWebhookWrapper
func NewManifestHookWrapper(manifestFile string, d WebhookDefaulter, v WebhookValidator) ManifestWebhookWrapper {
	return &manifestHookWrapper{
		manifestFile: manifestFile,
		defaulter:    d,
		validator:    v,
		watcher:      NewManifestWatcher(manifestFile, d, v),
	}
}

func (m *manifestHookWrapper) Initialize(webhookType WebhookType) error {
	return m.watcher.Init(webhookType)
}

func (m *manifestHookWrapper) WrapHookSource(s Source) Source {
	return NewWrappedSource(s, m.watcher)
}
