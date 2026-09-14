/*
Copyright The Kubernetes Authors.

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

// Package source provides a Source implementation that loads webhook configurations from manifest files.
package source

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	stagingloader "k8s.io/apiserver/pkg/admission/plugin/webhook/manifest/loader"
	"k8s.io/apiserver/pkg/util/filesystem"
	"k8s.io/klog/v2"
)

// defaultReloadInterval is the default interval at which the manifest directory is checked for changes.
var defaultReloadInterval = 1 * time.Minute

// SetReloadIntervalForTests sets the reload interval for testing and returns a function to restore the original value.
func SetReloadIntervalForTests(interval time.Duration) func() {
	original := defaultReloadInterval
	defaultReloadInterval = interval
	return func() {
		defaultReloadInterval = original
	}
}

// ValidatingWebhookLoadFunc loads validating webhook configurations from a directory.
// Returns the configurations, a hash string for change detection, and any error.
type ValidatingWebhookLoadFunc func(dir string) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, string, error)

// MutatingWebhookLoadFunc loads mutating webhook configurations from a directory.
// Returns the configurations, a hash string for change detection, and any error.
type MutatingWebhookLoadFunc func(dir string) ([]*admissionregistrationv1.MutatingWebhookConfiguration, string, error)

// accessorBuilder builds webhook accessors from configurations.
type accessorBuilder[T runtime.Object] func(configs []T) []webhook.WebhookAccessor

// webhookData holds the currently loaded webhook configurations and pre-built accessors.
type webhookData[T runtime.Object] struct {
	configurations []T
	accessors      []webhook.WebhookAccessor
}

// staticWebhookSource provides webhook configurations loaded from manifest files.
type staticWebhookSource[T runtime.Object] struct {
	manifestsDir   string
	apiServerID    string
	reloadInterval time.Duration
	webhookType    metrics.ManifestType
	loadFunc       func(dir string) ([]T, string, error)
	buildAccessors accessorBuilder[T]

	current      atomic.Pointer[webhookData[T]]
	lastReadHash atomic.Pointer[string] // hash of last file content read (for short-circuiting)
	hasSynced    atomic.Bool
}

// ValidatingSource provides validating webhook configurations loaded from manifest files.
type ValidatingSource struct {
	*staticWebhookSource[*admissionregistrationv1.ValidatingWebhookConfiguration]
}

// MutatingSource provides mutating webhook configurations loaded from manifest files.
type MutatingSource struct {
	*staticWebhookSource[*admissionregistrationv1.MutatingWebhookConfiguration]
}

// NewValidatingSource creates a new validating webhook source that loads configurations from the specified directory.
func NewValidatingSource(manifestsDir, apiServerID string, loadFunc ValidatingWebhookLoadFunc) *ValidatingSource {
	metrics.RegisterMetrics()
	return &ValidatingSource{
		staticWebhookSource: &staticWebhookSource[*admissionregistrationv1.ValidatingWebhookConfiguration]{
			manifestsDir:   manifestsDir,
			apiServerID:    apiServerID,
			reloadInterval: defaultReloadInterval,
			webhookType:    metrics.ValidatingWebhookManifestType,
			loadFunc:       loadFunc,
			buildAccessors: stagingloader.BuildValidatingAccessors,
		},
	}
}

// NewMutatingSource creates a new mutating webhook source that loads configurations from the specified directory.
func NewMutatingSource(manifestsDir, apiServerID string, loadFunc MutatingWebhookLoadFunc) *MutatingSource {
	metrics.RegisterMetrics()
	return &MutatingSource{
		staticWebhookSource: &staticWebhookSource[*admissionregistrationv1.MutatingWebhookConfiguration]{
			manifestsDir:   manifestsDir,
			apiServerID:    apiServerID,
			reloadInterval: defaultReloadInterval,
			webhookType:    metrics.MutatingWebhookManifestType,
			loadFunc:       loadFunc,
			buildAccessors: stagingloader.BuildMutatingAccessors,
		},
	}
}

// LoadInitial performs the initial load of webhook manifests.
func (s *staticWebhookSource[T]) LoadInitial() error {
	configs, hash, err := s.loadFunc(s.manifestsDir)
	if err != nil {
		return err
	}

	accessors := s.buildAccessors(configs)

	// Validate accessors eagerly to catch selector parse errors at startup.
	if err := validateAccessors(accessors); err != nil {
		return err
	}

	s.current.Store(&webhookData[T]{configurations: configs, accessors: accessors})
	s.lastReadHash.Store(&hash)
	s.hasSynced.Store(true)

	klog.InfoS("Loaded manifest-based webhook configurations", "plugin", string(s.webhookType),
		"configurations", len(configs))
	metrics.RecordAutomaticReloadSuccess(s.webhookType, s.apiServerID, hash)
	return nil
}

// RunReloadLoop watches for configuration changes and reloads when detected.
// It blocks until ctx is canceled.
func (s *staticWebhookSource[T]) RunReloadLoop(ctx context.Context) {
	filesystem.WatchUntil(
		ctx,
		s.reloadInterval,
		s.manifestsDir,
		func() {
			s.checkAndReload()
		},
		func(err error) {
			klog.ErrorS(err, "watching manifest directory", "plugin", string(s.webhookType), "dir", s.manifestsDir)
		},
	)
}

func (s *staticWebhookSource[T]) checkAndReload() {
	configs, hash, err := s.loadFunc(s.manifestsDir)
	if err != nil {
		klog.ErrorS(err, "reloading admission manifest config", "plugin", string(s.webhookType), "dir", s.manifestsDir)
		metrics.RecordAutomaticReloadFailure(s.webhookType, s.apiServerID)
		return
	}

	// Short-circuit if file content hasn't changed since last read.
	if last := s.lastReadHash.Load(); last != nil && hash == *last {
		return
	}
	s.lastReadHash.Store(&hash)

	// Build and validate accessors (configs may be empty if files deleted)
	accessors := s.buildAccessors(configs)
	if err := validateAccessors(accessors); err != nil {
		klog.ErrorS(err, "manifest webhook accessor validation failed after reload", "plugin", string(s.webhookType), "dir", s.manifestsDir)
		metrics.RecordAutomaticReloadFailure(s.webhookType, s.apiServerID)
		return
	}

	s.current.Store(&webhookData[T]{configurations: configs, accessors: accessors})
	klog.InfoS("reloaded admission manifest config", "plugin", string(s.webhookType), "dir", s.manifestsDir)
	metrics.RecordAutomaticReloadSuccess(s.webhookType, s.apiServerID, hash)
}

// HasSynced returns true if the initial load has completed.
func (s *staticWebhookSource[T]) HasSynced() bool {
	return s.hasSynced.Load()
}

// Webhooks returns the list of webhook accessors.
func (s *staticWebhookSource[T]) Webhooks() []webhook.WebhookAccessor {
	current := s.current.Load()
	if current == nil {
		return nil
	}
	return current.accessors
}

// validateAccessors exercises lazy getters on each accessor to surface latent
// errors (e.g. invalid label selectors) early.
// Note: We intentionally skip GetRESTClient validation here because it requires
// a ClientManager. API validation (applied during defaulting) already ensures
// CA bundles parse and URLs are present and valid, and the manifest loader
// rejects service references, which covers the inputs needed to construct a
// REST client at admission time.
func validateAccessors(accessors []webhook.WebhookAccessor) error {
	var errs []error
	for _, a := range accessors {
		if _, err := a.GetParsedNamespaceSelector(); err != nil {
			errs = append(errs, fmt.Errorf("webhook %q: invalid namespaceSelector: %w", a.GetName(), err))
		}
		if _, err := a.GetParsedObjectSelector(); err != nil {
			errs = append(errs, fmt.Errorf("webhook %q: invalid objectSelector: %w", a.GetName(), err))
		}
	}
	return errors.Join(errs...)
}
