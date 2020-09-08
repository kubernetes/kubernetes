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
	"bytes"
	"crypto/sha256"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/klog/v2"
)

var (
	scheme = runtime.NewScheme()
)

const pollDuration = time.Minute

func init() {
	utilruntime.Must(v1.AddToScheme(scheme))
	scheme.AddKnownTypes(metav1.SchemeGroupVersion, &metav1.List{})
}

// ManifestWatcher watches a manifest for provided webhook configuration.
type ManifestWatcher struct {
	sync.RWMutex
	accessors    *atomic.Value
	manifestFile string
	defaulter    WebhookDefaulter
	validator    WebhookValidator
	prevSha      string
	lastErr      error
	webhookType  WebhookType
	gen          uint64
}

// NewManifestWatcher returns a new ManifestWatcher that watches manifest file
// for webhook configurations. Manifest file can contain any combination of
// * `admissionregistration.k8s.io/v1.ValidatingWebhookConfigurationList`
// * `admissionregistration.k8s.io/v1.MutatingWebhookConfigurationList`
// * v1.List with items of admissionregistration.k8s.io/v1.ValidatingWebhookConfiguration
// * v1.List  with items of admissionregistration.k8s.io/v1.MutatingWebhookConfiguration.
func NewManifestWatcher(manifestFile string, defaulter WebhookDefaulter, validator WebhookValidator) *ManifestWatcher {
	return &ManifestWatcher{
		manifestFile: manifestFile,
		defaulter:    defaulter,
		validator:    validator,
		accessors:    &atomic.Value{},
	}
}

// Init load the config from the manifest file and initialises watchers for it.
func (w *ManifestWatcher) Init(webhookType WebhookType) error {
	if w.manifestFile == "" {
		return nil
	}
	var err error
	w.webhookType = webhookType

	watcher, err := w.setUpWatcher()
	if err != nil {
		return err
	}

	go w.watchForChanges(watcher)

	if err = w.loadConfig(); err != nil {
		return err
	}
	return nil
}

func (w *ManifestWatcher) setUpWatcher() (*fsnotify.Watcher, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return &fsnotify.Watcher{}, fmt.Errorf("error creating a watcher for webhook manifest file; %v", err)
	}

	// watch the parent dir of the config file to observe atomic renames
	parentDir := filepath.Dir(w.manifestFile)
	err = watcher.Add(parentDir)
	if err != nil {
		return &fsnotify.Watcher{}, fmt.Errorf("error adding a watcher on the parent directory of webhook manifest file; %v", err)
	}

	if err := watcher.Add(w.manifestFile); err != nil {
		return &fsnotify.Watcher{}, fmt.Errorf("error adding a watcher for webhook manifest config file; %v", err)
	}
	// resolve symlinks to observe the underlying file as well
	if linkedPath, err := filepath.EvalSymlinks(w.manifestFile); err == nil && linkedPath != w.manifestFile {
		// watch the linked file to observe writes
		err = watcher.Add(linkedPath)
		if err != nil {
			return &fsnotify.Watcher{}, fmt.Errorf("error adding a watcher on linked path for webhook manifest config file; %v", err)
		}

		// watch the parent dir of the linked file to observe replaces
		linkedParentDir := filepath.Dir(linkedPath)
		if linkedParentDir != parentDir {
			err = watcher.Add(linkedParentDir)
			if err != nil {
				return &fsnotify.Watcher{}, fmt.Errorf("error adding a watcher on linked parent directory for webhook manifest config file; %v", err)
			}
		}
	}
	return watcher, nil
}

func (w *ManifestWatcher) generation() uint64 {
	return atomic.LoadUint64(&w.gen)
}

func (w *ManifestWatcher) watchForChanges(watcher *fsnotify.Watcher) {
	fileEventsCh := watcher.Events
	fileErrorCh := watcher.Errors
	defer watcher.Close()
	for {
		select {
		case event := <-fileEventsCh:
			if event.Op&fsnotify.Write == fsnotify.Write || event.Op&fsnotify.Create == fsnotify.Create || event.Op&fsnotify.Rename == fsnotify.Rename {
				exportError(w.loadConfig())
			}
		case <-time.After(pollDuration):
			exportError(w.loadConfig())
		case <-fileErrorCh:
			watcherSetupErr := watcher.Close()
			var newWatcher *fsnotify.Watcher
			if watcherSetupErr == nil {
				newWatcher, watcherSetupErr = w.setUpWatcher()
			}

			if watcherSetupErr != nil {
				klog.Error("Failed to re-initalize watcher for manifest webhooks; falling back to time based polling")
				// No op watcher
				newWatcher = &fsnotify.Watcher{
					Events:   make(chan fsnotify.Event),
					Errors:   make(chan error),
				}
			}
			fileEventsCh = newWatcher.Events
			fileErrorCh = newWatcher.Errors
			exportError(w.loadConfig())
		}
	}
}

func exportError(err error) {
	if err != nil {
		klog.Errorf("Error encountered while watching webhook manifest: %v", err)
		admissionmetrics.Metrics.ObserveManifestLoadingError(true)
	} else {
		admissionmetrics.Metrics.ObserveManifestLoadingError(false)
	}
}

func (w *ManifestWatcher) loadConfig() error {
	w.Lock()
	defer w.Unlock()

	accessors, foundNew, sha, err := w.extractWebhookAccessors(w.manifestFile, w.defaulter, w.validator)
	if foundNew {
		w.lastErr = err
		if err == nil {
			w.accessors.Store(accessors)
			atomic.AddUint64(&w.gen, 1)
		}
		w.prevSha = sha
	}
	return w.lastErr
}

// extractWebhookAccessors parses the manifest file, defaults and validates the
// webhook and returns the webhooks parsed, a bool representing if it found a new
// manifest file and any err encountered while parsing the file.
func (w *ManifestWatcher) extractWebhookAccessors(manifestFile string, defaulter WebhookDefaulter, validator WebhookValidator) ([]webhook.WebhookAccessor, bool, string, error) {
	if manifestFile == "" {
		return []webhook.WebhookAccessor{}, false, w.prevSha, nil
	}

	var err error
	data, err := ioutil.ReadFile(manifestFile)
	if err != nil {
		return []webhook.WebhookAccessor{}, false, w.prevSha, err
	}

	currentSha := fmt.Sprintf("%x", sha256.Sum256(data))
	if currentSha == w.prevSha {
		klog.V(5).Infof("Content of manifest file at %s has not changed", manifestFile)
		return []webhook.WebhookAccessor{}, false, currentSha, nil
	}

	builder := resource.NewLocalBuilder().
		WithScheme(scheme, v1.SchemeGroupVersion, metav1.SchemeGroupVersion).
		Stream(bytes.NewBuffer(data), "input").
		Flatten()

	result := builder.Do()

	if err = result.Err(); err != nil {
		return []webhook.WebhookAccessor{}, true, currentSha, err
	}
	items, err := result.Infos()
	if err != nil {
		return []webhook.WebhookAccessor{}, true, currentSha, err
	}

	var accessors []webhook.WebhookAccessor
	for _, item := range items {
		obj := item.Object
		switch kind := obj.GetObjectKind().GroupVersionKind().String(); kind {
		case "admissionregistration.k8s.io/v1, Kind=ValidatingWebhookConfiguration":
			if w.webhookType != ValidatingWebhook {
				continue
			}
			config := (obj).(*v1.ValidatingWebhookConfiguration)
			a, err := w.extractValidatingAccessors(config, defaulter, validator)
			if err != nil {
				return []webhook.WebhookAccessor{}, true, currentSha, err
			}
			accessors = append(accessors, a...)
		case "admissionregistration.k8s.io/v1, Kind=MutatingWebhookConfiguration":
			if w.webhookType != MutatingWebhook {
				continue
			}
			config := (obj).(*v1.MutatingWebhookConfiguration)
			a, err := w.extractMutatingAccessors(config, defaulter, validator)
			if err != nil {
				return []webhook.WebhookAccessor{}, true, currentSha, err
			}
			accessors = append(accessors, a...)
		default:
			return []webhook.WebhookAccessor{}, true, currentSha, fmt.Errorf("unexpected kind: %s", kind)
		}
	}

	return accessors, true, currentSha, nil
}

func (w *ManifestWatcher) extractMutatingAccessors(m *v1.MutatingWebhookConfiguration, defaulter WebhookDefaulter, validator WebhookValidator) ([]webhook.WebhookAccessor, error) {
	var accessors []webhook.WebhookAccessor
	if defaulter != nil {
		defaulter.SetDefaultsForMutatingWebhookConfiguration(m)
	}

	if validator != nil {
		if err := validator.ValidateMutatingWebhookConfiguration(m); err != nil {
			return nil, err
		}
	}

	// webhook names are not validated for uniqueness, so we check for duplicates and
	// add a int suffix to distinguish between them
	names := map[string]int{}
	for i := range m.Webhooks {
		n := m.Webhooks[i].Name
		uid := fmt.Sprintf("manifest/%s/%s/%d", m.Name, n, names[n])
		names[n]++
		accessors = append(accessors, webhook.NewManifestBasedMutatingWebhookAccessor(uid, m.Name, &m.Webhooks[i]))
	}
	return accessors, nil
}

func (w *ManifestWatcher) extractValidatingAccessors(v *v1.ValidatingWebhookConfiguration, defaulter WebhookDefaulter, validator WebhookValidator) ([]webhook.WebhookAccessor, error) {
	var accessors []webhook.WebhookAccessor
	if defaulter != nil {
		defaulter.SetDefaultsForValidatingWebhookConfiguration(v)
	}

	if validator != nil {
		if err := validator.ValidateValidatingWebhookConfiguration(v); err != nil {
			return nil, err
		}
	}

	// webhook names are not validated for uniqueness, so we check for duplicates and
	// add a int suffix to distinguish between them
	names := map[string]int{}
	for i := range v.Webhooks {
		n := v.Webhooks[i].Name
		uid := fmt.Sprintf("manifest/%s/%s/%d", v.Name, n, names[n])
		names[n]++
		accessors = append(accessors, webhook.NewManifestBasedValidatingWebhookAccessor(uid, v.Name, &v.Webhooks[i]))
	}
	return accessors, nil
}

func (w *ManifestWatcher) getWebhookAccessors() []webhook.WebhookAccessor {
	acc := w.accessors.Load()
	if acc == nil {
		return nil
	}
	return acc.([]webhook.WebhookAccessor)
}
