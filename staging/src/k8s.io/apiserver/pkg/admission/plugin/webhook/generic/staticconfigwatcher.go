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
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/fsnotify/fsnotify"
	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/klog"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(v1.AddToScheme(scheme))
	scheme.AddKnownTypes(metav1.SchemeGroupVersion, &metav1.List{})
}

type staticConfigWatcher struct {
	sync.Mutex
	accessors        []webhook.WebhookAccessor
	staticConfigFile string
	defaultor        WebhookDefaultor
	validator        WebhookValidator
	watcher          *fsnotify.Watcher
}

func NewStaticConfigWatcher(staticConfigFile string, defaultor WebhookDefaultor, validator WebhookValidator) *staticConfigWatcher {
	return &staticConfigWatcher{
		staticConfigFile: staticConfigFile,
		defaultor:        defaultor,
		validator:        validator,
	}
}

func (w *staticConfigWatcher) Init() {
	var err error
	w.loadConfig()
	if w.watcher, err = fsnotify.NewWatcher(); err != nil {
		klog.Errorf("Error creating a watcher for static webhook config file: %v", err)
	}
	if err := w.watcher.Add(w.staticConfigFile); err != nil {
		klog.Errorf("Error adding a watcher for static webhook config file: %v", err)
	}

	go w.watchForChanges()
}

func (w *staticConfigWatcher) watchForChanges() {
	defer w.watcher.Close()

	for {
		select {
		case event := <-w.watcher.Events:
			if event.Op == fsnotify.Remove {
				// workaround for symlinks; remove and add
				if err := w.watcher.Remove(event.Name); err != nil {
					klog.Errorf("Error removing watcher for static webhook config file: %v", err)
				}
				if err := w.watcher.Add(w.staticConfigFile); err != nil {
					klog.Errorf("Error adding a watcher for static webhook config file: %v", err)
				}
				w.loadConfig()
			}
			if event.Op&fsnotify.Write == fsnotify.Write {
				w.loadConfig()
			}
		case err := <-w.watcher.Errors:
			klog.Errorf("Error watching static webhook config file: %v", err)
		}
	}
}

func (w *staticConfigWatcher) loadConfig() {
	var err error
	w.Lock()
	defer w.Unlock()
	if w.accessors, err = w.extractWebhookAccessors(w.staticConfigFile, w.defaultor, w.validator); err != nil {
		klog.Errorf("Error reading static webhooks from file: %v", err)
	}
}

func (w *staticConfigWatcher) extractWebhookAccessors(staticConfigFile string, defaultor WebhookDefaultor, validator WebhookValidator) ([]webhook.WebhookAccessor, error) {
	if staticConfigFile == "" {
		return []webhook.WebhookAccessor{}, nil
	}

	r, err := os.Open(staticConfigFile)
	if err != nil {
		return nil, err
	}

	reader := utilyaml.NewDocumentDecoder(r)
	decoder := codecs.UniversalDecoder(v1.SchemeGroupVersion, metav1.SchemeGroupVersion)
	var objs []runtime.Object
	for {
		data := make([]byte, 200000)
		n, err := reader.Read(data)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		decodedObj, err := runtime.Decode(decoder, data[:n])
		if err != nil {
			return nil, err
		}
		objs = append(objs, decodedObj)
	}

	var accessors []webhook.WebhookAccessor
	for _, obj := range objs {
		switch o := obj.(type) {
		case *metav1.List:
			if len(o.Items) == 0 {
				break
			}
			for i := range o.Items {
				o.Items[i].Object, err = runtime.Decode(decoder, o.Items[i].Raw)
				if err != nil {
					return nil, err
				}
			}
			if _, ok := o.Items[0].Object.(*v1.ValidatingWebhookConfiguration); ok {
				for _, item := range o.Items {
					vc, ok := item.Object.(*v1.ValidatingWebhookConfiguration)
					if !ok {
						return nil, fmt.Errorf("unexpected type: %T", item.Object)
					}
					a, err := w.extractAccessorsFromValidatingWebhookConfiguration(vc, defaultor, validator)
					if err != nil {
						return nil, err
					}
					accessors = append(accessors, a...)
				}
			} else if _, ok := o.Items[0].Object.(*v1.MutatingWebhookConfiguration); ok {
				for _, item := range o.Items {
					mc, ok := item.Object.(*v1.MutatingWebhookConfiguration)
					if !ok {
						return nil, fmt.Errorf("unexpected type: %T", item.Object)
					}
					a, err := w.extractAccessorsFromMutatingWebhookConfiguration(mc, defaultor, validator)
					if err != nil {
						return nil, err
					}
					accessors = append(accessors, a...)
				}
			} else {
				return nil, fmt.Errorf("unexpected type: %T", o.Items[0].Object)
			}
		case *v1.ValidatingWebhookConfiguration:
			a, err := w.extractAccessorsFromValidatingWebhookConfiguration(o, defaultor, validator)
			if err != nil {
				return nil, err
			}
			accessors = append(accessors, a...)
		case *v1.ValidatingWebhookConfigurationList:
			for _, vc := range o.Items {
				a, err := w.extractAccessorsFromValidatingWebhookConfiguration(&vc, defaultor, validator)
				if err != nil {
					return nil, err
				}
				accessors = append(accessors, a...)
			}
		case *v1.MutatingWebhookConfiguration:
			a, err := w.extractAccessorsFromMutatingWebhookConfiguration(o, defaultor, validator)
			if err != nil {
				return nil, err
			}
			accessors = append(accessors, a...)
		case *v1.MutatingWebhookConfigurationList:
			for _, mc := range o.Items {
				a, err := w.extractAccessorsFromMutatingWebhookConfiguration(&mc, defaultor, validator)
				if err != nil {
					return nil, err
				}
				accessors = append(accessors, a...)
			}
		default:
			return nil, fmt.Errorf("unexpected type: %T", obj)
		}
	}

	return accessors, nil
}

func (w *staticConfigWatcher) extractAccessorsFromMutatingWebhookConfiguration(m *v1.MutatingWebhookConfiguration, defaultor WebhookDefaultor, validator WebhookValidator) ([]webhook.WebhookAccessor, error) {
	var accessors []webhook.WebhookAccessor
	if defaultor != nil {
		defaultor.SetDefaultForMutatingWebhookConfiguration(m)
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
		uid := fmt.Sprintf("%s/%s/%d", m.Name, n, names[n])
		names[n]++
		accessors = append(accessors, webhook.NewMutatingWebhookAccessor(uid, m.Name, &m.Webhooks[i]))
	}
	return accessors, nil
}

func (w *staticConfigWatcher) extractAccessorsFromValidatingWebhookConfiguration(v *v1.ValidatingWebhookConfiguration, defaultor WebhookDefaultor, validator WebhookValidator) ([]webhook.WebhookAccessor, error) {
	var accessors []webhook.WebhookAccessor
	if defaultor != nil {
		defaultor.SetDefaultForValidatingWebhookConfiguration(v)
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
		uid := fmt.Sprintf("%s/%s/%d", v.Name, n, names[n])
		names[n]++
		accessors = append(accessors, webhook.NewValidatingWebhookAccessor(uid, v.Name, &v.Webhooks[i]))
	}
	return accessors, nil
}

func (w *staticConfigWatcher) getWebhookAccessors() []webhook.WebhookAccessor {
	w.Lock()
	defer w.Unlock()
	return w.accessors
}
