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

// Package loader provides generic functionality to load webhook configurations
// from manifest files. It handles file reading, YAML/JSON decoding, and generic
// manifest validation. Type-specific defaulting and validation (e.g., scheme-based
// defaulting via internal types) are injected by callers through the AcceptObjectFunc
// callback.
package loader

import (
	"fmt"
	"sort"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/manifest"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

// AcceptObjectFunc extracts typed items from a decoded runtime.Object, applying
// defaulting and validation. Returns the extracted items, or an error if the
// object type is not recognized or processing fails.
type AcceptObjectFunc[T metav1.Object] func(obj runtime.Object) ([]T, error)

// LoadManifests loads webhook configurations from manifest files in dir using
// the provided decoder and acceptObject callback. It handles file I/O, YAML
// splitting, decoding, v1.List unwrapping, manifest name validation, webhook
// client config validation, and deterministic sorting by name.
func LoadManifests[T metav1.Object](
	dir string,
	decoder runtime.Decoder,
	acceptObject AcceptObjectFunc[T],
) ([]T, string, error) {
	fileDocs, hash, err := manifest.LoadFiles(dir)
	if err != nil {
		return nil, "", err
	}

	configs := make([]T, 0)
	seenNames := map[string]string{}

	for _, fd := range fileDocs {
		obj, _, err := decoder.Decode(fd.Doc, nil, nil)
		if err != nil {
			return nil, "", fmt.Errorf("error loading %s: %w", fd.FilePath, err)
		}

		items, err := acceptFileObject(obj, fd.FilePath, seenNames, decoder, acceptObject)
		if err != nil {
			return nil, "", fmt.Errorf("error loading %s: %w", fd.FilePath, err)
		}
		configs = append(configs, items...)
	}

	sort.Slice(configs, func(i, j int) bool {
		return configs[i].GetName() < configs[j].GetName()
	})

	return configs, hash, nil
}

// acceptFileObject handles type dispatch for a decoded object, including
// generic v1.List unwrapping, manifest name validation, item validation,
// and the default unsupported-type error.
func acceptFileObject[T metav1.Object](
	obj runtime.Object,
	filePath string,
	seenNames map[string]string,
	decoder runtime.Decoder,
	acceptObject AcceptObjectFunc[T],
) ([]T, error) {
	// Handle generic v1.List by recursing into each item
	if list, ok := obj.(*metav1.List); ok {
		var allItems []T
		for _, rawItem := range list.Items {
			itemObj, _, err := decoder.Decode(rawItem.Raw, nil, nil)
			if err != nil {
				return nil, fmt.Errorf("failed to decode list item: %w", err)
			}
			items, err := acceptFileObject(itemObj, filePath, seenNames, decoder, acceptObject)
			if err != nil {
				return nil, err
			}
			allItems = append(allItems, items...)
		}
		return allItems, nil
	}

	items, err := acceptObject(obj)
	if err != nil {
		return nil, err
	}
	for _, item := range items {
		if err := validateAcceptedItem(item, filePath, seenNames); err != nil {
			return nil, err
		}
	}
	return items, nil
}

// validateAcceptedItem runs manifest name validation and webhook client config validation.
func validateAcceptedItem[T metav1.Object](item T, filePath string, seenNames map[string]string) error {
	name := item.GetName()
	if err := manifest.ValidateManifestName(name, filePath, seenNames); err != nil {
		return err
	}
	// Validate webhook client configs for webhook configuration types.
	obj, ok := any(item).(runtime.Object)
	if !ok {
		return fmt.Errorf("type %T does not implement runtime.Object", item)
	}
	return validateWebhookClientConfigs(obj, name, filePath)
}

// validateWebhookClientConfigs validates that all webhooks in a configuration
// use URL-based client config (service references are not supported for manifests).
func validateWebhookClientConfigs(obj runtime.Object, configName, filePath string) error {
	switch c := obj.(type) {
	case *admissionregistrationv1.ValidatingWebhookConfiguration:
		for _, wh := range c.Webhooks {
			if err := ValidateWebhookClientConfig(wh.Name, configName, filePath, wh.ClientConfig); err != nil {
				return err
			}
		}
	case *admissionregistrationv1.MutatingWebhookConfiguration:
		for _, wh := range c.Webhooks {
			if err := ValidateWebhookClientConfig(wh.Name, configName, filePath, wh.ClientConfig); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported webhook configuration type %T in file %q", obj, filePath)
	}
	return nil
}

// ValidateWebhookClientConfig checks that a webhook uses URL-based client config
// (service references are not supported for static manifests).
func ValidateWebhookClientConfig(webhookName, configName, filePath string, cc admissionregistrationv1.WebhookClientConfig) error {
	if cc.Service != nil {
		return fmt.Errorf("webhook %q in %q (file %q): clientConfig.service is not supported for static manifests; use clientConfig.url instead", webhookName, configName, filePath)
	}
	if cc.URL == nil || len(*cc.URL) == 0 {
		return fmt.Errorf("webhook %q in %q (file %q): clientConfig.url is required for static manifests", webhookName, configName, filePath)
	}
	return nil
}

// ValidatingLoadResult holds the validating webhook configurations loaded from manifest files.
type ValidatingLoadResult struct {
	// Configurations is the list of loaded validating webhook configurations.
	Configurations []*admissionregistrationv1.ValidatingWebhookConfiguration
	// Hash is the sha256 hash of all loaded files, used for change detection.
	Hash string
}

// MutatingLoadResult holds the mutating webhook configurations loaded from manifest files.
type MutatingLoadResult struct {
	// Configurations is the list of loaded mutating webhook configurations.
	Configurations []*admissionregistrationv1.MutatingWebhookConfiguration
	// Hash is the sha256 hash of all loaded files, used for change detection.
	Hash string
}

// BuildValidatingAccessors builds webhook accessors from validating webhook configurations.
func BuildValidatingAccessors(configs []*admissionregistrationv1.ValidatingWebhookConfiguration) []webhook.WebhookAccessor {
	var accessors []webhook.WebhookAccessor
	for _, config := range configs {
		names := map[string]int{}
		for i := range config.Webhooks {
			w := &config.Webhooks[i]
			n := w.Name
			uid := fmt.Sprintf("manifest/%s/%s/%d", config.Name, n, names[n])
			names[n]++
			accessors = append(accessors, webhook.NewValidatingWebhookAccessor(uid, config.Name, w))
		}
	}
	return accessors
}

// BuildMutatingAccessors builds webhook accessors from mutating webhook configurations.
func BuildMutatingAccessors(configs []*admissionregistrationv1.MutatingWebhookConfiguration) []webhook.WebhookAccessor {
	var accessors []webhook.WebhookAccessor
	for _, config := range configs {
		names := map[string]int{}
		for i := range config.Webhooks {
			w := &config.Webhooks[i]
			n := w.Name
			uid := fmt.Sprintf("manifest/%s/%s/%d", config.Name, n, names[n])
			names[n]++
			accessors = append(accessors, webhook.NewMutatingWebhookAccessor(uid, config.Name, w))
		}
	}
	return accessors
}

// GetWebhookAccessors returns webhook accessors for all validating webhooks.
func (r *ValidatingLoadResult) GetWebhookAccessors() []webhook.WebhookAccessor {
	return BuildValidatingAccessors(r.Configurations)
}

// GetWebhookAccessors returns webhook accessors for all mutating webhooks.
func (r *MutatingLoadResult) GetWebhookAccessors() []webhook.WebhookAccessor {
	return BuildMutatingAccessors(r.Configurations)
}
