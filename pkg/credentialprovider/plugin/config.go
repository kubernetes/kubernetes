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

package plugin

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	credentialproviderv1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

var (
	validCacheTypes = sets.New[string](
		string(kubeletconfig.ServiceAccountServiceAccountTokenCacheType),
		string(kubeletconfig.TokenServiceAccountTokenCacheType),
	)
)

// readCredentialProviderConfig receives a path to a config file or directory.
// If the path is a directory, it reads all "*.json", "*.yaml" and "*.yml" files in lexicographic order,
// decodes them, and merges their entries into a single CredentialProviderConfig object.
// If the path is a file, it decodes the file into a CredentialProviderConfig object directly.
// It also returns the SHA256 hash of all the raw file content. This hash is exposed via metrics
// as an external API to allow monitoring of configuration changes.
// The hash format follows container digest conventions (sha256:hexstring) for consistency.
func readCredentialProviderConfig(configPath string) (*kubeletconfig.CredentialProviderConfig, string, error) {
	if configPath == "" {
		return nil, "", fmt.Errorf("credential provider config path is empty")
	}

	fileInfo, err := os.Stat(configPath)
	if err != nil {
		return nil, "", fmt.Errorf("unable to access path %q: %w", configPath, err)
	}

	var configs []*kubeletconfig.CredentialProviderConfig
	var configFiles []string

	if fileInfo.IsDir() {
		entries, err := os.ReadDir(configPath)
		if err != nil {
			return nil, "", fmt.Errorf("unable to read directory %q: %w", configPath, err)
		}

		// Filter and sort *.json/*.yaml/*.yml files in lexicographic order
		for _, entry := range entries {
			ext := filepath.Ext(entry.Name())
			if !entry.IsDir() && (ext == ".json" || ext == ".yaml" || ext == ".yml") {
				configFiles = append(configFiles, filepath.Join(configPath, entry.Name()))
			}
		}
		sort.Strings(configFiles)

		if len(configFiles) == 0 {
			return nil, "", fmt.Errorf("no configuration files found in directory %q", configPath)
		}
	} else {
		configFiles = append(configFiles, configPath)
	}

	hasher := sha256.New()
	for _, filePath := range configFiles {
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, "", fmt.Errorf("unable to read file %q: %w", filePath, err)
		}

		// Use length prefix to prevent hash collisions
		dataLen := uint64(len(data))
		if err := binary.Write(hasher, binary.BigEndian, dataLen); err != nil {
			return nil, "", fmt.Errorf("error writing length prefix for file %q: %w", filePath, err)
		}
		hasher.Write(data)

		config, err := decode(data)
		if err != nil {
			return nil, "", fmt.Errorf("error decoding config %q: %w", filePath, err)
		}
		configs = append(configs, config)
	}

	// Merge all configs into a single CredentialProviderConfig
	mergedConfig := &kubeletconfig.CredentialProviderConfig{}
	providerNames := sets.NewString()
	for _, config := range configs {
		for _, provider := range config.Providers {
			if providerNames.Has(provider.Name) {
				return nil, "", fmt.Errorf("duplicate provider name %q found in configuration file(s)", provider.Name)
			}
			providerNames.Insert(provider.Name)
			mergedConfig.Providers = append(mergedConfig.Providers, provider)
		}
	}

	configHash := fmt.Sprintf("sha256:%x", hasher.Sum(nil))
	return mergedConfig, configHash, nil
}

// decode decodes data into the internal CredentialProviderConfig type.
func decode(data []byte) (*kubeletconfig.CredentialProviderConfig, error) {
	obj, gvk, err := codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}

	if gvk.Kind != "CredentialProviderConfig" {
		return nil, fmt.Errorf("failed to decode %q (wrong Kind)", gvk.Kind)
	}

	if gvk.Group != kubeletconfig.GroupName {
		return nil, fmt.Errorf("failed to decode CredentialProviderConfig, unexpected Group: %s", gvk.Group)
	}

	if internalConfig, ok := obj.(*kubeletconfig.CredentialProviderConfig); ok {
		return internalConfig, nil
	}

	return nil, fmt.Errorf("unable to convert %T to *CredentialProviderConfig", obj)
}

// validateCredentialProviderConfig validates CredentialProviderConfig.
func validateCredentialProviderConfig(config *kubeletconfig.CredentialProviderConfig, saTokenForCredentialProviders bool) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(config.Providers) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("providers"), "at least 1 item in plugins is required"))
	}

	fieldPath := field.NewPath("providers")
	seenProviderNames := sets.NewString()
	for _, provider := range config.Providers {
		if strings.Contains(provider.Name, "/") {
			allErrs = append(allErrs, field.Invalid(fieldPath.Child("name"), provider.Name, "provider name cannot contain '/'"))
		}

		if strings.Contains(provider.Name, " ") {
			allErrs = append(allErrs, field.Invalid(fieldPath.Child("name"), provider.Name, "provider name cannot contain spaces"))
		}

		if provider.Name == "." {
			allErrs = append(allErrs, field.Invalid(fieldPath.Child("name"), provider.Name, "provider name cannot be '.'"))
		}

		if provider.Name == ".." {
			allErrs = append(allErrs, field.Invalid(fieldPath.Child("name"), provider.Name, "provider name cannot be '..'"))
		}

		if seenProviderNames.Has(provider.Name) {
			allErrs = append(allErrs, field.Duplicate(fieldPath.Child("name"), provider.Name))
		}
		seenProviderNames.Insert(provider.Name)

		if provider.APIVersion == "" {
			allErrs = append(allErrs, field.Required(fieldPath.Child("apiVersion"), ""))
		} else if _, ok := apiVersions[provider.APIVersion]; !ok {
			validAPIVersions := sets.StringKeySet(apiVersions).List()
			allErrs = append(allErrs, field.NotSupported(fieldPath.Child("apiVersion"), provider.APIVersion, validAPIVersions))
		}

		if len(provider.MatchImages) == 0 {
			allErrs = append(allErrs, field.Required(fieldPath.Child("matchImages"), "at least 1 item in matchImages is required"))
		}

		for _, matchImage := range provider.MatchImages {
			if _, err := credentialprovider.ParseSchemelessURL(matchImage); err != nil {
				allErrs = append(allErrs, field.Invalid(fieldPath.Child("matchImages"), matchImage, fmt.Sprintf("match image is invalid: %s", err.Error())))
			}
		}

		if provider.DefaultCacheDuration == nil {
			allErrs = append(allErrs, field.Required(fieldPath.Child("defaultCacheDuration"), ""))
		}

		if provider.DefaultCacheDuration != nil && provider.DefaultCacheDuration.Duration < 0 {
			allErrs = append(allErrs, field.Invalid(fieldPath.Child("defaultCacheDuration"), provider.DefaultCacheDuration, "must be greater than or equal to 0"))
		}

		if provider.TokenAttributes != nil {
			fldPath := fieldPath.Child("tokenAttributes")
			if !saTokenForCredentialProviders {
				allErrs = append(allErrs, field.Forbidden(fldPath, "tokenAttributes is not supported when KubeletServiceAccountTokenForCredentialProviders feature gate is disabled"))
			}
			if len(provider.TokenAttributes.ServiceAccountTokenAudience) == 0 {
				allErrs = append(allErrs, field.Required(fldPath.Child("serviceAccountTokenAudience"), ""))
			}
			if provider.TokenAttributes.RequireServiceAccount == nil {
				allErrs = append(allErrs, field.Required(fldPath.Child("requireServiceAccount"), ""))
			}
			if provider.APIVersion != credentialproviderv1.SchemeGroupVersion.String() {
				allErrs = append(allErrs, field.Forbidden(fldPath, fmt.Sprintf("tokenAttributes is only supported for %s API version", credentialproviderv1.SchemeGroupVersion.String())))
			}

			if provider.TokenAttributes.RequireServiceAccount != nil && !*provider.TokenAttributes.RequireServiceAccount && len(provider.TokenAttributes.RequiredServiceAccountAnnotationKeys) > 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("requiredServiceAccountAnnotationKeys"), "requireServiceAccount cannot be false when requiredServiceAccountAnnotationKeys is set"))
			}

			allErrs = append(allErrs, validateServiceAccountAnnotationKeys(fldPath.Child("requiredServiceAccountAnnotationKeys"), provider.TokenAttributes.RequiredServiceAccountAnnotationKeys)...)
			allErrs = append(allErrs, validateServiceAccountAnnotationKeys(fldPath.Child("optionalServiceAccountAnnotationKeys"), provider.TokenAttributes.OptionalServiceAccountAnnotationKeys)...)

			requiredServiceAccountAnnotationKeys := sets.New[string](provider.TokenAttributes.RequiredServiceAccountAnnotationKeys...)
			optionalServiceAccountAnnotationKeys := sets.New[string](provider.TokenAttributes.OptionalServiceAccountAnnotationKeys...)
			duplicateAnnotationKeys := requiredServiceAccountAnnotationKeys.Intersection(optionalServiceAccountAnnotationKeys)
			if duplicateAnnotationKeys.Len() > 0 {
				allErrs = append(allErrs, field.Invalid(fldPath, sets.List(duplicateAnnotationKeys), "annotation keys cannot be both required and optional"))
			}

			switch {
			case len(provider.TokenAttributes.CacheType) == 0:
				allErrs = append(allErrs, field.Required(fldPath.Child("cacheType"), fmt.Sprintf("cacheType is required to be set when tokenAttributes is specified. Supported values are: %s", strings.Join(sets.List(validCacheTypes), ", "))))
			case validCacheTypes.Has(string(provider.TokenAttributes.CacheType)):
				// ok
			default:
				allErrs = append(allErrs, field.NotSupported(fldPath.Child("cacheType"), provider.TokenAttributes.CacheType, sets.List(validCacheTypes)))
			}
		}
	}

	return allErrs
}

// validateServiceAccountAnnotationKeys validates the service account annotation keys.
func validateServiceAccountAnnotationKeys(fldPath *field.Path, keys []string) field.ErrorList {
	allErrs := field.ErrorList{}

	seenAnnotationKeys := sets.New[string]()
	// Using the validation logic for keys from https://github.com/kubernetes/kubernetes/blob/69dbc74417304328a9fd3c161643dc4f0a057f41/staging/src/k8s.io/apimachinery/pkg/api/validation/objectmeta.go#L46-L51
	for _, k := range keys {
		// The rule is QualifiedName except that case doesn't matter, so convert to lowercase before checking.
		for _, msg := range validation.IsQualifiedName(strings.ToLower(k)) {
			allErrs = append(allErrs, field.Invalid(fldPath, k, msg))
		}
		if seenAnnotationKeys.Has(k) {
			allErrs = append(allErrs, field.Duplicate(fldPath, k))
		}
		seenAnnotationKeys.Insert(k)
	}
	return allErrs
}
