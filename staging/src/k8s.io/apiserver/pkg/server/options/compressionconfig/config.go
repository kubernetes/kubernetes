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

// Package compressionconfig loads the StorageCompressionConfiguration file that
// --storage-compression-config points at.
package compressionconfig

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserver"
	apiserverv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
	"k8s.io/apiserver/pkg/internal/storagecompression"
	"k8s.io/apiserver/pkg/storage/value/compression"
)

var (
	configScheme = runtime.NewScheme()
	codecs       serializer.CodecFactory
)

func init() {
	utilruntime.Must(apiserver.AddToScheme(configScheme))
	utilruntime.Must(apiserverv1alpha1.AddToScheme(configScheme))
	codecs = serializer.NewCodecFactory(configScheme, serializer.EnableStrict)
}

// CompressionConfiguration is a StorageCompressionConfiguration that passed
// validation. Only LoadCompressionConfig fills in its rules, so the zero value
// and a nil pointer compress nothing.
type CompressionConfiguration struct {
	// CompressionFileContentHash is the sha256 of the file's bytes, the value
	// to pass to metrics.RecordLastConfigInfo in compressionconfig/metrics.
	CompressionFileContentHash string

	// rules maps each selector to its entry's Rule. Validation rejects masked
	// selectors, so the most specific match is also the first one listed.
	rules map[schema.GroupResource]compression.Rule
}

// LoadCompressionConfig reads, decodes and validates the file. The file is read
// once, at startup.
func LoadCompressionConfig(filepath string) (*CompressionConfiguration, error) {
	config, contentHash, err := loadConfig(filepath)
	if err != nil {
		return nil, err
	}
	c, err := newCompressionConfiguration(config, contentHash)
	if err != nil {
		return nil, fmt.Errorf("invalid storage compression configuration file %q: %w", filepath, err)
	}
	return c, nil
}

// RuleFor returns the Rule of the first entry with a selector that covers gr.
// If there's none, or c is nil, it returns the zero Rule, which compresses
// nothing. Pass the result to compression.WithCompression for gr.
func (c *CompressionConfiguration) RuleFor(gr schema.GroupResource) compression.Rule {
	if c == nil {
		return compression.Rule{}
	}
	for _, selector := range storagecompression.CoveringSelectors(gr) {
		if r, ok := c.rules[selector]; ok {
			return r
		}
	}
	return compression.Rule{}
}

// newCompressionConfiguration defaults and validates the versioned object, and
// fills in rules only if it passes. Nothing else fills them in, so no
// CompressionConfiguration skips either step. Decoding has usually applied the
// defaults already, and applying them again changes nothing.
func newCompressionConfiguration(versioned *apiserverv1alpha1.StorageCompressionConfiguration, contentHash string) (*CompressionConfiguration, error) {
	if versioned == nil {
		return nil, errors.New("StorageCompressionConfiguration can't be nil")
	}
	configScheme.Default(versioned)
	if err := validateConfig(versioned).ToAggregate(); err != nil {
		return nil, err
	}
	config := &apiserver.StorageCompressionConfiguration{}
	if err := configScheme.Convert(versioned, config, nil); err != nil {
		return nil, fmt.Errorf("converting %s to the internal version: %w", apiserverv1alpha1.ConfigSchemeGroupVersion, err)
	}
	c := &CompressionConfiguration{
		CompressionFileContentHash: contentHash,
		rules:                      map[schema.GroupResource]compression.Rule{},
	}
	for _, entry := range config.CompressionResources {
		r := entryRule(entry)
		for _, selector := range entry.Resources {
			c.rules[schema.ParseGroupResource(selector)] = r
		}
	}
	return c, nil
}

// validateConfig runs the validation generated from the v1alpha1 tags, then the
// selector checks that no tag can express.
func validateConfig(versioned *apiserverv1alpha1.StorageCompressionConfiguration) field.ErrorList {
	allErrs := apiserverv1alpha1.Validate_StorageCompressionConfiguration(context.Background(), operation.Operation{Type: operation.Create}, nil, versioned, nil)
	compressionResourcesPath := field.NewPath("compressionResources")
	selectorsByEntry := make([][]string, 0, len(versioned.CompressionResources))
	for i, entry := range versioned.CompressionResources {
		allErrs = append(allErrs, storagecompression.ValidateSelectors(entry.Resources, compressionResourcesPath.Index(i).Child("resources"))...)
		selectorsByEntry = append(selectorsByEntry, entry.Resources)
	}
	return append(allErrs, storagecompression.ValidateMasking(selectorsByEntry, compressionResourcesPath)...)
}

// entryRule maps one validated entry to its Rule. Defaulting has already set
// minSize, and Value rounds a fraction up.
func entryRule(entry apiserver.CompressionResourceConfiguration) compression.Rule {
	var r compression.Rule
	if entry.Algorithm == apiserver.CompressionAlgorithmDeflate {
		r.Algorithm = compression.AlgorithmDeflate
	}
	if entry.MinSize != nil {
		r.MinSize = int(entry.MinSize.Value())
	}
	return r
}

// loadConfig decodes to v1alpha1, the version validation runs on.
// UniversalDecoder converts from any version the scheme knows, so it accepts
// the same files that decoding to the internal version would.
func loadConfig(filepath string) (*apiserverv1alpha1.StorageCompressionConfiguration, string, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, "", fmt.Errorf("error reading storage compression configuration file %q: %w", filepath, err)
	}
	if len(data) == 0 {
		return nil, "", fmt.Errorf("storage compression configuration file %q is empty", filepath)
	}

	configObj, gvk, err := codecs.UniversalDecoder(apiserverv1alpha1.ConfigSchemeGroupVersion).Decode(data, nil, nil)
	if err != nil {
		return nil, "", fmt.Errorf("error decoding storage compression configuration file %q: %w", filepath, err)
	}
	config, ok := configObj.(*apiserverv1alpha1.StorageCompressionConfiguration)
	if !ok {
		return nil, "", fmt.Errorf("got unexpected config type %v in storage compression configuration file %q", gvk, filepath)
	}
	return config, fmt.Sprintf("sha256:%x", sha256.Sum256(data)), nil
}
