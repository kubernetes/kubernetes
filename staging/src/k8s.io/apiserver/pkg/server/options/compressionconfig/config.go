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
	"crypto/sha256"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/apiserver"
	apiserverv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
	"k8s.io/apiserver/pkg/apis/apiserver/validation"
	"k8s.io/apiserver/pkg/storage/value/compression"
)

var codecs serializer.CodecFactory

func init() {
	configScheme := runtime.NewScheme()
	utilruntime.Must(apiserver.AddToScheme(configScheme))
	utilruntime.Must(apiserverv1alpha1.AddToScheme(configScheme))
	codecs = serializer.NewCodecFactory(configScheme, serializer.EnableStrict)
}

// LoadCompressionConfig reads, decodes and validates the file, then builds the
// policy. The hash covers the file's bytes and is what metrics.RecordLastConfigInfo
// in compressionconfig/metrics reports. The file is read once, at startup.
func LoadCompressionConfig(filepath string) (*compression.Policy, string, error) {
	config, contentHash, err := loadConfig(filepath)
	if err != nil {
		return nil, "", err
	}
	return compression.NewPolicy(config), contentHash, nil
}

func loadConfig(filepath string) (*apiserver.StorageCompressionConfiguration, string, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, "", fmt.Errorf("error reading storage compression configuration file %q: %w", filepath, err)
	}
	if len(data) == 0 {
		return nil, "", fmt.Errorf("storage compression configuration file %q is empty", filepath)
	}

	configObj, gvk, err := codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, "", fmt.Errorf("error decoding storage compression configuration file %q: %w", filepath, err)
	}
	config, ok := configObj.(*apiserver.StorageCompressionConfiguration)
	if !ok {
		return nil, "", fmt.Errorf("got unexpected config type %v in storage compression configuration file %q", gvk, filepath)
	}
	if err := validation.ValidateStorageCompressionConfiguration(config).ToAggregate(); err != nil {
		return nil, "", fmt.Errorf("invalid storage compression configuration file %q: %w", filepath, err)
	}

	return config, fmt.Sprintf("sha256:%x", sha256.Sum256(data)), nil
}
