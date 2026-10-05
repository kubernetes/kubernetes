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

package config

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/manifest"
	"k8s.io/apiserver/pkg/admission/plugin/policy/config/apis/policyconfig"
	v1 "k8s.io/apiserver/pkg/admission/plugin/policy/config/apis/policyconfig/v1"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme, serializer.EnableStrict)
)

func init() {
	utilruntime.Must(policyconfig.AddToScheme(scheme))
	utilruntime.Must(v1.AddToScheme(scheme))
}

// PolicyConfig holds the configuration loaded from the config file.
type PolicyConfig struct {
	// StaticManifestsDir is the path to the directory containing static policy manifests.
	StaticManifestsDir string
}

// staticManifestsDirAccessor is implemented by config types that have a StaticManifestsDir field.
type staticManifestsDirAccessor interface {
	runtime.Object
	GetStaticManifestsDir() string
}

// LoadValidatingConfig extracts the validating admission policy configuration from configFile.
func LoadValidatingConfig(configFile io.Reader) (PolicyConfig, error) {
	return loadConfig(configFile, func(obj runtime.Object) bool {
		_, ok := obj.(*policyconfig.ValidatingAdmissionPolicyConfiguration)
		return ok
	})
}

// LoadMutatingConfig extracts the mutating admission policy configuration from configFile.
func LoadMutatingConfig(configFile io.Reader) (PolicyConfig, error) {
	return loadConfig(configFile, func(obj runtime.Object) bool {
		_, ok := obj.(*policyconfig.MutatingAdmissionPolicyConfiguration)
		return ok
	})
}

func loadConfig(configFile io.Reader, isExpectedType func(runtime.Object) bool) (PolicyConfig, error) {
	var cfg PolicyConfig
	if configFile == nil {
		return cfg, nil
	}

	data, err := io.ReadAll(configFile)
	if err != nil {
		return cfg, err
	}
	decoder := codecs.UniversalDecoder()
	decodedObj, err := runtime.Decode(decoder, data)
	if err != nil {
		return cfg, err
	}
	if !isExpectedType(decodedObj) {
		return cfg, fmt.Errorf("unexpected type: %T", decodedObj)
	}
	config, ok := decodedObj.(staticManifestsDirAccessor)
	if !ok {
		return cfg, fmt.Errorf("type %T does not implement staticManifestsDirAccessor", decodedObj)
	}

	if err := manifest.ValidateStaticManifestsDir(config.GetStaticManifestsDir()); err != nil {
		return cfg, err
	}
	cfg.StaticManifestsDir = config.GetStaticManifestsDir()
	return cfg, nil
}
