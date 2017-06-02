/*
Copyright 2016 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/api"
)

// ConfigMapGeneratorV1 supports stable generation of a configMap.
type ConfigMapGeneratorV1 struct {
	// Name of configMap (required)
	Name string
	// Type of configMap (optional)
	Type string
	// FileSources to derive the configMap from (optional)
	FileSources []string
	// LiteralSources to derive the configMap from (optional)
	LiteralSources []string
	// EnvFileSource to derive the configMap from (optional)
	EnvFileSource string
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ Generator = &ConfigMapGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ StructuredGenerator = &ConfigMapGeneratorV1{}

// Generate returns a configMap using the specified parameters.
func (s ConfigMapGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &ConfigMapGeneratorV1{}
	fromFileStrings, found := genericParams["from-file"]
	if found {
		fromFileArray, isArray := fromFileStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", fromFileStrings)
		}
		delegate.FileSources = fromFileArray
		delete(genericParams, "from-file")
	}
	fromLiteralStrings, found := genericParams["from-literal"]
	if found {
		fromLiteralArray, isArray := fromLiteralStrings.([]string)
		if !isArray {
			return nil, fmt.Errorf("expected []string, found :%v", fromLiteralStrings)
		}
		delegate.LiteralSources = fromLiteralArray
		delete(genericParams, "from-literal")
	}
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	fromEnvFileString, found := genericParams["from-env-file"]
	if found {
		fromEnvFile, isString := fromEnvFileString.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, found :%v", fromEnvFileString)
		}
		delegate.EnvFileSource = fromEnvFile
		delete(genericParams, "from-env-file")
	}
	delegate.Name = params["name"]
	delegate.Type = params["type"]
	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s ConfigMapGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"type", false},
		{"from-file", false},
		{"from-literal", false},
		{"from-env-file", false},
		{"force", false},
	}
}

// StructuredGenerate outputs a configMap object using the configured fields.
func (s ConfigMapGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	configMap := &api.ConfigMap{}
	configMap.Name = s.Name
	configMap.Data = map[string]string{}
	if len(s.FileSources) > 0 {
		if err := handleConfigMapFromFileSources(configMap, s.FileSources); err != nil {
			return nil, err
		}
	}
	if len(s.LiteralSources) > 0 {
		if err := handleConfigMapFromLiteralSources(configMap, s.LiteralSources); err != nil {
			return nil, err
		}
	}
	if len(s.EnvFileSource) > 0 {
		if err := handleConfigMapFromEnvFileSource(configMap, s.EnvFileSource); err != nil {
			return nil, err
		}
	}
	return configMap, nil
}

// validate validates required fields are set to support structured generation.
func (s ConfigMapGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.EnvFileSource) > 0 && (len(s.FileSources) > 0 || len(s.LiteralSources) > 0) {
		return fmt.Errorf("from-env-file cannot be combined with from-file or from-literal")
	}
	return nil
}

// handleConfigMapFromLiteralSources adds the specified literal source
// information into the provided configMap.
func handleConfigMapFromLiteralSources(configMap *api.ConfigMap, literalSources []string) error {
	for _, literalSource := range literalSources {
		keyName, value, err := parseLiteralSource(literalSource)
		if err != nil {
			return err
		}
		err = addKeyFromLiteralToConfigMap(configMap, keyName, value)
		if err != nil {
			return err
		}
	}
	return nil
}

// handleConfigMapFromFileSources adds the specified file source information
// into the provided configMap
func handleConfigMapFromFileSources(configMap *api.ConfigMap, fileSources []string) error {
	for _, fileSource := range fileSources {
		keyName, filePath, err := parseFileSource(fileSource)
		if err != nil {
			return err
		}
		info, err := os.Stat(filePath)
		if err != nil {
			switch err := err.(type) {
			case *os.PathError:
				return fmt.Errorf("error reading %s: %v", filePath, err.Err)
			default:
				return fmt.Errorf("error reading %s: %v", filePath, err)
			}
		}
		if info.IsDir() {
			if strings.Contains(fileSource, "=") {
				return fmt.Errorf("cannot give a key name for a directory path.")
			}
			fileList, err := ioutil.ReadDir(filePath)
			if err != nil {
				return fmt.Errorf("error listing files in %s: %v", filePath, err)
			}
			for _, item := range fileList {
				itemPath := path.Join(filePath, item.Name())
				if item.Mode().IsRegular() {
					keyName = item.Name()
					err = addKeyFromFileToConfigMap(configMap, keyName, itemPath)
					if err != nil {
						return err
					}
				}
			}
		} else {
			if err := addKeyFromFileToConfigMap(configMap, keyName, filePath); err != nil {
				return err
			}
		}
	}

	return nil
}

// handleConfigMapFromEnvFileSource adds the specified env file source information
// into the provided configMap
func handleConfigMapFromEnvFileSource(configMap *api.ConfigMap, envFileSource string) error {
	info, err := os.Stat(envFileSource)
	if err != nil {
		switch err := err.(type) {
		case *os.PathError:
			return fmt.Errorf("error reading %s: %v", envFileSource, err.Err)
		default:
			return fmt.Errorf("error reading %s: %v", envFileSource, err)
		}
	}
	if info.IsDir() {
		return fmt.Errorf("env config file cannot be a directory")
	}

	return addFromEnvFile(envFileSource, func(key, value string) error {
		return addKeyFromLiteralToConfigMap(configMap, key, value)
	})
}

// addKeyFromFileToConfigMap adds a key with the given name to a ConfigMap, populating
// the value with the content of the given file path, or returns an error.
func addKeyFromFileToConfigMap(configMap *api.ConfigMap, keyName, filePath string) error {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return err
	}
	return addKeyFromLiteralToConfigMap(configMap, keyName, string(data))
}

// addKeyFromLiteralToConfigMap adds the given key and data to the given config map,
// returning an error if the key is not valid or if the key already exists.
func addKeyFromLiteralToConfigMap(configMap *api.ConfigMap, keyName, data string) error {
	// Note, the rules for ConfigMap keys are the exact same as the ones for SecretKeys.
	if errs := validation.IsConfigMapKey(keyName); len(errs) != 0 {
		return fmt.Errorf("%q is not a valid key name for a ConfigMap: %s", keyName, strings.Join(errs, ";"))
	}
	if _, entryExists := configMap.Data[keyName]; entryExists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists: %v.", keyName, configMap.Data)
	}
	configMap.Data[keyName] = data
	return nil
}
