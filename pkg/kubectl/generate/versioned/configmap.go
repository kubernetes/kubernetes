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

package versioned

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"
	"unicode/utf8"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/kubectl/generate"
	"k8s.io/kubernetes/pkg/kubectl/util"
	"k8s.io/kubernetes/pkg/kubectl/util/hash"
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
	// AppendHash; if true, derive a hash from the ConfigMap and append it to the name
	AppendHash bool
}

// Ensure it supports the generator pattern that uses parameter injection.
var _ generate.Generator = &ConfigMapGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction.
var _ generate.StructuredGenerator = &ConfigMapGeneratorV1{}

// Generate returns a configMap using the specified parameters.
func (s ConfigMapGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := generate.ValidateParams(s.ParamNames(), genericParams)
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
	fromEnvFileString, found := genericParams["from-env-file"]
	if found {
		fromEnvFile, isString := fromEnvFileString.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, found :%v", fromEnvFileString)
		}
		delegate.EnvFileSource = fromEnvFile
		delete(genericParams, "from-env-file")
	}
	hashParam, found := genericParams["append-hash"]
	if found {
		hashBool, isBool := hashParam.(bool)
		if !isBool {
			return nil, fmt.Errorf("expected bool, found :%v", hashParam)
		}
		delegate.AppendHash = hashBool
		delete(genericParams, "append-hash")
	}
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	delegate.Name = params["name"]
	delegate.Type = params["type"]

	return delegate.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern.
func (s ConfigMapGeneratorV1) ParamNames() []generate.GeneratorParam {
	return []generate.GeneratorParam{
		{Name: "name", Required: true},
		{Name: "type", Required: false},
		{Name: "from-file", Required: false},
		{Name: "from-literal", Required: false},
		{Name: "from-env-file", Required: false},
		{Name: "force", Required: false},
		{Name: "hash", Required: false},
	}
}

// StructuredGenerate outputs a configMap object using the configured fields.
func (s ConfigMapGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	configMap := &v1.ConfigMap{}
	configMap.Name = s.Name
	configMap.Data = map[string]string{}
	configMap.BinaryData = map[string][]byte{}
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
	if s.AppendHash {
		h, err := hash.ConfigMapHash(configMap)
		if err != nil {
			return nil, err
		}
		configMap.Name = fmt.Sprintf("%s-%s", configMap.Name, h)
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
func handleConfigMapFromLiteralSources(configMap *v1.ConfigMap, literalSources []string) error {
	for _, literalSource := range literalSources {
		keyName, value, err := util.ParseLiteralSource(literalSource)
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
func handleConfigMapFromFileSources(configMap *v1.ConfigMap, fileSources []string) error {
	for _, fileSource := range fileSources {
		keyName, filePath, err := util.ParseFileSource(fileSource)
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
func handleConfigMapFromEnvFileSource(configMap *v1.ConfigMap, envFileSource string) error {
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
func addKeyFromFileToConfigMap(configMap *v1.ConfigMap, keyName, filePath string) error {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return err
	}

	if utf8.Valid(data) {
		return addKeyFromLiteralToConfigMap(configMap, keyName, string(data))
	}

	err = validateNewConfigMap(configMap, keyName)
	if err != nil {
		return err
	}
	configMap.BinaryData[keyName] = data
	return nil
}

// addKeyFromLiteralToConfigMap adds the given key and data to the given config map,
// returning an error if the key is not valid or if the key already exists.
func addKeyFromLiteralToConfigMap(configMap *v1.ConfigMap, keyName, data string) error {
	err := validateNewConfigMap(configMap, keyName)
	if err != nil {
		return err
	}
	configMap.Data[keyName] = data
	return nil
}

func validateNewConfigMap(configMap *v1.ConfigMap, keyName string) error {
	// Note, the rules for ConfigMap keys are the exact same as the ones for SecretKeys.
	if errs := validation.IsConfigMapKey(keyName); len(errs) != 0 {
		return fmt.Errorf("%q is not a valid key name for a ConfigMap: %s", keyName, strings.Join(errs, ";"))
	}

	if _, exists := configMap.Data[keyName]; exists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists in data: %v", keyName, configMap.Data)
	}
	if _, exists := configMap.BinaryData[keyName]; exists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists in binaryData: %v", keyName, configMap.BinaryData)
	}
	return nil
}
