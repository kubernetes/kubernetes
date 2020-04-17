/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubectl/pkg/generate"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/hash"
)

// SecretGeneratorV1 supports stable generation of an opaque secret
type SecretGeneratorV1 struct {
	// Name of secret (required)
	Name string
	// Type of secret (optional)
	Type string
	// FileSources to derive the secret from (optional)
	FileSources []string
	// LiteralSources to derive the secret from (optional)
	LiteralSources []string
	// EnvFileSource to derive the secret from (optional)
	EnvFileSource string
	// AppendHash; if true, derive a hash from the Secret data and type and append it to the name
	AppendHash bool
}

// Ensure it supports the generator pattern that uses parameter injection
var _ generate.Generator = &SecretGeneratorV1{}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ generate.StructuredGenerator = &SecretGeneratorV1{}

// Generate returns a secret using the specified parameters
func (s SecretGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := generate.ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	delegate := &SecretGeneratorV1{}
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

// ParamNames returns the set of supported input parameters when using the parameter injection generator pattern
func (s SecretGeneratorV1) ParamNames() []generate.GeneratorParam {
	return []generate.GeneratorParam{
		{Name: "name", Required: true},
		{Name: "type", Required: false},
		{Name: "from-file", Required: false},
		{Name: "from-literal", Required: false},
		{Name: "from-env-file", Required: false},
		{Name: "force", Required: false},
		{Name: "append-hash", Required: false},
	}
}

// StructuredGenerate outputs a secret object using the configured fields
func (s SecretGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	secret := &v1.Secret{}
	secret.SetGroupVersionKind(v1.SchemeGroupVersion.WithKind("Secret"))
	secret.Name = s.Name
	secret.Data = map[string][]byte{}
	if len(s.Type) > 0 {
		secret.Type = v1.SecretType(s.Type)
	}
	if len(s.FileSources) > 0 {
		if err := handleFromFileSources(secret, s.FileSources); err != nil {
			return nil, err
		}
	}
	if len(s.LiteralSources) > 0 {
		if err := handleFromLiteralSources(secret, s.LiteralSources); err != nil {
			return nil, err
		}
	}
	if len(s.EnvFileSource) > 0 {
		if err := handleFromEnvFileSource(secret, s.EnvFileSource); err != nil {
			return nil, err
		}
	}
	if s.AppendHash {
		h, err := hash.SecretHash(secret)
		if err != nil {
			return nil, err
		}
		secret.Name = fmt.Sprintf("%s-%s", secret.Name, h)
	}
	return secret, nil
}

// validate validates required fields are set to support structured generation
func (s SecretGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.EnvFileSource) > 0 && (len(s.FileSources) > 0 || len(s.LiteralSources) > 0) {
		return fmt.Errorf("from-env-file cannot be combined with from-file or from-literal")
	}
	return nil
}

// handleFromLiteralSources adds the specified literal source information into the provided secret
func handleFromLiteralSources(secret *v1.Secret, literalSources []string) error {
	for _, literalSource := range literalSources {
		keyName, value, err := util.ParseLiteralSource(literalSource)
		if err != nil {
			return err
		}
		if err = addKeyFromLiteralToSecret(secret, keyName, []byte(value)); err != nil {
			return err
		}
	}
	return nil
}

// handleFromFileSources adds the specified file source information into the provided secret
func handleFromFileSources(secret *v1.Secret, fileSources []string) error {
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
				return fmt.Errorf("cannot give a key name for a directory path")
			}
			fileList, err := ioutil.ReadDir(filePath)
			if err != nil {
				return fmt.Errorf("error listing files in %s: %v", filePath, err)
			}
			for _, item := range fileList {
				itemPath := path.Join(filePath, item.Name())
				if item.Mode().IsRegular() {
					keyName = item.Name()
					if err = addKeyFromFileToSecret(secret, keyName, itemPath); err != nil {
						return err
					}
				}
			}
		} else {
			if err := addKeyFromFileToSecret(secret, keyName, filePath); err != nil {
				return err
			}
		}
	}

	return nil
}

// handleFromEnvFileSource adds the specified env file source information
// into the provided secret
func handleFromEnvFileSource(secret *v1.Secret, envFileSource string) error {
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
		return fmt.Errorf("env secret file cannot be a directory")
	}

	return addFromEnvFile(envFileSource, func(key, value string) error {
		return addKeyFromLiteralToSecret(secret, key, []byte(value))
	})
}

func addKeyFromFileToSecret(secret *v1.Secret, keyName, filePath string) error {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return err
	}
	return addKeyFromLiteralToSecret(secret, keyName, data)
}

func addKeyFromLiteralToSecret(secret *v1.Secret, keyName string, data []byte) error {
	if errs := validation.IsConfigMapKey(keyName); len(errs) != 0 {
		return fmt.Errorf("%q is not a valid key name for a Secret: %s", keyName, strings.Join(errs, ";"))
	}

	if _, entryExists := secret.Data[keyName]; entryExists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists", keyName)
	}
	secret.Data[keyName] = data
	return nil
}
