/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/runtime"
)

type SecretGeneratorV1 struct{}

func (SecretGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"default-name", false},
		{"name", true},
		{"type", false},
		{"from-file", false},
		{"from-literal", false},
	}
}

func (SecretGeneratorV1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	secret := &api.Secret{
		Data: map[string][]byte{},
	}

	fromFileStrings, found := genericParams["from-file"]
	if found {
		if fromFileArray, isArray := fromFileStrings.([]string); isArray {
			if err := handleFromFileSources(secret, fromFileArray); err != nil {
				return nil, err
			}
			delete(genericParams, "from-file")
		} else {
			return nil, fmt.Errorf("expected []string, found :%v", fromFileStrings)
		}
	}
	fromLiteralStrings, found := genericParams["from-literal"]
	if found {
		if fromLiteralArray, isArray := fromLiteralStrings.([]string); isArray {
			if err := handleFromLiteralSources(secret, fromLiteralArray); err != nil {
				return nil, err
			}
			delete(genericParams, "from-literal")
		} else {
			return nil, fmt.Errorf("expected []string, found :%v", fromFileStrings)
		}
	}

	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}

	name, found := params["name"]
	secret.Name = name

	if secretType, found := params["type"]; found {
		secret.Type = api.SecretType(secretType)
	}

	return secret, nil
}

// handleFromLiteralSources adds the specified literal source information into the provided secret
func handleFromLiteralSources(secret *api.Secret, literalSources []string) error {
	for _, literalSource := range literalSources {
		keyName, value, err := parseLiteralSource(literalSource)
		if err != nil {
			return err
		}
		addKeyFromLiteralToSecret(secret, keyName, []byte(value))
	}
	return nil
}

// handleFromFileSources adds the specified file source information into the provided secret
func handleFromFileSources(secret *api.Secret, fileSources []string) error {
	for _, fileSource := range fileSources {
		keyName, filePath, err := parseFileSource(fileSource)
		fmt.Fprintf(os.Stderr, "FILE %v\n", filePath)
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
			fmt.Fprintf(os.Stderr, "ISDIR\n")
			if strings.Contains(fileSource, "=") {
				return fmt.Errorf("cannot give a key name for a directory path.")
			}
			fileList, err := ioutil.ReadDir(filePath)
			if err != nil {
				return fmt.Errorf("error listing files in %s: %v", filePath, err)
			}
			for _, item := range fileList {
				itemPath := path.Join(filePath, item.Name())
				fmt.Fprintf(os.Stderr, "FILES %v\n", itemPath)
				if !item.Mode().IsRegular() {
					// NEED A FLAG for quiet
					if true {
						fmt.Fprintf(os.Stderr, "Skipping resource %s\n", itemPath)
					}
				} else {
					keyName = item.Name()
					err = addKeyFromFileToSecret(secret, keyName, itemPath)
					if err != nil {
						return err
					}
				}
			}
		} else {
			fmt.Fprintf(os.Stderr, "KEY %v FILE %v\n", keyName, filePath)
			err = addKeyFromFileToSecret(secret, keyName, filePath)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func addKeyFromFileToSecret(secret *api.Secret, keyName, filePath string) error {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return err
	}
	addKeyFromLiteralToSecret(secret, keyName, data)
	return nil
}

func addKeyFromLiteralToSecret(secret *api.Secret, keyName string, data []byte) error {
	if !validation.IsSecretKey(keyName) {
		return fmt.Errorf("%v is not a valid key name for a secret", keyName)
	}
	if _, entryExists := secret.Data[keyName]; entryExists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists: %v.", keyName, secret.Data)
	}
	secret.Data[keyName] = data
	return nil
}

// parseFileSource parses the source given. Acceptable formats include:
// source-name=source-path, where source-name will become the key name and source-path is the path to the key file
// source-path, where source-path is a path to a file or directory, and key names will default to file names
// Key names cannot include '='.
func parseFileSource(source string) (keyName, filePath string, err error) {
	numSeparators := strings.Count(source, "=")
	switch {
	case numSeparators == 0:
		return path.Base(source), source, nil
	case numSeparators == 1 && strings.HasPrefix(source, "="):
		return "", "", fmt.Errorf("key name for file path %v missing.", strings.TrimPrefix(source, "="))
	case numSeparators == 1 && strings.HasSuffix(source, "="):
		return "", "", fmt.Errorf("file path for key name %v missing.", strings.TrimSuffix(source, "="))
	case numSeparators > 1:
		return "", "", errors.New("Key names or file paths cannot contain '='.")
	default:
		components := strings.Split(source, "=")
		return components[0], components[1], nil
	}
}

// parseLiteralSource parses the source key=val pair
func parseLiteralSource(source string) (keyName, value string, err error) {
	items := strings.Split(source, "=")
	if len(items) != 2 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}
	return items[0], items[1], nil
}
