/*
Copyright 2019 The Kubernetes Authors.

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

package util

import (
	"encoding/json"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/yaml"
)

// TextOutput interface contains methods required to be implemented for the data structures
// to be able to use ConvertToOutputFormat API
type TextOutput interface {
	Text() string
	Short() string
}

// ConvertToOutputFormat returns data representation in the specified output format
func ConvertToOutputFormat(data interface{}, outputFormat string) (string, error) {
	of := strings.ToLower(outputFormat)
	if of == "text" {
		return data.(TextOutput).Text(), nil
	} else if of == "short" {
		return data.(TextOutput).Short(), nil
	} else if of == "json" || of == "yaml" {
		bytes, err := json.MarshalIndent(data, "", "  ")
		if err != nil {
			return "", err
		}
		if outputFormat == "yaml" {
			bytes, err = yaml.JSONToYAML(bytes)
			if err != nil {
				return "", errors.Wrap(err, "failed to convert JSON output to YAML")
			}
		}
		return string(bytes), nil
	}

	return "", errors.Errorf("invalid output format: %s", outputFormat)
}
