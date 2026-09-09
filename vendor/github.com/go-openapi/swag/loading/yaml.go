// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package loading

import (
	"encoding/json"
	"path/filepath"

	"github.com/go-openapi/swag/yamlutils"
)

// YAMLMatcher matches yaml for a file loader.
func YAMLMatcher(path string) bool {
	ext := filepath.Ext(path)
	return ext == ".yaml" || ext == ".yml"
}

// YAMLDoc loads a yaml document from either http or a file and converts it to json.
func YAMLDoc(path string, opts ...Option) (json.RawMessage, error) {
	yamlDoc, err := YAMLData(path, opts...)
	if err != nil {
		return nil, err
	}

	return yamlutils.YAMLToJSON(yamlDoc)
}

// YAMLData loads a yaml document from either http or a file.
func YAMLData(path string, opts ...Option) (any, error) {
	data, err := LoadFromFileOrHTTP(path, opts...)
	if err != nil {
		return nil, err
	}

	return yamlutils.BytesToYAMLDoc(data)
}
