// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package yamlutils provides utilities to work with YAML documents.
//
//   - [BytesToYAMLDoc] to construct a [yaml.Node] document
//   - [YAMLToJSON] to convert a [yaml.Node] document to JSON bytes
//   - [YAMLMapSlice] to serialize and deserialize YAML with the order of keys maintained
package yamlutils

import (
	_ "go.yaml.in/yaml/v3" // for documentation purpose only
)
