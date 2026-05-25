// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import (
	"encoding/json"

	"github.com/go-openapi/swag/yamlutils"
)

// YAMLToJSON converts YAML unmarshaled data into json compatible data
//
// Deprecated: use [yamlutils.YAMLToJSON] instead.
func YAMLToJSON(data any) (json.RawMessage, error) { return yamlutils.YAMLToJSON(data) }

// BytesToYAMLDoc converts a byte slice into a YAML document
//
// Deprecated: use [yamlutils.BytesToYAMLDoc] instead.
func BytesToYAMLDoc(data []byte) (any, error) { return yamlutils.BytesToYAMLDoc(data) }
