// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package error has contextual error types.
package kusterr

import (
	"fmt"
	"strings"
)

// YamlFormatError represents error with yaml file name where json/yaml format error happens.
type YamlFormatError struct {
	Path     string
	ErrorMsg string
}

func (e YamlFormatError) Error() string {
	return fmt.Sprintf("YAML file [%s] encounters a format error.\n%s\n", e.Path, e.ErrorMsg)
}

// MalformedYamlError represents an error that occurred while trying to decode a given YAML.
type MalformedYamlError struct {
	Path     string
	ErrorMsg string
}

func (e MalformedYamlError) Error() string {
	return fmt.Sprintf("%s in File: %s", e.ErrorMsg, e.Path)
}

// Handler handles YamlFormatError
func Handler(e error, path string) error {
	if isYAMLSyntaxError(e) {
		return YamlFormatError{
			Path:     path,
			ErrorMsg: e.Error(),
		}
	}
	if IsMalformedYAMLError(e) {
		return MalformedYamlError{
			Path:     path,
			ErrorMsg: e.Error(),
		}
	}
	return e
}

func isYAMLSyntaxError(e error) bool {
	return strings.Contains(e.Error(), "error converting YAML to JSON") || strings.Contains(e.Error(), "error unmarshaling JSON")
}

func IsMalformedYAMLError(e error) bool {
	return strings.Contains(e.Error(), "MalformedYAMLError")
}
