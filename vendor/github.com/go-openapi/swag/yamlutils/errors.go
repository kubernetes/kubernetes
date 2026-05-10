// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package yamlutils

type yamlError string

const (
	// ErrYAML is an error raised by YAML utilities
	ErrYAML yamlError = "yaml error"
)

func (e yamlError) Error() string {
	return string(e)
}
