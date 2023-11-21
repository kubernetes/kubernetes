//go:build go1.21
// +build go1.21

/*
Copyright 2023 The logr Authors.

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

package logs

import (
	"log/slog"
)

// ErrorWithDetails wraps an error and adds additional information about it
// which is only used when logging the error. Text and JSON log output in
// Kubernetes supports this. Other logging backends treat it like a normal
// error.
//
// The details can be a simple value
//
//	slog.IntValue(42))
//
// or a group of values
//
//	slog.GroupValue(slog.Int("answer", 42), slog.StringValue("thank", "fish")))
func ErrorWithDetails(err error, details slog.Value) error {
	return structuredError{error: err, Value: details}
}

type structuredError struct {
	error
	slog.Value
}

var _ error = structuredError{}
var _ slog.LogValuer = structuredError{}

func (err structuredError) LogValue() slog.Value {
	return err.Value
}

func (err structuredError) Unwrap() error {
	return err.error
}
