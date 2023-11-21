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

package logs_test

import (
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/pkg/util/logs"
)

func TestErrorWithDetails(t *testing.T) {
	baseErr := errors.New("hello")
	details := slog.AnyValue(42)
	wrappedErr := logs.ErrorWithDetails(baseErr, details)

	require.ErrorIs(t, wrappedErr, baseErr)
	require.Implements(t, (*slog.LogValuer)(nil), wrappedErr)
	require.Equal(t, details, wrappedErr.(slog.LogValuer).LogValue())
}
