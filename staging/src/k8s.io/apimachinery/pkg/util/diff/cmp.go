//go:build usegocmp
// +build usegocmp

/*
Copyright 2025 The Kubernetes Authors.

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

package diff

import (
	"github.com/google/go-cmp/cmp" //nolint:depguard
)

// Diff returns a string representation of the difference between two objects.
// When built with the usegocmp tag, it uses go-cmp/cmp to generate a diff
// between the objects.
func Diff(a, b any) string {
	return cmp.Diff(a, b)
}
