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

// Package limits contains Kubernetes-specific size and length limits
// used for validation throughout the system.
package limits

// Kubernetes-specific size limits for labels, annotations, and fields.
const (
	// LabelValueMaxLength is the maximum length of a Kubernetes label value.
	// Label values must be 63 characters or less (can be empty), unless empty,
	// must begin and end with an alphanumeric character ([a-z0-9A-Z]), and
	// could contain dashes (-), underscores (_), dots (.), and alphanumerics between.
	LabelValueMaxLength int = 63

	// LabelKeyMaxLength is the maximum length of a Kubernetes label key.
	// The key segment is required and must be 63 characters or less, beginning
	// and ending with an alphanumeric character with dashes, underscores, dots,
	// and alphanumerics between.
	LabelKeyMaxLength int = 63

	// FieldManagerMaxLength is the maximum length of a field manager name
	// in server-side apply operations.
	FieldManagerMaxLength int = 128
)
