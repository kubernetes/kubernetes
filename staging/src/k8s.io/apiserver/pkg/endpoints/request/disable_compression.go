/*
Copyright 2022 The Kubernetes Authors.

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

package request

import (
	"context"
)

type disableCompressionIDKeyType int

const disableCompressionIDKey disableCompressionIDKeyType = iota

// WithCompressionDisabled stores bool in context.
func WithCompressionDisabled(parent context.Context, disableCompression bool) context.Context {
	return WithValue(parent, disableCompressionIDKey, disableCompression)
}

// CompressionDisabledFrom retrieves bool from context.
// Defaults to false if not set.
func CompressionDisabledFrom(ctx context.Context) bool {
	decision, _ := ctx.Value(disableCompressionIDKey).(bool)
	return decision
}
