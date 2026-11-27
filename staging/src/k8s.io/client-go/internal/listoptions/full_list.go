/*
Copyright The Kubernetes Authors.

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

// Package listoptions preserves the pager's full-list fallback intent across
// list-options callbacks without changing ordinary zero-limit requests.
package listoptions

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type fullListKey struct{}

// WithFullList marks list requests made with the returned context as unlimited.
func WithFullList(ctx context.Context) context.Context {
	return context.WithValue(ctx, fullListKey{}, true)
}

// PreserveFullList restores the unlimited request after options callbacks have
// run. Other list options, including selectors, are left unchanged.
func PreserveFullList(ctx context.Context, options *metav1.ListOptions) {
	if fullList, _ := ctx.Value(fullListKey{}).(bool); fullList {
		options.Limit = 0
	}
}
