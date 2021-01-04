/*
Copyright 2021 The Kubernetes Authors.

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

package injection

import (
	"context"
	"testing"

	"k8s.io/client-go/rest"
)

func TestContextNamespace(t *testing.T) {
	ctx := context.Background()

	if HasNamespaceScope(ctx) {
		t.Error("HasNamespaceScope() = true, wanted false")
	}

	want := "this-is-the-best-ns-evar"
	ctx = WithNamespaceScope(ctx, want)

	if !HasNamespaceScope(ctx) {
		t.Error("HasNamespaceScope() = false, wanted true")
	}

	if got := GetNamespaceScope(ctx); got != want {
		t.Errorf("GetNamespaceScope() = %v, wanted %v", got, want)
	}
}

func TestContextConfig(t *testing.T) {
	ctx := context.Background()

	if cfg := GetConfig(ctx); cfg != nil {
		t.Errorf("GetConfig() = %v, wanted nil", cfg)
	}

	want := &rest.Config{}
	ctx = WithConfig(ctx, want)

	if cfg := GetConfig(ctx); cfg != want {
		t.Errorf("GetConfig() = %v, wanted %v", cfg, want)
	}
}
