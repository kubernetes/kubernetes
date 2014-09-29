/*
Copyright 2014 Google Inc. All rights reserved.

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

package api_test

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// TestNamespaceContext validates that a namespace can be get/set on a context object
func TestNamespaceContext(t *testing.T) {
	ctx := api.NewDefaultContext()
	result, ok := api.NamespaceFrom(ctx)
	if !ok {
		t.Errorf("Error getting namespace")
	}
	if api.NamespaceDefault != result {
		t.Errorf("Expected: %v, Actual: %v", api.NamespaceDefault, result)
	}
}

func TestValidNamespaceOnCreateOrUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	namespace, _ := api.NamespaceFrom(ctx)
	resource := api.ReplicationController{}
	if !api.ValidNamespaceOnCreateOrUpdate(ctx, &resource.JSONBase) {
		t.Errorf("expected success")
	}
	if namespace != resource.Namespace {
		t.Errorf("expected resource to have the default namespace assigned during validation")
	}
	resource = api.ReplicationController{JSONBase: api.JSONBase{Namespace: "other"}}
	if api.ValidNamespaceOnCreateOrUpdate(ctx, &resource.JSONBase) {
		t.Errorf("Expected error that resource and context errors do not match")
	}
}
