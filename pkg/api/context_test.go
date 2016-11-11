/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
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

	ctx = api.NewContext()
	result, ok = api.NamespaceFrom(ctx)
	if ok {
		t.Errorf("Should not be ok because there is no namespace on the context")
	}
}

// TestValidNamespace validates that namespace rules are enforced on a resource prior to create or update
func TestValidNamespace(t *testing.T) {
	ctx := api.NewDefaultContext()
	namespace, _ := api.NamespaceFrom(ctx)
	resource := api.ReplicationController{}
	if !api.ValidNamespace(ctx, &resource.ObjectMeta) {
		t.Errorf("expected success")
	}
	if namespace != resource.Namespace {
		t.Errorf("expected resource to have the default namespace assigned during validation")
	}
	resource = api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: "other"}}
	if api.ValidNamespace(ctx, &resource.ObjectMeta) {
		t.Errorf("Expected error that resource and context errors do not match because resource has different namespace")
	}
	ctx = api.NewContext()
	if api.ValidNamespace(ctx, &resource.ObjectMeta) {
		t.Errorf("Expected error that resource and context errors do not match since context has no namespace")
	}

	ctx = api.NewContext()
	ns := api.NamespaceValue(ctx)
	if ns != "" {
		t.Errorf("Expected the empty string")
	}
}

//TestUIDContext validates that a UID can be get/set on a context object
func TestUIDContext(t *testing.T) {
	ctx := api.NewContext()
	_, ok := api.UIDFrom(ctx)
	if ok {
		t.Errorf("Should not be ok because there is no UID on the context")
	}
	ctx = api.WithUID(
		ctx,
		types.UID{"testUID"},
	)
	_, ok = api.UIDFrom(ctx)
	if !ok {
		t.Errorf("Error getting UID")
	}
}

//TestUserAgentContext validates that a useragent can be get/set on a context object
func TestUserAgentContext(t *testing.T) {
	ctx := api.NewContext()
	_, ok := api.UserAgentFrom(ctx)
	if ok {
		t.Errorf("Should not be ok because there is no UserAgent on the context")
	}

	ctx = api.WithUserAgent(
		ctx,
		"TestUserAgent",
	)
	result, ok := api.UserAgentFrom(ctx)
	if !ok {
		t.Errorf("Error getting UserAgent")
	}
	if result != "TestUserAgent" {
		t.Errorf("Expected: TestUserAgent, Actual: %v", result)
	}
}
