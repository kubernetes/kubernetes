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

package namespace

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestNamespaceStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Errorf("Namespaces should not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Namespaces should not allow create on update")
	}
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.NamespaceStatus{Phase: api.NamespaceTerminating},
	}
	Strategy.ResetBeforeCreate(namespace)
	if namespace.Status.Phase != api.NamespaceActive {
		t.Errorf("Namespaces do not allow setting phase on create")
	}
}
