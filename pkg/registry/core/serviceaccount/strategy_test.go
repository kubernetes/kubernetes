/*
Copyright 2024 The Kubernetes Authors.

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

package serviceaccount

import (
	"context"
	"fmt"
	"testing"

	api "k8s.io/kubernetes/pkg/apis/core"
	sa "k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
)

func TestWarningsOnCreate(t *testing.T) {
	ctx := context.TODO()
	serviceAccount := &api.ServiceAccount{}

	warnings := Strategy.WarningsOnCreate(ctx, serviceAccount)
	if len(warnings) != 0 {
		t.Errorf("expected no warnings, got %v", warnings)
	}
	serviceAccount.Annotations = map[string]string{sa.EnforceMountableSecretsAnnotation: "true"}
	warnings = Strategy.WarningsOnCreate(ctx, serviceAccount)
	if len(warnings) != 1 || warnings[0] != fmt.Sprintf("metadata.annotations[%s]: deprecated in v1.32+; prefer separate namespaces to isolate access to mounted secrets", sa.EnforceMountableSecretsAnnotation) {
		t.Errorf("expected warnings, got %v", warnings)
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	ctx := context.TODO()
	serviceAccount := &api.ServiceAccount{}
	oldServiceAccount := &api.ServiceAccount{}

	warnings := Strategy.WarningsOnUpdate(ctx, serviceAccount, oldServiceAccount)
	if len(warnings) != 0 {
		t.Errorf("expected no warnings, got %v", warnings)
	}
	serviceAccount.Annotations = map[string]string{sa.EnforceMountableSecretsAnnotation: "true"}
	warnings = Strategy.WarningsOnUpdate(ctx, serviceAccount, oldServiceAccount)
	if len(warnings) != 1 || warnings[0] != fmt.Sprintf("metadata.annotations[%s]: deprecated in v1.32+; prefer separate namespaces to isolate access to mounted secrets", sa.EnforceMountableSecretsAnnotation) {
		t.Errorf("expected warnings, got %v", warnings)
	}

	oldServiceAccount.Annotations = map[string]string{sa.EnforceMountableSecretsAnnotation: "true"}
	warnings = Strategy.WarningsOnUpdate(ctx, serviceAccount, oldServiceAccount)
	if len(warnings) != 0 {
		t.Errorf("expected no warnings if request isn't newly setting the annotation, got %v", warnings)
	}
}
