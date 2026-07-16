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

package runtimeclass

import (
	"fmt"
	"strings"
	"testing"

	nodev1 "k8s.io/api/node/v1"
	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/test/coverage"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/node/runtimeclass"
)

func TestRuntimeClassCheckpointPolicyValidation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	keys := make([]string, 65)
	for i := range keys {
		keys[i] = fmt.Sprintf("option-%d", i)
	}
	for _, mode := range []struct {
		name      string
		beta, all bool
	}{
		{name: "beta enabled", beta: true},
		{name: "beta disabled"},
		{name: "all rules enforced", beta: true, all: true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiserverfeatures.DeclarativeValidationBeta, mode.beta)
			for _, operation := range []string{"create", "update"} {
				t.Run(operation, func(t *testing.T) {
					ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
						APIGroup: "node.k8s.io", APIVersion: "v1", IsResourceRequest: true, Verb: operation,
					})
					if mode.all {
						ctx = validate.WithAllDeclarativeEnforcedForTest(ctx)
					}
					for _, fieldName := range []string{"allowedCheckpointOptions", "allowedRestoreOptions"} {
						t.Run(fieldName, func(t *testing.T) {
							path := field.NewPath("podCheckpoint", fieldName)
							for _, tc := range []struct {
								name string
								keys []string
								errs field.ErrorList
							}{
								{name: "omitted"},
								{name: "empty", keys: []string{}},
								{name: "valid", keys: []string{"compression", "tcp-close"}},
								{name: "64 keys", keys: keys[:64]},
								{name: "256 bytes", keys: []string{strings.Repeat("é", 128)}},
								{name: "too many", keys: keys, errs: field.ErrorList{field.TooMany(path, 65, 64).WithOrigin("maxItems")}},
								{name: "empty key", keys: []string{""}, errs: field.ErrorList{field.TooShort(path.Index(0), "", 1).WithOrigin("minLength")}},
								{name: "257 bytes", keys: []string{strings.Repeat("x", 257)}, errs: field.ErrorList{field.TooLong(path.Index(0), "", 256).WithOrigin("maxBytes")}},
								{name: "258 multibyte bytes", keys: []string{strings.Repeat("é", 129)}, errs: field.ErrorList{field.TooLong(path.Index(0), "", 256).WithOrigin("maxBytes")}},
								{name: "duplicate", keys: []string{"compression", "compression"}, errs: field.ErrorList{field.Duplicate(path.Index(1), "compression")}},
							} {
								t.Run(tc.name, func(t *testing.T) {
									obj := mkRuntimeClassHandlerOnly()
									obj.PodCheckpoint = &node.RuntimeClassPodCheckpoint{}
									if fieldName == "allowedCheckpointOptions" {
										obj.PodCheckpoint.AllowedCheckpointOptions = tc.keys
									} else {
										obj.PodCheckpoint.AllowedRestoreOptions = tc.keys
									}
									var errs field.ErrorList
									if operation == "create" {
										errs = rest.ValidateCreate(ctx, &obj, registry.Strategy)
									} else {
										old := mkRuntimeClassHandlerOnly()
										old.ResourceVersion, obj.ResourceVersion = "1", "1"
										errs = rest.ValidateUpdate(ctx, &obj, &old, registry.Strategy)
									}
									field.ErrorMatcher{}.ByType().ByField().ByOrigin().Test(t, tc.errs, errs)
									coverage.RecordObservedRules(nodev1.SchemeGroupVersion.WithKind("RuntimeClass"), errs)
								})
							}
						})
					}
				})
			}
		})
	}
}
