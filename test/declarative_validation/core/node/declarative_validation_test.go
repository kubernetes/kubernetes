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

package node

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	core "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/node"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateNodeUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateNodeUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "nodes",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkValidNode()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "nodes",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	updateObj := mkValidNode()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateNodeUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "api",
		APIGroup:          "",
		APIVersion:        apiVersion,
		Resource:          "nodes",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		old          core.Node
		update       core.Node
		expectedErrs field.ErrorList
	}{
		"no change": {
			old:    mkValidNode(),
			update: mkValidNode(),
		},
		"no change with providerID": {
			old: mkValidNode(func(n *core.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
			update: mkValidNode(func(n *core.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
		},
		"set providerID from empty": {
			old: mkValidNode(),
			update: mkValidNode(func(n *core.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
		},
		"modify providerID": {
			old: mkValidNode(func(n *core.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
			update: mkValidNode(func(n *core.Node) {
				n.Spec.ProviderID = "provider:///node-2"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "providerID"), nil, "field cannot be modified once set").WithOrigin("update").MarkAlpha(),
			},
		},
		"clear providerID": {
			old: mkValidNode(func(n *core.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
			update: mkValidNode(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "providerID"), nil, "field cannot be cleared once set").WithOrigin("update").MarkAlpha(),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkValidNode(tweaks ...func(n *core.Node)) core.Node {
	node := core.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
	}
	for _, tweak := range tweaks {
		tweak(&node)
	}
	return node
}
