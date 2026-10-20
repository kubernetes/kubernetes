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

package pod

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// RunDeclarativeValidateDefaultNetworkTestCases exercises the declarative enum
// validation of spec.defaultNetwork for any object that embeds a PodSpec.
// setDefaultNetwork must set the field on baseObj; hostNetwork is set alongside
// "Host" so the hand-written cross-field validation stays quiet.
func RunDeclarativeValidateDefaultNetworkTestCases[T runtime.Object](t *testing.T, ctx context.Context, strategy rest.RESTCreateStrategy, specPath *field.Path, baseObj T, setDefaultNetwork func(baseObj T, defaultNetwork *api.PodDefaultNetwork, hostNetwork bool)) {
	testCases := map[string]struct {
		defaultNetwork *api.PodDefaultNetwork
		hostNetwork    bool
		expectedErrs   field.ErrorList
	}{
		"unset": {},
		"Pod": {
			defaultNetwork: new(api.PodDefaultNetworkPod),
		},
		"Host": {
			defaultNetwork: new(api.PodDefaultNetworkHost),
			hostNetwork:    true,
		},
		"None": {
			defaultNetwork: new(api.PodDefaultNetworkNone),
		},
		"unsupported value": {
			defaultNetwork: new(api.PodDefaultNetwork("Bridge")),
			expectedErrs: field.ErrorList{
				field.NotSupported(specPath.Child("defaultNetwork"), api.PodDefaultNetwork("Bridge"), []api.PodDefaultNetwork{}),
			},
		},
	}
	for k, tc := range testCases {
		t.Run("defaultNetwork/"+k, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			setDefaultNetwork(obj, tc.defaultNetwork, tc.hostNetwork)
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.expectedErrs)
		})
	}
}
