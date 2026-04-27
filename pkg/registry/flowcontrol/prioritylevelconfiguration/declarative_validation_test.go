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

package prioritylevelconfiguration

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

var apiVersions = []string{"v1", "v1beta1", "v1beta2", "v1beta3"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "flowcontrol.apiserver.k8s.io",
		APIVersion: apiVersion,
	})

	specPath := field.NewPath("spec")

	testCases := map[string]struct {
		input        flowcontrol.PriorityLevelConfiguration
		expectedErrs field.ErrorList
	}{
		"valid: Limited allows Reject without queuing": {
			input: mkPLC(),
		},
		"valid: Queue requires queuing": {
			input: mkPLC(tweakLimitResponseType(flowcontrol.LimitResponseTypeQueue)),
		},
		"valid: Exempt without exempt config": {
			input: mkPLC(tweakExempt()),
		},
		"valid: Exempt with exempt config": {
			input: mkPLC(tweakExempt(), tweakExemptConfig(&flowcontrol.ExemptPriorityLevelConfiguration{
				NominalConcurrencyShares: ptrInt32(100),
			})),
		},
		"spec.type: Limited with limited=nil": {
			input: mkPLC(tweakLimited(nil)),
			expectedErrs: field.ErrorList{
				field.Required(specPath.Child("limited"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"spec.type: Exempt with limited set": {
			input: mkPLC(tweakExempt(), tweakLimited(&flowcontrol.LimitedPriorityLevelConfiguration{
				NominalConcurrencyShares: 100,
				LimitResponse: flowcontrol.LimitResponse{
					Type: flowcontrol.LimitResponseTypeReject,
				},
			})),
			expectedErrs: field.ErrorList{
				// Mandatory object check: spec of 'exempt' differs from bootstrap (HW-only)
				field.Invalid(specPath, nil, "").MarkFromImperative(),
				field.Forbidden(specPath.Child("limited"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"spec.type: Limited with exempt set": {
			input: mkPLC(tweakExemptConfig(&flowcontrol.ExemptPriorityLevelConfiguration{})),
			expectedErrs: field.ErrorList{
				field.Forbidden(specPath.Child("exempt"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"limitResponse.type: Queue with queuing=nil": {
			input: mkPLC(tweakLimitResponseType(flowcontrol.LimitResponseTypeQueue), tweakQueuing(nil)),
			expectedErrs: field.ErrorList{
				field.Required(specPath.Child("limited", "limitResponse", "queuing"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"limitResponse.type: Reject with queuing set": {
			input: mkPLC(tweakQueuing(&flowcontrol.QueuingConfiguration{
				Queues:           64,
				HandSize:         8,
				QueueLengthLimit: 50,
			})),
			expectedErrs: field.ErrorList{
				field.Forbidden(specPath.Child("limited", "limitResponse", "queuing"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
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

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	specPath := field.NewPath("spec")

	testCases := map[string]struct {
		old          flowcontrol.PriorityLevelConfiguration
		update       flowcontrol.PriorityLevelConfiguration
		expectedErrs field.ErrorList
	}{
		"valid update (no changes)": {
			old:    mkPLC(),
			update: mkPLC(),
		},
		"valid update: Reject to Queue": {
			old:    mkPLC(),
			update: mkPLC(tweakLimitResponseType(flowcontrol.LimitResponseTypeQueue)),
		},
		"update: limited set to nil": {
			old:    mkPLC(),
			update: mkPLC(tweakLimited(nil)),
			expectedErrs: field.ErrorList{
				field.Required(specPath.Child("limited"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"update: add exempt field to Limited PLC": {
			old:    mkPLC(),
			update: mkPLC(tweakExemptConfig(&flowcontrol.ExemptPriorityLevelConfiguration{})),
			expectedErrs: field.ErrorList{
				field.Forbidden(specPath.Child("exempt"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"update: Queue with queuing set to nil": {
			old:    mkPLC(tweakLimitResponseType(flowcontrol.LimitResponseTypeQueue)),
			update: mkPLC(tweakLimitResponseType(flowcontrol.LimitResponseTypeQueue), tweakQueuing(nil)),
			expectedErrs: field.ErrorList{
				field.Required(specPath.Child("limited", "limitResponse", "queuing"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"update: Reject with queuing added": {
			old: mkPLC(),
			update: mkPLC(tweakQueuing(&flowcontrol.QueuingConfiguration{
				Queues:           64,
				HandSize:         8,
				QueueLengthLimit: 50,
			})),
			expectedErrs: field.ErrorList{
				field.Forbidden(specPath.Child("limited", "limitResponse", "queuing"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "flowcontrol.apiserver.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "prioritylevelconfigurations",
				Name:              tc.old.Name,
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkPLC creates a valid Limited PriorityLevelConfiguration with Reject limit response.
func mkPLC(tweaks ...func(*flowcontrol.PriorityLevelConfiguration)) flowcontrol.PriorityLevelConfiguration {
	obj := flowcontrol.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-limited",
		},
		Spec: flowcontrol.PriorityLevelConfigurationSpec{
			Type: flowcontrol.PriorityLevelEnablementLimited,
			Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
				NominalConcurrencyShares: 100,
				LimitResponse: flowcontrol.LimitResponse{
					Type: flowcontrol.LimitResponseTypeReject,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

// tweakExempt switches the PLC to Exempt type with the mandatory "exempt" name.
func tweakExempt() func(*flowcontrol.PriorityLevelConfiguration) {
	return func(obj *flowcontrol.PriorityLevelConfiguration) {
		obj.Name = "exempt"
		obj.Spec.Type = flowcontrol.PriorityLevelEnablementExempt
		obj.Spec.Limited = nil
	}
}

// tweakLimited sets the Limited field directly.
func tweakLimited(limited *flowcontrol.LimitedPriorityLevelConfiguration) func(*flowcontrol.PriorityLevelConfiguration) {
	return func(obj *flowcontrol.PriorityLevelConfiguration) {
		obj.Spec.Limited = limited
	}
}

// tweakExemptConfig sets the Exempt configuration field.
func tweakExemptConfig(exempt *flowcontrol.ExemptPriorityLevelConfiguration) func(*flowcontrol.PriorityLevelConfiguration) {
	return func(obj *flowcontrol.PriorityLevelConfiguration) {
		obj.Spec.Exempt = exempt
	}
}

// tweakLimitResponseType sets the LimitResponse type and auto-populates Queuing if Queue.
func tweakLimitResponseType(lrt flowcontrol.LimitResponseType) func(*flowcontrol.PriorityLevelConfiguration) {
	return func(obj *flowcontrol.PriorityLevelConfiguration) {
		if obj.Spec.Limited == nil {
			return
		}
		obj.Spec.Limited.LimitResponse.Type = lrt
		if lrt == flowcontrol.LimitResponseTypeQueue {
			obj.Spec.Limited.LimitResponse.Queuing = &flowcontrol.QueuingConfiguration{
				Queues:           64,
				HandSize:         8,
				QueueLengthLimit: 50,
			}
		} else {
			obj.Spec.Limited.LimitResponse.Queuing = nil
		}
	}
}

// tweakQueuing sets the Queuing field directly, overriding auto-population.
func tweakQueuing(queuing *flowcontrol.QueuingConfiguration) func(*flowcontrol.PriorityLevelConfiguration) {
	return func(obj *flowcontrol.PriorityLevelConfiguration) {
		if obj.Spec.Limited == nil {
			return
		}
		obj.Spec.Limited.LimitResponse.Queuing = queuing
	}
}

func ptrInt32(v int32) *int32 {
	return &v
}
