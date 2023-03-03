/*
Copyright 2022 The Kubernetes Authors.

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

package validatingadmissionpolicy

import (
	"fmt"
	"k8s.io/klog/v2"
	"strings"

	celtypes "github.com/google/cel-go/common/types"

	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

// validator implements the Validator interface
type validator struct {
	filter     cel.Filter
	failPolicy *v1.FailurePolicyType
}

func NewValidator(filter cel.Filter, failPolicy *v1.FailurePolicyType) Validator {
	return &validator{
		filter:     filter,
		failPolicy: failPolicy,
	}
}

func policyDecisionActionForError(f v1.FailurePolicyType) PolicyDecisionAction {
	if f == v1.Ignore {
		return ActionAdmit
	}
	return ActionDeny
}

// Validate takes a list of Evaluation and a failure policy and converts them into actionable PolicyDecisions
func (v *validator) Validate(versionedAttr *generic.VersionedAttributes, versionedParams runtime.Object) []PolicyDecision {
	var f v1.FailurePolicyType
	if v.failPolicy == nil {
		f = v1.Fail
	} else {
		f = *v.failPolicy
	}

	evalResults, err := v.filter.ForInput(versionedAttr, versionedParams, cel.CreateAdmissionRequest(versionedAttr.Attributes))
	if err != nil {
		return []PolicyDecision{
			{
				Action:     policyDecisionActionForError(f),
				Evaluation: EvalError,
				Message:    err.Error(),
			},
		}
	}
	decisions := make([]PolicyDecision, len(evalResults))

	for i, evalResult := range evalResults {
		var decision = &decisions[i]
		// TODO: move this to generics
		validation, ok := evalResult.ExpressionAccessor.(*ValidationCondition)
		if !ok {
			klog.Error("Invalid type conversion to ValidationCondition")
			decision.Action = policyDecisionActionForError(f)
			decision.Evaluation = EvalError
			decision.Message = "Invalid type sent to validator, expected ValidationCondition"
			continue
		}

		if evalResult.Error != nil {
			decision.Action = policyDecisionActionForError(f)
			decision.Evaluation = EvalError
			decision.Message = evalResult.Error.Error()
		} else if evalResult.EvalResult != celtypes.True {
			decision.Action = ActionDeny
			if validation.Reason == nil {
				decision.Reason = metav1.StatusReasonInvalid
			} else {
				decision.Reason = *validation.Reason
			}
			if len(validation.Message) > 0 {
				decision.Message = strings.TrimSpace(validation.Message)
			} else {
				decision.Message = fmt.Sprintf("failed expression: %v", strings.TrimSpace(validation.Expression))
			}

		} else {
			decision.Action = ActionAdmit
			decision.Evaluation = EvalAdmit
		}
	}
	return decisions
}
