/*
Copyright 2026 The Kubernetes Authors.

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

package authorizer

import (
	"errors"
	"testing"
)

func TestEvaluateConditionsMap(t *testing.T) {
	evalErr := errors.New("eval error")

	tests := []struct {
		name            string
		conditionsMap   ConditionsMap
		supportedTarget ConditionsTarget
		supportedType   ConditionType
		eval            func(string) (bool, error)
		wantDecision    string
		wantSuccessful  bool
	}{

		// wrong condition target
		{
			name: "wrong target with only allow conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: "wrong-target",
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `NoOpinion("failed closed", "unsupported condition target: \"wrong-target\", expected: \"AdmissionControl\"")`,
			wantSuccessful:  false,
		},
		{
			name: "wrong target with only noopinion conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: "wrong-target",
				conditionType:   "test",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `NoOpinion("failed closed", "unsupported condition target: \"wrong-target\", expected: \"AdmissionControl\"")`,
			wantSuccessful:  false,
		},
		{
			name: "wrong target with deny conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: "wrong-target",
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `Deny("failed closed", "unsupported condition target: \"wrong-target\", expected: \"AdmissionControl\"")`,
			wantSuccessful:  false,
		},

		// wrong condition type
		{
			name: "wrong type with only allow conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "wrong-type",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `NoOpinion("failed closed", "unsupported condition type: \"wrong-type\", expected: \"test\"")`,
			wantSuccessful:  false,
		},
		{
			name: "wrong type with only noopinion conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "wrong-type",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `NoOpinion("failed closed", "unsupported condition type: \"wrong-type\", expected: \"test\"")`,
			wantSuccessful:  false,
		},
		{
			name: "wrong type with deny conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "wrong-type",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `Deny("failed closed", "unsupported condition type: \"wrong-type\", expected: \"test\"")`,
			wantSuccessful:  false,
		},

		// Deny conditions
		{
			name: "deny condition matches, one match is enough",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-yes": {Condition: "match", Effect: ConditionEffectDeny, Description: "access denied"},
					"deny-no":  {Condition: "no-match", Effect: ConditionEffectDeny},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				return cond == "match", nil
			},
			wantDecision:   `Deny("[condition \"deny-yes\" denied the request with description \"access denied\"]", <nil>)`,
			wantSuccessful: true,
		},
		{
			name: "deny condition does not match",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond": {Condition: "x", Effect: ConditionEffectDeny},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return false, nil },
			wantDecision:    `NoOpinion("no conditions matched", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "deny condition eval error; deny error trumps matching allow",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "err", Effect: ConditionEffectDeny},
					"allow-cond": {Condition: "ok", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				if cond == "err" {
					return false, evalErr
				}
				return true, nil
			},
			wantDecision:   `Deny("one or more conditional evaluation errors occurred", "Deny condition \"deny-cond\" produced error: eval error")`,
			wantSuccessful: false,
		},

		// NoOpinion conditions
		{
			name: "noopinion condition matches",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion, Description: "not relevant"},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return true, nil },
			wantDecision:    `NoOpinion("[condition \"nop-cond\" evaluated to NoOpinion with description \"not relevant\"]", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "noopinion condition does not match",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return false, nil },
			wantDecision:    `NoOpinion("no conditions matched", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "noopinion condition eval error, noopinion error trumps matching allow",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"nop-cond":   {Condition: "err", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "ok", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				if cond == "err" {
					return false, evalErr
				}
				return true, nil
			},
			wantDecision:   `NoOpinion("one or more conditional evaluation errors occurred", "NoOpinion condition \"nop-cond\" produced error: eval error")`,
			wantSuccessful: false,
		},
		{
			name: "first noopinion no match second noopinion matches",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"nop-no":  {Condition: "no-match", Effect: ConditionEffectNoOpinion},
					"nop-yes": {Condition: "match", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				return cond == "match", nil
			},
			wantDecision:   `NoOpinion("[condition \"nop-yes\" evaluated to NoOpinion]", <nil>)`,
			wantSuccessful: true,
		},

		// Allow conditions
		{
			name: "allow condition matches",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow, Description: "access granted"},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return true, nil },
			wantDecision:    `Allow("[condition \"allow-cond\" allowed the request with description \"access granted\"]", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "allow condition does not match",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return false, nil },
			wantDecision:    `NoOpinion("no conditions matched", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "allow conditions both eval error",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-err1": {Condition: "x", Effect: ConditionEffectAllow},
					"allow-err2": {Condition: "y", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return false, evalErr },
			wantDecision:    `NoOpinion("one or more conditional evaluation errors occurred", "[Allow condition \"allow-err1\" produced error: eval error, Allow condition \"allow-err2\" produced error: eval error]")`,
			wantSuccessful:  false,
		},
		{
			name: "allow one errors, one matches",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-err": {Condition: "error-cond", Effect: ConditionEffectAllow},
					"allow-ok":  {Condition: "match-cond", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				if cond == "error-cond" {
					return false, evalErr
				}
				return true, nil
			},
			wantDecision:   `Allow("[condition \"allow-ok\" allowed the request]", "Allow condition \"allow-err\" produced error: eval error")`,
			wantSuccessful: true,
		},

		// Precedence: Deny > NoOpinion > Allow
		{
			name: "deny takes precedence over allow",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return true, nil },
			wantDecision:    `Deny("[condition \"deny-cond\" denied the request]", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "deny takes precedence over noopinion and allow",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
					"nop-cond":   {Condition: "x", Effect: ConditionEffectNoOpinion},
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return true, nil },
			wantDecision:    `Deny("[condition \"deny-cond\" denied the request]", <nil>)`,
			wantSuccessful:  true,
		},
		{
			name: "noopinion takes precedence over allow",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
					"nop-cond":   {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return true, nil },
			wantDecision:    `NoOpinion("[condition \"nop-cond\" evaluated to NoOpinion]", <nil>)`,
			wantSuccessful:  true,
		},

		// Selective matching across effect types
		{
			name: "deny no match, noopinion matches, allow matches",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "deny-check", Effect: ConditionEffectDeny},
					"nop-cond":   {Condition: "nop-check", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "allow-check", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				return cond != "deny-check", nil
			},
			wantDecision:   `NoOpinion("[condition \"nop-cond\" evaluated to NoOpinion]", <nil>)`,
			wantSuccessful: true,
		},
		{
			name: "only allow matches",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "deny-check", Effect: ConditionEffectDeny},
					"nop-cond":   {Condition: "nop-check", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "allow-check", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				return cond == "allow-check", nil
			},
			wantDecision:   `Allow("[condition \"allow-cond\" allowed the request]", <nil>)`,
			wantSuccessful: true,
		},
		{
			name: "no conditions match across all effects",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
					"nop-cond":   {Condition: "x", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { return false, nil },
			wantDecision:    `NoOpinion("no conditions matched", <nil>)`,
			wantSuccessful:  true,
		},

		// Empty condition set (non-nil but no conditions)
		{
			name: "empty condition set with no conditions",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions:      map[string]Condition{},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval:            func(string) (bool, error) { panic("should not be called") },
			wantDecision:    `NoOpinion("no conditions matched", <nil>)`,
			wantSuccessful:  true,
		},

		// Deny no match, noopinion errors: fail closed to NoOpinion
		{
			name: "deny no match noopinion errors",
			conditionsMap: ConditionsMap{
				conditionTarget: ConditionsTargetAdmissionControl,
				conditionType:   "test",
				conditions: map[string]Condition{
					"deny-cond": {Condition: "deny-check", Effect: ConditionEffectDeny},
					"nop-cond":  {Condition: "nop-check", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedTarget: ConditionsTargetAdmissionControl,
			supportedType:   "test",
			eval: func(cond string) (bool, error) {
				if cond == "nop-check" {
					return false, evalErr
				}
				return false, nil
			},
			wantDecision: `NoOpinion("one or more conditional evaluation errors occurred", "NoOpinion condition \"nop-cond\" produced error: eval error")`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision, successful := EvaluateConditionsMap(tt.conditionsMap, tt.supportedTarget, tt.supportedType, tt.eval)

			if decision.String() != tt.wantDecision {
				t.Errorf("got decision %s, want %s", decision.String(), tt.wantDecision)
			}
			if successful != tt.wantSuccessful {
				t.Errorf("got successful %t, want %t", successful, tt.wantSuccessful)
			}
		})
	}
}
