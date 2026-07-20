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

package v1

import (
	"fmt"
	"iter"

	authorizationv1 "k8s.io/api/authorization/v1"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// SerializeConditionsAwareDecision serializes a authorizer.ConditionsAwareDecision into a v1.ConditionsAwareDecision
func SerializeConditionsAwareDecision(decision authorizer.ConditionsAwareDecision) authorizationv1.ConditionsAwareDecision {
	var errString string
	if decision.Error() != nil {
		errString = decision.Error().Error()
	}
	switch {
	case decision.IsDeny():
		return authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
			Deny: &authorizationv1.UnconditionalDecision{
				Reason:          decision.Reason(),
				EvaluationError: errString,
			},
		}
	case decision.IsNoOpinion():
		return authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
			NoOpinion: &authorizationv1.UnconditionalDecision{
				Reason:          decision.Reason(),
				EvaluationError: errString,
			},
		}
	case decision.IsAllow():
		return authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeAllow,
			Allow: &authorizationv1.UnconditionalDecision{
				Reason:          decision.Reason(),
				EvaluationError: errString,
			},
		}
	case decision.IsConditionsMap():
		return authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				DenyConditions:      collectConditions(decision.ConditionsMap().DenyConditions()),
				NoOpinionConditions: collectConditions(decision.ConditionsMap().NoOpinionConditions()),
				AllowConditions:     collectConditions(decision.ConditionsMap().AllowConditions()),
			},
		}
	case decision.IsUnion():
		subDecisions := []authorizationv1.NamedConditionsAwareDecision{}
		for authorizerName, subDecision := range decision.UnionedDecisions() {
			subDecisions = append(subDecisions, authorizationv1.NamedConditionsAwareDecision{
				AuthorizerName: authorizerName,
				Decision:       SerializeConditionsAwareDecision(subDecision),
			})
		}
		return authorizationv1.ConditionsAwareDecision{
			Type:  authorizationv1.ConditionsAwareDecisionTypeUnion,
			Union: subDecisions,
		}
	default: // unrecognized mode
		return authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
			Deny: &authorizationv1.UnconditionalDecision{
				Reason: decision.Reason(),
				EvaluationError: utilerrors.NewAggregate([]error{
					decision.Error(), // dropped by NewAggregate if nil
					fmt.Errorf("unrecognized ConditionsAwareDecision: %s", decision.String()),
				}).Error(), // NewAggregate is always non-nil, as there is at least one error in the list always
			},
		}
	}
}

func collectConditions(condIter iter.Seq[authorizer.Condition]) []authorizationv1.Condition {
	conds := []authorizationv1.Condition{}
	for condition := range condIter {
		conds = append(conds, authorizationv1.Condition{
			ID:          condition.GetID(),
			Condition:   condition.GetCondition(),
			Type:        condition.GetType(),
			Description: condition.GetDescription(),
		})
	}
	return conds
}
