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
	"errors"
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

// DeserializeEvaluationError extracts the error from an authorizationv1.UnconditionalDecision that could be nil.
func DeserializeEvaluationError(ud *authorizationv1.UnconditionalDecision) error {
	if ud == nil || len(ud.EvaluationError) == 0 {
		return nil
	}
	return errors.New(ud.EvaluationError)
}

// DeserializeReason extracts the reason from an authorizationv1.UnconditionalDecision that could be nil.
func DeserializeReason(ud *authorizationv1.UnconditionalDecision) string {
	if ud == nil {
		return ""
	}
	return ud.Reason
}

// DeserializeConditionsAwareDecision deserializes an authorizationv1.ConditionsAwareDecision into the authorizer types.
func DeserializeConditionsAwareDecision(serializedDecision authorizationv1.ConditionsAwareDecision, failClosed func(error) authorizer.ConditionsAwareDecision) authorizer.ConditionsAwareDecision {
	switch serializedDecision.Type {
	case authorizationv1.ConditionsAwareDecisionTypeAllow:
		return authorizer.ConditionsAwareDecisionAllow(DeserializeReason(serializedDecision.Allow), DeserializeEvaluationError(serializedDecision.Allow))
	case authorizationv1.ConditionsAwareDecisionTypeDeny:
		return authorizer.ConditionsAwareDecisionDeny(DeserializeReason(serializedDecision.Deny), DeserializeEvaluationError(serializedDecision.Deny))
	case authorizationv1.ConditionsAwareDecisionTypeNoOpinion:
		return authorizer.ConditionsAwareDecisionNoOpinion(DeserializeReason(serializedDecision.NoOpinion), DeserializeEvaluationError(serializedDecision.NoOpinion))
	case authorizationv1.ConditionsAwareDecisionTypeConditionsMap:
		if serializedDecision.ConditionsMap != nil {
			return authorizer.ConditionsAwareDecisionConditionsMap(
				deserializeConditions(serializedDecision.ConditionsMap.DenyConditions),
				deserializeConditions(serializedDecision.ConditionsMap.NoOpinionConditions),
				deserializeConditions(serializedDecision.ConditionsMap.AllowConditions),
			)
		}
		return authorizer.ConditionsAwareDecisionConditionsMap(nil, nil, nil)
	case authorizationv1.ConditionsAwareDecisionTypeUnion:
		chain := authorizer.ConditionsAwareDecisionUnion{}
		for _, namedDecision := range serializedDecision.Union {
			chain.Add(namedDecision.AuthorizerName, DeserializeConditionsAwareDecision(namedDecision.Decision, failClosed))
		}
		return chain.ToDecision()
	default:
		return failClosed(fmt.Errorf("unrecognized ConditionsAwareDecision.type=%q", serializedDecision.Type))
	}
}

func deserializeConditions(serializedConditions []authorizationv1.Condition) []authorizer.Condition {
	deserializedConditions := make([]authorizer.Condition, len(serializedConditions))
	for i, serialized := range serializedConditions {
		deserializedConditions[i] = authorizer.GenericCondition{
			ID:          serialized.ID,
			Condition:   serialized.Condition,
			Type:        serialized.Type,
			Description: serialized.Description,
		}
	}
	return deserializedConditions
}
