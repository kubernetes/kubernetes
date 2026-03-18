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

package conditionsenforcer

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"sync"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// BuiltinConditionsEvaluator is a method that can evaluate conditions with some types fast in-process, without need for webhooks.
type BuiltinConditionsEvaluator interface {
	EvaluateCondition(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult
}

// BuiltinConditionsEvaluatorFunc implements BuiltinConditionsEvaluator
var _ BuiltinConditionsEvaluator = BuiltinConditionsEvaluatorFunc(nil)

// BuiltinConditionsEvaluatorFunc is a function that implements BuiltinConditionsEvaluator.
type BuiltinConditionsEvaluatorFunc func(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult

func (f BuiltinConditionsEvaluatorFunc) EvaluateCondition(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
	return f(ctx, condition, data)
}

// BuiltinConditionsEvaluators implements BuiltinConditionsEvaluator
var _ BuiltinConditionsEvaluator = BuiltinConditionsEvaluators(nil)

type BuiltinConditionsEvaluators []BuiltinConditionsEvaluator

func (evaluators BuiltinConditionsEvaluators) EvaluateCondition(ctx context.Context, condition authorizer.Condition, data authorizer.ConditionsData) authorizer.ConditionEvaluationResult {
	for _, evaluator := range evaluators {
		res := evaluator.EvaluateCondition(ctx, condition, data)
		if !res.IsUnevaluatable() {
			return res
		}
	}
	return authorizer.ConditionsEvaluationResultUnevaluatable()
}

var evaluatorMap = map[string]BuiltinConditionsEvaluator{}
var evaluatorMapMu = &sync.RWMutex{}

func RegisteredBuiltinConditionsEvaluators() BuiltinConditionsEvaluators {
	evaluatorMapMu.RLock()
	defer evaluatorMapMu.RUnlock()
	// Always return a fresh slice, so there's no shared memory returned.
	return slices.Collect(maps.Values(evaluatorMap))
}

func RegisterBuiltinConditionsEvaluator(name string, evaluator BuiltinConditionsEvaluator, overwrite bool) error {
	evaluatorMapMu.Lock()
	defer evaluatorMapMu.Unlock()
	_, existing := evaluatorMap[name]
	if existing && !overwrite {
		return fmt.Errorf("already existing conditions evaluator with name %q registered, and overwrite is false", name)
	}
	evaluatorMap[name] = evaluator
	return nil
}
