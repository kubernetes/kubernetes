/*
Copyright 2023 The Kubernetes Authors.

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

package cel

import (
	celgo "github.com/google/cel-go/cel"
)

type ExpressionAccessor interface {
	GetExpression() string
	ReturnTypes() []*celgo.Type
}

var _ ExpressionAccessor = &SubjectAccessReviewMatchCondition{}

// SubjectAccessReviewMatchCondition is a CEL expression that maps a SubjectAccessReview request to a list of values.
type SubjectAccessReviewMatchCondition struct {
	Expression string
}

func (v *SubjectAccessReviewMatchCondition) GetExpression() string {
	return v.Expression
}

func (v *SubjectAccessReviewMatchCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}
