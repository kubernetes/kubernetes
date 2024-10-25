/*
Copyright 2024 The Kubernetes Authors.

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

package mutating

import (
	celgo "github.com/google/cel-go/cel"
	celtypes "github.com/google/cel-go/common/types"

	"k8s.io/apiserver/pkg/admission/plugin/cel"
)

var _ cel.ExpressionAccessor = &ApplyConfigurationCondition{}

// ApplyConfigurationCondition contains the inputs needed to compile and evaluate a cel expression
// that returns an apply configuration
type ApplyConfigurationCondition struct {
	Expression string
}

func (v *ApplyConfigurationCondition) GetExpression() string {
	return v.Expression
}

func (v *ApplyConfigurationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{applyConfigObjectType}
}

var applyConfigObjectType = celtypes.NewObjectType("Object")

var _ cel.ExpressionAccessor = &JSONPatchCondition{}

// JSONPatchCondition contains the inputs needed to compile and evaluate a cel expression
// that returns a JSON patch value.
type JSONPatchCondition struct {
	Expression string
}

func (v *JSONPatchCondition) GetExpression() string {
	return v.Expression
}

func (v *JSONPatchCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.ListType(jsonPatchType)}
}

var jsonPatchType = celtypes.NewObjectType("JSONPatch")
