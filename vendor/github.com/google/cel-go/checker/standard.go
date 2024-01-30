// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package checker

import (
	"github.com/google/cel-go/common/stdlib"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// StandardFunctions returns the Decls for all functions in the evaluator.
//
// Deprecated: prefer stdlib.FunctionExprDecls()
func StandardFunctions() []*exprpb.Decl {
	return stdlib.FunctionExprDecls()
}

// StandardTypes returns the set of type identifiers for standard library types.
//
// Deprecated: prefer stdlib.TypeExprDecls()
func StandardTypes() []*exprpb.Decl {
	return stdlib.TypeExprDecls()
}
