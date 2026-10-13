// Copyright 2022 Google LLC
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

package interpreter

import (
	"regexp"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// MatchesRegexOptimization optimizes the 'matches' standard library function by compiling the regex pattern and
// reporting any compilation errors at program creation time, and using the compiled regex pattern for all function
// call invocations.
var MatchesRegexOptimization = &RegexOptimization{
	Function:   "matches",
	RegexIndex: 1,
	Factory: func(call InterpretableCall, regexPattern string) (InterpretableCall, error) {
		compiledRegex, err := regexp.Compile(regexPattern)
		if err != nil {
			return nil, err
		}
		return NewCall(call.ID(), call.Function(), call.OverloadID(), call.Args(), func(values ...ref.Val) ref.Val {
			if len(values) != 2 {
				return types.NoSuchOverloadErr()
			}
			in, ok := values[0].Value().(string)
			if !ok {
				return types.NoSuchOverloadErr()
			}
			return types.Bool(compiledRegex.MatchString(in))
		}), nil
	},
}
