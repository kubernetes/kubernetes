// Copyright 2026 Google LLC
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

package types

import (
	"fmt"
	"regexp"
	"regexp/syntax"
)

// RegexProgramSize calculates the instruction count (program plan size) of a regex pattern.
func RegexProgramSize(pattern string) (int, error) {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return 0, err
	}
	prog, err := syntax.Compile(re)
	if err != nil {
		return 0, err
	}
	return len(prog.Inst), nil
}

// CompileRegexWithLimit compiles a regex pattern and verifies that its program plan size does not exceed limit.
// A limit <= 0 means unbounded.
func CompileRegexWithLimit(pattern string, limit int) (*regexp.Regexp, error) {
	if limit > 0 {
		sz, err := RegexProgramSize(pattern)
		if err != nil {
			return nil, err
		}
		if sz > limit {
			return nil, fmt.Errorf("regex program size %d exceeds limit of %d", sz, limit)
		}
	}
	return regexp.Compile(pattern)
}
