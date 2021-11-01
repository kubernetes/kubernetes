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

package parser

import (
	"fmt"

	"github.com/google/cel-go/common"
)

// parseErrors is a specialization of Errors.
type parseErrors struct {
	*common.Errors
}

func (e *parseErrors) syntaxError(l common.Location, message string) {
	e.ReportError(l, fmt.Sprintf("Syntax error: %s", message))
}
