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
	errs *common.Errors
}

// errorCount indicates the number of errors reported.
func (e *parseErrors) errorCount() int {
	return len(e.errs.GetErrors())
}

func (e *parseErrors) internalError(message string) {
	e.errs.ReportErrorAtID(0, common.NoLocation, message)
}

func (e *parseErrors) syntaxError(l common.Location, message string) {
	e.errs.ReportErrorAtID(0, l, fmt.Sprintf("Syntax error: %s", message))
}

func (e *parseErrors) reportErrorAtID(id int64, l common.Location, message string, args ...any) {
	e.errs.ReportErrorAtID(id, l, message, args...)
}
