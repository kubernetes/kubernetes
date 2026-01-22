// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import "sync"

// Opts specifies validation options for a SpecValidator.
//
// NOTE: other options might be needed, for example a go-swagger specific mode.
type Opts struct {
	ContinueOnErrors bool // true: continue reporting errors, even if spec is invalid
}

var (
	defaultOpts      = Opts{ContinueOnErrors: false} // default is to stop validation on errors
	defaultOptsMutex = &sync.Mutex{}
)

// SetContinueOnErrors sets global default behavior regarding spec validation errors reporting.
//
// For extended error reporting, you most likely want to set it to true.
// For faster validation, it's better to give up early when a spec is detected as invalid: set it to false (this is the default).
//
// Setting this mode does NOT affect the validation status.
//
// NOTE: this method affects global defaults. It is not suitable for a concurrent usage.
func SetContinueOnErrors(c bool) {
	defer defaultOptsMutex.Unlock()
	defaultOptsMutex.Lock()
	defaultOpts.ContinueOnErrors = c
}
