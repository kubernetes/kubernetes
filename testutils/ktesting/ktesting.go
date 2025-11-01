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

package ktesting

import (
	"flag"
	"fmt"
	"testing"

	"k8s.io/klog/v2"

	// Initialize command line parameters.
	_ "k8s.io/component-base/logs/testinit"
)

func init() {
	// This is a good default for unit tests. Benchmarks should add their own
	// init function or TestMain to lower the default, for example to 2.
	SetDefaultVerbosity(5)
}

// SetDefaultVerbosity can be called during init to modify the default
// log verbosity of the program.
//
// Note that this immediately reconfigures the klog verbosity, already before
// flag parsing. If the verbosity is non-zero and SetDefaultVerbosity is called
// during init, then other init functions might start logging where normally
// they wouldn't log anything. Should this occur, then the right fix is to
// remove those log calls because logging during init is discouraged. It leads
// to unpredictable output (init order is not specified) and/or is useless
// (logging not initialized during init and thus conditional log output gets
// omitted).
func SetDefaultVerbosity(v int) {
	f := flag.CommandLine.Lookup("v")
	_ = f.Value.Set(fmt.Sprintf("%d", v))
}

// NewTestContext is a replacement for ktesting.NewTestContext
// which returns a more versatile context.
func NewTestContext(tb testing.TB) (klog.Logger, TContext) {
	tCtx := Init(tb)
	return tCtx.Logger(), tCtx
}
