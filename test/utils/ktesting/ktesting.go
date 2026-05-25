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
	"context"
	"flag"
	"fmt"
	"os"
	"testing"

	"k8s.io/klog/v2"
	_ "k8s.io/kubernetes/test/utils/ktesting/format" // Activate YAML format in Gomega.
)

var klogVFlag *flag.Flag
var haveVerbosityFromEnv = false

func init() {
	// Register "v" and "vmodule" in the global command line.
	//
	// This is where ktesting is opinionated: all tests using
	// it should have those flags, without unit test authors
	// having to manually ask for it.
	var klogFlags flag.FlagSet
	klog.InitFlags(&klogFlags)
	klogVFlag = klogFlags.Lookup("v")
	for _, name := range []string{"v", "vmodule"} {
		f := klogFlags.Lookup(name)

		// Some other code might have done this already,
		// for example ./staging/src/k8s.io/component-base/logs/testinit/testinit.go.
		//
		// As of https://tip.golang.org/doc/go1.21#language,
		// the init from k8s.io/component-base is guaranteed to run
		// before ours because that package name sorts first.
		// By checking here whether the flags are already added
		// ktesting works with and without component-base/logs/testinit.
		//
		// We could require that a test binary only uses either ktesting
		// or testinit, but this approach is more flexible.
		if flag.CommandLine.Lookup(name) != nil {
			continue
		}

		flag.CommandLine.Var(f.Value, name, f.DefValue)
	}

	// CI jobs which run a large collection of tests cannot
	// assume that all tests have these command line flags.
	// They can set the KTESTING_VERBOSITY env variable
	// to change the default verbosity in those tests which
	// use ktesting, without breaking other tests which don't.
	if v, ok := os.LookupEnv("KTESTING_VERBOSITY"); ok {
		haveVerbosityFromEnv = true
		if err := klogVFlag.Value.Set(v); err != nil {
			panic(fmt.Sprintf("KTESTING_VERBOSITY: %v", err))
		}
	}
}

// SetDefaultVerbosity can be called during init to modify the default
// log verbosity of the program. If the KTESTING_VERBOSITY env variable
// is set, then the value from that variable is used.
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
	if !haveVerbosityFromEnv {
		_ = klogVFlag.Value.Set(fmt.Sprintf("%d", v))
	}
}

// NewTestContext is a drop-in replacement for ktesting.NewTestContext.
//
// The result can be cast to a TContext and has the same functionality
// (for example, it gets cancelled at the end of the test).
// It's returned as a context.Context to avoid breaking code which expects that type.
func NewTestContext(tb testing.TB) (klog.Logger, context.Context) {
	tCtx := Init(tb)
	return tCtx.Logger(), tCtx
}
