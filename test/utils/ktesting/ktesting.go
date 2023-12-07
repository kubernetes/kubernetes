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

// Package ktesting is a wrapper around k8s.io/klog/v2/ktesting. It provides
// those (and only those) functions that test code in Kubernetes should use,
// plus better dumping of complex datatypes. It adds the klog command line
// flags and increases the default verbosity to 5.
package ktesting

import (
	"context"
	"flag"
	"fmt"

	_ "k8s.io/component-base/logs/testinit"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	// "k8s.io/kubernetes/test/utils/format"
)

func init() {
	// This is a good default for unit tests. Benchmarks should add their own
	// init function or TestMain to lower the default, for example to 2.
	SetDefaultVerbosity(5)
}

// SetDefaultVerbosity can be called during init to modify the default
// log verbosity of the program.
func SetDefaultVerbosity(v int) {
	f := flag.CommandLine.Lookup("v")
	_ = f.Value.Set(fmt.Sprintf("%d", v))
}

// NewTestContext is a wrapper around ktesting.NewTestContext with settings
// specific to Kubernetes.
func NewTestContext(tl ktesting.TL) (klog.Logger, context.Context) {
	config := ktesting.NewConfig(
		// TODO (pohly): merge
		// https://github.com/kubernetes/klog/pull/363, new klog
		// release, update and merge
		// https://github.com/kubernetes/kubernetes/pull/115277, then
		// uncomment this.
		//
		// ktesting.AnyToString(format.AnyToString),
		ktesting.VerbosityFlagName("v"),
		ktesting.VModuleFlagName("vmodule"),
	)

	// Copy klog settings instead of making the ktesting logger
	// configurable directly.
	var fs flag.FlagSet
	config.AddFlags(&fs)
	for _, name := range []string{"v", "vmodule"} {
		from := flag.CommandLine.Lookup(name)
		to := fs.Lookup(name)
		if err := to.Value.Set(from.Value.String()); err != nil {
			panic(err)
		}
	}

	logger := ktesting.NewLogger(tl, config)
	ctx := klog.NewContext(context.Background(), logger)
	return logger, ctx
}
