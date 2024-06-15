/*
Copyright 2017 The Kubernetes Authors.

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

package benchmark

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	_ "k8s.io/component-base/logs/json/register"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestMain(m *testing.M) {
	// Run with -v=2, this is the default log level in production.
	ktesting.SetDefaultVerbosity(5)

	// test/integration/framework/flags.go unconditionally initializes the
	// logging flags. That's correct for most tests, but in the
	// scheduler_perf test we want more control over the flags, therefore
	// here strip them out.
	var fs flag.FlagSet
	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		switch f.Name {
		case "log-flush-frequency", "v", "vmodule":
			// These will be added below ourselves, don't copy.
		default:
			fs.Var(f.Value, f.Name, f.Usage)
		}
	})
	flag.CommandLine = &fs

	featureGate := featuregate.NewFeatureGate()
	runtime.Must(logsapi.AddFeatureGates(featureGate))
	flag.Var(featureGate, "feature-gate",
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
			"Options are:\n"+strings.Join(featureGate.KnownFeatures(), "\n"))
	c := logsapi.NewLoggingConfiguration()

	// This would fail if we hadn't removed the logging flags above.
	logsapi.AddGoFlags(c, flag.CommandLine)

	flag.Parse()

	logs.InitLogs()
	if err := logsapi.ValidateAndApply(c, featureGate); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	m.Run()
}
