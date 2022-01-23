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
	"testing"

	"github.com/spf13/pflag"

	logsapi "k8s.io/component-base/logs/api/v1"
	_ "k8s.io/component-base/logs/json/register"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestMain(m *testing.M) {
	ktesting.DefaultConfig = ktesting.NewConfig(ktesting.Verbosity(0)) // Run with -v=0 by default.
	ktesting.DefaultConfig.AddFlags(flag.CommandLine)

	c := logsapi.NewLoggingConfiguration()

	// component-base only supports pflag at the moment.
	var fs pflag.FlagSet
	logsapi.AddFlags(c, &fs)
	// Not ideal. https://github.com/spf13/pflag/pull/330 would be better.
	fs.VisitAll(func(f *pflag.Flag) {
		if flag.CommandLine.Lookup(f.Name) == nil {
			flag.CommandLine.Var(f.Value, f.Name, f.Usage)
		}
	})
	flag.Parse()
	if err := logsapi.ValidateAndApply(c, nil /* no feature gates */); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	framework.EtcdMain(m.Run)
}
