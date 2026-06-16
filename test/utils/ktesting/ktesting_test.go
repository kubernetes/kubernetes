/*
Copyright The Kubernetes Authors.

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
	"os"
	"strings"
	"testing"

	"github.com/onsi/gomega"
)

func TestLogFlags(t *testing.T) {
	klogVFlag := flag.CommandLine.Lookup("v")
	klogVModuleFlag := flag.CommandLine.Lookup("vmodule")

	tCtx := Init(t)
	tCtx.Assert(klogVFlag).NotTo(gomega.BeNil(), "v flag")
	tCtx.Assert(klogVModuleFlag).NotTo(gomega.BeNil(), "vmodule flag")

	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		if f.Name == "v" ||
			f.Name == "vmodule" ||
			strings.HasPrefix(f.Name, "test.") {
			return
		}
		tCtx.Errorf("unexpected command line flag: %s", f.Name)
	})

	// The behavior of init and SetDefaultVerbosity depend on whether
	// "KTESTING_VERBOSITY" is set. We don't know how this test gets
	// invoked, so we have to adapt the expectations based on that.
	v, ok := os.LookupEnv("KTESTING_VERBOSITY")
	if !ok {
		v = "0"
	}
	tCtx.Assert(klogVFlag.Value.String()).To(gomega.Equal(v), "initial v")
	SetDefaultVerbosity(42)
	defer SetDefaultVerbosity(0)
	if !ok {
		v = "42"
	}
	tCtx.Assert(klogVFlag.Value.String()).To(gomega.Equal(v), "updated v")
}
