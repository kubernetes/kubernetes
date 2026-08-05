//go:build linux

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

package options

import (
	"flag"
	"slices"
	"testing"

	"github.com/google/cadvisor/lib/cadvisorflags"
	"github.com/spf13/pflag"
)

// TestPinnedCadvisorFlagsResolve is a contract test between the kubelet and
// github.com/google/cadvisor/lib. addCadvisorFlags re-registers the flag names in
// cadvisorflags by looking them up on the global flag set; register() panics if
// a name is missing. This test turns that potential kubelet STARTUP PANIC into a
// build-time test failure: if libcadvisor removes or renames a flag without
// updating cadvisorflags, this fails here instead.
func TestPinnedCadvisorFlagsResolve(t *testing.T) {
	names := cadvisorflags.Kept()
	if len(names) == 0 {
		t.Fatal("cadvisorflags exposes no pinned flag names; expected at least housekeeping_interval")
	}
	for _, name := range names {
		if flag.CommandLine.Lookup(name) == nil {
			t.Errorf("cAdvisor flag %q is pinned by the kubelet (cadvisorflags) but is not registered on the global flag set; update github.com/google/cadvisor/lib or its cadvisorflags package", name)
		}
	}
}

// TestAddCadvisorFlagsDoesNotPanic exercises the real startup path end-to-end:
// register() panics if the pinned flag is missing, so a clean run proves
// addCadvisorFlags will not panic the kubelet at startup.
func TestAddCadvisorFlagsDoesNotPanic(t *testing.T) {
	addCadvisorFlags(pflag.NewFlagSet("test", pflag.ContinueOnError))
}

// TestKeptIncludesHousekeepingInterval ensures the flag addCadvisorFlags binds
// explicitly is still listed by the library's Kept().
func TestKeptIncludesHousekeepingInterval(t *testing.T) {
	if !slices.Contains(cadvisorflags.Kept(), cadvisorflags.HousekeepingInterval) {
		t.Errorf("cadvisorflags.Kept() must include HousekeepingInterval (%q); addCadvisorFlags binds it as a normal kubelet flag", cadvisorflags.HousekeepingInterval)
	}
}
