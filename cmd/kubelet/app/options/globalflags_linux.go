//go:build linux

/*
Copyright 2018 The Kubernetes Authors.

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
	"os"

	"github.com/spf13/pflag"

	"github.com/google/cadvisor/lib/cadvisorflags"

	// ensure libs have a chance to globally register their flags
	_ "github.com/google/cadvisor/lib/container/common"
	_ "github.com/google/cadvisor/lib/container/containerd"
	_ "github.com/google/cadvisor/lib/container/raw"
	_ "github.com/google/cadvisor/lib/machine"
	_ "github.com/google/cadvisor/lib/manager"
	_ "github.com/google/cadvisor/lib/storage"
)

// addCadvisorFlags adds flags from cadvisor
func addCadvisorFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with our flagset
	global := flag.CommandLine
	local := pflag.NewFlagSet(os.Args[0], pflag.ExitOnError)

	// Flag names come from github.com/google/cadvisor/lib/cadvisorflags, the single
	// source of truth in the library. Sourcing them from there instead of string
	// literals means a removed or renamed cAdvisor flag surfaces as a build/test
	// failure (see globalflags_linux_test.go) rather than a kubelet startup panic.

	// Bind only HousekeepingInterval, explicitly, so the kubelet's flags can't grow
	// just because cadvisorflags.Kept() gains a name (a test asserts it stays there).
	register(global, local, cadvisorflags.HousekeepingInterval)

	// These flags were implicit from cadvisor, and are mistakes that should be registered deprecated:
	const deprecated = "This is a cadvisor flag that was mistakenly registered with the Kubelet. Due to legacy concerns, it will follow the standard CLI deprecation timeline before being removed."
	for _, name := range cadvisorflags.Deprecated() {
		registerDeprecated(global, local, name, deprecated)
	}

	// finally, add cadvisor flags to the provided flagset
	fs.AddFlagSet(local)
}
