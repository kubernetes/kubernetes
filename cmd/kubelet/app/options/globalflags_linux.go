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

	// registers housekeeping_interval on the global flag set
	_ "github.com/google/cadvisor/lib/manager"
)

// addCadvisorFlags adds flags from cadvisor
func addCadvisorFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with our flagset
	global := flag.CommandLine
	local := pflag.NewFlagSet(os.Args[0], pflag.ExitOnError)

	// e2e node tests rely on this
	register(global, local, cadvisorflags.HousekeepingInterval)

	// finally, add cadvisor flags to the provided flagset
	fs.AddFlagSet(local)
}
