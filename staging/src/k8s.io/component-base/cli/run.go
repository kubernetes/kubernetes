/*
Copyright 2021 The Kubernetes Authors.

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

package cli

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/component-base/logs"
	cliflag "k8s.io/component-base/cli/flag"
)

// Run provides the common boilerplate code around executing a cobra command.
// For example, it ensures that logging is set up properly. Logging
// flags get added to the command line if not added already. Flags get normalized
// so that help texts show them with hyphens. Underscores are accepted
// as alternative for the command parameters.
func Run(cmd *cobra.Command) int {
	rand.Seed(time.Now().UnixNano())
	defer logs.FlushLogs()

	cmd.SetGlobalNormalizationFunc(cliflag.WordSepNormalizeFunc)

	// This is idempotent.
	logs.AddFlags(cmd.PersistentFlags())

	// Inject logs.InitLogs after command line parsing into one of the
	// PersistentPre* functions.
	switch {
	case cmd.PersistentPreRun != nil:
		pre := cmd.PersistentPreRun
		cmd.PersistentPreRun = func(cmd *cobra.Command, args []string) {
			logs.InitLogs()
			pre(cmd, args)
		}
	case cmd.PersistentPreRunE != nil:
		pre := cmd.PersistentPreRunE
		cmd.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
			logs.InitLogs()
			return pre(cmd, args)
		}
	default:
		cmd.PersistentPreRun = func(cmd *cobra.Command, args []string) {
			logs.InitLogs()
		}
	}

	if err := cmd.Execute(); err != nil {
		// The error may be about the command line ("unknown shorthand
		// flag: 'e' in -elp" in kube-scheduler). Printing that without
		// any decoration like the one that klog would add ("E0923
		// 23:02:03.219216 4168816 run.go:61] unknown shorthand flag")
		// is more readable.
		//
		// We also don't know in which state logging (configuration may
		// have failed) is whereas os.Stderr always should be usable.
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return 1
	}
	return 0
}
