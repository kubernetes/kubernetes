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

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
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

	// When error printing is enabled for the Cobra command, a flag parse
	// error gets printed first, then optionally the often long usage
	// text. This is very unreadable in a console because the last few
	// lines that will be visible on screen don't include the error.
	//
	// The recommendation from #sig-cli was to print the usage text, then
	// the error. We implement this consistently for all commands here.
	// However, we don't want to print the usage text when command
	// execution fails for other reasons than parsing. We detect this via
	// the FlagParseError callback.
	//
	// A variable is used instead of wrapping the error with a special
	// error type because the variable is simpler and less fragile: the
	// original FlagErrorFunc might replace the wrapped error.
	parsingFailed := false
	if cmd.SilenceUsage {
		// Some commands, like kubectl, already deal with this themselves.
		// We don't change the behavior for those and just track whether
		// parsing failed for the error output below.
		flagErrorFunc := cmd.FlagErrorFunc()
		cmd.SetFlagErrorFunc(func(c *cobra.Command, err error) error {
			parsingFailed = true
			return flagErrorFunc(c, err)
		})
	} else {
		cmd.SilenceUsage = true
		cmd.SetFlagErrorFunc(func(c *cobra.Command, err error) error {
			parsingFailed = true

			// Re-enable usage printing.
			c.SilenceUsage = false
			return err
		})
	}

	// In all cases error printing is done below.
	cmd.SilenceErrors = true

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
		// If the error is about flag parsing, then printing that error
		// with the decoration that klog would add ("E0923
		// 23:02:03.219216 4168816 run.go:61] unknown shorthand flag")
		// is less readable. Using klog.Fatal is even worse because it
		// dumps a stack trace that isn't about the error.
		//
		// But if it is some other error encountered at runtime, then
		// we want to log it as error.
		//
		// We can distinguish these two cases depending on whether
		// our FlagErrorFunc above was called.
		if parsingFailed {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		} else {
			klog.ErrorS(err, "command failed")
		}
		return 1
	}
	return 0
}
