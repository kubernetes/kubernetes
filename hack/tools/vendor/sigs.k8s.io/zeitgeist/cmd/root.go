/*
Copyright 2020 The Kubernetes Authors.

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

/* Zeitgeist is a language-agnostic dependency checker */

package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"sigs.k8s.io/zeitgeist/internal/log"
)

type rootOptions struct {
	logLevel string
}

var (
	rootOpts = &rootOptions{}

	// TODO: Implement these as a separate function or subcommand to avoid the
	//       deadcode,unused,varcheck nolints
	// Variables set by GoReleaser on release
	version = "dev"     // nolint: deadcode,unused,varcheck
	commit  = "none"    // nolint: deadcode,unused,varcheck
	date    = "unknown" // nolint: deadcode,unused,varcheck
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:               "zeitgeist",
	Short:             "Zeitgeist is a language-agnostic dependency checker",
	PersistentPreRunE: initLogging,
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		logrus.Fatal(err)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(
		&rootOpts.logLevel,
		"log-level",
		"info",
		"the logging verbosity, either 'panic', 'fatal', 'error', 'warn', 'warning', 'info', 'debug' or 'trace'",
	)
}

func initLogging(*cobra.Command, []string) error {
	return log.SetupGlobalLogger(rootOpts.logLevel)
}
