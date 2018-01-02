//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package cmds

import (
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
)

var (
	HEKETI_CLI_VERSION = "(dev)"
	stderr             io.Writer
	stdout             io.Writer
	options            Options
	version            bool
)

// Main arguments
type Options struct {
	Url, Key, User string
	Json           bool
}

var RootCmd = &cobra.Command{
	Use:   "heketi-cli",
	Short: "Command line program for Heketi",
	Long:  "Command line program for Heketi",
	Example: `  $ export HEKETI_CLI_SERVER=http://localhost:8080
  $ heketi-cli volume list`,
	Run: func(cmd *cobra.Command, args []string) {
		if version {
			fmt.Printf("heketi-cli %v\n", HEKETI_CLI_VERSION)
		} else {
			cmd.Usage()
		}
	},
}

func init() {
	cobra.OnInitialize(initConfig)
	RootCmd.PersistentFlags().StringVarP(&options.Url, "server", "s", "",
		"\n\tHeketi server. Can also be set using the"+
			"\n\tenvironment variable HEKETI_CLI_SERVER")
	RootCmd.PersistentFlags().StringVar(&options.Key, "secret", "",
		"\n\tSecret key for specified user.  Can also be"+
			"\n\tset using the environment variable HEKETI_CLI_KEY")
	RootCmd.PersistentFlags().StringVar(&options.User, "user", "",
		"\n\tHeketi user.  Can also be set using the"+
			"\n\tenvironment variable HEKETI_CLI_USER")
	RootCmd.PersistentFlags().BoolVar(&options.Json, "json", false,
		"\n\tPrint response as JSON")
	RootCmd.Flags().BoolVarP(&version, "version", "v", false,
		"\n\tPrint version")
	RootCmd.SilenceUsage = true
}

func initConfig() {
	// Check server
	if options.Url == "" {
		options.Url = os.Getenv("HEKETI_CLI_SERVER")
		args := os.Args[1:]
		if options.Url == "" && !version && len(args) > 0 {
			fmt.Fprintf(stderr, "Server must be provided\n")
			os.Exit(3)
		}
	}

	// Check user
	if options.Key == "" {
		options.Key = os.Getenv("HEKETI_CLI_KEY")
	}

	// Check key
	if options.User == "" {
		options.User = os.Getenv("HEKETI_CLI_USER")
	}
}

func NewHeketiCli(heketiVersion string, mstderr io.Writer, mstdout io.Writer) *cobra.Command {
	stderr = mstderr
	stdout = mstdout
	HEKETI_CLI_VERSION = heketiVersion
	return RootCmd
}
