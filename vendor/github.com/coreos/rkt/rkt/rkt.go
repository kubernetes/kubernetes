// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/keystore"
	"github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/pkg/multicall"
	"github.com/coreos/rkt/rkt/config"
	rktflag "github.com/coreos/rkt/rkt/flag"
	"github.com/spf13/cobra"
)

const (
	cliName        = "rkt"
	cliDescription = "rkt, the application container runner"

	defaultDataDir = "/var/lib/rkt"

	bash_completion_func = `__rkt_parse_image()
{
    local rkt_output
    if rkt_output=$(rkt image list --no-legend 2>/dev/null); then
        out=($(echo "${rkt_output}" | awk '{print $1}'))
        COMPREPLY=( $( compgen -W "${out[*]}" -- "$cur" ) )
    fi
}

__rkt_parse_list()
{
    local rkt_output
    if rkt_output=$(rkt list --no-legend 2>/dev/null); then
        if [[ -n "$1" ]]; then
            out=($(echo "${rkt_output}" | grep ${1} | awk '{print $1}'))
        else
            out=($(echo "${rkt_output}" | awk '{print $1}'))
        fi
        COMPREPLY=( $( compgen -W "${out[*]}" -- "$cur" ) )
    fi
}

__custom_func() {
    case ${last_command} in
        rkt_image_export | \
        rkt_image_extract | \
        rkt_image_cat-manifest | \
        rkt_image_render | \
        rkt_image_rm | \
        rkt_run | \
        rkt_prepare)
            __rkt_parse_image
            return
            ;;
        rkt_run-prepared)
            __rkt_parse_list prepared
            return
            ;;
        rkt_enter)
            __rkt_parse_list running
            return
            ;;
        rkt_rm)
            __rkt_parse_list "prepare\|exited"
            return
            ;;
        rkt_status)
            __rkt_parse_list
            return
            ;;
        *)
            ;;
    esac
}
`
)

type absDir string

func (d *absDir) String() string {
	return (string)(*d)
}

func (d *absDir) Set(str string) error {
	if str == "" {
		return fmt.Errorf(`"" is not a valid directory`)
	}

	dir, err := filepath.Abs(str)
	if err != nil {
		return err
	}
	*d = (absDir)(dir)
	return nil
}

func (d *absDir) Type() string {
	return "absolute-directory"
}

var (
	tabOut      *tabwriter.Writer
	globalFlags = struct {
		Dir                string
		SystemConfigDir    string
		LocalConfigDir     string
		UserConfigDir      string
		Debug              bool
		Help               bool
		InsecureFlags      *rktflag.SecFlags
		TrustKeysFromHTTPS bool
	}{
		Dir:             defaultDataDir,
		SystemConfigDir: common.DefaultSystemConfigDir,
		LocalConfigDir:  common.DefaultLocalConfigDir,
	}

	cachedConfig  *config.Config
	cachedDataDir string
	cmdExitCode   int

	stderr *log.Logger
	stdout *log.Logger
)

var cmdRkt = &cobra.Command{
	Use:   "rkt [command]",
	Short: cliDescription,
	Long: `A CLI for running app containers on Linux.

To get the help on any specific command, run "rkt help command".`,
	BashCompletionFunction: bash_completion_func,
	Run: runMissingCommand,
}

func init() {
	sf, err := rktflag.NewSecFlags("none")
	if err != nil {
		fmt.Fprintf(os.Stderr, "rkt: problem initializing: %v", err)
		os.Exit(1)
	}

	globalFlags.InsecureFlags = sf

	cmdRkt.PersistentFlags().BoolVar(&globalFlags.Debug, "debug", false, "print out more debug information to stderr")
	cmdRkt.PersistentFlags().Var((*absDir)(&globalFlags.Dir), "dir", "rkt data directory")
	cmdRkt.PersistentFlags().Var((*absDir)(&globalFlags.SystemConfigDir), "system-config", "system configuration directory")
	cmdRkt.PersistentFlags().Var((*absDir)(&globalFlags.LocalConfigDir), "local-config", "local configuration directory")
	cmdRkt.PersistentFlags().Var((*absDir)(&globalFlags.UserConfigDir), "user-config", "user configuration directory")
	cmdRkt.PersistentFlags().Var(globalFlags.InsecureFlags, "insecure-options",
		fmt.Sprintf("comma-separated list of security features to disable. Allowed values: %s",
			globalFlags.InsecureFlags.PermissibleString()))
	cmdRkt.PersistentFlags().BoolVar(&globalFlags.TrustKeysFromHTTPS, "trust-keys-from-https",
		false, "automatically trust gpg keys fetched from https")

	// Run this before the execution of each subcommand to set up output
	cmdRkt.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		stderr = log.New(os.Stderr, cmd.Name(), globalFlags.Debug)
		stdout = log.New(os.Stdout, "", false)
	}

	cobra.EnablePrefixMatching = true
}

func getTabOutWithWriter(writer io.Writer) *tabwriter.Writer {
	aTabOut := new(tabwriter.Writer)
	aTabOut.Init(writer, 0, 8, 1, '\t', 0)
	return aTabOut
}

// runWrapper returns a func(cmd *cobra.Command, args []string) that internally
// will add command function return code and the reinsertion of the "--" flag
// terminator.
func runWrapper(cf func(cmd *cobra.Command, args []string) (exit int)) func(cmd *cobra.Command, args []string) {
	return func(cmd *cobra.Command, args []string) {
		cmdExitCode = cf(cmd, args)
	}
}

// ensureSuperuser will error out if the effective UID of the current process
// is not zero. Otherwise, it will invoke the supplied cobra command.
func ensureSuperuser(cf func(cmd *cobra.Command, args []string)) func(cmd *cobra.Command, args []string) {
	return func(cmd *cobra.Command, args []string) {
		if os.Geteuid() != 0 {
			stderr.Print("cannot run as unprivileged user")
			cmdExitCode = 1
			return
		}
		cf(cmd, args)
	}
}

func runMissingCommand(cmd *cobra.Command, args []string) {
	stderr.Print("missing command")
	cmd.HelpFunc()(cmd, args)
	cmdExitCode = 2 // invalid argument
}

func main() {
	// check if rkt is executed with a multicall command
	multicall.MaybeExec()

	cmdRkt.SetUsageFunc(usageFunc)

	// Make help just show the usage
	cmdRkt.SetHelpTemplate(`{{.UsageString}}`)

	// // Uncomment to update rkt.bash
	// stdout.Print("Generating rkt.bash")
	// cmdRkt.GenBashCompletionFile("dist/bash_completion/rkt.bash")
	// os.Exit(0)

	if err := cmdRkt.Execute(); err != nil && cmdExitCode == 0 {
		cmdExitCode = 2 // invalid argument
	}
	os.Exit(cmdExitCode)
}

// where pod directories are created and locked before moving to prepared
func embryoDir() string {
	return filepath.Join(getDataDir(), "pods", "embryo")
}

// where pod trees reside during (locked) and after failing to complete preparation (unlocked)
func prepareDir() string {
	return filepath.Join(getDataDir(), "pods", "prepare")
}

// where pod trees reside upon successful preparation
func preparedDir() string {
	return filepath.Join(getDataDir(), "pods", "prepared")
}

// where pod trees reside once run
func runDir() string {
	return filepath.Join(getDataDir(), "pods", "run")
}

// where pod trees reside once exited & marked as garbage by a gc pass
func exitedGarbageDir() string {
	return filepath.Join(getDataDir(), "pods", "exited-garbage")
}

// where never-executed pod trees reside once marked as garbage by a gc pass (failed prepares, expired prepareds)
func garbageDir() string {
	return filepath.Join(getDataDir(), "pods", "garbage")
}

func getKeystore() *keystore.Keystore {
	if globalFlags.InsecureFlags.SkipImageCheck() {
		return nil
	}
	config := keystore.NewConfig(globalFlags.SystemConfigDir, globalFlags.LocalConfigDir)
	return keystore.New(config)
}

func getDataDir() string {
	if cachedDataDir == "" {
		cachedDataDir = calculateDataDir()
	}
	return cachedDataDir
}

func calculateDataDir() string {
	// If --dir parameter is passed, then use this value.
	if dirFlag := cmdRkt.PersistentFlags().Lookup("dir"); dirFlag != nil {
		if dirFlag.Changed {
			return globalFlags.Dir
		}
	} else {
		// should not happen
		panic(`"--dir" flag not found`)
	}

	// If above fails, then try to get the value from configuration.
	if config, err := getConfig(); err != nil {
		stderr.PrintE("cannot get configuration", err)
		os.Exit(1)
	} else {
		if config.Paths.DataDir != "" {
			return config.Paths.DataDir
		}
	}

	// If above fails, then use the default.
	return defaultDataDir
}

func getConfig() (*config.Config, error) {
	if cachedConfig == nil {
		dirs := []string{
			globalFlags.SystemConfigDir,
			globalFlags.LocalConfigDir,
		}
		if globalFlags.UserConfigDir != "" {
			dirs = append(dirs, globalFlags.UserConfigDir)
		}
		cfg, err := config.GetConfigFrom(dirs...)
		if err != nil {
			return nil, err
		}
		cachedConfig = cfg
	}
	return cachedConfig, nil
}

func lockDir() string {
	return filepath.Join(getDataDir(), "locks")
}
