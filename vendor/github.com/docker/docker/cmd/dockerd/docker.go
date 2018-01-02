package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/docker/docker/cli"
	"github.com/docker/docker/daemon/config"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/docker/pkg/term"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

func newDaemonCommand() *cobra.Command {
	opts := newDaemonOptions(config.New())

	cmd := &cobra.Command{
		Use:           "dockerd [OPTIONS]",
		Short:         "A self-sufficient runtime for containers.",
		SilenceUsage:  true,
		SilenceErrors: true,
		Args:          cli.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.flags = cmd.Flags()
			return runDaemon(opts)
		},
	}
	cli.SetupRootCommand(cmd)

	flags := cmd.Flags()
	flags.BoolVarP(&opts.version, "version", "v", false, "Print version information and quit")
	flags.StringVar(&opts.configFile, "config-file", defaultDaemonConfigFile, "Daemon configuration file")
	opts.InstallFlags(flags)
	installConfigFlags(opts.daemonConfig, flags)
	installServiceFlags(flags)

	return cmd
}

func runDaemon(opts *daemonOptions) error {
	if opts.version {
		showVersion()
		return nil
	}

	daemonCli := NewDaemonCli()

	// Windows specific settings as these are not defaulted.
	if runtime.GOOS == "windows" {
		if opts.daemonConfig.Pidfile == "" {
			opts.daemonConfig.Pidfile = filepath.Join(opts.daemonConfig.Root, "docker.pid")
		}
		if opts.configFile == "" {
			opts.configFile = filepath.Join(opts.daemonConfig.Root, `config\daemon.json`)
		}
	}

	// On Windows, this may be launching as a service or with an option to
	// register the service.
	stop, runAsService, err := initService(daemonCli)
	if err != nil {
		logrus.Fatal(err)
	}

	if stop {
		return nil
	}

	// If Windows SCM manages the service - no need for PID files
	if runAsService {
		opts.daemonConfig.Pidfile = ""
	}

	err = daemonCli.start(opts)
	notifyShutdown(err)
	return err
}

func showVersion() {
	fmt.Printf("Docker version %s, build %s\n", dockerversion.Version, dockerversion.GitCommit)
}

func main() {
	if reexec.Init() {
		return
	}

	// Set terminal emulation based on platform as required.
	_, stdout, stderr := term.StdStreams()

	// @jhowardmsft - maybe there is a historic reason why on non-Windows, stderr is used
	// here. However, on Windows it makes no sense and there is no need.
	if runtime.GOOS == "windows" {
		logrus.SetOutput(stdout)
	} else {
		logrus.SetOutput(stderr)
	}

	cmd := newDaemonCommand()
	cmd.SetOutput(stdout)
	if err := cmd.Execute(); err != nil {
		fmt.Fprintf(stderr, "%s\n", err)
		os.Exit(1)
	}
}
