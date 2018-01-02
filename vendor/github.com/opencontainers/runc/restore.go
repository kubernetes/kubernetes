// +build linux

package main

import (
	"fmt"
	"os"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/urfave/cli"
)

var restoreCommand = cli.Command{
	Name:  "restore",
	Usage: "restore a container from a previous checkpoint",
	ArgsUsage: `<container-id>

Where "<container-id>" is the name for the instance of the container to be
restored.`,
	Description: `Restores the saved state of the container instance that was previously saved
using the runc checkpoint command.`,
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "console-socket",
			Value: "",
			Usage: "path to an AF_UNIX socket which will receive a file descriptor referencing the master end of the console's pseudoterminal",
		},
		cli.StringFlag{
			Name:  "image-path",
			Value: "",
			Usage: "path to criu image files for restoring",
		},
		cli.StringFlag{
			Name:  "work-path",
			Value: "",
			Usage: "path for saving work files and logs",
		},
		cli.BoolFlag{
			Name:  "tcp-established",
			Usage: "allow open tcp connections",
		},
		cli.BoolFlag{
			Name:  "ext-unix-sk",
			Usage: "allow external unix sockets",
		},
		cli.BoolFlag{
			Name:  "shell-job",
			Usage: "allow shell jobs",
		},
		cli.BoolFlag{
			Name:  "file-locks",
			Usage: "handle file locks, for safety",
		},
		cli.StringFlag{
			Name:  "manage-cgroups-mode",
			Value: "",
			Usage: "cgroups mode: 'soft' (default), 'full' and 'strict'",
		},
		cli.StringFlag{
			Name:  "bundle, b",
			Value: "",
			Usage: "path to the root of the bundle directory",
		},
		cli.BoolFlag{
			Name:  "detach,d",
			Usage: "detach from the container's process",
		},
		cli.StringFlag{
			Name:  "pid-file",
			Value: "",
			Usage: "specify the file to write the process id to",
		},
		cli.BoolFlag{
			Name:  "no-subreaper",
			Usage: "disable the use of the subreaper used to reap reparented processes",
		},
		cli.BoolFlag{
			Name:  "no-pivot",
			Usage: "do not use pivot root to jail process inside rootfs.  This should be used whenever the rootfs is on top of a ramdisk",
		},
		cli.StringSliceFlag{
			Name:  "empty-ns",
			Usage: "create a namespace, but don't restore its properties",
		},
		cli.BoolFlag{
			Name:  "auto-dedup",
			Usage: "enable auto deduplication of memory images",
		},
	},
	Action: func(context *cli.Context) error {
		if err := checkArgs(context, 1, exactArgs); err != nil {
			return err
		}
		// XXX: Currently this is untested with rootless containers.
		if isRootless() {
			return fmt.Errorf("runc restore requires root")
		}

		spec, err := setupSpec(context)
		if err != nil {
			return err
		}
		options := criuOptions(context)
		status, err := startContainer(context, spec, CT_ACT_RESTORE, options)
		if err != nil {
			return err
		}
		// exit with the container's exit status so any external supervisor is
		// notified of the exit with the correct exit status.
		os.Exit(status)
		return nil
	},
}

func criuOptions(context *cli.Context) *libcontainer.CriuOpts {
	imagePath := getCheckpointImagePath(context)
	if err := os.MkdirAll(imagePath, 0655); err != nil {
		fatal(err)
	}
	return &libcontainer.CriuOpts{
		ImagesDirectory:         imagePath,
		WorkDirectory:           context.String("work-path"),
		ParentImage:             context.String("parent-path"),
		LeaveRunning:            context.Bool("leave-running"),
		TcpEstablished:          context.Bool("tcp-established"),
		ExternalUnixConnections: context.Bool("ext-unix-sk"),
		ShellJob:                context.Bool("shell-job"),
		FileLocks:               context.Bool("file-locks"),
		PreDump:                 context.Bool("pre-dump"),
		AutoDedup:               context.Bool("auto-dedup"),
	}
}
