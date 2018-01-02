// +build linux

package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/urfave/cli"

	"golang.org/x/sys/unix"
)

var checkpointCommand = cli.Command{
	Name:  "checkpoint",
	Usage: "checkpoint a running container",
	ArgsUsage: `<container-id>

Where "<container-id>" is the name for the instance of the container to be
checkpointed.`,
	Description: `The checkpoint command saves the state of the container instance.`,
	Flags: []cli.Flag{
		cli.StringFlag{Name: "image-path", Value: "", Usage: "path for saving criu image files"},
		cli.StringFlag{Name: "work-path", Value: "", Usage: "path for saving work files and logs"},
		cli.StringFlag{Name: "parent-path", Value: "", Usage: "path for previous criu image files in pre-dump"},
		cli.BoolFlag{Name: "leave-running", Usage: "leave the process running after checkpointing"},
		cli.BoolFlag{Name: "tcp-established", Usage: "allow open tcp connections"},
		cli.BoolFlag{Name: "ext-unix-sk", Usage: "allow external unix sockets"},
		cli.BoolFlag{Name: "shell-job", Usage: "allow shell jobs"},
		cli.StringFlag{Name: "page-server", Value: "", Usage: "ADDRESS:PORT of the page server"},
		cli.BoolFlag{Name: "file-locks", Usage: "handle file locks, for safety"},
		cli.BoolFlag{Name: "pre-dump", Usage: "dump container's memory information only, leave the container running after this"},
		cli.StringFlag{Name: "manage-cgroups-mode", Value: "", Usage: "cgroups mode: 'soft' (default), 'full' and 'strict'"},
		cli.StringSliceFlag{Name: "empty-ns", Usage: "create a namespace, but don't restore its properties"},
		cli.BoolFlag{Name: "auto-dedup", Usage: "enable auto deduplication of memory images"},
	},
	Action: func(context *cli.Context) error {
		if err := checkArgs(context, 1, exactArgs); err != nil {
			return err
		}
		// XXX: Currently this is untested with rootless containers.
		if isRootless() {
			return fmt.Errorf("runc checkpoint requires root")
		}

		container, err := getContainer(context)
		if err != nil {
			return err
		}
		status, err := container.Status()
		if err != nil {
			return err
		}
		if status == libcontainer.Created {
			fatalf("Container cannot be checkpointed in created state")
		}
		defer destroy(container)
		options := criuOptions(context)
		// these are the mandatory criu options for a container
		setPageServer(context, options)
		setManageCgroupsMode(context, options)
		if err := setEmptyNsMask(context, options); err != nil {
			return err
		}
		if err := container.Checkpoint(options); err != nil {
			return err
		}
		return nil
	},
}

func getCheckpointImagePath(context *cli.Context) string {
	imagePath := context.String("image-path")
	if imagePath == "" {
		imagePath = getDefaultImagePath(context)
	}
	return imagePath
}

func setPageServer(context *cli.Context, options *libcontainer.CriuOpts) {
	// xxx following criu opts are optional
	// The dump image can be sent to a criu page server
	if psOpt := context.String("page-server"); psOpt != "" {
		addressPort := strings.Split(psOpt, ":")
		if len(addressPort) != 2 {
			fatal(fmt.Errorf("Use --page-server ADDRESS:PORT to specify page server"))
		}
		portInt, err := strconv.Atoi(addressPort[1])
		if err != nil {
			fatal(fmt.Errorf("Invalid port number"))
		}
		options.PageServer = libcontainer.CriuPageServerInfo{
			Address: addressPort[0],
			Port:    int32(portInt),
		}
	}
}

func setManageCgroupsMode(context *cli.Context, options *libcontainer.CriuOpts) {
	if cgOpt := context.String("manage-cgroups-mode"); cgOpt != "" {
		switch cgOpt {
		case "soft":
			options.ManageCgroupsMode = libcontainer.CRIU_CG_MODE_SOFT
		case "full":
			options.ManageCgroupsMode = libcontainer.CRIU_CG_MODE_FULL
		case "strict":
			options.ManageCgroupsMode = libcontainer.CRIU_CG_MODE_STRICT
		default:
			fatal(fmt.Errorf("Invalid manage cgroups mode"))
		}
	}
}

var namespaceMapping = map[specs.LinuxNamespaceType]int{
	specs.NetworkNamespace: unix.CLONE_NEWNET,
}

func setEmptyNsMask(context *cli.Context, options *libcontainer.CriuOpts) error {
	var nsmask int

	for _, ns := range context.StringSlice("empty-ns") {
		f, exists := namespaceMapping[specs.LinuxNamespaceType(ns)]
		if !exists {
			return fmt.Errorf("namespace %q is not supported", ns)
		}
		nsmask |= f
	}

	options.EmptyNs = uint32(nsmask)
	return nil
}
