// +build !solaris

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/opencontainers/runc/libcontainer"
	"github.com/urfave/cli"

	"golang.org/x/sys/unix"
)

func killContainer(container libcontainer.Container) error {
	_ = container.Signal(unix.SIGKILL, false)
	for i := 0; i < 100; i++ {
		time.Sleep(100 * time.Millisecond)
		if err := container.Signal(syscall.Signal(0), false); err != nil {
			destroy(container)
			return nil
		}
	}
	return fmt.Errorf("container init still running")
}

var deleteCommand = cli.Command{
	Name:  "delete",
	Usage: "delete any resources held by the container often used with detached container",
	ArgsUsage: `<container-id>

Where "<container-id>" is the name for the instance of the container.

EXAMPLE:
For example, if the container id is "ubuntu01" and runc list currently shows the
status of "ubuntu01" as "stopped" the following will delete resources held for
"ubuntu01" removing "ubuntu01" from the runc list of containers:

       # runc delete ubuntu01`,
	Flags: []cli.Flag{
		cli.BoolFlag{
			Name:  "force, f",
			Usage: "Forcibly deletes the container if it is still running (uses SIGKILL)",
		},
	},
	Action: func(context *cli.Context) error {
		if err := checkArgs(context, 1, exactArgs); err != nil {
			return err
		}

		id := context.Args().First()
		force := context.Bool("force")
		container, err := getContainer(context)
		if err != nil {
			if lerr, ok := err.(libcontainer.Error); ok && lerr.Code() == libcontainer.ContainerNotExists {
				// if there was an aborted start or something of the sort then the container's directory could exist but
				// libcontainer does not see it because the state.json file inside that directory was never created.
				path := filepath.Join(context.GlobalString("root"), id)
				if e := os.RemoveAll(path); e != nil {
					fmt.Fprintf(os.Stderr, "remove %s: %v\n", path, e)
				}
				if force {
					return nil
				}
			}
			return err
		}
		s, err := container.Status()
		if err != nil {
			return err
		}
		switch s {
		case libcontainer.Stopped:
			destroy(container)
		case libcontainer.Created:
			return killContainer(container)
		default:
			if force {
				return killContainer(container)
			} else {
				return fmt.Errorf("cannot delete container %s that is not stopped: %s\n", id, s)
			}
		}

		return nil
	},
}
