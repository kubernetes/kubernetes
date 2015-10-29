package main

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"

	"github.com/codegangsta/cli"
	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/utils"
)

var restoreCommand = cli.Command{
	Name:  "restore",
	Usage: "restore a container from a previous checkpoint",
	Flags: []cli.Flag{
		cli.StringFlag{Name: "id", Value: "nsinit", Usage: "specify the ID for a container"},
		cli.StringFlag{Name: "image-path", Value: "", Usage: "path to criu image files for restoring"},
		cli.StringFlag{Name: "work-path", Value: "", Usage: "path for saving work files and logs"},
		cli.BoolFlag{Name: "tcp-established", Usage: "allow open tcp connections"},
		cli.BoolFlag{Name: "ext-unix-sk", Usage: "allow external unix sockets"},
		cli.BoolFlag{Name: "shell-job", Usage: "allow shell jobs"},
	},
	Action: func(context *cli.Context) {
		imagePath := context.String("image-path")
		if imagePath == "" {
			fatal(fmt.Errorf("The --image-path option isn't specified"))
		}

		var (
			container libcontainer.Container
			err       error
		)

		factory, err := loadFactory(context)
		if err != nil {
			fatal(err)
		}

		config, err := loadConfig(context)
		if err != nil {
			fatal(err)
		}

		created := false
		container, err = factory.Load(context.String("id"))
		if err != nil {
			created = true
			if container, err = factory.Create(context.String("id"), config); err != nil {
				fatal(err)
			}
		}

		process := &libcontainer.Process{
			Stdin:  os.Stdin,
			Stdout: os.Stdout,
			Stderr: os.Stderr,
		}
		//rootuid, err := config.HostUID()
		//if err != nil {
		//fatal(err)
		//}
		rootuid := 0 // XXX
		tty, err := newTty(context, process, rootuid)
		if err != nil {
			fatal(err)
		}
		if err := tty.attach(process); err != nil {
			fatal(err)
		}
		go handleSignals(process, tty)

		err = container.Restore(process, &libcontainer.CriuOpts{
			ImagesDirectory:         imagePath,
			WorkDirectory:           context.String("work-path"),
			TcpEstablished:          context.Bool("tcp-established"),
			ExternalUnixConnections: context.Bool("ext-unix-sk"),
			ShellJob:                context.Bool("shell-job"),
		})
		if err != nil {
			tty.Close()
			if created {
				container.Destroy()
			}
			fatal(err)
		}

		status, err := process.Wait()
		if err != nil {
			exitError, ok := err.(*exec.ExitError)
			if ok {
				status = exitError.ProcessState
			} else {
				tty.Close()
				if created {
					container.Destroy()
				}
				fatal(err)
			}
		}

		if created {
			status, err := container.Status()
			if err != nil {
				tty.Close()
				fatal(err)
			}
			if status != libcontainer.Checkpointed {
				if err := container.Destroy(); err != nil {
					tty.Close()
					fatal(err)
				}
			}
		}

		tty.Close()
		os.Exit(utils.ExitStatus(status.Sys().(syscall.WaitStatus)))
	},
}
