package main

import (
	"fmt"
	"github.com/codegangsta/cli"
	"github.com/docker/libcontainer"
	"strconv"
	"strings"
)

var checkpointCommand = cli.Command{
	Name:  "checkpoint",
	Usage: "checkpoint a running container",
	Flags: []cli.Flag{
		cli.StringFlag{Name: "id", Value: "nsinit", Usage: "specify the ID for a container"},
		cli.StringFlag{Name: "image-path", Value: "", Usage: "path for saving criu image files"},
		cli.StringFlag{Name: "work-path", Value: "", Usage: "path for saving work files and logs"},
		cli.BoolFlag{Name: "leave-running", Usage: "leave the process running after checkpointing"},
		cli.BoolFlag{Name: "tcp-established", Usage: "allow open tcp connections"},
		cli.BoolFlag{Name: "ext-unix-sk", Usage: "allow external unix sockets"},
		cli.BoolFlag{Name: "shell-job", Usage: "allow shell jobs"},
		cli.StringFlag{Name: "page-server", Value: "", Usage: "ADDRESS:PORT of the page server"},
	},
	Action: func(context *cli.Context) {
		imagePath := context.String("image-path")
		if imagePath == "" {
			fatal(fmt.Errorf("The --image-path option isn't specified"))
		}

		container, err := getContainer(context)
		if err != nil {
			fatal(err)
		}

		// these are the mandatory criu options for a container
		criuOpts := &libcontainer.CriuOpts{
			ImagesDirectory:         imagePath,
			WorkDirectory:           context.String("work-path"),
			LeaveRunning:            context.Bool("leave-running"),
			TcpEstablished:          context.Bool("tcp-established"),
			ExternalUnixConnections: context.Bool("ext-unix-sk"),
			ShellJob:                context.Bool("shell-job"),
		}

		// xxx following criu opts are optional
		// The dump image can be sent to a criu page server
		if psOpt := context.String("page-server"); psOpt != "" {
			addressPort := strings.Split(psOpt, ":")
			if len(addressPort) != 2 {
				fatal(fmt.Errorf("Use --page-server ADDRESS:PORT to specify page server"))
			}

			port_int, err := strconv.Atoi(addressPort[1])
			if err != nil {
				fatal(fmt.Errorf("Invalid port number"))
			}
			criuOpts.PageServer = libcontainer.CriuPageServerInfo{
				Address: addressPort[0],
				Port:    int32(port_int),
			}
		}

		if err := container.Checkpoint(criuOpts); err != nil {
			fatal(err)
		}
	},
}
