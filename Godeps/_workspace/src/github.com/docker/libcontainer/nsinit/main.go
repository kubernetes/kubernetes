package main

import (
	"log"
	"os"
	"strings"

	"github.com/codegangsta/cli"
)

var (
	logPath = os.Getenv("log")
	argvs   = make(map[string]*rFunc)
)

func init() {
	argvs["exec"] = &rFunc{
		Usage:  "execute a process inside an existing container",
		Action: nsenterExec,
	}

	argvs["mknod"] = &rFunc{
		Usage:  "mknod a device inside an existing container",
		Action: nsenterMknod,
	}

	argvs["ip"] = &rFunc{
		Usage:  "display the container's network interfaces",
		Action: nsenterIp,
	}
}

func main() {
	// we need to check our argv 0 for any registred functions to run instead of the
	// normal cli code path
	f, exists := argvs[strings.TrimPrefix(os.Args[0], "nsenter-")]
	if exists {
		runFunc(f)

		return
	}

	app := cli.NewApp()

	app.Name = "nsinit"
	app.Version = "0.1"
	app.Author = "libcontainer maintainers"
	app.Flags = []cli.Flag{
		cli.StringFlag{Name: "nspid"},
		cli.StringFlag{Name: "console"},
	}

	app.Before = preload

	app.Commands = []cli.Command{
		execCommand,
		initCommand,
		statsCommand,
		configCommand,
		pauseCommand,
		unpauseCommand,
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
