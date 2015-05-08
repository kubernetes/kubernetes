package main

import (
	"log"

	"github.com/codegangsta/cli"
)

var pauseCommand = cli.Command{
	Name:  "pause",
	Usage: "pause the container's processes",
	Flags: []cli.Flag{
		cli.StringFlag{Name: "id", Value: "nsinit", Usage: "specify the ID for a container"},
	},
	Action: func(context *cli.Context) {
		container, err := getContainer(context)
		if err != nil {
			log.Fatal(err)
		}
		if err = container.Pause(); err != nil {
			log.Fatal(err)
		}
	},
}

var unpauseCommand = cli.Command{
	Name:  "unpause",
	Usage: "unpause the container's processes",
	Flags: []cli.Flag{
		cli.StringFlag{Name: "id", Value: "nsinit", Usage: "specify the ID for a container"},
	},
	Action: func(context *cli.Context) {
		container, err := getContainer(context)
		if err != nil {
			log.Fatal(err)
		}
		if err = container.Resume(); err != nil {
			log.Fatal(err)
		}
	},
}
