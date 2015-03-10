package main

import (
	"log"

	"github.com/codegangsta/cli"
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/docker/libcontainer/cgroups/systemd"
)

var pauseCommand = cli.Command{
	Name:   "pause",
	Usage:  "pause the container's processes",
	Action: pauseAction,
}

var unpauseCommand = cli.Command{
	Name:   "unpause",
	Usage:  "unpause the container's processes",
	Action: unpauseAction,
}

func pauseAction(context *cli.Context) {
	if err := toggle(cgroups.Frozen); err != nil {
		log.Fatal(err)
	}
}

func unpauseAction(context *cli.Context) {
	if err := toggle(cgroups.Thawed); err != nil {
		log.Fatal(err)
	}
}

func toggle(state cgroups.FreezerState) error {
	container, err := loadConfig()
	if err != nil {
		return err
	}

	if systemd.UseSystemd() {
		err = systemd.Freeze(container.Cgroups, state)
	} else {
		err = fs.Freeze(container.Cgroups, state)
	}

	return err
}
