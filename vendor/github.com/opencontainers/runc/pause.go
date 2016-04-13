// +build linux

package main

import "github.com/codegangsta/cli"

var pauseCommand = cli.Command{
	Name:  "pause",
	Usage: "pause suspends all processes inside the container",
	Action: func(context *cli.Context) {
		container, err := getContainer(context)
		if err != nil {
			fatal(err)
		}
		if err := container.Pause(); err != nil {
			fatal(err)
		}
	},
}

var resumeCommand = cli.Command{
	Name:  "resume",
	Usage: "resume resumes all processes that have been previously paused",
	Action: func(context *cli.Context) {
		container, err := getContainer(context)
		if err != nil {
			fatal(err)
		}
		if err := container.Resume(); err != nil {
			fatal(err)
		}
	},
}
