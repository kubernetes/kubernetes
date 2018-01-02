package tasks

import (
	gocontext "context"

	"github.com/urfave/cli"
)

type resizer interface {
	Resize(ctx gocontext.Context, w, h uint32) error
}

// Command is the cli command for managing tasks
var Command = cli.Command{
	Name:    "tasks",
	Usage:   "manage tasks",
	Aliases: []string{"t"},
	Subcommands: []cli.Command{
		attachCommand,
		checkpointCommand,
		deleteCommand,
		execCommand,
		listCommand,
		killCommand,
		pauseCommand,
		psCommand,
		resumeCommand,
		startCommand,
	},
}
