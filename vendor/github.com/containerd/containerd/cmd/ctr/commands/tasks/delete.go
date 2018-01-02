package tasks

import (
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/urfave/cli"
)

var deleteCommand = cli.Command{
	Name:      "delete",
	Usage:     "delete a task",
	ArgsUsage: "CONTAINER",
	Action: func(context *cli.Context) error {
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		container, err := client.LoadContainer(ctx, context.Args().First())
		if err != nil {
			return err
		}
		task, err := container.Task(ctx, nil)
		if err != nil {
			return err
		}
		status, err := task.Delete(ctx)
		if err != nil {
			return err
		}
		if ec := status.ExitCode(); ec != 0 {
			return cli.NewExitError("", int(ec))
		}
		return nil
	},
}
